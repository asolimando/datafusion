// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`DFParquetMetadata`] for fetching Parquet file metadata, statistics
//! and schema information.

use crate::file_format::ObjectStoreFetch;
use crate::{Int96Coercer, apply_file_schema_type_coercions};
use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::compute::kernels::cmp::{eq, lt_eq};
use arrow::compute::{and, sum};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion_common::encryption::FileDecryptionProperties;
use datafusion_common::stats::Precision;
use datafusion_common::{
    ColumnStatistics, DataFusionError, HashMap, HashSet, Result, ScalarValue, Statistics,
    internal_datafusion_err,
};
use datafusion_execution::cache::cache_manager::{
    CachedFileMetadataEntry, FileMetadata, FileMetadataCache,
};
use datafusion_functions_aggregate_common::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::Accumulator;
use log::debug;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::{parquet_column, parquet_to_arrow_schema};
use parquet::basic::{ColumnOrder, Compression, SortOrder, Type as PhysicalType};
use parquet::file::metadata::{
    ColumnChunkMetaData, PageIndexPolicy, ParquetMetaData, ParquetMetaDataPushDecoder,
    ParquetMetaDataReader, RowGroupMetaData, SortingColumn,
};
use parquet::file::statistics::Statistics as ParquetStatistics;
use parquet::schema::types::{ColumnDescriptor, SchemaDescriptor};
use std::any::Any;
use std::f64::consts::LN_2;
use std::sync::Arc;

/// Minimum fraction of row groups that must report NDV statistics for the
/// merged result to be `Inexact` rather than `Absent`, as the estimate
/// would be too unreliable otherwise.
const PARTIAL_NDV_THRESHOLD: f64 = 0.75;

fn requires_unsigned_byte_array_order(column: &ColumnDescriptor) -> bool {
    matches!(
        column.physical_type(),
        PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY
    ) && column.sort_order() != SortOrder::SIGNED
}

/// Whether a column's min/max bounds lack a comparison order matching Arrow's.
///
/// The deprecated Parquet `min`/`max` fields use signed comparison, unlike
/// Arrow's string and binary comparisons. Even the modern bounds cannot be
/// interpreted without the corresponding footer `column_orders` entry.
/// Signed logical types, such as decimals, retain their existing behavior.
/// Columns with undefined sort orders, such as `INT96`, never have usable
/// min/max bounds regardless of their physical type. The `INT96` check is
/// defensive because parquet-rs does not currently expose those bounds.
pub(crate) fn has_untrusted_min_max_order(
    parquet_schema: &SchemaDescriptor,
    column_orders: Option<&[ColumnOrder]>,
    parquet_column_index: usize,
) -> bool {
    let column = parquet_schema.column(parquet_column_index);
    // As of arrow 60, INT96 columns report `SortOrder::INT96_TIMESTAMP`
    // rather than `UNDEFINED`; keep treating their min/max as untrusted.
    // until <https://github.com/apache/datafusion/issues/25484>
    if matches!(
        column.sort_order(),
        SortOrder::UNDEFINED | SortOrder::INT96_TIMESTAMP
    ) {
        return true;
    }
    requires_unsigned_byte_array_order(&column)
        && (column.sort_order() != SortOrder::UNSIGNED
            || column_orders
                .and_then(|orders| orders.get(parquet_column_index))
                .copied()
                != Some(ColumnOrder::TYPE_DEFINED_ORDER(SortOrder::UNSIGNED)))
}

/// Whether any row group provides byte-array bounds in the deprecated signed
/// order, or the column's logical order is undefined.
pub(crate) fn has_untrusted_byte_array_stats<'a>(
    parquet_schema: &SchemaDescriptor,
    parquet_column_index: Option<usize>,
    row_groups: impl IntoIterator<Item = &'a RowGroupMetaData>,
) -> bool {
    parquet_column_index.is_some_and(|index| {
        let column = parquet_schema.column(index);
        requires_unsigned_byte_array_order(&column)
            && (column.sort_order() != SortOrder::UNSIGNED
                || row_groups.into_iter().any(|group| {
                    group.column(index).statistics().is_some_and(|stats| {
                        stats.is_min_max_deprecated()
                            && (stats.min_bytes_opt().is_some()
                                || stats.max_bytes_opt().is_some())
                    })
                }))
    })
}

/// Handles fetching Parquet file schema, metadata and statistics
/// from object store.
///
/// This component is exposed for low level integrations through
/// [`ParquetFileReaderFactory`].
///
/// [`ParquetFileReaderFactory`]: crate::ParquetFileReaderFactory
#[derive(Debug)]
pub struct DFParquetMetadata<'a> {
    /// Source of the Parquet file's bytes.
    store: &'a dyn ObjectStore,
    /// Location, size and last-modified time of the target Parquet file.
    object_meta: &'a ObjectMeta,
    /// Hint for the number of trailing bytes to prefetch before parsing the
    /// footer, mirroring [`ParquetMetaDataReader::with_prefetch_hint`].
    metadata_size_hint: Option<usize>,
    /// Decryption properties used to read files encrypted with Parquet
    /// Modular Encryption, mirroring
    /// [`ParquetMetaDataReader::with_decryption_properties`].
    decryption_properties: Option<Arc<FileDecryptionProperties>>,
    /// Optional cache of previously fetched [`ParquetMetaData`], keyed by
    /// file location.
    file_metadata_cache: Option<Arc<FileMetadataCache>>,
    /// Policy controlling whether the Parquet page index (column and offset
    /// indexes) is fetched, mirroring
    /// [`ParquetMetaDataReader::with_page_index_policy`].
    ///
    /// `None` means the effective policy is chosen automatically, see
    /// [`DFParquetMetadata::effective_page_index_policy`].
    page_index_policy: Option<PageIndexPolicy>,
    /// timeunit to coerce INT96 timestamps to
    pub coerce_int96: Option<TimeUnit>,
    /// Optional timezone applied to INT96-coerced timestamps.
    pub coerce_int96_tz: Option<Arc<str>>,
    /// If true, promote string/binary columns with dictionary pages to `Dictionary(Int32, ...)`.
    enable_rle_to_dictionary: bool,
    /// If true, estimate the distinct count of columns that do not carry a written
    /// `distinct_count` from dictionary page sizes and row group min/max statistics.
    estimate_distinct_count: bool,
}

impl<'a> DFParquetMetadata<'a> {
    /// Create a new `DFParquetMetadata` for the given file.
    ///
    /// Use the `with_*` builder methods to customize behavior
    /// before calling [`Self::fetch_metadata`] or [`Self::fetch_schema`].
    pub fn new(store: &'a dyn ObjectStore, object_meta: &'a ObjectMeta) -> Self {
        Self {
            store,
            object_meta,
            metadata_size_hint: None,
            decryption_properties: None,
            file_metadata_cache: None,
            page_index_policy: None,
            coerce_int96: None,
            coerce_int96_tz: None,
            enable_rle_to_dictionary: false,
            estimate_distinct_count: false,
        }
    }

    /// Estimate the distinct count of columns that do not carry a written
    /// `distinct_count`, using dictionary page sizes and row group min/max
    /// statistics. The estimate is always [`Precision::Inexact`].
    pub fn with_estimate_distinct_count(mut self, estimate: bool) -> Self {
        self.estimate_distinct_count = estimate;
        self
    }

    /// Promote string/binary columns with dictionary pages to `Dictionary(Int32, ...)`.
    pub fn with_enable_rle_to_dictionary(mut self, enable: bool) -> Self {
        self.enable_rle_to_dictionary = enable;
        self
    }

    /// Set a hint for the number of trailing bytes to prefetch from the end
    /// of the file, equivalent to
    /// [`ParquetMetaDataReader::with_prefetch_hint`].
    ///
    /// Providing a good estimate of the footer (and, if requested, page index)
    /// size can save an extra I/O round trip when fetching metadata from the
    /// store.
    pub fn with_metadata_size_hint(mut self, metadata_size_hint: Option<usize>) -> Self {
        self.metadata_size_hint = metadata_size_hint;
        self
    }

    /// Set the decryption properties used to read an encrypted Parquet file,
    /// equivalent to [`ParquetMetaDataReader::with_decryption_properties`].
    ///
    /// Only needed when the target file was written with Parquet Modular
    /// Encryption.
    pub fn with_decryption_properties(
        mut self,
        decryption_properties: Option<Arc<FileDecryptionProperties>>,
    ) -> Self {
        self.decryption_properties = decryption_properties;
        self
    }

    /// Set an optional [`FileMetadataCache`] used to avoid re-fetching
    /// [`ParquetMetaData`] for files that have already been read.
    pub fn with_file_metadata_cache(
        mut self,
        file_metadata_cache: Option<Arc<FileMetadataCache>>,
    ) -> Self {
        self.file_metadata_cache = file_metadata_cache;
        self
    }

    /// Sets the policy for loading parquet page index structures (column and
    /// offset indexes), equivalent to
    /// [`ParquetMetaDataReader::with_page_index_policy`].
    ///
    /// Passing `None` uses a default automatically, based on whether a metadata
    /// cache is configured.
    pub fn with_page_index_policy(
        mut self,
        page_index_policy: Option<PageIndexPolicy>,
    ) -> Self {
        self.page_index_policy = page_index_policy;
        self
    }

    /// Set the [`TimeUnit`] that INT96 timestamp columns should be coerced
    /// to when reading the schema.
    ///
    /// INT96 in Parquet has no defined unit or timezone, so leaving this
    /// `None` reads INT96 columns as nanosecond timestamps with no timezone
    /// — DataFusion's default behavior.
    pub fn with_coerce_int96(mut self, time_unit: Option<TimeUnit>) -> Self {
        self.coerce_int96 = time_unit;
        self
    }

    /// Set the optional timezone applied to INT96-coerced timestamps.
    ///
    /// Only used when [`Self::with_coerce_int96`] has also been set, and
    /// otherwise has no effect.
    pub fn with_coerce_int96_tz(mut self, timezone: Option<Arc<str>>) -> Self {
        self.coerce_int96_tz = timezone;
        self
    }

    /// Fetch the [`ParquetMetaData`] for this file.
    ///
    /// Consults the [`FileMetadataCache`] first when one is configured and
    /// falls back to reading from the object store via
    /// [`ParquetMetaDataPushDecoder`] on a cache miss.
    pub async fn fetch_metadata(&self) -> Result<Arc<ParquetMetaData>> {
        // fetch_metadata
        // │
        // ├─ cache_metadata = encryption check
        // ├─ page_index_policy = caller override OR default
        // │
        // ├─ CACHE HIT?
        // │   │
        // │   ├─ has index OR policy=Skip?  → return cache
        // │   │
        // │   └─ else (footer only, wants index)
        // │         → load_page_index (index bytes only)
        // │         → cache_metadata() if allowed
        // │         → return
        // │
        // └─ CACHE MISS
        //       → fetch_metadata_from_store(policy)
        //       → cache_metadata() if allowed
        //       → return
        let cache_metadata =
            !cfg!(feature = "parquet_encryption") || self.decryption_properties.is_none();
        let page_index_policy = self.effective_page_index_policy(cache_metadata);

        if cache_metadata
            && let Some(file_metadata_cache) = self.file_metadata_cache.as_ref()
            && let Some(cached) = file_metadata_cache.get(&self.object_meta.location)
            && cached.is_valid_for(self.object_meta)
            && let Some(cached_parquet) = cached
                .file_metadata
                .as_any()
                .downcast_ref::<CachedParquetMetaData>()
        {
            let cached_metadata = Arc::clone(cached_parquet.parquet_metadata());
            // Reuse the cache when it already has page index, or when the caller
            // asked to skip page index I/O (footer-only metadata is sufficient).
            if Self::metadata_has_page_index(cached_metadata.as_ref())
                || page_index_policy == PageIndexPolicy::Skip
            {
                return Ok(cached_metadata);
            }
            let metadata =
                Self::load_page_index(self.store, self.object_meta, cached_metadata)
                    .await?;
            if cache_metadata {
                self.cache_metadata(Arc::clone(&metadata))?;
            }
            return Ok(metadata);
        }

        let metadata = self.fetch_metadata_from_store(page_index_policy).await?;
        if cache_metadata {
            self.cache_metadata(Arc::clone(&metadata))?;
        }
        Ok(metadata)
    }

    /// Resolve the [`PageIndexPolicy`] to use for a fetch.
    fn effective_page_index_policy(&self, cache_metadata: bool) -> PageIndexPolicy {
        self.page_index_policy.unwrap_or_else(|| {
            // fetching the page index often requires a second IO (after the
            // main metadata), so it is not free.
            if cache_metadata && self.file_metadata_cache.is_some() {
                // When there is a cache available, retrieve the page index
                // heuristically on the assumption it will be used multiple
                // times
                PageIndexPolicy::Optional
            } else {
                PageIndexPolicy::Skip
            }
        })
    }

    /// Check whether `metadata` already has both the column index and the
    /// offset index populated (see [`ParquetMetaData::page_index`]).
    ///
    /// Used to decide whether page index I/O can be skipped.
    fn metadata_has_page_index(metadata: &ParquetMetaData) -> bool {
        metadata
            .page_index()
            .is_some_and(|page_index| page_index.is_complete())
    }

    /// Store `metadata` in the configured [`FileMetadataCache`], keyed by
    /// the file's location.
    ///
    /// This is a no-op unless a cache has been configured via
    /// [`Self::with_file_metadata_cache`].
    fn cache_metadata(&self, metadata: Arc<ParquetMetaData>) -> Result<()> {
        if let Some(file_metadata_cache) = &self.file_metadata_cache {
            file_metadata_cache.put(
                &self.object_meta.location,
                CachedFileMetadataEntry::new(
                    self.object_meta.clone(),
                    Arc::new(CachedParquetMetaData::new(metadata)),
                ),
            );
        }
        Ok(())
    }

    /// Fetch the full [`ParquetMetaData`] (including footer, and optional
    /// page index) from the object store.
    async fn fetch_metadata_from_store(
        &self,
        page_index_policy: PageIndexPolicy,
    ) -> Result<Arc<ParquetMetaData>> {
        let file_size = self.object_meta.size;
        let mut decoder = ParquetMetaDataPushDecoder::try_new(file_size)
            .map_err(DataFusionError::from)?;

        #[cfg(feature = "parquet_encryption")]
        if let Some(decryption_properties) = &self.decryption_properties {
            decoder = decoder
                .with_file_decryption_properties(Some(Arc::clone(decryption_properties)));
        }

        decoder = decoder.with_page_index_policy(page_index_policy);

        if let Some(hint) = self.metadata_size_hint {
            let prefetch_start = file_size.saturating_sub(hint as u64);
            let prefetch_range = prefetch_start..file_size;
            let data = self
                .store
                .get_ranges(
                    &self.object_meta.location,
                    std::slice::from_ref(&prefetch_range),
                )
                .await
                .map_err(DataFusionError::from)?;
            decoder
                .push_ranges(vec![prefetch_range], data)
                .map_err(DataFusionError::from)?;
        }

        let metadata = loop {
            match decoder.try_decode().map_err(DataFusionError::from)? {
                DecodeResult::Data(metadata) => break metadata,
                DecodeResult::NeedsData(ranges) => {
                    let buffers = self
                        .store
                        .get_ranges(&self.object_meta.location, &ranges)
                        .await
                        .map_err(DataFusionError::from)?;
                    decoder
                        .push_ranges(ranges, buffers)
                        .map_err(DataFusionError::from)?;
                }
                DecodeResult::Finished => {
                    return Err(DataFusionError::Internal(
                        "ParquetMetaDataPushDecoder finished without producing metadata"
                            .to_string(),
                    ));
                }
            }
        };

        Ok(Arc::new(metadata))
    }

    /// If `metadata` does not already have a page index, fetch and attach the
    /// column and offset indexes.
    async fn load_page_index(
        store: &dyn ObjectStore,
        object_meta: &ObjectMeta,
        metadata: Arc<ParquetMetaData>,
    ) -> Result<Arc<ParquetMetaData>> {
        if metadata
            .page_index()
            .is_some_and(|page_index| page_index.is_complete())
        {
            return Ok(metadata);
        }
        let metadata =
            Arc::try_unwrap(metadata).unwrap_or_else(|shared| (*shared).clone());
        let mut reader = ParquetMetaDataReader::new_with_metadata(metadata)
            .with_page_index_policy(PageIndexPolicy::Optional);
        let fetch = ObjectStoreFetch::new(store, object_meta);
        reader
            .load_page_index(fetch)
            .await
            .map_err(DataFusionError::from)?;
        Ok(Arc::new(reader.finish().map_err(DataFusionError::from)?))
    }

    /// Fetch this file's [`ParquetMetaData`] and convert its embedded Thrift
    /// schema into an Arrow [`Schema`].
    pub async fn fetch_schema(&self) -> Result<Schema> {
        let metadata = self.fetch_metadata().await?;

        let file_metadata = metadata.file_metadata();
        let schema = parquet_to_arrow_schema(
            file_metadata.schema_descr(),
            file_metadata.key_value_metadata(),
        )?;
        let schema = self
            .coerce_int96
            .as_ref()
            .and_then(|time_unit| {
                Int96Coercer::new(file_metadata.schema_descr(), &schema, time_unit)
                    .with_timezone(self.coerce_int96_tz.clone())
                    .coerce()
            })
            .unwrap_or(schema);

        let schema = if self.enable_rle_to_dictionary {
            let schema_descr = file_metadata.schema_descr();
            // Top-level columns that have a dictionary page in at least one row group.
            let dict_cols: HashSet<String> = metadata
                .row_groups()
                .iter()
                .flat_map(|rg| {
                    rg.columns()
                        .iter()
                        .enumerate()
                        .filter_map(|(col_idx, col)| {
                            col.dictionary_page_offset()?;
                            let col_desc = schema_descr.column(col_idx);
                            let parts = col_desc.path().parts();
                            // Skip nested columns: their leaf name doesn't match the
                            // Arrow top-level field name.
                            (parts.len() == 1).then(|| parts[0].clone())
                        })
                })
                .collect();
            if dict_cols.is_empty() {
                schema
            } else {
                let promoted: Vec<_> = schema
                    .fields()
                    .iter()
                    .map(|field| {
                        if !dict_cols.contains(field.name()) {
                            return Arc::clone(field);
                        }
                        let dict_value_type = match field.data_type() {
                            DataType::Utf8 => Some(DataType::Utf8),
                            DataType::LargeUtf8 => Some(DataType::LargeUtf8),
                            DataType::Binary => Some(DataType::Binary),
                            DataType::LargeBinary => Some(DataType::LargeBinary),
                            _ => None,
                        };
                        dict_value_type.map_or_else(
                            || Arc::clone(field),
                            |value_type| {
                                Arc::new(
                                    Field::new(
                                        field.name(),
                                        DataType::Dictionary(
                                            Box::new(DataType::Int32),
                                            Box::new(value_type),
                                        ),
                                        field.is_nullable(),
                                    )
                                    .with_metadata(field.metadata().clone()),
                                )
                            },
                        )
                    })
                    .collect();
                Schema::new_with_metadata(promoted, schema.metadata().clone())
            }
        } else {
            schema
        };

        Ok(schema)
    }

    /// Convenience wrapper around [`Self::fetch_schema`] that also returns
    /// the file's object store [`Path`].
    pub(crate) async fn fetch_schema_with_location(&self) -> Result<(Path, Schema)> {
        let loc_path = self.object_meta.location.clone();
        let schema = self.fetch_schema().await?;
        Ok((loc_path, schema))
    }

    /// Fetch the metadata from the Parquet file via [`Self::fetch_metadata`] and convert
    /// the statistics in the metadata using [`Self::statistics_from_parquet_metadata`]
    pub async fn fetch_statistics(&self, table_schema: &SchemaRef) -> Result<Statistics> {
        let metadata = self.fetch_metadata().await?;
        Self::statistics_from_parquet_metadata_with_options(
            &metadata,
            table_schema,
            self.estimate_distinct_count,
        )
    }

    /// Convert statistics in [`ParquetMetaData`] into [`Statistics`] using [`StatisticsConverter`]
    ///
    /// The statistics are calculated for each column in the table schema
    /// using the row group statistics in the parquet metadata.
    ///
    /// # Key behaviors:
    ///
    /// 1. Extracts row counts and byte sizes from all row groups
    /// 2. Applies schema type coercions to align file schema with table schema
    /// 3. Collects and aggregates statistics across row groups when available
    ///
    /// # When there are no statistics:
    ///
    /// If the Parquet file doesn't contain any statistics (has_statistics is false), the function returns a Statistics object with:
    /// - Exact row count
    /// - Exact byte size
    /// - All column statistics marked as unknown via Statistics::unknown_column(&table_schema)
    /// - Column byte sizes are still calculated and recorded
    ///
    /// # When only some columns have statistics:
    ///
    /// For columns with statistics:
    /// - Min/max values are properly extracted and represented as Precision::Exact
    /// - Null counts are calculated by summing across row groups
    /// - Byte sizes are calculated and recorded
    ///
    /// For columns without statistics,
    /// - For min/max, there are two situations:
    ///     1. The column isn't in arrow schema, then min/max values are set to Precision::Absent
    ///     2. The column is in arrow schema, but not in parquet schema due to schema revolution, min/max values are set to Precision::Exact(null)
    /// - Null counts are set to Precision::Exact(num_rows) (conservatively assuming all values could be null)
    ///
    /// # Byte Size Calculation:
    ///
    /// - For primitive types with known fixed size, exact byte size is calculated as (byte width * number of rows)
    /// - For other types, uncompressed Parquet size is used as an estimate for in-memory size
    /// - If neither method is applicable, byte size is marked as Precision::Absent
    pub fn statistics_from_parquet_metadata(
        metadata: &ParquetMetaData,
        logical_file_schema: &SchemaRef,
    ) -> Result<Statistics> {
        Self::statistics_from_parquet_metadata_with_options(
            metadata,
            logical_file_schema,
            false,
        )
    }

    /// Same as [`Self::statistics_from_parquet_metadata`]. When
    /// `estimate_distinct_count` is true, columns without a written
    /// `distinct_count` get an inexact estimate, see
    /// [`estimate_distinct_count_from_metadata`].
    pub(crate) fn statistics_from_parquet_metadata_with_options(
        metadata: &ParquetMetaData,
        logical_file_schema: &SchemaRef,
        estimate_distinct_count: bool,
    ) -> Result<Statistics> {
        let row_groups_metadata = metadata.row_groups();

        // Use Statistics::default() as opposed to Statistics::new_unknown()
        // because we are going to replace the column statistics below
        // and we don't want to initialize them twice.
        let mut statistics = Statistics::default();
        let mut has_statistics = false;
        let mut num_rows = 0_usize;
        for row_group_meta in row_groups_metadata {
            num_rows += row_group_meta.num_rows() as usize;

            if !has_statistics {
                has_statistics = row_group_meta
                    .columns()
                    .iter()
                    .any(|column| column.statistics().is_some());
            }
        }
        statistics.num_rows = Precision::Exact(num_rows);

        let file_metadata = metadata.file_metadata();
        let mut physical_file_schema = parquet_to_arrow_schema(
            file_metadata.schema_descr(),
            file_metadata.key_value_metadata(),
        )?;

        if let Some(merged) =
            apply_file_schema_type_coercions(logical_file_schema, &physical_file_schema)
        {
            physical_file_schema = merged;
        }

        statistics.column_statistics =
            if has_statistics {
                let (mut max_accs, mut min_accs) =
                    create_max_min_accs(logical_file_schema);
                let mut null_counts_array =
                    vec![Precision::Absent; logical_file_schema.fields().len()];
                let mut column_byte_sizes =
                    vec![Precision::Absent; logical_file_schema.fields().len()];
                let mut is_max_value_exact =
                    vec![Some(true); logical_file_schema.fields().len()];
                let mut is_min_value_exact =
                    vec![Some(true); logical_file_schema.fields().len()];
                let mut distinct_counts_array =
                    vec![Precision::Absent; logical_file_schema.fields().len()];
                logical_file_schema.fields().iter().enumerate().for_each(
                    |(idx, field)| match StatisticsConverter::try_new(
                        field.name(),
                        &physical_file_schema,
                        file_metadata.schema_descr(),
                    ) {
                        Ok(stats_converter) => {
                            // An omitted count must not become an exact zero in
                            // file statistics used for pruning and aggregates.
                            let stats_converter =
                                stats_converter.with_missing_null_counts_as_zero(false);
                            let parquet_index = stats_converter.parquet_column_index();
                            if parquet_index.is_some_and(|index| {
                                has_untrusted_min_max_order(
                                    file_metadata.schema_descr(),
                                    file_metadata.column_orders().map(Vec::as_slice),
                                    index,
                                )
                            }) || has_untrusted_byte_array_stats(
                                file_metadata.schema_descr(),
                                parquet_index,
                                row_groups_metadata,
                            ) {
                                // The remaining row groups cannot establish bounds
                                // for the whole file. Keep unrelated statistics.
                                min_accs[idx] = None;
                                max_accs[idx] = None;
                            }
                            let mut accumulators = StatisticsAccumulators {
                                min_accs: &mut min_accs,
                                max_accs: &mut max_accs,
                                null_counts_array: &mut null_counts_array,
                                is_min_value_exact: &mut is_min_value_exact,
                                is_max_value_exact: &mut is_max_value_exact,
                                column_byte_sizes: &mut column_byte_sizes,
                                distinct_counts_array: &mut distinct_counts_array,
                            };
                            summarize_column_statistics(
                                logical_file_schema,
                                &mut accumulators,
                                idx,
                                &stats_converter,
                                row_groups_metadata,
                                num_rows,
                                estimate_distinct_count,
                            )
                            .ok();
                        }
                        Err(e) => {
                            debug!("Failed to create statistics converter: {e}");
                            null_counts_array[idx] = Precision::Exact(num_rows);
                        }
                    },
                );

                let mut accumulators = StatisticsAccumulators {
                    min_accs: &mut min_accs,
                    max_accs: &mut max_accs,
                    null_counts_array: &mut null_counts_array,
                    is_min_value_exact: &mut is_min_value_exact,
                    is_max_value_exact: &mut is_max_value_exact,
                    column_byte_sizes: &mut column_byte_sizes,
                    distinct_counts_array: &mut distinct_counts_array,
                };
                accumulators.build_column_statistics(logical_file_schema)
            } else {
                // Record column sizes
                logical_file_schema
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(logical_file_schema_index, field)| {
                        let arrow_field =
                            logical_file_schema.field(logical_file_schema_index);
                        let parquet_idx = parquet_column(
                            file_metadata.schema_descr(),
                            &physical_file_schema,
                            arrow_field.name(),
                        )
                        .map(|(idx, _)| idx);
                        let byte_size = compute_arrow_column_size(
                            field.data_type(),
                            row_groups_metadata,
                            parquet_idx,
                            num_rows,
                        );
                        ColumnStatistics::new_unknown().with_byte_size(byte_size)
                    })
                    .collect()
            };

        #[cfg(debug_assertions)]
        {
            // Check that the column statistics length matches the table schema fields length
            assert_eq!(
                statistics.column_statistics.len(),
                logical_file_schema.fields().len(),
                "Column statistics length does not match table schema fields length"
            );
        }

        Ok(statistics)
    }
}

/// Min/max aggregation can take Dictionary encode input but always produces unpacked
/// (aka non Dictionary) output. We need to adjust the output data type to reflect this.
/// The reason min/max aggregate produces unpacked output because there is only one
/// min/max value per group; there is no needs to keep them Dictionary encoded
fn min_max_aggregate_data_type(input_type: &DataType) -> &DataType {
    if let DataType::Dictionary(_, value_type) = input_type {
        value_type.as_ref()
    } else {
        input_type
    }
}

fn create_max_min_accs(
    schema: &Schema,
) -> (Vec<Option<MaxAccumulator>>, Vec<Option<MinAccumulator>>) {
    let max_values: Vec<Option<MaxAccumulator>> = schema
        .fields()
        .iter()
        .map(|field| {
            MaxAccumulator::try_new(min_max_aggregate_data_type(field.data_type())).ok()
        })
        .collect();
    let min_values: Vec<Option<MinAccumulator>> = schema
        .fields()
        .iter()
        .map(|field| {
            MinAccumulator::try_new(min_max_aggregate_data_type(field.data_type())).ok()
        })
        .collect();
    (max_values, min_values)
}

/// Holds the accumulator state for collecting statistics from row groups
struct StatisticsAccumulators<'a> {
    min_accs: &'a mut [Option<MinAccumulator>],
    max_accs: &'a mut [Option<MaxAccumulator>],
    null_counts_array: &'a mut [Precision<usize>],
    is_min_value_exact: &'a mut [Option<bool>],
    is_max_value_exact: &'a mut [Option<bool>],
    column_byte_sizes: &'a mut [Precision<usize>],
    distinct_counts_array: &'a mut [Precision<usize>],
}

impl StatisticsAccumulators<'_> {
    /// Converts the accumulated statistics into a vector of `ColumnStatistics`
    fn build_column_statistics(&mut self, schema: &Schema) -> Vec<ColumnStatistics> {
        (0..schema.fields().len())
            .map(|i| {
                let max_value = match (
                    self.max_accs.get_mut(i).unwrap(),
                    self.is_max_value_exact.get(i).unwrap(),
                ) {
                    (Some(max_value), Some(true)) => {
                        max_value.evaluate().ok().map(Precision::Exact)
                    }
                    (Some(max_value), Some(false)) | (Some(max_value), None) => {
                        max_value.evaluate().ok().map(Precision::Inexact)
                    }
                    (None, _) => None,
                };
                let min_value = match (
                    self.min_accs.get_mut(i).unwrap(),
                    self.is_min_value_exact.get(i).unwrap(),
                ) {
                    (Some(min_value), Some(true)) => {
                        min_value.evaluate().ok().map(Precision::Exact)
                    }
                    (Some(min_value), Some(false)) | (Some(min_value), None) => {
                        min_value.evaluate().ok().map(Precision::Inexact)
                    }
                    (None, _) => None,
                };
                ColumnStatistics {
                    null_count: self.null_counts_array[i],
                    max_value: max_value.unwrap_or(Precision::Absent),
                    min_value: min_value.unwrap_or(Precision::Absent),
                    sum_value: Precision::Absent,
                    distinct_count: self.distinct_counts_array[i],
                    byte_size: self.column_byte_sizes[i],
                }
            })
            .collect()
    }
}

fn summarize_column_statistics(
    logical_file_schema: &Schema,
    accumulators: &mut StatisticsAccumulators,
    logical_schema_index: usize,
    stats_converter: &StatisticsConverter,
    row_groups_metadata: &[RowGroupMetaData],
    num_rows: usize,
    estimate_distinct_count: bool,
) -> Result<()> {
    let parquet_index = stats_converter.parquet_column_index();

    if accumulators.max_accs[logical_schema_index].is_some() {
        accumulators.is_max_value_exact[logical_schema_index] = summarize_bound(
            &mut accumulators.max_accs[logical_schema_index],
            &stats_converter.row_group_maxes(row_groups_metadata)?,
            parquet_index,
            row_groups_metadata,
            ParquetStatistics::max_is_exact,
            || Ok(stats_converter.row_group_is_max_value_exact(row_groups_metadata)?),
        )?;
    }

    if accumulators.min_accs[logical_schema_index].is_some() {
        accumulators.is_min_value_exact[logical_schema_index] = summarize_bound(
            &mut accumulators.min_accs[logical_schema_index],
            &stats_converter.row_group_mins(row_groups_metadata)?,
            parquet_index,
            row_groups_metadata,
            ParquetStatistics::min_is_exact,
            || Ok(stats_converter.row_group_is_min_value_exact(row_groups_metadata)?),
        )?;
    }

    accumulators.null_counts_array[logical_schema_index] =
        summarize_null_counts(stats_converter, row_groups_metadata)?;

    accumulators.distinct_counts_array[logical_schema_index] =
        summarize_distinct_counts(parquet_index, row_groups_metadata);

    if estimate_distinct_count
        && accumulators.distinct_counts_array[logical_schema_index] == Precision::Absent
        && let Some(parquet_index) = parquet_index
    {
        let min = accumulators.min_accs[logical_schema_index]
            .as_mut()
            .and_then(|acc| acc.evaluate().ok());
        let max = accumulators.max_accs[logical_schema_index]
            .as_mut()
            .and_then(|acc| acc.evaluate().ok());
        let value_range = min
            .zip(max)
            .and_then(|(min, max)| integer_value_range(&min, &max));
        // The accumulators are cleared when the min/max order is untrusted
        let shared_boundaries = if accumulators.min_accs[logical_schema_index].is_some()
            && accumulators.max_accs[logical_schema_index].is_some()
        {
            disjoint_row_groups_shared_boundaries(
                &stats_converter.row_group_mins(row_groups_metadata)?,
                &stats_converter.row_group_maxes(row_groups_metadata)?,
            )?
        } else {
            None
        };
        if let Some(estimate) = estimate_distinct_count_from_metadata(
            parquet_index,
            row_groups_metadata,
            value_range,
            shared_boundaries,
        ) {
            accumulators.distinct_counts_array[logical_schema_index] =
                Precision::Inexact(estimate);
        }
    }

    let arrow_field = logical_file_schema.field(logical_schema_index);
    accumulators.column_byte_sizes[logical_schema_index] = compute_arrow_column_size(
        arrow_field.data_type(),
        row_groups_metadata,
        parquet_index,
        num_rows,
    );

    Ok(())
}

/// Feed a column's per-row-group min or max `values` into `acc` and decide
/// whether the resulting bound is exact across all row groups.
///
/// `is_exact` reads the per-row-group exactness flag straight from the raw
/// parquet statistics. `row_group_exactness` rebuilds the exactness as a Boolean
/// array and is only called for the rare case where row groups disagree.
fn summarize_bound<A: Accumulator>(
    acc: &mut Option<A>,
    values: &ArrayRef,
    parquet_index: Option<usize>,
    row_groups_metadata: &[RowGroupMetaData],
    is_exact: impl Fn(&ParquetStatistics) -> bool,
    row_group_exactness: impl FnOnce() -> Result<BooleanArray>,
) -> Result<Option<bool>> {
    // A NULL converted bound can mean missing statistics, not just all-NULL
    // data. Ignoring it in MIN/MAX would let another row group's exact endpoint
    // incorrectly establish an exact bound for the whole file. Drop this bound
    // unless the row group is empty or is proven to contain only NULLs.
    if values.null_count() > 0
        && parquet_index.is_some_and(|column_index| {
            row_groups_metadata
                .iter()
                .enumerate()
                .any(|(index, group)| {
                    if values.is_valid(index) || group.num_rows() == 0 {
                        return false;
                    }
                    let column = group.column(column_index);
                    let all_null = column
                        .statistics()
                        .and_then(|stats| stats.null_count_opt())
                        .is_some_and(|nulls| nulls == column.num_values() as u64);
                    !all_null
                })
        })
    {
        *acc = None;
        return Ok(None);
    }
    let Some(acc) = acc.as_mut() else {
        return Ok(None);
    };
    acc.update_batch(&[Arc::clone(values)])?;

    Ok(
        match summarize_row_group_exactness(parquet_index, row_groups_metadata, is_exact)
        {
            ExactnessSummary::AllExact => Some(true),
            ExactnessSummary::NoneExact => Some(false),
            ExactnessSummary::Mixed => {
                let exactness = row_group_exactness()?;
                has_any_exact_match(&acc.evaluate()?, values, &exactness)
            }
        },
    )
}

fn summarize_null_counts(
    stats_converter: &StatisticsConverter,
    row_groups_metadata: &[RowGroupMetaData],
) -> Result<Precision<usize>> {
    if row_groups_metadata.is_empty() {
        return Ok(Precision::Exact(0));
    }

    let null_counts = stats_converter.row_group_null_counts(row_groups_metadata)?;

    match sum(&null_counts) {
        Some(count) => {
            // If any row group has an unknown null_count, either because column
            // statistics are absent or because the null_count field is omitted,
            // report the aggregate as inexact.
            if null_counts.null_count() > 0 {
                Ok(Precision::Inexact(count as usize))
            } else {
                Ok(Precision::Exact(count as usize))
            }
        }
        None => match null_counts.len() {
            // If sum() returned None we either have no rows or all values are null
            0 => Ok(Precision::Exact(0)),
            _ => Ok(Precision::Absent),
        },
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ExactnessSummary {
    AllExact,
    NoneExact,
    Mixed,
}

fn summarize_row_group_exactness(
    parquet_idx: Option<usize>,
    row_groups_metadata: &[RowGroupMetaData],
    exactness: impl Fn(&ParquetStatistics) -> bool,
) -> ExactnessSummary {
    let Some(parquet_idx) = parquet_idx else {
        return ExactnessSummary::NoneExact;
    };

    summarize_exactness(row_groups_metadata.iter().map(|row_group| {
        row_group
            .columns()
            .get(parquet_idx)
            .and_then(|column| column.statistics())
            .map(&exactness)
    }))
}

fn summarize_exactness<I>(exactness: I) -> ExactnessSummary
where
    I: IntoIterator<Item = Option<bool>>,
{
    let mut has_true = false;
    let mut has_false_or_null = false;

    for exactness in exactness {
        match exactness {
            Some(true) => has_true = true,
            Some(false) | None => has_false_or_null = true,
        }

        if has_true && has_false_or_null {
            return ExactnessSummary::Mixed;
        }
    }

    if has_true {
        ExactnessSummary::AllExact
    } else {
        ExactnessSummary::NoneExact
    }
}

/// Extract distinct counts from row group column statistics.
fn summarize_distinct_counts(
    parquet_idx: Option<usize>,
    row_groups_metadata: &[RowGroupMetaData],
) -> Precision<usize> {
    let Some(parquet_idx) = parquet_idx else {
        return Precision::Absent;
    };

    let num_row_groups = row_groups_metadata.len();
    if num_row_groups == 0 {
        return Precision::Absent;
    }

    let required_count = (num_row_groups as f64 * PARTIAL_NDV_THRESHOLD).ceil() as usize;
    let mut ndv_count = 0;
    let mut max_distinct_count: Option<u64> = None;

    for (row_group_idx, row_group) in row_groups_metadata.iter().enumerate() {
        if let Some(distinct_count) = row_group
            .columns()
            .get(parquet_idx)
            .and_then(|col| col.statistics())
            .and_then(|stats| stats.distinct_count_opt())
        {
            ndv_count += 1;
            max_distinct_count = Some(match max_distinct_count {
                Some(max) => max.max(distinct_count),
                None => distinct_count,
            });
        }

        // Return early if there's no chance to reach the required coverage.
        let remaining = num_row_groups - row_group_idx - 1;
        if ndv_count + remaining < required_count {
            return Precision::Absent;
        }
    }

    match max_distinct_count {
        Some(distinct_count) if num_row_groups == 1 => {
            Precision::Exact(distinct_count as usize)
        }
        Some(distinct_count) => Precision::Inexact(distinct_count as usize),
        None => Precision::Absent,
    }
}

/// Bytes that precede each value in a PLAIN encoded `BYTE_ARRAY` dictionary page.
const BYTE_ARRAY_LENGTH_PREFIX_BYTES: f64 = 4.0;

/// Maximum number of iterations of the Newton-Raphson solvers.
const NEWTON_MAX_ITERATIONS: usize = 50;

/// Convergence tolerance of the Newton-Raphson solvers.
const NEWTON_TOLERANCE: f64 = 1e-6;

/// Fraction of a row group's non-null values that must be distinct for the
/// row group to count as nearly unique.
const NEARLY_UNIQUE_FRACTION: f64 = 0.9;

/// Upper estimate of the size of the Thrift encoded dictionary page header,
/// which is usually 15 to 30 bytes.
const DICTIONARY_PAGE_HEADER_BYTES: i64 = 32;

/// Minimum number of non-null values per possible value of an integer column
/// for the range fill floor to apply, see [`range_fill_floor`].
const RANGE_FILL_MIN_ROWS_PER_VALUE: f64 = 10.0;

/// Ranges up to this size get the range fill floor without dictionary
/// evidence, see [`range_fill_floor`].
const RANGE_FILL_MAX_UNCHECKED_RANGE: u128 = 16;

/// Minimum stored dictionary bytes per possible value for the range fill floor
/// to apply to larger ranges, see [`range_fill_floor`].
const RANGE_FILL_MIN_DICTIONARY_BYTES_PER_VALUE: f64 = 0.5;

/// Estimate the number of distinct non-null values of a column from Parquet
/// metadata only, following "Zero-Cost NDV Estimation from Columnar File
/// Metadata" (<https://arxiv.org/abs/2603.24606>).
///
/// Two signals are computed and the larger one is used:
///
/// 1. Dictionary size: the uncompressed size of a dictionary encoded column
///    chunk is inverted to a distinct count, see [`invert_dictionary_size`].
///    A lower bound from the dictionary page size corrects the inversion for
///    run length encoded indexes. The paper does not say how to combine row
///    groups. When `shared_boundaries` is set the row groups share no values
///    except that many boundary values, so the per row group counts are added
///    and the shared values are subtracted. The counts are also added when
///    every row group is nearly unique. Otherwise the maximum is used, which
///    assumes the values are spread across row groups.
/// 2. Row group min/max diversity: the number of distinct row group minimums
///    and maximums is inverted with a coupon collector model, see
///    [`invert_coupon_collector`].
///
/// The number of distinct extreme values is a lower bound of the distinct
/// count. It is applied to the result, but is not used as an estimate on its
/// own: with few row groups it is far below the real count.
///
/// The result is capped by the number of non-null values and, for integer like
/// columns, by `value_range`, the number of integers between the file minimum
/// and maximum. Returns `None` when neither signal is available.
fn estimate_distinct_count_from_metadata(
    parquet_idx: usize,
    row_groups_metadata: &[RowGroupMetaData],
    value_range: Option<u128>,
    shared_boundaries: Option<usize>,
) -> Option<usize> {
    let chunks: Vec<(&RowGroupMetaData, &ColumnChunkMetaData)> = row_groups_metadata
        .iter()
        .filter_map(|row_group| {
            row_group
                .columns()
                .get(parquet_idx)
                .map(|chunk| (row_group, chunk))
        })
        .collect();
    let first_chunk = chunks.first()?.1;
    let physical_type = first_chunk.column_type();
    if physical_type == PhysicalType::BOOLEAN {
        return None;
    }

    let mut distinct_mins: HashSet<&[u8]> = HashSet::new();
    let mut distinct_maxs: HashSet<&[u8]> = HashSet::new();
    let mut row_groups_with_stats = 0_usize;
    for stats in chunks.iter().filter_map(|(_, chunk)| chunk.statistics()) {
        if let (Some(min), Some(max)) = (stats.min_bytes_opt(), stats.max_bytes_opt()) {
            row_groups_with_stats += 1;
            distinct_mins.insert(min);
            distinct_maxs.insert(max);
        }
    }
    let extrema: HashSet<&[u8]> = distinct_mins.union(&distinct_maxs).copied().collect();

    let value_length = match physical_type {
        PhysicalType::BYTE_ARRAY => (!extrema.is_empty()).then(|| {
            let total: usize = extrema.iter().map(|value| value.len()).sum();
            total as f64 / extrema.len() as f64 + BYTE_ARRAY_LENGTH_PREFIX_BYTES
        }),
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            Some(first_chunk.column_descr().type_length() as f64)
        }
        PhysicalType::INT32 | PhysicalType::FLOAT => Some(4.0),
        PhysicalType::INT64 | PhysicalType::DOUBLE => Some(8.0),
        PhysicalType::INT96 => Some(12.0),
        PhysicalType::BOOLEAN => None,
    };

    // A chunk that starts dictionary encoded and falls back to PLAIN pages is
    // not treated specially: its size is larger, so the inverted count is high
    // and is capped by the number of non-null values below.
    //
    // Each entry is the estimate of one row group and its non-null count.
    let mut row_group_estimates: Vec<(f64, f64)> = vec![];
    // Stored dictionary bytes without the fixed overhead, over all row groups
    let mut total_dictionary_bytes = 0_i64;
    if let Some(value_length) = value_length {
        for (row_group, chunk) in &chunks {
            let Some(dictionary_offset) = chunk.dictionary_page_offset() else {
                continue;
            };
            let nulls = chunk
                .statistics()
                .and_then(|stats| stats.null_count_opt())
                .unwrap_or(0) as i64;
            let non_null = row_group.num_rows().saturating_sub(nulls);
            if non_null <= 0 || chunk.uncompressed_size() <= 0 {
                continue;
            }
            let non_null = non_null as f64;
            let inverted = invert_dictionary_size(
                chunk.uncompressed_size() as f64,
                non_null,
                value_length,
            );
            // The size model assumes bit packed indexes. Clustered values give
            // run length encoded indexes that are much smaller, so the inverted
            // count collapses. The dictionary page holds every distinct value,
            // and compression only makes it smaller, so its stored size divided
            // by the value length is a lower bound. The stored size also includes
            // a fixed overhead, which would otherwise dominate small dictionaries.
            let dictionary_bytes = chunk.data_page_offset()
                - dictionary_offset
                - dictionary_page_overhead_bytes(chunk.compression());
            total_dictionary_bytes += dictionary_bytes.max(0);
            let lower_bound = dictionary_bytes.max(0) as f64 / value_length;
            row_group_estimates.push((inverted.max(lower_bound).min(non_null), non_null));
        }
    }

    // When almost every value of each row group is distinct, row groups are
    // unlikely to share values, so the counts are added like for disjoint
    // row groups. This overestimates when the same unique values repeat in
    // every row group.
    let all_nearly_unique = row_group_estimates
        .iter()
        .all(|(estimate, non_null)| *estimate >= NEARLY_UNIQUE_FRACTION * non_null);
    let ndv_dictionary = (!row_group_estimates.is_empty()).then(|| {
        let sum: f64 = row_group_estimates
            .iter()
            .map(|(estimate, _)| estimate)
            .sum();
        let max = row_group_estimates
            .iter()
            .map(|(estimate, _)| *estimate)
            .fold(0.0, f64::max);
        match shared_boundaries {
            Some(shared) => (sum - shared as f64).max(max),
            None if all_nearly_unique => sum,
            None => max,
        }
    });

    let ndv_coupon = [&distinct_mins, &distinct_maxs]
        .into_iter()
        .filter_map(|distinct| {
            invert_coupon_collector(distinct.len(), row_groups_with_stats)
        })
        .reduce(f64::max);

    if ndv_dictionary.is_none() && ndv_coupon.is_none() {
        return None;
    }
    let estimate = ndv_dictionary
        .unwrap_or(0.0)
        .max(ndv_coupon.unwrap_or(0.0))
        .max(extrema.len() as f64);

    let total_rows: u64 = row_groups_metadata
        .iter()
        .map(|row_group| row_group.num_rows() as u64)
        .sum();
    let null_counts: Option<u64> = chunks
        .iter()
        .map(|(_, chunk)| chunk.statistics().and_then(|stats| stats.null_count_opt()))
        .sum();
    let non_null_values = total_rows.saturating_sub(null_counts.unwrap_or(0));
    if non_null_values == 0 {
        return None;
    }

    let mut estimate = estimate;
    if let Some(fill) = value_range.and_then(|range| {
        range_fill_floor(range, non_null_values, total_dictionary_bytes)
    }) {
        estimate = estimate.max(fill);
    }
    let mut ndv = estimate.round().clamp(1.0, non_null_values as f64) as u128;
    if let Some(range) = value_range {
        ndv = ndv.min(range);
    }
    Some(ndv as usize)
}

/// Fixed size overhead of a stored dictionary page: the page header plus the
/// framing added by the compression codec. For example a GZIP compressed
/// dictionary page with a single value takes 43 to 47 bytes.
fn dictionary_page_overhead_bytes(codec: Compression) -> i64 {
    let framing = match codec {
        Compression::UNCOMPRESSED => 0,
        Compression::GZIP(_) => 20,
        Compression::ZSTD(_) => 16,
        _ => 8,
    };
    DICTIONARY_PAGE_HEADER_BYTES + framing
}

/// Expected number of distinct values when `non_null` values are drawn
/// uniformly from the `range` possible values of an integer column:
/// `range * (1 - exp(-non_null / range))`.
///
/// Small integer domains (years, months, hours) compress so well that the
/// dictionary signals underestimate them, while the value range is known. The
/// floor applies only when there are at least [`RANGE_FILL_MIN_ROWS_PER_VALUE`]
/// values per possible value, where the expectation is close to `range`.
///
/// A domain with gaps, e.g. multiples of 500 between 0 and 9500, has a small
/// dictionary compared to its range. For ranges above
/// [`RANGE_FILL_MAX_UNCHECKED_RANGE`], the floor therefore also requires
/// `dictionary_bytes` (stored dictionary bytes without the fixed overhead) to
/// hold at least [`RANGE_FILL_MIN_DICTIONARY_BYTES_PER_VALUE`] per possible
/// value. On TPC-DS, dense domains have at least 1 byte per value and domains
/// with gaps at most 0.25. Smaller ranges are not checked because their
/// dictionaries are close to the fixed overhead, and the overestimate is at
/// most the range.
fn range_fill_floor(range: u128, non_null: u64, dictionary_bytes: i64) -> Option<f64> {
    let dense = range <= RANGE_FILL_MAX_UNCHECKED_RANGE
        || dictionary_bytes as f64
            >= RANGE_FILL_MIN_DICTIONARY_BYTES_PER_VALUE * range as f64;
    let (range, non_null) = (range as f64, non_null as f64);
    (dense && non_null >= RANGE_FILL_MIN_ROWS_PER_VALUE * range)
        .then(|| range * (1.0 - (-non_null / range).exp()))
}

/// Number of integers in `min..=max` when both are integer like scalars.
fn integer_value_range(min: &ScalarValue, max: &ScalarValue) -> Option<u128> {
    fn as_i128(value: &ScalarValue) -> Option<i128> {
        match value {
            ScalarValue::Int8(Some(v)) => Some(i128::from(*v)),
            ScalarValue::Int16(Some(v)) => Some(i128::from(*v)),
            ScalarValue::Int32(Some(v)) | ScalarValue::Date32(Some(v)) => {
                Some(i128::from(*v))
            }
            ScalarValue::Int64(Some(v)) | ScalarValue::Date64(Some(v)) => {
                Some(i128::from(*v))
            }
            ScalarValue::UInt8(Some(v)) => Some(i128::from(*v)),
            ScalarValue::UInt16(Some(v)) => Some(i128::from(*v)),
            ScalarValue::UInt32(Some(v)) => Some(i128::from(*v)),
            ScalarValue::UInt64(Some(v)) => Some(i128::from(*v)),
            _ => None,
        }
    }
    let (min, max) = (as_i128(min)?, as_i128(max)?);
    max.checked_sub(min)
        .and_then(|diff| diff.checked_add(1))
        .and_then(|range| u128::try_from(range).ok())
}

/// When there are at least two row groups and their `[min, max]` ranges follow
/// each other in ascending or descending order, returns the number of
/// boundary values shared by adjacent row groups. Such row groups share no
/// other values. This is the case for sorted or partitioned columns.
///
/// For example the ranges `[0, 10]`, `[10, 19]`, `[20, 30]` share one boundary
/// value, and three row groups with the range `["x", "x"]` share two.
fn disjoint_row_groups_shared_boundaries(
    mins: &ArrayRef,
    maxes: &ArrayRef,
) -> Result<Option<usize>> {
    let n = mins.len();
    if n < 2 || mins.null_count() > 0 || maxes.null_count() > 0 {
        return Ok(None);
    }
    let all_true = |result: &BooleanArray| result.true_count() == result.len();
    let (head_mins, tail_mins) = (mins.slice(0, n - 1), mins.slice(1, n - 1));
    let (head_maxes, tail_maxes) = (maxes.slice(0, n - 1), maxes.slice(1, n - 1));
    for (lower_maxes, upper_mins) in
        [(&head_maxes, &tail_mins), (&tail_maxes, &head_mins)]
    {
        if all_true(&lt_eq(lower_maxes, upper_mins)?) {
            return Ok(Some(eq(lower_maxes, upper_mins)?.true_count()));
        }
    }
    Ok(None)
}

/// Invert the size of a dictionary encoded column chunk to a distinct count.
///
/// The uncompressed chunk size is modeled as the dictionary page plus the
/// bit packed indexes of the data pages:
///
/// `size = ndv * value_length + non_null * ceil(log2(ndv)) / 8`
///
/// where `non_null` is the number of non-null values in the chunk and
/// `value_length` is the mean encoded length of a value in bytes. The equation
/// is solved with Newton-Raphson. The size is not smooth because of the
/// `ceil`, so if Newton-Raphson does not reach the tolerance the result is
/// found by bisection, which works because the size grows with `ndv`.
/// The result is in `[1, non_null]`.
fn invert_dictionary_size(size: f64, non_null: f64, value_length: f64) -> f64 {
    let residual = |ndv: f64| {
        ndv * value_length + non_null * ndv.log2().ceil().max(0.0) / 8.0 - size
    };
    let upper = non_null.max(1.0);

    let mut ndv = (size / value_length).clamp(1.0, upper);
    for _ in 0..NEWTON_MAX_ITERATIONS {
        let value = residual(ndv);
        if value.abs() < NEWTON_TOLERANCE {
            return ndv;
        }
        let slope = value_length + non_null / (8.0 * ndv * LN_2);
        ndv = (ndv - value / slope).clamp(1.0, upper);
    }

    if residual(1.0) >= 0.0 {
        return 1.0;
    }
    if residual(upper) <= 0.0 {
        return upper;
    }
    let (mut low, mut high) = (1.0, upper);
    for _ in 0..100 {
        let mid = f64::midpoint(low, high);
        if residual(mid) < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

/// Invert the coupon collector model to a distinct count.
///
/// Drawing `draws` values from `ndv` equally likely distinct values yields in
/// expectation `ndv * (1 - exp(-draws / ndv))` distinct values. Given the
/// observed number of distinct values `observed`, solve for `ndv` with
/// Newton-Raphson, starting at `observed`. The function is concave and
/// increasing, so the iteration approaches the root from below.
///
/// Returns `None` when there is no finite solution, which happens when
/// `observed >= draws` or `draws < 2`.
fn invert_coupon_collector(observed: usize, draws: usize) -> Option<f64> {
    if draws < 2 || observed == 0 || observed >= draws {
        return None;
    }
    let (m, n) = (observed as f64, draws as f64);
    let mut ndv = m;
    for _ in 0..NEWTON_MAX_ITERATIONS * 4 {
        let decay = (-n / ndv).exp();
        let value = ndv * (1.0 - decay) - m;
        let slope = 1.0 - decay * (1.0 + n / ndv);
        if slope <= 0.0 {
            return None;
        }
        let next = ndv - value / slope;
        if !next.is_finite() {
            return None;
        }
        if (next - ndv).abs() < NEWTON_TOLERANCE {
            return Some(next);
        }
        ndv = next;
    }
    Some(ndv)
}

/// Compute the Arrow in-memory size for a single column
fn compute_arrow_column_size(
    data_type: &DataType,
    row_groups_metadata: &[RowGroupMetaData],
    parquet_idx: Option<usize>,
    num_rows: usize,
) -> Precision<usize> {
    // For primitive types with known fixed size, compute exact size
    if let Some(byte_width) = data_type.primitive_width() {
        return Precision::Exact(byte_width * num_rows);
    }

    // Use the uncompressed Parquet size as an estimate for other types
    if let Some(parquet_idx) = parquet_idx {
        let uncompressed_bytes: i64 = row_groups_metadata
            .iter()
            .filter_map(|rg| rg.columns().get(parquet_idx))
            .map(|col| col.uncompressed_size())
            .sum();
        return Precision::Inexact(uncompressed_bytes as usize);
    }

    // Otherwise, we cannot determine the size
    Precision::Absent
}

/// Checks if any occurrence of `value` in `array` corresponds to a `true`
/// entry in the `exactness` array.
///
/// This is used to determine if a calculated statistic (e.g., min or max)
/// is exact, by checking if at least one of its source values was exact.
///
/// # Example
/// - `value`: `0`
/// - `array`: `[0, 1, 0, 3, 0, 5]`
/// - `exactness`: `[true, false, false, false, false, false]`
///
/// The value `0` appears at indices `[0, 2, 4]`. The corresponding exactness
/// values are `[true, false, false]`. Since at least one is `true`, the
/// function returns `Some(true)`.
fn has_any_exact_match(
    value: &ScalarValue,
    array: &ArrayRef,
    exactness: &BooleanArray,
) -> Option<bool> {
    if value.is_null() {
        return Some(false);
    }

    // Shortcut for single row group
    if array.len() == 1 {
        return Some(exactness.is_valid(0) && exactness.value(0));
    }

    let scalar_array = value.to_scalar().ok()?;
    let eq_mask = eq(&scalar_array, &array).ok()?;
    let combined_mask = and(&eq_mask, exactness).ok()?;
    Some(combined_mask.has_true())
}

/// Wrapper to implement [`FileMetadata`] for [`ParquetMetaData`].
pub struct CachedParquetMetaData(Arc<ParquetMetaData>);

impl CachedParquetMetaData {
    pub fn new(metadata: Arc<ParquetMetaData>) -> Self {
        Self(metadata)
    }

    pub fn parquet_metadata(&self) -> &Arc<ParquetMetaData> {
        &self.0
    }
}

impl FileMetadata for CachedParquetMetaData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn memory_size(&self) -> usize {
        self.0.memory_size()
    }

    fn extra_info(&self) -> HashMap<String, String> {
        let page_index = self
            .0
            .page_index()
            .is_some_and(|page_index| page_index.is_complete());
        HashMap::from([("page_index".to_owned(), page_index.to_string())])
    }
}

/// Convert a [`PhysicalSortExpr`] to a Parquet [`SortingColumn`].
///
/// Returns `Ok(None)` if the referenced column is not in `writer_schema`, such as a
/// hive partition column that is removed before writing the Parquet file. Returns
/// `Err` if the expression is not a simple column reference or references a column
/// outside `input_schema`.
pub(crate) fn sort_expr_to_sorting_column(
    sort_expr: &PhysicalSortExpr,
    input_schema: &Schema,
    writer_schema: &Schema,
) -> Result<Option<SortingColumn>> {
    let column = sort_expr.expr.downcast_ref::<Column>().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "Parquet sorting_columns only supports simple column references, \
                 but got expression: {}",
            sort_expr.expr
        ))
    })?;

    let input_field = input_schema.fields().get(column.index()).ok_or_else(|| {
        internal_datafusion_err!(
            "Parquet sorting column '{}' references index {} but the input schema has {} columns",
            column.name(),
            column.index(),
            input_schema.fields().len()
        )
    })?;
    let Some((writer_index, _)) = writer_schema.column_with_name(input_field.name())
    else {
        return Ok(None);
    };

    let column_idx: i32 = writer_index.try_into().map_err(|_| {
        DataFusionError::Plan(format!(
            "Column index {writer_index} is too large to be represented as i32"
        ))
    })?;

    Ok(Some(SortingColumn {
        column_idx,
        descending: sort_expr.options.descending,
        nulls_first: sort_expr.options.nulls_first,
    }))
}

/// Convert a [`LexOrdering`] to `Vec<SortingColumn>` for Parquet.
///
/// Columns that are not present in `writer_schema` are omitted from the resulting
/// metadata. Returns `Err` if any expression is not a simple column reference or
/// references a column outside `input_schema`.
pub(crate) fn lex_ordering_to_sorting_columns(
    ordering: &LexOrdering,
    input_schema: &Schema,
    writer_schema: &Schema,
) -> Result<Vec<SortingColumn>> {
    ordering
        .iter()
        .filter_map(|sort_expr| {
            sort_expr_to_sorting_column(sort_expr, input_schema, writer_schema)
                .transpose()
        })
        .collect()
}

/// Extracts ordering information from Parquet metadata.
///
/// This function reads the sorting_columns from the first row group's metadata
/// and converts them into a [`LexOrdering`] that can be used by the query engine.
///
/// # Arguments
/// * `metadata` - The Parquet metadata containing sorting_columns information
/// * `schema` - The Arrow schema to use for column lookup
///
/// # Returns
/// * `Ok(Some(ordering))` if valid ordering information was found
/// * `Ok(None)` if no sorting columns were specified or they couldn't be resolved
pub fn ordering_from_parquet_metadata(
    metadata: &ParquetMetaData,
    schema: &SchemaRef,
) -> Result<Option<LexOrdering>> {
    // Get the sorting columns from the first row group metadata.
    // If no row groups exist or no sorting columns are specified, return None.
    let sorting_columns = metadata
        .row_groups()
        .first()
        .and_then(|rg| rg.sorting_columns())
        .filter(|cols| !cols.is_empty());

    let Some(sorting_columns) = sorting_columns else {
        return Ok(None);
    };

    let parquet_schema = metadata.file_metadata().schema_descr();

    let sort_exprs =
        sorting_columns_to_physical_exprs(sorting_columns, parquet_schema, schema);

    if sort_exprs.is_empty() {
        return Ok(None);
    }

    Ok(LexOrdering::new(sort_exprs))
}

/// Converts Parquet sorting columns to physical sort expressions.
fn sorting_columns_to_physical_exprs(
    sorting_columns: &[SortingColumn],
    parquet_schema: &SchemaDescriptor,
    arrow_schema: &SchemaRef,
) -> Vec<PhysicalSortExpr> {
    use arrow::compute::SortOptions;

    sorting_columns
        .iter()
        .filter_map(|sc| {
            let parquet_column = parquet_schema.column(sc.column_idx as usize);
            let name = parquet_column.name();

            // Find the column in the arrow schema
            let (index, _) = arrow_schema.column_with_name(name)?;

            let expr = Arc::new(Column::new(name, index));
            let options = SortOptions {
                descending: sc.descending,
                nulls_first: sc.nulls_first,
            };
            Some(PhysicalSortExpr::new(expr, options))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::compute::SortOptions;
    use arrow::datatypes::Field;

    #[test]
    fn test_lex_ordering_to_sorting_columns_uses_writer_schema() -> Result<()> {
        let input_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("part", DataType::Utf8, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        let writer_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        let ordering = LexOrdering::new(vec![
            PhysicalSortExpr::new(
                Arc::new(Column::new("part", 1)),
                SortOptions::default(),
            ),
            PhysicalSortExpr::new(Arc::new(Column::new("a", 0)), SortOptions::default()),
            PhysicalSortExpr::new(
                Arc::new(Column::new("b", 2)),
                SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            ),
        ])
        .unwrap();

        let sorting_columns =
            lex_ordering_to_sorting_columns(&ordering, &input_schema, &writer_schema)?;

        assert_eq!(
            sorting_columns,
            vec![
                SortingColumn {
                    column_idx: 0,
                    descending: false,
                    nulls_first: true,
                },
                SortingColumn {
                    column_idx: 1,
                    descending: true,
                    nulls_first: false,
                },
            ]
        );

        Ok(())
    }

    #[test]
    fn test_has_any_exact_match() {
        // Case 1: Mixed exact and inexact matches
        {
            let computed_min = ScalarValue::Int32(Some(0));
            let row_group_mins =
                Arc::new(Int32Array::from(vec![0, 1, 0, 3, 0, 5])) as ArrayRef;
            let exactness =
                BooleanArray::from(vec![true, false, false, false, false, false]);

            let result = has_any_exact_match(&computed_min, &row_group_mins, &exactness);
            assert_eq!(result, Some(true));
        }
        // Case 2: All inexact matches
        {
            let computed_min = ScalarValue::Int32(Some(0));
            let row_group_mins =
                Arc::new(Int32Array::from(vec![0, 1, 0, 3, 0, 5])) as ArrayRef;
            let exactness =
                BooleanArray::from(vec![false, false, false, false, false, false]);

            let result = has_any_exact_match(&computed_min, &row_group_mins, &exactness);
            assert_eq!(result, Some(false));
        }
        // Case 3: All exact matches
        {
            let computed_max = ScalarValue::Int32(Some(5));
            let row_group_maxes =
                Arc::new(Int32Array::from(vec![1, 5, 3, 5, 2, 5])) as ArrayRef;
            let exactness =
                BooleanArray::from(vec![false, true, true, true, false, true]);

            let result = has_any_exact_match(&computed_max, &row_group_maxes, &exactness);
            assert_eq!(result, Some(true));
        }
        // Case 4: All maxes are null values
        {
            let computed_max = ScalarValue::Int32(None);
            let row_group_maxes =
                Arc::new(Int32Array::from(vec![None, None, None, None])) as ArrayRef;
            let exactness = BooleanArray::from(vec![None, Some(true), None, Some(false)]);

            let result = has_any_exact_match(&computed_max, &row_group_maxes, &exactness);
            assert_eq!(result, Some(false));
        }
    }

    #[test]
    fn test_summarize_exactness() {
        assert_eq!(
            summarize_exactness([Some(true), Some(true)]),
            ExactnessSummary::AllExact
        );
        assert_eq!(
            summarize_exactness([Some(false), None]),
            ExactnessSummary::NoneExact
        );
        assert_eq!(
            summarize_exactness([Some(true), Some(false)]),
            ExactnessSummary::Mixed
        );
        assert_eq!(
            summarize_exactness([Some(true), None]),
            ExactnessSummary::Mixed
        );
        assert_eq!(
            summarize_exactness(std::iter::empty()),
            ExactnessSummary::NoneExact
        );
    }

    mod statistics_tests {
        use super::*;
        use arrow::array::{Int32Array, Int64Array, StringArray};
        use arrow::datatypes::Field;
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::basic::{Compression, GzipLevel};
        use parquet::file::properties::WriterProperties;
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::file::statistics::Statistics as ParquetStatistics;
        use parquet::schema::types::{ColumnPath, Type as SchemaType};
        use std::fs::File;
        use std::path::PathBuf;

        #[test]
        fn test_invert_coupon_collector() {
            // 100 equally likely values drawn 50 times give about 39.35 distinct values
            let expected = 100.0 * (1.0 - (-0.5_f64).exp());
            assert!((expected - 39.35).abs() < 0.01);
            let ndv = invert_coupon_collector(39, 50).unwrap();
            assert!((ndv - 99.0).abs() < 5.0, "unexpected estimate {ndv}");

            // The expected count of the model is recovered exactly
            let ndv = invert_coupon_collector(5, 10).unwrap();
            let forward = ndv * (1.0 - (-10.0 / ndv).exp());
            assert!((forward - 5.0).abs() < 1e-4);

            // No finite solution when every draw is distinct, or with fewer than 2 draws
            assert_eq!(invert_coupon_collector(50, 50), None);
            assert_eq!(invert_coupon_collector(60, 50), None);
            assert_eq!(invert_coupon_collector(1, 1), None);
            assert_eq!(invert_coupon_collector(0, 10), None);

            // Almost every draw is distinct, the solution is large but finite
            let ndv = invert_coupon_collector(49, 50).unwrap();
            assert!(ndv.is_finite() && ndv > 100.0);
        }

        #[test]
        fn test_invert_dictionary_size() {
            for (ndv, non_null, value_length) in [
                (1.0, 1000.0, 4.0),
                (20.0, 10_000.0, 12.0),
                (256.0, 10_000.0, 8.0),
                (1000.0, 100_000.0, 4.0),
                (5000.0, 100_000.0, 16.0),
            ] {
                let size =
                    ndv * value_length + non_null * f64::ceil(f64::log2(ndv)) / 8.0;
                let estimate = invert_dictionary_size(size, non_null, value_length);
                assert!(
                    (estimate - ndv).abs() <= ndv * 0.01,
                    "ndv {ndv}: estimate {estimate}"
                );
            }

            // The estimate stays within [1, non_null]
            assert_eq!(invert_dictionary_size(1.0, 100.0, 8.0), 1.0);
            assert_eq!(invert_dictionary_size(1e9, 100.0, 8.0), 100.0);
        }

        #[test]
        fn test_range_fill_floor() {
            // 12 months in 73049 rows, small range: no dictionary check
            let months = range_fill_floor(12, 73_049, 0).unwrap();
            assert!((months - 12.0).abs() < 1e-6, "{months}");
            // 10 rows per value with a dense dictionary: 1 - exp(-10) of the range
            let dense = range_fill_floor(100, 1000, 150).unwrap();
            assert!((dense - 100.0 * (1.0 - (-10.0_f64).exp())).abs() < 1e-9);
            // Fewer rows per value: no floor
            assert_eq!(range_fill_floor(100, 999, 150), None);
            assert_eq!(range_fill_floor(1 << 40, 1_000_000, 1 << 30), None);
            // Domain with gaps (TPC-DS cd_purchase_estimate: 20 values, range
            // 9501, 70 dictionary bytes): no floor
            assert_eq!(range_fill_floor(9501, 1_920_800, 70), None);
            // Dense domain (TPC-DS d_year: range 201, 310 dictionary bytes)
            assert!(range_fill_floor(201, 73_049, 310).is_some());
        }

        #[test]
        fn test_dictionary_page_overhead_bytes() {
            assert_eq!(
                dictionary_page_overhead_bytes(Compression::UNCOMPRESSED),
                32
            );
            assert_eq!(
                dictionary_page_overhead_bytes(Compression::GZIP(GzipLevel::default())),
                52
            );
            assert_eq!(dictionary_page_overhead_bytes(Compression::SNAPPY), 40);
        }

        #[test]
        fn test_integer_value_range() {
            assert_eq!(
                integer_value_range(
                    &ScalarValue::Int32(Some(-5)),
                    &ScalarValue::Int32(Some(4))
                ),
                Some(10)
            );
            assert_eq!(
                integer_value_range(
                    &ScalarValue::Int64(Some(i64::MIN)),
                    &ScalarValue::Int64(Some(i64::MAX))
                ),
                Some(1_u128 << 64)
            );
            assert_eq!(
                integer_value_range(
                    &ScalarValue::Float64(Some(1.0)),
                    &ScalarValue::Float64(Some(2.0))
                ),
                None
            );
            assert_eq!(
                integer_value_range(
                    &ScalarValue::Int32(None),
                    &ScalarValue::Int32(Some(1))
                ),
                None
            );
        }

        /// Write a Parquet file without `distinct_count` statistics: 3 row groups of
        /// 5000 rows with
        /// - category: 20 distinct strings present in every row group
        /// - sorted: unique, increasing integers, 15000 distinct values
        /// - plain: unique, unordered integers, without dictionary encoding
        /// - clustered: 1250 distinct integers, each repeated in 12 consecutive
        ///   rows, so the dictionary indexes are run length encoded
        /// - unique_strings: unique, unordered strings
        /// - constant: a single string value (no value range cap), GZIP compressed
        fn write_estimation_test_file() -> ParquetMetaData {
            let rows = 15_000_i64;
            let category: StringArray = (0..rows)
                .map(|i| Some(format!("category_{:02}", (i * 7) % 20)))
                .collect();
            let sorted: Int64Array = (0..rows).collect();
            let plain: Int64Array = (0..rows).map(|i| (i * 7919) % rows).collect();
            let clustered: Int64Array =
                (0..rows).map(|i| ((i / 12) * 7919) % 1250).collect();
            let unique_strings: StringArray = (0..rows)
                .map(|i| Some(format!("value_{:05}", (i * 7919) % rows)))
                .collect();
            let schema = Arc::new(Schema::new(vec![
                Field::new("category", DataType::Utf8, false),
                Field::new("sorted", DataType::Int64, false),
                Field::new("plain", DataType::Int64, false),
                Field::new("clustered", DataType::Int64, false),
                Field::new("unique_strings", DataType::Utf8, false),
                Field::new("constant", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(category),
                    Arc::new(sorted),
                    Arc::new(plain),
                    Arc::new(clustered),
                    Arc::new(unique_strings),
                    Arc::new(StringArray::from(vec!["x"; rows as usize])),
                ],
            )
            .unwrap();

            let props = WriterProperties::builder()
                .set_max_row_group_row_count(Some(5000))
                .set_column_dictionary_enabled(ColumnPath::from("plain"), false)
                .set_column_compression(
                    ColumnPath::from("constant"),
                    Compression::GZIP(GzipLevel::default()),
                )
                .build();
            let mut buffer = Vec::new();
            let mut writer =
                ArrowWriter::try_new(&mut buffer, schema, Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();

            ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(buffer))
                .unwrap()
                .metadata()
                .as_ref()
                .clone()
        }

        #[test]
        fn test_estimate_distinct_count_from_real_parquet_file() {
            let metadata = write_estimation_test_file();
            assert_eq!(metadata.num_row_groups(), 3);
            let schema = Arc::new(
                parquet_to_arrow_schema(metadata.file_metadata().schema_descr(), None)
                    .unwrap(),
            );

            // Flag off: no written distinct count means Absent
            let stats =
                DFParquetMetadata::statistics_from_parquet_metadata(&metadata, &schema)
                    .unwrap();
            assert_eq!(stats.column_statistics[0].distinct_count, Precision::Absent);
            assert_eq!(stats.column_statistics[1].distinct_count, Precision::Absent);

            let stats = DFParquetMetadata::statistics_from_parquet_metadata_with_options(
                &metadata, &schema, true,
            )
            .unwrap();

            // Truth is 20, the accepted band is [10, 40]
            let Precision::Inexact(category) = stats.column_statistics[0].distinct_count
            else {
                panic!("expected an inexact estimate");
            };
            assert!(
                (10..=40).contains(&category),
                "category estimate {category}"
            );

            // Truth is 15000. The row groups do not overlap, so their counts are
            // added. The accepted band is [12000, 15000]
            let Precision::Inexact(sorted) = stats.column_statistics[1].distinct_count
            else {
                panic!("expected an inexact estimate");
            };
            assert!(
                (12_000..=15_000).contains(&sorted),
                "sorted estimate {sorted}"
            );

            // No dictionary, and every row group has a different minimum, so
            // neither signal applies
            assert_eq!(stats.column_statistics[2].distinct_count, Precision::Absent);

            // Truth is 1250, about 417 per row group, in overlapping ranges, so the
            // maximum over row groups is used. The dictionary page size bound keeps
            // the run length encoded indexes from collapsing the estimate. The
            // accepted band is [350, 1250]
            let Precision::Inexact(clustered) = stats.column_statistics[3].distinct_count
            else {
                panic!("expected an inexact estimate");
            };
            assert!(
                (350..=1250).contains(&clustered),
                "clustered estimate {clustered}"
            );

            // Truth is 15000. Every row group is nearly unique, so the counts are
            // added. The accepted band is [12000, 15000]
            let Precision::Inexact(unique) = stats.column_statistics[4].distinct_count
            else {
                panic!("expected an inexact estimate");
            };
            assert!(
                (12_000..=15_000).contains(&unique),
                "unique_strings estimate {unique}"
            );

            // Truth is 1. The fixed dictionary page overhead must not inflate the
            // dictionary page size bound. The accepted band is [1, 2]
            let Precision::Inexact(constant) = stats.column_statistics[5].distinct_count
            else {
                panic!("expected an inexact estimate");
            };
            assert!((1..=2).contains(&constant), "constant estimate {constant}");
        }

        #[test]
        fn test_disjoint_row_groups_shared_boundaries() {
            let ints = |values: Vec<Option<i32>>| -> ArrayRef {
                Arc::new(Int32Array::from(values))
            };
            let shared = |mins: ArrayRef, maxes: ArrayRef| {
                disjoint_row_groups_shared_boundaries(&mins, &maxes).unwrap()
            };
            // Ascending, sharing the boundary value 10
            assert_eq!(
                shared(
                    ints(vec![Some(0), Some(10), Some(20)]),
                    ints(vec![Some(10), Some(19), Some(30)]),
                ),
                Some(1)
            );
            // Descending, no shared boundary
            assert_eq!(
                shared(
                    ints(vec![Some(20), Some(0)]),
                    ints(vec![Some(30), Some(10)])
                ),
                Some(0)
            );
            // The same single value in every row group
            let constant: ArrayRef = Arc::new(StringArray::from(vec!["x"; 3]));
            assert_eq!(shared(Arc::clone(&constant), constant), Some(2));
            // Overlapping
            assert_eq!(
                shared(ints(vec![Some(0), Some(5)]), ints(vec![Some(10), Some(15)])),
                None
            );
            // Single row group or missing statistics
            assert_eq!(shared(ints(vec![Some(0)]), ints(vec![Some(1)])), None);
            assert_eq!(
                shared(ints(vec![Some(0), None]), ints(vec![Some(1), Some(5)])),
                None
            );
        }

        fn create_schema_descr(num_columns: usize) -> Arc<SchemaDescriptor> {
            let fields: Vec<Arc<SchemaType>> = (0..num_columns)
                .map(|i| {
                    Arc::new(
                        SchemaType::primitive_type_builder(
                            &format!("col_{i}"),
                            PhysicalType::INT32,
                        )
                        .build()
                        .unwrap(),
                    )
                })
                .collect();

            let schema = SchemaType::group_type_builder("schema")
                .with_fields(fields)
                .build()
                .unwrap();

            Arc::new(SchemaDescriptor::new(Arc::new(schema)))
        }

        fn create_arrow_schema(num_columns: usize) -> SchemaRef {
            let fields: Vec<Field> = (0..num_columns)
                .map(|i| Field::new(format!("col_{i}"), DataType::Int32, true))
                .collect();
            Arc::new(Schema::new(fields))
        }

        fn create_row_group_with_stats(
            schema_descr: &Arc<SchemaDescriptor>,
            column_stats: Vec<Option<ParquetStatistics>>,
            num_rows: i64,
        ) -> RowGroupMetaData {
            let columns: Vec<ColumnChunkMetaData> = column_stats
                .into_iter()
                .enumerate()
                .map(|(i, stats)| {
                    let mut builder =
                        ColumnChunkMetaData::builder(schema_descr.column(i));
                    if let Some(s) = stats {
                        builder = builder.set_statistics(s);
                    }
                    builder.set_num_values(num_rows).build().unwrap()
                })
                .collect();

            RowGroupMetaData::builder(schema_descr.clone())
                .set_num_rows(num_rows)
                .set_total_byte_size(1000)
                .set_column_metadata(columns)
                .build()
                .unwrap()
        }

        fn create_parquet_metadata(
            schema_descr: Arc<SchemaDescriptor>,
            row_groups: Vec<RowGroupMetaData>,
        ) -> ParquetMetaData {
            use parquet::file::metadata::FileMetaData;

            let num_rows: i64 = row_groups.iter().map(|rg| rg.num_rows()).sum();
            let file_meta = FileMetaData::new(
                1,            // version
                num_rows,     // num_rows
                None,         // created_by
                None,         // key_value_metadata
                schema_descr, // schema_descr
                None,         // column_orders
            );

            ParquetMetaData::new(file_meta, row_groups)
        }

        #[test]
        fn test_statistics_preserve_missing_null_counts() {
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);
            for (null_counts, expected) in [
                (vec![None], Precision::Absent),
                (vec![Some(0), None], Precision::Inexact(0)),
                (vec![Some(2), None], Precision::Inexact(2)),
                (vec![Some(0), Some(0)], Precision::Exact(0)),
            ] {
                let row_groups = null_counts
                    .into_iter()
                    .map(|null_count| {
                        create_row_group_with_stats(
                            &schema_descr,
                            vec![Some(ParquetStatistics::int32(
                                Some(1),
                                Some(10),
                                None,
                                null_count,
                                false,
                            ))],
                            10,
                        )
                    })
                    .collect();
                let metadata =
                    create_parquet_metadata(Arc::clone(&schema_descr), row_groups);
                let statistics = DFParquetMetadata::statistics_from_parquet_metadata(
                    &metadata,
                    &arrow_schema,
                )
                .unwrap();
                assert_eq!(statistics.column_statistics[0].null_count, expected);
            }
        }

        #[test]
        fn test_summarize_null_counts() {
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(2);
            let stats_with_count =
                ParquetStatistics::int32(Some(1), Some(10), None, Some(2), false);
            let stats_without_count =
                ParquetStatistics::int32(Some(1), Some(10), None, None, false);

            let row_groups = vec![
                create_row_group_with_stats(
                    &schema_descr,
                    vec![Some(stats_with_count)],
                    10,
                ),
                create_row_group_with_stats(
                    &schema_descr,
                    vec![Some(stats_without_count.clone())],
                    10,
                ),
                create_row_group_with_stats(&schema_descr, vec![None], 10),
            ];
            let stats_converter =
                StatisticsConverter::try_new("col_0", &arrow_schema, &schema_descr)
                    .unwrap();
            let missing_column_converter =
                StatisticsConverter::try_new("col_1", &arrow_schema, &schema_descr)
                    .unwrap();

            assert_eq!(
                summarize_null_counts(&stats_converter, &row_groups).unwrap(),
                Precision::Inexact(2)
            );
            assert_eq!(
                summarize_null_counts(&missing_column_converter, &row_groups).unwrap(),
                Precision::Absent
            );
            assert_eq!(
                summarize_null_counts(&stats_converter, &[]).unwrap(),
                Precision::Exact(0)
            );
            assert_eq!(
                summarize_null_counts(&missing_column_converter, &[]).unwrap(),
                Precision::Exact(0)
            );

            let missing_counts_unknown_converter =
                StatisticsConverter::try_new("col_0", &arrow_schema, &schema_descr)
                    .unwrap()
                    .with_missing_null_counts_as_zero(false);
            assert_eq!(
                summarize_null_counts(&missing_counts_unknown_converter, &row_groups)
                    .unwrap(),
                Precision::Inexact(2)
            );

            let row_groups_without_count = vec![
                create_row_group_with_stats(
                    &schema_descr,
                    vec![Some(stats_without_count.clone())],
                    10,
                ),
                create_row_group_with_stats(
                    &schema_descr,
                    vec![Some(stats_without_count)],
                    10,
                ),
            ];
            assert_eq!(
                summarize_null_counts(&stats_converter, &row_groups_without_count)
                    .unwrap(),
                Precision::Exact(0)
            );

            let missing_counts_unknown_converter =
                stats_converter.with_missing_null_counts_as_zero(false);
            assert_eq!(
                summarize_null_counts(
                    &missing_counts_unknown_converter,
                    &row_groups_without_count,
                )
                .unwrap(),
                Precision::Absent
            );
        }

        #[test]
        fn test_distinct_count_single_row_group_with_ndv() {
            // Single row group with distinct count should return Exact
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);

            // Create statistics with distinct_count = 42
            let stats = ParquetStatistics::int32(
                Some(1),   // min
                Some(100), // max
                Some(42),  // distinct_count
                Some(0),   // null_count
                false,     // is_deprecated
            );

            let row_group =
                create_row_group_with_stats(&schema_descr, vec![Some(stats)], 1000);
            let metadata = create_parquet_metadata(schema_descr, vec![row_group]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Exact(42)
            );
        }

        #[test]
        fn test_distinct_count_multiple_row_groups_with_ndv() {
            // Multiple row groups with distinct counts should return Inexact (sum)
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);

            // Row group 1: distinct_count = 10
            let stats1 = ParquetStatistics::int32(
                Some(1),
                Some(50),
                Some(10), // distinct_count
                Some(0),
                false,
            );

            // Row group 2: distinct_count = 20
            let stats2 = ParquetStatistics::int32(
                Some(51),
                Some(100),
                Some(20), // distinct_count
                Some(0),
                false,
            );

            let row_group1 =
                create_row_group_with_stats(&schema_descr, vec![Some(stats1)], 500);
            let row_group2 =
                create_row_group_with_stats(&schema_descr, vec![Some(stats2)], 500);
            let metadata =
                create_parquet_metadata(schema_descr, vec![row_group1, row_group2]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            // Max of distinct counts (lower bound since we can't accurately merge NDV)
            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Inexact(20)
            );
        }

        #[test]
        fn test_distinct_count_no_ndv_available() {
            // No distinct count in statistics should return Absent
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);

            // Create statistics without distinct_count (None)
            let stats = ParquetStatistics::int32(
                Some(1),
                Some(100),
                None, // no distinct_count
                Some(0),
                false,
            );

            let row_group =
                create_row_group_with_stats(&schema_descr, vec![Some(stats)], 1000);
            let metadata = create_parquet_metadata(schema_descr, vec![row_group]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Absent
            );
        }

        #[test]
        fn test_distinct_count_partial_ndv_below_threshold() {
            // 1 of 2 row groups has NDV (50% < 75% threshold) -> Absent
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);

            let stats1 =
                ParquetStatistics::int32(Some(1), Some(50), Some(15), Some(0), false);
            let stats2 =
                ParquetStatistics::int32(Some(51), Some(100), None, Some(0), false);

            let row_group1 =
                create_row_group_with_stats(&schema_descr, vec![Some(stats1)], 500);
            let row_group2 =
                create_row_group_with_stats(&schema_descr, vec![Some(stats2)], 500);
            let metadata =
                create_parquet_metadata(schema_descr, vec![row_group1, row_group2]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Absent
            );
        }

        #[test]
        fn test_distinct_count_partial_ndv_above_threshold() {
            // 3 of 4 row groups have NDV (75% >= 75% threshold) -> Inexact
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);

            let stats_with = |ndv| {
                ParquetStatistics::int32(Some(1), Some(100), Some(ndv), Some(0), false)
            };
            let stats_without =
                ParquetStatistics::int32(Some(1), Some(100), None, Some(0), false);

            let rg1 = create_row_group_with_stats(
                &schema_descr,
                vec![Some(stats_with(10))],
                250,
            );
            let rg2 = create_row_group_with_stats(
                &schema_descr,
                vec![Some(stats_with(20))],
                250,
            );
            let rg3 = create_row_group_with_stats(
                &schema_descr,
                vec![Some(stats_with(15))],
                250,
            );
            let rg4 = create_row_group_with_stats(
                &schema_descr,
                vec![Some(stats_without)],
                250,
            );
            let metadata =
                create_parquet_metadata(schema_descr, vec![rg1, rg2, rg3, rg4]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Inexact(20)
            );
        }

        #[test]
        fn test_distinct_count_multiple_columns() {
            // Test with multiple columns, each with different NDV
            let schema_descr = create_schema_descr(3);
            let arrow_schema = create_arrow_schema(3);

            // col_0: distinct_count = 5
            let stats0 =
                ParquetStatistics::int32(Some(1), Some(10), Some(5), Some(0), false);
            // col_1: no distinct_count
            let stats1 =
                ParquetStatistics::int32(Some(1), Some(100), None, Some(0), false);
            // col_2: distinct_count = 100
            let stats2 =
                ParquetStatistics::int32(Some(1), Some(1000), Some(100), Some(0), false);

            let row_group = create_row_group_with_stats(
                &schema_descr,
                vec![Some(stats0), Some(stats1), Some(stats2)],
                1000,
            );
            let metadata = create_parquet_metadata(schema_descr, vec![row_group]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Exact(5)
            );
            assert_eq!(
                result.column_statistics[1].distinct_count,
                Precision::Absent
            );
            assert_eq!(
                result.column_statistics[2].distinct_count,
                Precision::Exact(100)
            );
        }

        #[test]
        fn test_min_max_require_complete_row_group_bounds() {
            let known =
                || ParquetStatistics::int32(Some(1), Some(3), None, Some(0), false);
            let exact = |v| Precision::Exact(ScalarValue::Int32(Some(v)));
            let cases = [
                (None, 3, Precision::Absent, Precision::Absent),
                (
                    Some(ParquetStatistics::int32(
                        None,
                        Some(6),
                        None,
                        Some(0),
                        false,
                    )),
                    3,
                    Precision::Absent,
                    exact(6),
                ),
                (
                    Some(ParquetStatistics::int32(
                        Some(4),
                        None,
                        None,
                        Some(0),
                        false,
                    )),
                    3,
                    exact(1),
                    Precision::Absent,
                ),
                (
                    Some(ParquetStatistics::int32(None, None, None, None, false)),
                    3,
                    Precision::Absent,
                    Precision::Absent,
                ),
                (
                    Some(ParquetStatistics::int32(None, None, None, Some(2), false)),
                    3,
                    Precision::Absent,
                    Precision::Absent,
                ),
                // Empty and proven all-NULL row groups do not contribute extrema.
                (None, 0, exact(1), exact(3)),
                (
                    Some(ParquetStatistics::int32(None, None, None, Some(3), false)),
                    3,
                    exact(1),
                    exact(3),
                ),
                (
                    Some(ParquetStatistics::int32(
                        Some(4),
                        Some(6),
                        None,
                        Some(0),
                        false,
                    )),
                    3,
                    exact(1),
                    exact(6),
                ),
            ];
            for (other, rows, expected_min, expected_max) in cases {
                for reverse in [false, true] {
                    let schema_descr = create_schema_descr(1);
                    let arrow_schema = create_arrow_schema(1);
                    let mut groups = vec![
                        create_row_group_with_stats(
                            &schema_descr,
                            vec![Some(known())],
                            3,
                        ),
                        create_row_group_with_stats(
                            &schema_descr,
                            vec![other.clone()],
                            rows,
                        ),
                    ];
                    if reverse {
                        groups.reverse();
                    }
                    let metadata = create_parquet_metadata(schema_descr, groups);
                    let stats = DFParquetMetadata::statistics_from_parquet_metadata(
                        &metadata,
                        &arrow_schema,
                    )
                    .unwrap();
                    assert_eq!(
                        stats.column_statistics[0].min_value, expected_min,
                        "other={other:?}, rows={rows}, reverse={reverse}"
                    );
                    assert_eq!(
                        stats.column_statistics[0].max_value, expected_max,
                        "other={other:?}, rows={rows}, reverse={reverse}"
                    );
                    assert_eq!(stats.num_rows, Precision::Exact((3 + rows) as usize));
                }
            }
        }

        #[test]
        fn test_distinct_count_no_statistics_at_all() {
            // No statistics in row group should return Absent for all stats
            let schema_descr = create_schema_descr(1);
            let arrow_schema = create_arrow_schema(1);

            // Create row group without any statistics
            let row_group = create_row_group_with_stats(&schema_descr, vec![None], 1000);
            let metadata = create_parquet_metadata(schema_descr, vec![row_group]);

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                &metadata,
                &arrow_schema,
            )
            .unwrap();

            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Absent
            );
        }

        /// Integration test that reads a real Parquet file with distinct_count statistics.
        /// The test file was created with DuckDB and has known NDV values:
        /// - id: NULL (high cardinality, not tracked)
        /// - category: 10 distinct values
        /// - name: 5 distinct values
        #[test]
        fn test_distinct_count_from_real_parquet_file() {
            // Path to test file created by DuckDB with distinct_count statistics
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/test_data/ndv_test.parquet");

            let file = File::open(&path).expect("Failed to open test parquet file");
            let reader =
                SerializedFileReader::new(file).expect("Failed to create reader");
            let parquet_metadata = reader.metadata();

            // Derive Arrow schema from parquet file metadata
            let arrow_schema = Arc::new(
                parquet_to_arrow_schema(
                    parquet_metadata.file_metadata().schema_descr(),
                    None,
                )
                .expect("Failed to convert schema"),
            );

            let result = DFParquetMetadata::statistics_from_parquet_metadata(
                parquet_metadata,
                &arrow_schema,
            )
            .expect("Failed to extract statistics");

            // id: no distinct_count (high cardinality)
            assert_eq!(
                result.column_statistics[0].distinct_count,
                Precision::Absent,
                "id column should have Absent distinct_count"
            );

            // category: 10 distinct values
            assert_eq!(
                result.column_statistics[1].distinct_count,
                Precision::Exact(10),
                "category column should have Exact(10) distinct_count"
            );

            // name: 5 distinct values
            assert_eq!(
                result.column_statistics[2].distinct_count,
                Precision::Exact(5),
                "name column should have Exact(5) distinct_count"
            );
        }
    }
}
