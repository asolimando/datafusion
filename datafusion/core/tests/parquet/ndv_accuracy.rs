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

//! Accuracy of the distinct count estimated from Parquet metadata, measured on
//! TPC-H data generated in process and compared with the exact counts.
//!
//! `tpchgen` is deterministic, so the snapshot only changes when the estimate
//! does: a change to the estimator shows, column by column, which estimates got
//! closer to the truth and which moved away.

use std::fmt::Write as _;
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::stats::Precision;
use datafusion::physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tpchgen::generators::{
    CustomerGenerator, LineItemGenerator, NationGenerator, OrderGenerator, PartGenerator,
    PartSuppGenerator, RegionGenerator, SupplierGenerator,
};

/// Scale factor of the generated data: 60,175 `lineitem` rows
const SCALE_FACTOR: f64 = 0.01;

/// The worst q-error, `max(estimate / truth, truth / estimate)`, any column
/// may have. A column beyond it is a regression of the estimator, unless the
/// snapshot change explains why.
const MAX_Q_ERROR: f64 = 2.0;

/// The geometric mean of the q-errors of all columns may not exceed this
const MAX_GEOMEAN_Q_ERROR: f64 = 1.1;

fn decimal() -> DataType {
    DataType::Decimal128(15, 2)
}

/// The TPC-H tables, with the Arrow schema `tpchgen-cli` writes, as rows in the
/// `|` separated `dbgen` format
fn tpch_tables() -> Vec<(&'static str, SchemaRef, String)> {
    use DataType::{Date32, Int32, Int64, Utf8};
    let schema = |columns: &[(&str, DataType)]| {
        Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, data_type)| Field::new(*name, data_type.clone(), false))
                .collect::<Vec<_>>(),
        ))
    };
    fn rows<T: std::fmt::Display>(rows: impl IntoIterator<Item = T>) -> String {
        let mut text = String::new();
        for row in rows {
            writeln!(text, "{row}").unwrap();
        }
        text
    }
    let sf = SCALE_FACTOR;
    vec![
        (
            "nation",
            schema(&[
                ("n_nationkey", Int64),
                ("n_name", Utf8),
                ("n_regionkey", Int64),
                ("n_comment", Utf8),
            ]),
            rows(NationGenerator::new(sf, 1, 1)),
        ),
        (
            "region",
            schema(&[
                ("r_regionkey", Int64),
                ("r_name", Utf8),
                ("r_comment", Utf8),
            ]),
            rows(RegionGenerator::new(sf, 1, 1)),
        ),
        (
            "part",
            schema(&[
                ("p_partkey", Int64),
                ("p_name", Utf8),
                ("p_mfgr", Utf8),
                ("p_brand", Utf8),
                ("p_type", Utf8),
                ("p_size", Int32),
                ("p_container", Utf8),
                ("p_retailprice", decimal()),
                ("p_comment", Utf8),
            ]),
            rows(PartGenerator::new(sf, 1, 1)),
        ),
        (
            "supplier",
            schema(&[
                ("s_suppkey", Int64),
                ("s_name", Utf8),
                ("s_address", Utf8),
                ("s_nationkey", Int64),
                ("s_phone", Utf8),
                ("s_acctbal", decimal()),
                ("s_comment", Utf8),
            ]),
            rows(SupplierGenerator::new(sf, 1, 1)),
        ),
        (
            "partsupp",
            schema(&[
                ("ps_partkey", Int64),
                ("ps_suppkey", Int64),
                ("ps_availqty", Int32),
                ("ps_supplycost", decimal()),
                ("ps_comment", Utf8),
            ]),
            rows(PartSuppGenerator::new(sf, 1, 1)),
        ),
        (
            "customer",
            schema(&[
                ("c_custkey", Int64),
                ("c_name", Utf8),
                ("c_address", Utf8),
                ("c_nationkey", Int64),
                ("c_phone", Utf8),
                ("c_acctbal", decimal()),
                ("c_mktsegment", Utf8),
                ("c_comment", Utf8),
            ]),
            rows(CustomerGenerator::new(sf, 1, 1)),
        ),
        (
            "orders",
            schema(&[
                ("o_orderkey", Int64),
                ("o_custkey", Int64),
                ("o_orderstatus", Utf8),
                ("o_totalprice", decimal()),
                ("o_orderdate", Date32),
                ("o_orderpriority", Utf8),
                ("o_clerk", Utf8),
                ("o_shippriority", Int32),
                ("o_comment", Utf8),
            ]),
            rows(OrderGenerator::new(sf, 1, 1)),
        ),
        (
            "lineitem",
            schema(&[
                ("l_orderkey", Int64),
                ("l_partkey", Int64),
                ("l_suppkey", Int64),
                ("l_linenumber", Int32),
                ("l_quantity", decimal()),
                ("l_extendedprice", decimal()),
                ("l_discount", decimal()),
                ("l_tax", decimal()),
                ("l_returnflag", Utf8),
                ("l_linestatus", Utf8),
                ("l_shipdate", Date32),
                ("l_commitdate", Date32),
                ("l_receiptdate", Date32),
                ("l_shipinstruct", Utf8),
                ("l_shipmode", Utf8),
                ("l_comment", Utf8),
            ]),
            rows(LineItemGenerator::new(sf, 1, 1)),
        ),
    ]
}

/// Parses `dbgen` rows: `|` separated, with a trailing `|`
fn parse(schema: &SchemaRef, text: &str) -> Vec<RecordBatch> {
    let mut fields: Vec<Field> =
        schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("trailing", DataType::Utf8, true));
    let with_trailing = Arc::new(Schema::new(fields));
    let projection: Vec<usize> = (0..schema.fields().len()).collect();
    ReaderBuilder::new(with_trailing)
        .with_delimiter(b'|')
        .with_header(false)
        .with_projection(projection)
        .build(Cursor::new(text.as_bytes()))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

/// File layouts: the estimate depends on how the data is written, the truth
/// does not
fn layouts() -> Vec<(&'static str, WriterProperties)> {
    vec![
        ("one row group", WriterProperties::builder().build()),
        (
            "row groups of 8192 rows, pages of 1024 rows",
            WriterProperties::builder()
                .set_max_row_group_row_count(Some(8192))
                .set_data_page_row_count_limit(1024)
                .build(),
        ),
    ]
}

/// The exact number of distinct values of a column
async fn exact_distinct_count(ctx: &SessionContext, table: &str, column: &str) -> u64 {
    let batches = ctx
        .sql(&format!("SELECT COUNT(DISTINCT {column}) FROM {table}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let counts = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    counts.value(0) as u64
}

#[tokio::test]
async fn tpch_distinct_count_accuracy() {
    let dir = TempDir::new().unwrap();
    let tables = tpch_tables();
    let mut report = String::new();
    for (layout, props) in layouts() {
        writeln!(report, "## {layout}").unwrap();
        writeln!(report, "column truth estimate q_error").unwrap();
        let mut config = SessionConfig::new();
        config
            .options_mut()
            .execution
            .parquet
            .estimate_distinct_count_from_metadata = true;
        let ctx = SessionContext::new_with_config(config);
        let mut q_errors = vec![];
        for (table, schema, text) in &tables {
            let path = dir
                .path()
                .join(format!("{table}-{}.parquet", q_errors.len()));
            let mut writer = ArrowWriter::try_new(
                std::fs::File::create(&path).unwrap(),
                Arc::clone(schema),
                Some(props.clone()),
            )
            .unwrap();
            for batch in parse(schema, text) {
                writer.write(&batch).unwrap();
            }
            writer.close().unwrap();
            ctx.register_parquet(
                *table,
                path.to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await
            .unwrap();

            let plan = ctx
                .table(*table)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let statistics = StatisticsContext::new()
                .compute(plan.as_ref(), &StatisticsArgs::new())
                .unwrap();
            let rows = statistics.num_rows.get_value().copied().unwrap();
            for (field, column) in
                schema.fields().iter().zip(&statistics.column_statistics)
            {
                let truth = exact_distinct_count(&ctx, table, field.name()).await;
                let estimate = match column.distinct_count {
                    Precision::Exact(n) | Precision::Inexact(n) => n as u64,
                    Precision::Absent => {
                        panic!("{table}.{} has no estimate", field.name())
                    }
                };
                assert!(
                    estimate <= rows as u64,
                    "{table}.{}: {estimate} distinct values in {rows} rows",
                    field.name()
                );
                let q_error = (estimate.max(1) as f64 / truth.max(1) as f64)
                    .max(truth.max(1) as f64 / estimate.max(1) as f64);
                q_errors.push(q_error);
                writeln!(
                    report,
                    "{table}.{} {truth} {estimate} {q_error:.2}",
                    field.name()
                )
                .unwrap();
                assert!(
                    q_error <= MAX_Q_ERROR,
                    "{layout}: {table}.{} estimated {estimate}, truth {truth}",
                    field.name()
                );
            }
        }
        let geomean =
            (q_errors.iter().map(|q| q.ln()).sum::<f64>() / q_errors.len() as f64).exp();
        writeln!(report, "geomean q_error {geomean:.3}\n").unwrap();
        assert!(
            geomean <= MAX_GEOMEAN_Q_ERROR,
            "{layout}: geomean q-error {geomean:.3}"
        );
    }
    insta::assert_snapshot!(report);
}
