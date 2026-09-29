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

//! Statistics computation for physical plans.
//!
//! [`StatisticsArgs`] provides external context to
//! [`ExecutionPlan::statistics_from_inputs`].

use crate::ExecutionPlan;
use crate::coalesce_partitions::CoalescePartitionsExec;
use crate::displayable;
use crate::operator_statistics::{
    ExtendedStatistics, StatisticsRegistry, StatisticsResult,
};
use crate::projection::ProjectionExec;
use crate::repartition::RepartitionExec;
use crate::sorts::sort::SortExec;
use datafusion_common::extensions::Extensions;
use datafusion_common::{
    Result, Statistics, assert_eq_or_internal_err, assert_or_internal_err,
};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::synopsis_registry::SynopsisRegistry;
use log::debug;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::from_ref;
use std::rc::Rc;
use std::sync::Arc;

type CacheKey = (usize, Option<usize>);

fn cache_key(plan: &dyn ExecutionPlan, partition: Option<usize>) -> CacheKey {
    (
        from_ref::<dyn ExecutionPlan>(plan).cast::<()>() as usize,
        partition,
    )
}

/// A provider's node-level and column-level extensions for one node, cached
/// together since they are always produced and consumed as a pair.
#[derive(Debug, Default, Clone)]
struct NodeExtensions {
    node: Extensions,
    columns: HashMap<usize, Extensions>,
}

/// Per-call memoization cache for statistics computation.
///
/// Keyed by `(plan node pointer address, partition)`. Shared across
/// a single statistics walk via [`StatisticsContext`].
///
/// The pointer-based key is safe within a single synchronous walk:
/// all `Arc<dyn ExecutionPlan>` nodes are held by the plan tree for
/// the duration of the walk, so addresses cannot be reused.
///
/// Core statistics and provider extensions are cached separately: the
/// `statistics` map is the hot path (populated on every walk); the `extensions`
/// map is populated only when a provider returns non-empty node-level or
/// column-level extensions, so a walk with no providers never touches it.
#[derive(Debug, Default)]
struct StatsCache {
    statistics: HashMap<CacheKey, Arc<Statistics>>,
    extensions: HashMap<CacheKey, NodeExtensions>,
}

/// Fields of [`StatisticsArgs`] that every node in a walk inherits, unlike
/// `partition`, which is chosen per node.
#[derive(Debug, Default, Clone)]
struct WalkScoped {
    synopsis_registry: Option<Arc<SynopsisRegistry>>,
}

/// Arguments passed to [`ExecutionPlan::statistics_from_inputs`] carrying
/// external information that operators can use when computing their
/// statistics.
#[derive(Debug, Default, Clone)]
pub struct StatisticsArgs {
    partition: Option<usize>,
    walk: WalkScoped,
}

impl StatisticsArgs {
    /// Creates new statistics arguments.
    ///
    /// By default the partition is set to `None` (statistics should be computed
    /// for the entire plan).
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the partition to compute statistics
    ///
    /// * `None` means statistics should be computed for the entire plan.
    /// * `Some(idx)` means statistics should be computed for the specified
    ///   partition index.
    pub fn set_partition(&mut self, partition: Option<usize>) {
        self.partition = partition;
    }

    /// Builder Style API for [`Self::set_partition`]
    pub fn with_partition(mut self, partition: Option<usize>) -> Self {
        self.set_partition(partition);
        self
    }

    /// Return the partition to compute statistics
    pub fn partition(&self) -> Option<usize> {
        self.partition
    }

    /// Sets the expression-level statistics providers for every node in the
    /// walk.
    pub(crate) fn with_synopsis_registry(
        mut self,
        synopsis_registry: Arc<SynopsisRegistry>,
    ) -> Self {
        self.walk.synopsis_registry = Some(synopsis_registry);
        self
    }

    /// The expression-level statistics providers for this walk, empty when
    /// none are registered.
    pub fn synopsis_registry(&self) -> &SynopsisRegistry {
        static EMPTY: SynopsisRegistry = SynopsisRegistry::new();
        self.walk.synopsis_registry.as_deref().unwrap_or(&EMPTY)
    }

    /// Arguments for a child resolved at `partition`, keeping the walk-scoped
    /// fields. [`Self::new`] would drop them.
    pub fn for_child(&self, partition: Option<usize>) -> Self {
        Self {
            partition,
            walk: self.walk.clone(),
        }
    }
}

/// Directive returned by [`ExecutionPlan::child_stats_requests`] describing
/// how the [`StatisticsContext`] should obtain each child's statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildStats {
    /// Compute the child's statistics at this partition (`None` = overall).
    At(Option<usize>),
    /// Skip this child; the parent does not need its statistics. A placeholder
    /// [`Statistics::new_unknown`] is supplied in its slot.
    Skip,
}

/// Owns the bottom-up traversal and per-walk memoization cache for statistics
/// computation. Call [`StatisticsContext::compute`] to walk a plan tree.
///
/// An optional [`StatisticsRegistry`] plugs providers into the walk: at each node
/// they are consulted before the operator's built-in
/// [`ExecutionPlan::statistics_from_inputs`]. An empty registry is the built-in
/// computation.
///
/// The walk carries [`ExtendedStatistics`], which holds both node-level
/// extensions and column-level extensions keyed by output column index. A node
/// has extensions only if a provider `Computed` them for it, with one
/// exception: a built-in node that moves values without changing them or the
/// rows (a projection, a coalesce, a sort or a repartition without a limit)
/// carries its child's column extensions to the output columns that hold the
/// same values. Any other built-in node yields no extensions and hides those of
/// everything beneath it.
/// [`Self::compute_extended`] observes extensions; [`Self::compute`] returns core
/// [`Statistics`] only.
pub struct StatisticsContext {
    cache: Rc<RefCell<StatsCache>>,
    registry: StatisticsRegistry,
    synopsis_registry: Option<Arc<SynopsisRegistry>>,
}

impl Default for StatisticsContext {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticsContext {
    /// Creates a context with an empty cache and no statistics providers.
    pub fn new() -> Self {
        Self::new_with_registry(StatisticsRegistry::new())
    }

    /// Creates a context whose walk consults `registry`'s provider chain.
    pub fn new_with_registry(registry: StatisticsRegistry) -> Self {
        Self {
            cache: Rc::new(RefCell::new(StatsCache::default())),
            registry,
            synopsis_registry: None,
        }
    }

    /// Makes `synopsis_registry` available to every node and provider in
    /// this walk.
    pub fn with_synopsis_registry(mut self, synopsis_registry: SynopsisRegistry) -> Self {
        self.synopsis_registry = Some(Arc::new(synopsis_registry));
        self
    }

    /// Clears the memoization cache.
    ///
    /// The cache is keyed by raw plan-node pointers, which are only stable
    /// while the current plan tree is alive. Reset between optimizer passes
    /// (which rewrite the plan) when reusing one context across them, so stale
    /// pointer keys cannot collide.
    pub fn reset_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        cache.statistics.clear();
        cache.extensions.clear();
    }

    /// Computes the core [`Statistics`] for `plan`, discarding any
    /// provider-supplied extensions (see [`Self::compute_extended`]).
    ///
    /// With no providers registered this is the plain built-in walk: only the
    /// `statistics` cache is touched, so it carries no extension overhead.
    ///
    /// # Example
    ///
    /// ```
    /// # use arrow::datatypes::{DataType, Field, Schema};
    /// # use datafusion_common::Statistics;
    /// # use datafusion_common::stats::Precision;
    /// # use datafusion_physical_plan::statistics::{StatisticsArgs, StatisticsContext};
    /// # use datafusion_physical_plan::test::exec::StatisticsExec;
    ///
    /// let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    /// let overall_stats =
    ///     Statistics::new_unknown(&schema).with_num_rows(Precision::Exact(100));
    /// let partition_stats = vec![
    ///     Statistics::new_unknown(&schema).with_num_rows(Precision::Exact(60)),
    ///     Statistics::new_unknown(&schema).with_num_rows(Precision::Exact(40)),
    /// ];
    /// let plan = StatisticsExec::new(overall_stats, schema)
    ///     .with_partition_statistics(partition_stats);
    ///
    /// let context = StatisticsContext::new();
    ///
    /// // Statistics for the whole plan (all partitions combined).
    /// let overall = context.compute(&plan, &StatisticsArgs::new())?;
    /// assert_eq!(overall.num_rows, Precision::Exact(100));
    ///
    /// // Statistics for a single partition.
    /// let args = StatisticsArgs::new().with_partition(Some(0));
    /// let per_partition = context.compute(&plan, &args)?;
    /// assert_eq!(per_partition.num_rows, Precision::Exact(60));
    /// # Ok::<(), datafusion_common::DataFusionError>(())
    /// ```
    pub fn compute(
        &self,
        plan: &dyn ExecutionPlan,
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        self.compute_base(plan, args)
    }

    /// Computes the [`ExtendedStatistics`] for `plan`: the core statistics plus
    /// any extensions a provider attached to this node (see the type-level docs
    /// for how extensions propagate up the tree).
    pub fn compute_extended(
        &self,
        plan: &dyn ExecutionPlan,
        args: &StatisticsArgs,
    ) -> Result<Arc<ExtendedStatistics>> {
        let statistics = self.compute_base(plan, args)?;
        let extensions = self
            .cached_extensions(plan, args.partition())
            .unwrap_or_default();
        Ok(Arc::new(ExtendedStatistics::new_with_extensions(
            statistics,
            extensions.node,
            extensions.columns,
        )))
    }

    /// Bottom-up walk producing the node's core statistics, resolving children
    /// first and consulting the provider chain before the operator's built-in
    /// [`ExecutionPlan::statistics_from_inputs`]. Any extensions a provider
    /// attaches are recorded in the extension cache for [`Self::compute_extended`].
    ///
    /// When `args.partition()` is `Some(idx)`, `idx` is validated against the
    /// plan's partition count.
    fn compute_base(
        &self,
        plan: &dyn ExecutionPlan,
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        // The context's registry applies to every node in the walk. Entry
        // points often pass bare `StatisticsArgs::new()`.
        let stamped;
        let args = match &self.synopsis_registry {
            Some(registry)
                if !args
                    .walk
                    .synopsis_registry
                    .as_ref()
                    .is_some_and(|r| Arc::ptr_eq(r, registry)) =>
            {
                stamped = args.clone().with_synopsis_registry(Arc::clone(registry));
                &stamped
            }
            _ => args,
        };
        let partition = args.partition();

        if let Some(idx) = partition {
            let partition_count = plan.properties().partitioning.partition_count();
            assert_or_internal_err!(
                idx < partition_count,
                "Invalid partition index: {}, the partition count is {}",
                idx,
                partition_count
            );
        }

        if let Some(cached) = self.cached_statistics(plan, partition) {
            return Ok(cached);
        }

        let children = plan.children();
        // Try providers before resolving the operator's own children, so a
        // provider that overrides this node is not blocked by the fallback walk.
        let statistics = match self.try_provider_stats(plan, &children, args)? {
            Some(statistics) => statistics,
            None => {
                let requests = plan.child_stats_requests(partition);
                self.validate_child_requests(plan, &children, &requests)?;
                let child_statistics =
                    self.resolve_children(plan, &children, &requests, args)?;
                let statistics = plan.statistics_from_inputs(&child_statistics, args)?;
                self.forward_column_extensions(plan, &children, &requests, partition);
                statistics
            }
        };
        self.store_statistics(plan, partition, Arc::clone(&statistics));
        Ok(statistics)
    }

    /// Validates child stat `requests` against `plan`'s children: the count must
    /// match, and each `At(Some(idx))` must be a valid partition of that child.
    fn validate_child_requests(
        &self,
        plan: &dyn ExecutionPlan,
        children: &[&Arc<dyn ExecutionPlan>],
        requests: &[ChildStats],
    ) -> Result<()> {
        assert_eq_or_internal_err!(
            requests.len(),
            children.len(),
            "{} child_stats_requests returned {} entries for {} children",
            plan.name(),
            requests.len(),
            children.len()
        );
        for (child, directive) in children.iter().zip(requests) {
            if let ChildStats::At(Some(idx)) = directive {
                let count = child.properties().partitioning.partition_count();
                assert_or_internal_err!(
                    *idx < count,
                    "{} requested invalid partition {idx} for child {} with {count} partitions",
                    plan.name(),
                    child.name()
                );
            }
        }
        Ok(())
    }

    /// Resolves each child's core statistics per `requests`: computes the child
    /// at the requested partition (memoized), or supplies a
    /// [`Statistics::new_unknown`] placeholder for [`ChildStats::Skip`]. Callers
    /// must validate `requests` via [`Self::validate_child_requests`] first.
    fn resolve_children(
        &self,
        plan: &dyn ExecutionPlan,
        children: &[&Arc<dyn ExecutionPlan>],
        requests: &[ChildStats],
        args: &StatisticsArgs,
    ) -> Result<Vec<Arc<Statistics>>> {
        children
            .iter()
            .zip(requests)
            .enumerate()
            .map(|(i, (child, directive))| match directive {
                ChildStats::At(p) => self
                    .compute_base(child.as_ref(), &args.for_child(*p))
                    .map_err(|e| {
                        e.context(format!(
                            "computing statistics for child {i} ({}) of {} at partition {p:?}",
                            child.name(),
                            plan.name()
                        ))
                    }),
                ChildStats::Skip => {
                    Ok(Arc::new(Statistics::new_unknown(child.schema().as_ref())))
                }
            })
            .collect()
    }

    /// Runs the provider chain, returning the first `Computed` result's core
    /// statistics (and recording its extensions), or `None` if the chain is empty
    /// or all delegate. A partition-blind provider applies only to overall stats
    /// (its default `compute_statistics_with_args` delegates per partition).
    ///
    /// Each provider's child statistics come from its own
    /// [`child_stats_requests`](crate::operator_statistics::StatisticsProvider::child_stats_requests)
    /// and are memoized, so a walk with no providers pays nothing.
    fn try_provider_stats(
        &self,
        plan: &dyn ExecutionPlan,
        children: &[&Arc<dyn ExecutionPlan>],
        args: &StatisticsArgs,
    ) -> Result<Option<Arc<Statistics>>> {
        let providers = self.registry.providers();
        if providers.is_empty() {
            return Ok(None);
        }
        let partition = args.partition();
        for provider in providers {
            if !provider.matches(plan) {
                continue;
            }
            let requests = provider.child_stats_requests(plan, partition);
            self.validate_child_requests(plan, children, &requests)?;
            // A provider's child walk is speculative: on failure, skip the provider
            // so a later one or the operator fallback can handle the node. Not
            // error-swallowing, whoever genuinely needs the child resolves it again
            // and the error resurfaces there; a matched provider's own `compute`
            // error below stays fatal.
            let child_statistics = match self
                .resolve_children(plan, children, &requests, args)
            {
                Ok(child_statistics) => child_statistics,
                Err(e) => {
                    debug!(
                        "Statistics provider {provider:?} skipped for {}: child statistics resolution failed: {e}",
                        displayable(plan).one_line().to_string().trim_end()
                    );
                    continue;
                }
            };
            let child_extended =
                self.child_extended_stats(children, &requests, &child_statistics);
            if let StatisticsResult::Computed(computed) =
                provider.compute_statistics_with_args(plan, &child_extended, args)?
            {
                if !computed.extensions().is_empty()
                    || !computed.column_extensions().is_empty()
                {
                    self.store_extensions(
                        plan,
                        partition,
                        computed.extensions().clone(),
                        computed.column_extensions().clone(),
                    );
                }
                return Ok(Some(Arc::clone(computed.base_arc())));
            }
        }
        Ok(None)
    }

    /// Carries the child's column extensions through a built-in operator that
    /// moves values without changing them or the rows: a projection keeps them
    /// on each output column that is an input column, and a repartition (for
    /// the whole output), coalesce or sort without a limit keeps them on the
    /// same column. Any other built-in operator drops them, since an extension
    /// of unknown type need not stay valid when rows change.
    fn forward_column_extensions(
        &self,
        plan: &dyn ExecutionPlan,
        children: &[&Arc<dyn ExecutionPlan>],
        requests: &[ChildStats],
        partition: Option<usize>,
    ) {
        let ([child], [ChildStats::At(child_partition)]) = (children, requests) else {
            return;
        };
        let Some(child_extensions) =
            self.cached_extensions(child.as_ref(), *child_partition)
        else {
            return;
        };
        let columns: HashMap<usize, Extensions> =
            if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
                let Ok(mapping) = projection
                    .projection_expr()
                    .projection_mapping(&child.schema())
                else {
                    return;
                };
                mapping
                    .iter()
                    .filter_map(|(source, targets)| {
                        let column = source.downcast_ref::<Column>()?;
                        let extensions = child_extensions.columns.get(&column.index())?;
                        Some(
                            targets
                                .iter()
                                .map(|(_, output)| (*output, extensions.clone())),
                        )
                    })
                    .flatten()
                    .collect()
            } else if plan.fetch().is_none()
                && (plan.downcast_ref::<CoalescePartitionsExec>().is_some()
                    || plan.downcast_ref::<SortExec>().is_some()
                    || (partition.is_none()
                        && plan.downcast_ref::<RepartitionExec>().is_some()))
            {
                child_extensions.columns
            } else {
                return;
            };
        if !columns.is_empty() {
            self.store_extensions(plan, partition, Extensions::default(), columns);
        }
    }

    /// Pairs each child's core statistics with any extensions cached for it,
    /// producing the [`ExtendedStatistics`] the provider chain consumes. Called
    /// only when providers exist, so an empty registry pays no extension cost.
    fn child_extended_stats(
        &self,
        children: &[&Arc<dyn ExecutionPlan>],
        requests: &[ChildStats],
        child_statistics: &[Arc<Statistics>],
    ) -> Vec<ExtendedStatistics> {
        children
            .iter()
            .zip(requests)
            .zip(child_statistics)
            .map(|((child, directive), statistics)| {
                let extensions = match directive {
                    ChildStats::At(p) => self.cached_extensions(child.as_ref(), *p),
                    ChildStats::Skip => None,
                };
                match extensions {
                    Some(extensions) => ExtendedStatistics::new_with_extensions(
                        Arc::clone(statistics),
                        extensions.node,
                        extensions.columns,
                    ),
                    None => ExtendedStatistics::new_arc(Arc::clone(statistics)),
                }
            })
            .collect()
    }

    fn cached_statistics(
        &self,
        plan: &dyn ExecutionPlan,
        partition: Option<usize>,
    ) -> Option<Arc<Statistics>> {
        self.cache
            .borrow()
            .statistics
            .get(&cache_key(plan, partition))
            .cloned()
    }

    fn store_statistics(
        &self,
        plan: &dyn ExecutionPlan,
        partition: Option<usize>,
        statistics: Arc<Statistics>,
    ) {
        self.cache
            .borrow_mut()
            .statistics
            .insert(cache_key(plan, partition), statistics);
    }

    fn cached_extensions(
        &self,
        plan: &dyn ExecutionPlan,
        partition: Option<usize>,
    ) -> Option<NodeExtensions> {
        self.cache
            .borrow()
            .extensions
            .get(&cache_key(plan, partition))
            .cloned()
    }

    fn store_extensions(
        &self,
        plan: &dyn ExecutionPlan,
        partition: Option<usize>,
        extensions: Extensions,
        column_extensions: HashMap<usize, Extensions>,
    ) {
        self.cache.borrow_mut().extensions.insert(
            cache_key(plan, partition),
            NodeExtensions {
                node: extensions,
                columns: column_extensions,
            },
        );
    }
}

#[cfg(all(test, feature = "test_utils"))]
mod tests {
    use super::*;
    use crate::filter::FilterExec;
    use crate::operator_statistics::StatisticsProvider;
    use crate::test::exec::StatisticsExec;
    use crate::union::UnionExec;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::{ColumnStatistics, stats::Precision};
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, col, lit};

    /// Overall-only provider: sets a fixed row count for any node.
    #[derive(Debug)]
    struct FixedRowCountProvider(usize);
    impl StatisticsProvider for FixedRowCountProvider {
        fn compute_statistics(
            &self,
            plan: &dyn ExecutionPlan,
            _child_stats: &[ExtendedStatistics],
        ) -> Result<StatisticsResult> {
            let mut stats = Statistics::new_unknown(&plan.schema());
            stats.num_rows = Precision::Exact(self.0);
            Ok(StatisticsResult::Computed(ExtendedStatistics::new(stats)))
        }
    }

    /// Partition-aware provider: encodes the requested partition in the row count.
    #[derive(Debug)]
    struct PartitionRowCountProvider;
    impl StatisticsProvider for PartitionRowCountProvider {
        fn compute_statistics_with_args(
            &self,
            plan: &dyn ExecutionPlan,
            _child_stats: &[ExtendedStatistics],
            args: &StatisticsArgs,
        ) -> Result<StatisticsResult> {
            let marker = 700 + args.partition().map_or(0, |p| p + 1);
            let mut stats = Statistics::new_unknown(&plan.schema());
            stats.num_rows = Precision::Exact(marker);
            Ok(StatisticsResult::Computed(ExtendedStatistics::new(stats)))
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Tag(u32);

    /// Leaf provider: sets a row count and attaches a `Tag` extension.
    #[derive(Debug)]
    struct TagLeafProvider {
        rows: usize,
        tag: u32,
    }
    impl StatisticsProvider for TagLeafProvider {
        fn compute_statistics(
            &self,
            plan: &dyn ExecutionPlan,
            child_stats: &[ExtendedStatistics],
        ) -> Result<StatisticsResult> {
            if !child_stats.is_empty() {
                return Ok(StatisticsResult::Delegate);
            }
            let mut stats = Statistics::new_unknown(&plan.schema());
            stats.num_rows = Precision::Exact(self.rows);
            let mut extended = ExtendedStatistics::new(stats);
            extended.set_extension(Tag(self.tag));
            Ok(StatisticsResult::Computed(extended))
        }
    }

    /// Non-leaf provider: re-emits a `Tag` doubled from the first child's `Tag`,
    /// proving the child's extension reached this provider.
    #[derive(Debug)]
    struct TagDoublingProvider;
    impl StatisticsProvider for TagDoublingProvider {
        fn compute_statistics(
            &self,
            plan: &dyn ExecutionPlan,
            child_stats: &[ExtendedStatistics],
        ) -> Result<StatisticsResult> {
            let Some(Tag(v)) = child_stats.first().and_then(|c| c.get_extension::<Tag>())
            else {
                return Ok(StatisticsResult::Delegate);
            };
            let mut extended =
                ExtendedStatistics::new(Statistics::new_unknown(&plan.schema()));
            extended.set_extension(Tag(v * 2));
            Ok(StatisticsResult::Computed(extended))
        }
    }

    fn ctx_with(provider: Arc<dyn StatisticsProvider>) -> StatisticsContext {
        StatisticsContext::new_with_registry(StatisticsRegistry::with_providers(vec![
            provider,
        ]))
    }

    fn make_stats_leaf(num_rows: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let col_stats = vec![ColumnStatistics {
            null_count: Precision::Exact(0),
            max_value: Precision::Absent,
            min_value: Precision::Absent,
            sum_value: Precision::Absent,
            distinct_count: Precision::Absent,
            byte_size: Precision::Absent,
        }];
        Arc::new(StatisticsExec::new(
            Statistics {
                num_rows: Precision::Exact(num_rows),
                total_byte_size: Precision::Absent,
                column_statistics: col_stats,
            },
            schema,
        ))
    }

    #[test]
    fn coalesce_returns_overall_stats_for_any_partition() {
        let leaf = make_stats_leaf(100);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(leaf));

        let ctx = StatisticsContext::new();
        let stats = ctx
            .compute(
                plan.as_ref(),
                &StatisticsArgs::new().with_partition(Some(0)),
            )
            .unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(100));

        let stats_none = ctx.compute(plan.as_ref(), &StatisticsArgs::new()).unwrap();
        assert_eq!(stats_none.num_rows, Precision::Exact(100));
    }

    #[test]
    fn context_caches_within_walk() {
        let leaf = make_stats_leaf(42);
        let ctx = StatisticsContext::new();
        let args = StatisticsArgs::new();

        let s1 = ctx.compute(leaf.as_ref(), &args).unwrap();
        assert!(!ctx.cache.borrow().statistics.is_empty());

        let s2 = ctx.compute(leaf.as_ref(), &args).unwrap();
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn reset_cache_clears_entries() {
        let leaf = make_stats_leaf(10);
        let ctx = StatisticsContext::new();
        let _ = ctx.compute(leaf.as_ref(), &StatisticsArgs::new()).unwrap();
        assert!(!ctx.cache.borrow().statistics.is_empty());
        ctx.reset_cache();
        assert!(ctx.cache.borrow().statistics.is_empty());
    }

    #[test]
    fn partition_aware_provider_applies_per_partition() {
        let leaf = make_stats_leaf(10);
        let ctx = ctx_with(Arc::new(PartitionRowCountProvider));

        let per_part = ctx
            .compute(
                leaf.as_ref(),
                &StatisticsArgs::new().with_partition(Some(0)),
            )
            .unwrap();
        assert_eq!(per_part.num_rows, Precision::Exact(701));
    }

    #[test]
    fn extensions_reach_parent_provider() {
        let leaf = make_stats_leaf(100);
        let parent: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(leaf));
        let ctx = StatisticsContext::new_with_registry(
            StatisticsRegistry::with_providers(vec![
                Arc::new(TagLeafProvider { rows: 100, tag: 7 }),
                Arc::new(TagDoublingProvider),
            ]),
        );
        let extended = ctx
            .compute_extended(parent.as_ref(), &StatisticsArgs::new())
            .unwrap();
        assert_eq!(extended.get_extension::<Tag>(), Some(&Tag(14)));
    }

    #[test]
    fn builtin_fallback_drops_extensions() {
        let leaf = make_stats_leaf(100);
        let parent: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&leaf)));
        let ctx = ctx_with(Arc::new(TagLeafProvider { rows: 100, tag: 7 }));

        let leaf_extended = ctx
            .compute_extended(leaf.as_ref(), &StatisticsArgs::new())
            .unwrap();
        assert_eq!(leaf_extended.get_extension::<Tag>(), Some(&Tag(7)));

        let parent_extended = ctx
            .compute_extended(parent.as_ref(), &StatisticsArgs::new())
            .unwrap();
        assert_eq!(parent_extended.get_extension::<Tag>(), None);
        assert_eq!(parent_extended.base().num_rows, Precision::Exact(100));
    }

    #[derive(Debug, Clone, PartialEq)]
    struct ColumnTag(u32);

    /// Leaf provider: sets a row count and attaches a `ColumnTag` extension to
    /// output column 0.
    #[derive(Debug)]
    struct ColumnTagLeafProvider {
        rows: usize,
        tag: u32,
    }
    impl StatisticsProvider for ColumnTagLeafProvider {
        fn compute_statistics(
            &self,
            plan: &dyn ExecutionPlan,
            child_stats: &[ExtendedStatistics],
        ) -> Result<StatisticsResult> {
            if !child_stats.is_empty() {
                return Ok(StatisticsResult::Delegate);
            }
            let mut stats = Statistics::new_unknown(&plan.schema());
            stats.num_rows = Precision::Exact(self.rows);
            let mut extended = ExtendedStatistics::new(stats);
            extended.set_column_extension(0, ColumnTag(self.tag));
            Ok(StatisticsResult::Computed(extended))
        }
    }

    /// Non-leaf provider: passes the first child's `ExtendedStatistics` through
    /// unchanged, so a column extension attached below it reaches this node
    /// only if the walk carries it there.
    #[derive(Debug)]
    struct ColumnTagPassthroughProvider;
    impl StatisticsProvider for ColumnTagPassthroughProvider {
        fn compute_statistics(
            &self,
            _plan: &dyn ExecutionPlan,
            child_stats: &[ExtendedStatistics],
        ) -> Result<StatisticsResult> {
            let Some(first) = child_stats.first() else {
                return Ok(StatisticsResult::Delegate);
            };
            Ok(StatisticsResult::Computed(first.clone()))
        }
    }

    /// Attaches a `ColumnTag` to output column 1 of a leaf.
    #[derive(Debug)]
    struct TagSecondColumn;
    impl StatisticsProvider for TagSecondColumn {
        fn compute_statistics(
            &self,
            plan: &dyn ExecutionPlan,
            child_stats: &[ExtendedStatistics],
        ) -> Result<StatisticsResult> {
            if !child_stats.is_empty() {
                return Ok(StatisticsResult::Delegate);
            }
            let mut extended =
                ExtendedStatistics::new(Statistics::new_unknown(&plan.schema()));
            extended.set_column_extension(1, ColumnTag(7));
            Ok(StatisticsResult::Computed(extended))
        }
    }

    // A built-in projection, which here removes `a`, inserts a computed column
    // and repeats `b`, carries a column extension to every output column that
    // holds the same values, and so does a coalesce. A filter, which removes
    // rows, drops it.
    #[test]
    fn column_extension_follows_its_column_through_value_preserving_operators()
    -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let leaf: Arc<dyn ExecutionPlan> = Arc::new(StatisticsExec::new(
            Statistics::new_unknown(&schema),
            schema.clone(),
        ));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
            vec![
                (col("b", &schema)?, "b".to_string()),
                (
                    Arc::new(BinaryExpr::new(
                        col("a", &schema)?,
                        Operator::Plus,
                        lit(1i32),
                    )),
                    "c".to_string(),
                ),
                (col("b", &schema)?, "b2".to_string()),
            ],
            leaf,
        )?);
        let coalesce: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&projection)));
        let ctx = ctx_with(Arc::new(TagSecondColumn));

        let out = ctx.compute_extended(coalesce.as_ref(), &StatisticsArgs::new())?;
        assert_eq!(
            out.get_column_extension::<ColumnTag>(0),
            Some(&ColumnTag(7))
        );
        assert_eq!(out.get_column_extension::<ColumnTag>(1), None);
        assert_eq!(
            out.get_column_extension::<ColumnTag>(2),
            Some(&ColumnTag(7))
        );

        let predicate = Arc::new(BinaryExpr::new(
            col("b", &projection.schema())?,
            Operator::Gt,
            lit(0i32),
        ));
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(predicate, coalesce)?);
        let out = ctx.compute_extended(filter.as_ref(), &StatisticsArgs::new())?;
        assert_eq!(out.get_column_extension::<ColumnTag>(0), None);
        Ok(())
    }

    #[test]
    fn column_extension_reaches_provider_parent_but_drops_at_builtin_filter() {
        let leaf = make_stats_leaf(100);
        let parent: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&leaf)));

        // A provider-handled parent preserves the column extension published
        // at the leaf.
        let ctx = StatisticsContext::new_with_registry(
            StatisticsRegistry::with_providers(vec![
                Arc::new(ColumnTagLeafProvider { rows: 100, tag: 9 }),
                Arc::new(ColumnTagPassthroughProvider),
            ]),
        );
        let leaf_extended = ctx
            .compute_extended(leaf.as_ref(), &StatisticsArgs::new())
            .unwrap();
        assert_eq!(
            leaf_extended.get_column_extension::<ColumnTag>(0),
            Some(&ColumnTag(9))
        );
        let parent_extended = ctx
            .compute_extended(parent.as_ref(), &StatisticsArgs::new())
            .unwrap();
        assert_eq!(
            parent_extended.get_column_extension::<ColumnTag>(0),
            Some(&ColumnTag(9))
        );

        // A built-in node that removes rows drops the column extension, the
        // same rule as for node-level extensions.
        let predicate = Arc::new(BinaryExpr::new(
            col("a", &leaf.schema()).unwrap(),
            Operator::Gt,
            lit(0i32),
        ));
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(predicate, Arc::clone(&leaf)).unwrap());
        let builtin_ctx = ctx_with(Arc::new(ColumnTagLeafProvider { rows: 100, tag: 9 }));
        let builtin_filter_extended = builtin_ctx
            .compute_extended(filter.as_ref(), &StatisticsArgs::new())
            .unwrap();
        assert_eq!(
            builtin_filter_extended.get_column_extension::<ColumnTag>(0),
            None
        );
    }

    #[test]
    fn per_partition_union_with_registry_no_out_of_bounds() {
        // Two 2-partition inputs -> 4 output partitions. Union owns output
        // partition 3 via its second input (owning_input(3) = (1, 1)); the first
        // input is Skipped, so the walk supplies a placeholder for it (never
        // resolving it at partition 3, which is out of that input's 0..2 range).
        // An overall-only provider delegates for a specific partition, so p3 keeps
        // the operator's honest per-partition answer while the overall request
        // picks up the provider's row count.
        let union =
            UnionExec::try_new(vec![make_stats_leaf(10), make_stats_leaf(20)]).unwrap();
        let ctx = ctx_with(Arc::new(FixedRowCountProvider(999)));

        let p3 = ctx
            .compute(
                union.as_ref(),
                &StatisticsArgs::new().with_partition(Some(3)),
            )
            .unwrap();
        assert_eq!(p3.num_rows, Precision::Absent);

        let overall = ctx.compute(union.as_ref(), &StatisticsArgs::new()).unwrap();
        assert_eq!(overall.num_rows, Precision::Exact(999));
    }
}
