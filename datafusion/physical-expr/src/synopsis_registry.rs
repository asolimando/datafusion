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

//! Expression-level statistics: the provider chain and the walk over an
//! expression tree.
//!
//! This mirrors the operator level one level down:
//!
//! | operator level           | expression level                |
//! |--------------------------|---------------------------------|
//! | `StatisticsContext`      | [`SynopsisContext`]             |
//! | `StatisticsRegistry`     | [`SynopsisRegistry`]            |
//! | `StatisticsProvider`     | [`SynopsisProvider`]            |
//! | `statistics_from_inputs` | `synopsis_from_inputs`          |
//!
//! An expression implements only its own rule, in
//! [`PhysicalExpr::synopsis_from_inputs`]. The [`SynopsisContext`]
//! walks the expression tree and caches the results. The
//! [`SynopsisRegistry`] holds the custom providers, which take precedence
//! over the built-in rules.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use datafusion_common::extensions::Extensions;
use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, Statistics};
use datafusion_expr_common::interval_arithmetic::Interval;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::synopsis::{ExprSynopsis, SynopsisArgs};

use crate::expressions::Column;
use crate::filter_statistics::{
    collect_equality_columns, collect_null_rejecting_columns,
    column_statistics_from_boundaries, column_statistics_from_selectivity,
    distinct_count_after_filter,
};
use crate::intervals::utils::check_support;
use crate::{AnalysisContext, ExprBoundaries, analyze};

/// Result of attempting to compute a synopsis with a [`SynopsisProvider`],
/// the expression-level counterpart of `StatisticsResult`.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "returned once per provider call and matched at once"
)]
pub enum SynopsisResult {
    /// The synopsis was computed by this provider.
    Computed(ExprSynopsis),
    /// This provider does not handle this expression; delegate to the next
    /// provider in the chain, and then to the built-in rules.
    Delegate,
}

/// A custom source of expression synopses, the expression-level counterpart
/// of `StatisticsProvider`.
///
/// A provider receives the [`SynopsisContext`], so it can compute
/// sub-expressions through the whole chain, including the other providers,
/// in the same way as the built-in rules. The context adds no extensions to a
/// provider's answer: the provider decides them, and can read the input's
/// through [`SynopsisContext::get_extension`] and
/// [`SynopsisContext::get_column_extension`].
pub trait SynopsisProvider: Debug + Send + Sync {
    /// Returns the synopsis of `expr`, or [`SynopsisResult::Delegate`] to
    /// leave `expr` to the next provider and then to the built-in rules.
    fn compute_synopsis(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        ctx: &SynopsisContext,
    ) -> SynopsisResult;
}

/// An ordered chain of [`SynopsisProvider`]s, the expression-level
/// counterpart of `StatisticsRegistry`.
///
/// The registry holds no per-walk state, so one registry can serve many
/// contexts. Providers are asked in order, and the first one that returns
/// [`SynopsisResult::Computed`] supplies the synopsis.
#[derive(Debug, Default, Clone)]
pub struct SynopsisRegistry {
    providers: Vec<Arc<dyn SynopsisProvider>>,
}

impl SynopsisRegistry {
    /// Creates a new empty registry.
    pub const fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Creates a registry with the given provider chain, in priority order
    /// (first match wins).
    pub fn with_providers(providers: Vec<Arc<dyn SynopsisProvider>>) -> Self {
        Self { providers }
    }

    /// Register a provider at the front of the chain (higher priority).
    pub fn register(&mut self, provider: Arc<dyn SynopsisProvider>) {
        self.providers.insert(0, provider);
    }

    /// Returns the current provider chain.
    pub fn providers(&self) -> &[Arc<dyn SynopsisProvider>] {
        &self.providers
    }
}

/// An empty registry for [`SynopsisContext::new`]. The context borrows its
/// registry, so the registry must outlive it.
static EMPTY: SynopsisRegistry = SynopsisRegistry {
    providers: Vec::new(),
};

/// One cache entry: the computed synopsis, or `None` while computation of the
/// same expression is still in progress (the re-entrancy guard) or when
/// nothing computed it.
type CacheEntry = Option<ExprSynopsis>;

/// The `analyze()` outcome for one expression: its selectivity and the
/// interval-analysis boundaries of every input column under it.
#[derive(Debug)]
struct AnalyzeResult {
    selectivity: Option<f64>,
    boundaries: Vec<ExprBoundaries>,
}

/// The `analyze()` outcome per expression, `None` when there is none.
type AnalysisCache = HashMap<Arc<dyn PhysicalExpr>, Option<Arc<AnalyzeResult>>>;

/// Walks an expression tree and caches the synopsis of each expression, the
/// expression-level counterpart of `StatisticsContext`.
///
/// The cache is keyed by the expression itself, through the `DynEq` and
/// `DynHash` bounds of `PhysicalExpr`, so two structurally equal
/// sub-expressions share one entry even when they are separate allocations.
/// Hashing a key walks its sub-tree.
#[derive(Debug)]
pub struct SynopsisContext<'a> {
    args: SynopsisArgs<'a>,
    registry: &'a SynopsisRegistry,
    cache: RefCell<HashMap<Arc<dyn PhysicalExpr>, CacheEntry>>,
    /// The providers' answer per expression, `None` when none answers.
    provider_cache: RefCell<HashMap<Arc<dyn PhysicalExpr>, Option<ExprSynopsis>>>,
    /// Per-expression cache of whether an expression's subtree (itself and
    /// every descendant) contains no provider answer.
    provider_free_cache: RefCell<HashMap<Arc<dyn PhysicalExpr>, bool>>,
    analysis_cache: RefCell<AnalysisCache>,
    /// The input's node-level extensions, when the caller supplies them.
    node_extensions: Option<&'a Extensions>,
    /// The input's per-column extensions, keyed by column index, when the
    /// caller supplies them.
    column_extensions: Option<&'a HashMap<usize, Extensions>>,
    /// The precomputed facts about the condition in [`Self::args`], when the
    /// context was derived with [`Self::given`].
    condition_estimate: Option<ConditionEstimate>,
}
/// The facts about a condition that every column computed under it reuses,
/// computed once when a context is derived with [`SynopsisContext::given`].
#[derive(Debug)]
struct ConditionEstimate {
    /// The fraction of input rows that satisfy the condition.
    selectivity: f64,
    /// The input row count scaled by `selectivity`.
    filtered_num_rows: Precision<usize>,
    /// The interval-analysis boundaries of every input column under the
    /// condition, or `None` when `check_support` rejects the condition or the
    /// analysis fails.
    boundaries: Option<Vec<ExprBoundaries>>,
    /// The columns that the condition equates to one non-null literal.
    equality_columns: HashSet<usize>,
    /// The columns that cannot be NULL in a row that satisfies the condition.
    null_rejecting_columns: HashSet<usize>,
}

impl ConditionEstimate {
    /// `boundaries` is the interval-analysis boundaries of every input
    /// column under `condition`, precomputed by the caller (normally reused
    /// from the context's `analyze()` cache), or `None` when `check_support`
    /// rejects `condition` or `analyze()` failed.
    fn new(
        args: &SynopsisArgs,
        condition: &Arc<dyn PhysicalExpr>,
        selectivity: f64,
        boundaries: Option<Vec<ExprBoundaries>>,
    ) -> Self {
        let input_stats = args.input_stats();
        let (equality_columns, _) = collect_equality_columns(condition);
        Self {
            selectivity,
            filtered_num_rows: input_stats
                .num_rows
                .with_estimated_selectivity(selectivity),
            boundaries,
            equality_columns,
            null_rejecting_columns: collect_null_rejecting_columns(condition),
        }
    }

    /// The statistics of the input column at `index`, restricted to the rows
    /// that satisfy the condition.
    fn column_statistics(
        &self,
        index: usize,
        args: &SynopsisArgs,
    ) -> Option<ColumnStatistics> {
        let input_stats = args.input_stats();
        let input = input_stats.column_statistics.get(index)?;
        let data_type = args.input_schema().fields().get(index)?.data_type();
        let null_rejecting = self.null_rejecting_columns.contains(&index);
        let mut column = match &self.boundaries {
            Some(boundaries) => column_statistics_from_boundaries(
                data_type,
                input,
                boundaries.get(index)?.clone(),
                self.selectivity,
                null_rejecting,
                self.filtered_num_rows,
            ),
            None => column_statistics_from_selectivity(
                input,
                self.selectivity,
                null_rejecting,
                self.equality_columns.contains(&index),
                self.filtered_num_rows,
            ),
        };
        column.distinct_count = distinct_count_after_filter(
            column.distinct_count,
            input_stats.num_rows,
            self.filtered_num_rows,
        );
        Some(column)
    }
}

impl<'a> SynopsisContext<'a> {
    /// Creates a context with no registered providers, so every expression
    /// is computed through the built-in rules.
    pub fn new(input_stats: &'a Statistics, input_schema: &'a Schema) -> Self {
        Self::new_with_registry(input_stats, input_schema, &EMPTY)
    }

    /// Creates a context whose walk consults `registry`'s provider chain
    /// before falling back to the built-in rules.
    pub fn new_with_registry(
        input_stats: &'a Statistics,
        input_schema: &'a Schema,
        registry: &'a SynopsisRegistry,
    ) -> Self {
        Self {
            args: SynopsisArgs::new(input_stats, input_schema),
            registry,
            cache: RefCell::new(HashMap::new()),
            provider_cache: RefCell::new(HashMap::new()),
            provider_free_cache: RefCell::new(HashMap::new()),
            analysis_cache: RefCell::new(HashMap::new()),
            node_extensions: None,
            column_extensions: None,
            condition_estimate: None,
        }
    }
    /// Derives a context that computes expressions over only the input rows
    /// that satisfy `condition`, with its own cache. Providers see `condition`
    /// through [`SynopsisArgs::condition`]. A condition already set on this
    /// context is replaced, not combined.
    ///
    /// `selectivity` is the fraction of rows that satisfy `condition`; the
    /// caller passes the value it uses for its row count, so the column
    /// statistics agree with it. Interval analysis of `condition` runs once and
    /// serves every column.
    pub fn given<'b>(
        &'b self,
        condition: &'b Arc<dyn PhysicalExpr>,
        selectivity: f64,
    ) -> SynopsisContext<'b> {
        let args = self.args.with_condition(condition);
        let boundaries = self
            .analyze_once(condition)
            .map(|analysis| analysis.boundaries.clone());
        SynopsisContext {
            args,
            registry: self.registry,
            cache: RefCell::new(HashMap::new()),
            provider_cache: RefCell::new(HashMap::new()),
            provider_free_cache: RefCell::new(HashMap::new()),
            analysis_cache: RefCell::new(HashMap::new()),
            node_extensions: self.node_extensions,
            column_extensions: self.column_extensions,
            condition_estimate: Some(ConditionEstimate::new(
                &args,
                condition,
                selectivity,
                boundaries,
            )),
        }
    }

    /// Returns this context with the caller's default selectivity, which a rule
    /// may use for a predicate that nothing estimates.
    pub fn with_default_selectivity(mut self, default_selectivity: f64) -> Self {
        self.args = self.args.with_default_selectivity(default_selectivity);
        self
    }

    /// The arguments every expression in this walk is computed against, the
    /// same ones the built-in rules see.
    pub fn args(&self) -> &SynopsisArgs<'a> {
        &self.args
    }

    /// Attach the input's node-level and per-column extensions, so a
    /// provider can read metadata the input carries beyond plain
    /// [`Statistics`] (for example, cross-column correlation).
    pub fn with_extensions(
        mut self,
        node: &'a Extensions,
        columns: &'a HashMap<usize, Extensions>,
    ) -> Self {
        self.node_extensions = Some(node);
        self.column_extensions = Some(columns);
        self
    }

    /// Get a reference to a node-level extension of the input, if the
    /// context was built with [`Self::with_extensions`] and the input
    /// carries one of type `T`.
    pub fn get_extension<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.node_extensions?.get::<T>()
    }

    /// Get a reference to an extension of type `T` that the input carries for
    /// the column at `index`, if the context was built with
    /// [`Self::with_extensions`].
    pub fn get_column_extension<T: 'static + Send + Sync>(
        &self,
        index: usize,
    ) -> Option<&T> {
        self.column_extensions?.get(&index)?.get::<T>()
    }

    /// Computes the synopsis of `expr`, caching it by structural expression
    /// identity.
    ///
    /// The registered providers are asked first. Otherwise a Boolean
    /// expression whose whole subtree interval analysis supports, with no
    /// provider answer inside, takes its selectivity from one `analyze()`
    /// call. Otherwise the expression's own rule combines its children's
    /// synopses. A missing range comes from `evaluate_bounds` on the
    /// children's ranges, for a provider's answer too.
    pub fn compute(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<ExprSynopsis> {
        // The borrow ends here, before any recursive call.
        if let Some(cached) = self.cache.borrow().get(expr) {
            return cached.clone();
        }

        // Insert `None` first, so a provider that asks for this same expression
        // receives `None` instead of recursing indefinitely.
        self.cache.borrow_mut().insert(Arc::clone(expr), None);

        // This fails only if the context was created with the wrong schema.
        let Ok(data_type) = expr.data_type(self.args.input_schema()) else {
            return None;
        };

        let mut result = self
            .provider_answer(expr, &data_type)
            .map(|synopsis| self.fill_missing_range(expr, &data_type, synopsis));
        if result.is_none() {
            result = self.compute_supported_subtree(expr, &data_type);
        }
        if result.is_none() {
            result = self.compute_from_children(expr, &data_type);
        }
        if let Some(synopsis) = result.as_mut() {
            self.fill_missing_distinct_count(synopsis);
        }

        self.cache
            .borrow_mut()
            .insert(Arc::clone(expr), result.clone());
        result
    }

    /// The answer is cached per expression, so a provider is asked at most
    /// once for the same expression in this context, whether the caller is
    /// [`Self::compute`] or [`Self::subtree_free_of_providers`].
    fn provider_answer(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        data_type: &DataType,
    ) -> Option<ExprSynopsis> {
        if let Some(cached) = self.provider_cache.borrow().get(expr) {
            return cached.clone();
        }
        // Guard against a provider that asks for its own expression's answer.
        self.provider_cache
            .borrow_mut()
            .insert(Arc::clone(expr), None);
        let result = self.registry.providers.iter().find_map(|p| {
            match p.compute_synopsis(expr, self) {
                SynopsisResult::Computed(synopsis) => {
                    let conformed = conform(synopsis, data_type);
                    if conformed.is_none() {
                        log::debug!(
                            "ignoring {p:?}: selectivity on non-Boolean expression {expr}"
                        );
                    }
                    conformed
                }
                SynopsisResult::Delegate => None,
            }
        });
        self.provider_cache
            .borrow_mut()
            .insert(Arc::clone(expr), result.clone());
        result
    }

    /// Whether `expr`'s subtree, `expr` itself and every descendant, contains
    /// no provider answer.
    fn subtree_free_of_providers(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        if let Some(cached) = self.provider_free_cache.borrow().get(expr) {
            return *cached;
        }
        let free = match expr.data_type(self.args.input_schema()) {
            Ok(data_type) => {
                self.provider_answer(expr, &data_type).is_none()
                    && expr
                        .children()
                        .iter()
                        .all(|child| self.subtree_free_of_providers(child))
            }
            Err(_) => false,
        };
        self.provider_free_cache
            .borrow_mut()
            .insert(Arc::clone(expr), free);
        free
    }

    /// Computes `expr`'s selectivity from one `analyze()` call on its own
    /// subtree. Returns `None` when `expr` is not Boolean, when a provider
    /// answers inside its subtree, or when `analyze()` gives no selectivity.
    /// A non-leaf `expr` does not compute its children, since `analyze()`
    /// already covers the whole subtree.
    fn compute_supported_subtree(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        data_type: &DataType,
    ) -> Option<ExprSynopsis> {
        if *data_type != DataType::Boolean || !self.subtree_free_of_providers(expr) {
            return None;
        }
        let analysis = self.analyze_once(expr)?;
        let selectivity = analysis.selectivity?;
        let children = expr.children();
        let mut synopsis = if children.is_empty() {
            expr.synopsis_from_inputs(&self.args, &[])
                .and_then(|s| conform(s, data_type))
                .unwrap_or_else(|| ExprSynopsis::unknown(data_type.clone()))
        } else {
            let mut synopsis = ExprSynopsis::unknown(data_type.clone());
            synopsis.column.null_count = self.operand_null_count(expr, &children);
            synopsis
        };
        if let Some(column) = self.conditioned_column(expr, data_type) {
            synopsis = column;
        }
        synopsis.selectivity = Some(selectivity);
        self.attach_column_extensions(expr, &mut synopsis);
        Some(synopsis)
    }

    /// Computes `expr` by computing its children through the full chain,
    /// applying `expr`'s own rule to their synopses, and filling a missing
    /// range from their ranges.
    fn compute_from_children(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        data_type: &DataType,
    ) -> Option<ExprSynopsis> {
        let children = expr.children();
        let child_synopses: Vec<Option<ExprSynopsis>> =
            children.iter().copied().map(|c| self.compute(c)).collect();

        // A child with no synopsis is passed to the rule as an unknown
        // synopsis of its type, so the rule can still use what the other
        // children know (for example, `AND` with one known side). The rule
        // is skipped only when a child's type cannot be computed.
        let computed: Option<Vec<ExprSynopsis>> = children
            .iter()
            .zip(&child_synopses)
            .map(|(child, synopsis)| match synopsis {
                Some(synopsis) => Some(synopsis.clone()),
                None => child
                    .data_type(self.args.input_schema())
                    .ok()
                    .map(ExprSynopsis::unknown),
            })
            .collect();
        let mut synopsis = computed.and_then(|computed| {
            expr.synopsis_from_inputs(&self.args, &computed)
                .and_then(|s| conform(s, data_type))
        });

        if let Some(column) = self.conditioned_column(expr, data_type) {
            synopsis = Some(column);
        }

        // Leaves (no children) keep their own min/max, set by the `Column` and
        // `Literal` rules.
        if !children.is_empty() {
            synopsis = self.merge_evaluated_range(
                expr,
                &children,
                &child_synopses,
                data_type,
                synopsis,
            );
        }

        if let Some(synopsis) = synopsis.as_mut() {
            self.attach_column_extensions(expr, synopsis);
        }
        synopsis
    }

    /// The null count of a predicate whose operands are not Boolean, such as a
    /// comparison, from its own rule over its operands, which are cheap to
    /// compute. Interval analysis gives no null count.
    fn operand_null_count(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        children: &[&Arc<dyn PhysicalExpr>],
    ) -> Precision<usize> {
        let schema = self.args.input_schema();
        let not_boolean = children.iter().all(|child| {
            child
                .data_type(schema)
                .is_ok_and(|data_type| data_type != DataType::Boolean)
        });
        if !not_boolean {
            return Precision::Absent;
        }
        let operands: Option<Vec<ExprSynopsis>> =
            children.iter().map(|child| self.compute(child)).collect();
        operands
            .and_then(|operands| expr.synopsis_from_inputs(&self.args, &operands))
            .map_or(Precision::Absent, |synopsis| synopsis.column.null_count)
    }

    /// Runs `analyze()` on `expr`'s own subtree and caches the outcome, so
    /// the same expression is analyzed at most once in this context. Returns
    /// `None` when `check_support` rejects `expr` or `analyze()` fails.
    fn analyze_once(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<Arc<AnalyzeResult>> {
        if let Some(cached) = self.analysis_cache.borrow().get(expr) {
            return cached.clone();
        }
        let schema = self.args.input_schema();
        let result = if check_support(expr, &Arc::new(schema.clone())) {
            AnalysisContext::try_from_statistics(
                schema,
                &self.args.input_stats().column_statistics,
            )
            .and_then(|input| analyze(expr, input, schema))
            .inspect_err(|e| {
                log::debug!(
                    "interval analysis failed for `{expr}`, estimating without it: {e}"
                )
            })
            .ok()
            .map(|analysis| {
                Arc::new(AnalyzeResult {
                    selectivity: analysis.selectivity,
                    boundaries: analysis.boundaries,
                })
            })
        } else {
            None
        };
        self.analysis_cache
            .borrow_mut()
            .insert(Arc::clone(expr), result.clone());
        result
    }

    /// The built-in synopsis of a `Column` under the condition of this
    /// context, or `None` when `expr` is not a `Column` or the context has no
    /// condition.
    fn conditioned_column(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        data_type: &DataType,
    ) -> Option<ExprSynopsis> {
        let estimate = self.condition_estimate.as_ref()?;
        let column = expr.downcast_ref::<Column>()?;
        let statistics = estimate.column_statistics(column.index(), &self.args)?;
        Some(ExprSynopsis::from_column(statistics, data_type.clone()))
    }

    /// Merges in the per-column extensions that the input carries, for a
    /// `Column` whose index has an entry. A built-in rule above a `Column`
    /// cannot carry an opaque extension forward, so this is the only place
    /// where a per-column extension enters a synopsis without a provider.
    fn attach_column_extensions(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        synopsis: &mut ExprSynopsis,
    ) {
        let Some(column_extensions) = self.column_extensions else {
            return;
        };
        let Some(column) = expr.downcast_ref::<Column>() else {
            return;
        };
        let Some(extensions) = column_extensions.get(&column.index()) else {
            return;
        };
        synopsis.extensions.merge(extensions);
    }

    /// Fills a missing distinct count with the number of values in the range,
    /// or in the type for a Boolean, when that is below the row count: the
    /// input's, or under a condition the rows that satisfy it. A larger number
    /// says nothing new, since there is at most one value per row.
    fn fill_missing_distinct_count(&self, synopsis: &mut ExprSynopsis) {
        if synopsis.column.distinct_count != Precision::Absent {
            return;
        }
        let interval = match (
            synopsis.column.min_value.get_value(),
            synopsis.column.max_value.get_value(),
        ) {
            (Some(min), Some(max)) => Interval::try_new(min.clone(), max.clone()).ok(),
            _ if synopsis.data_type == DataType::Boolean => {
                Interval::make_unbounded(&DataType::Boolean).ok()
            }
            _ => None,
        };
        let num_rows = match &self.condition_estimate {
            Some(estimate) => estimate.filtered_num_rows,
            None => self.args.input_stats().num_rows,
        };
        let (Some(cardinality), Some(&rows)) = (
            interval.and_then(|interval| interval.cardinality()),
            num_rows.get_value(),
        ) else {
            return;
        };
        if let Ok(cardinality) = usize::try_from(cardinality)
            && cardinality < rows
        {
            synopsis.column.distinct_count = Precision::Inexact(cardinality);
        }
    }

    /// Fills a missing range of a provider's answer for a non-leaf `expr`,
    /// computing its children only when needed.
    fn fill_missing_range(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        data_type: &DataType,
        synopsis: ExprSynopsis,
    ) -> ExprSynopsis {
        let children = expr.children();
        let has_range = synopsis.column.min_value != Precision::Absent
            && synopsis.column.max_value != Precision::Absent;
        if children.is_empty() || has_range {
            return synopsis;
        }
        let child_synopses: Vec<Option<ExprSynopsis>> =
            children.iter().map(|child| self.compute(child)).collect();
        self.merge_evaluated_range(
            expr,
            &children,
            &child_synopses,
            data_type,
            Some(synopsis.clone()),
        )
        .unwrap_or(synopsis)
    }

    /// Computes `expr`'s range from its children's ranges with
    /// [`PhysicalExpr::evaluate_bounds`] and fills the missing min/max of
    /// `synopsis`, creating one that carries only the range when `synopsis` is
    /// `None`.
    fn merge_evaluated_range(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        children: &[&Arc<dyn PhysicalExpr>],
        child_synopses: &[Option<ExprSynopsis>],
        data_type: &DataType,
        synopsis: Option<ExprSynopsis>,
    ) -> Option<ExprSynopsis> {
        let intervals: Option<Vec<Interval>> = children
            .iter()
            .copied()
            .zip(child_synopses)
            .map(|(child, child_synopsis)| {
                child_interval(child, child_synopsis.as_ref(), self.args.input_schema())
            })
            .collect();
        let Some(intervals) = intervals else {
            return synopsis;
        };
        let interval_refs: Vec<&Interval> = intervals.iter().collect();
        let Ok(range) = expr.evaluate_bounds(&interval_refs) else {
            return synopsis;
        };
        Some(apply_range(synopsis, &range, data_type))
    }
}

/// Builds the `Interval` a child contributes to `evaluate_bounds`: its own
/// bounds when both are known, otherwise an unbounded interval of its type.
/// A child with no synopsis at all also gets an unbounded interval, of the
/// child expression's own type.
fn child_interval(
    child: &Arc<dyn PhysicalExpr>,
    synopsis: Option<&ExprSynopsis>,
    input_schema: &Schema,
) -> Option<Interval> {
    match synopsis {
        Some(synopsis) => {
            match (
                synopsis.column.min_value.get_value(),
                synopsis.column.max_value.get_value(),
            ) {
                (Some(min), Some(max)) => {
                    Interval::try_new(min.clone(), max.clone()).ok()
                }
                _ => Interval::make_unbounded(&synopsis.data_type).ok(),
            }
        }
        None => {
            let data_type = child.data_type(input_schema).ok()?;
            Interval::make_unbounded(&data_type).ok()
        }
    }
}

/// Fills a missing min/max from `range`'s finite endpoints, as `Inexact`: an
/// envelope need not be reached, for example the minimum of `a + b` can be
/// above `min(a) + min(b)`. A min or max already set is kept.
fn apply_range(
    synopsis: Option<ExprSynopsis>,
    range: &Interval,
    data_type: &DataType,
) -> ExprSynopsis {
    let mut synopsis =
        synopsis.unwrap_or_else(|| ExprSynopsis::unknown(data_type.clone()));
    let lower = range.lower();
    let upper = range.upper();
    if !lower.is_null() && synopsis.column.min_value == Precision::Absent {
        synopsis.column.min_value = Precision::Inexact(lower.clone());
    }
    if !upper.is_null() && synopsis.column.max_value == Precision::Absent {
        synopsis.column.max_value = Precision::Inexact(upper.clone());
    }
    synopsis
}

/// Sets the synopsis type from the schema, whatever the provider or rule
/// declared, and rejects a selectivity on a non-Boolean expression, where it
/// has no meaning.
fn conform(mut synopsis: ExprSynopsis, data_type: &DataType) -> Option<ExprSynopsis> {
    synopsis.data_type = data_type.clone();
    (synopsis.selectivity.is_none() || *data_type == DataType::Boolean)
        .then_some(synopsis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::{BinaryExpr, CastExpr, Column, LikeExpr, NotExpr, lit};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::stats::Precision;
    use datafusion_common::{ColumnStatistics, ScalarValue};
    use datafusion_expr::Operator;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn cs(ndv: usize, min: f64, max: f64) -> ColumnStatistics {
        ColumnStatistics {
            distinct_count: Precision::Exact(ndv),
            min_value: Precision::Exact(ScalarValue::Float64(Some(min))),
            max_value: Precision::Exact(ScalarValue::Float64(Some(max))),
            ..ColumnStatistics::new_unknown()
        }
    }

    /// One column of the given type, with the given NDV, plus the schema it
    /// is defined against.
    fn one_col_stats(ndv: usize, data_type: DataType) -> (Statistics, Schema) {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![cs(ndv, 0.0, 100.0)],
        };
        let schema = Schema::new(vec![Field::new("col0", data_type, true)]);
        (stats, schema)
    }

    #[test]
    fn column_leaf_is_computed_and_structurally_equal_subexpressions_share_a_cache_entry()
    {
        let (stats, schema) = one_col_stats(7, DataType::Float64);
        let counter = Arc::new(AtomicUsize::new(0));
        let registry = SynopsisRegistry::with_providers(vec![Arc::new(
            CountComputations(Arc::clone(&counter)),
        )]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let x: Arc<dyn PhysicalExpr> = Arc::new(Column::new("x", 0));
        assert_eq!(
            ctx.compute(&x).and_then(|s| s.ndv()),
            Some(7),
            "a leaf Column's synopsis comes from the input statistics"
        );

        // Two separate allocations with identical structure.
        let build = || -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("x", 0)),
                Operator::Plus,
                lit(1.0_f64),
            ))
        };
        let first = build();
        let second = build();

        ctx.compute(&first);
        let after_first = counter.load(Ordering::Relaxed);
        assert!(after_first > 0, "the first computation consults the chain");

        ctx.compute(&second);
        assert_eq!(
            counter.load(Ordering::Relaxed),
            after_first,
            "the structurally equal expression is served from the cache"
        );
    }

    /// Counts how many expressions reach the provider chain, which happens only
    /// on a cache miss.
    #[derive(Debug)]
    struct CountComputations(Arc<AtomicUsize>);

    impl SynopsisProvider for CountComputations {
        fn compute_synopsis(
            &self,
            _expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            self.0.fetch_add(1, Ordering::Relaxed);
            SynopsisResult::Delegate
        }
    }

    #[test]
    fn injective_ndv_rule_through_nested_expression_type_aware_and_unknown_fallback() {
        let (float_stats, float_schema) = one_col_stats(42, DataType::Float64);
        let float_ctx = SynopsisContext::new(&float_stats, &float_schema);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let a_plus_1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&a),
            Operator::Plus,
            lit(1.0_f64),
        ));
        let nested: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(a_plus_1, Operator::Minus, lit(2.0_f64)));
        assert_eq!(
            float_ctx.compute(&nested).map(|s| s.column.distinct_count),
            Some(Precision::Inexact(42)),
            "the count carries through both operators, inexact for a floating \
             point column"
        );

        let a_mod_5: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(a, Operator::Modulo, lit(5.0_f64)));
        assert!(
            float_ctx.compute(&a_mod_5).is_none(),
            "Modulo has no rule, so it has an unknown synopsis even though both \
             operands are known"
        );

        let (int_stats, int_schema) = one_col_stats(42, DataType::Int64);
        let int_ctx = SynopsisContext::new(&int_stats, &int_schema);
        let b: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 0));
        let b_plus_1: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(b, Operator::Plus, lit(1_i64)));
        assert_eq!(
            int_ctx.compute(&b_plus_1).map(|s| s.column.distinct_count),
            Some(Precision::Exact(42)),
            "the count carries through unchanged, exact for an integer column"
        );

        let b_plus_float: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 0)),
            Operator::Plus,
            lit(1.0_f64),
        ));
        assert_eq!(
            int_ctx
                .compute(&b_plus_float)
                .map(|s| s.column.distinct_count),
            Some(Precision::Inexact(42)),
            "an integer operand with a floating point result is inexact"
        );
    }

    // Adding a non-NULL constant is NULL exactly where the other operand is,
    // so the result keeps that operand's null count.
    #[test]
    fn adding_a_constant_keeps_the_null_count() {
        let (mut stats, schema) = one_col_stats(42, DataType::Int64);
        stats.column_statistics[0].null_count = Precision::Exact(3);
        let ctx = SynopsisContext::new(&stats, &schema);
        let b_plus_1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 0)),
            Operator::Plus,
            lit(1_i64),
        ));
        assert_eq!(
            ctx.compute(&b_plus_1).map(|s| s.column.null_count),
            Some(Precision::Exact(3))
        );
    }

    // A NULL constant makes every result NULL, so the other operand's count
    // carries over only as an estimate, even in integer arithmetic.
    #[test]
    fn adding_a_null_constant_is_inexact() {
        let (stats, schema) = one_col_stats(42, DataType::Int64);
        let ctx = SynopsisContext::new(&stats, &schema);
        let b_plus_null: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 0)),
            Operator::Plus,
            lit(ScalarValue::Int64(None)),
        ));
        assert_eq!(
            ctx.compute(&b_plus_null).map(|s| s.column.distinct_count),
            Some(Precision::Inexact(42))
        );
    }

    /// Overrides any binary expression, so the built-in rule for it is bypassed.
    #[derive(Debug)]
    struct OverrideBinary {
        ndv: usize,
    }

    impl SynopsisProvider for OverrideBinary {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(_) = expr.downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            SynopsisResult::Computed(ExprSynopsis::from_column(
                ColumnStatistics {
                    distinct_count: Precision::Exact(self.ndv),
                    ..ColumnStatistics::new_unknown()
                },
                DataType::Float64,
            ))
        }
    }

    /// Asks the context for its own expression.
    #[derive(Debug)]
    struct AsksForItself;

    impl SynopsisProvider for AsksForItself {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            ctx: &SynopsisContext,
        ) -> SynopsisResult {
            match ctx.compute(expr) {
                Some(synopsis) => SynopsisResult::Computed(synopsis),
                None => SynopsisResult::Delegate,
            }
        }
    }

    // The provider receives `None` for its own expression, so the built-in rule
    // supplies the synopsis.
    #[test]
    fn provider_computing_its_own_expression_terminates() {
        let (stats, schema) = one_col_stats(42, DataType::Float64);
        let registry = SynopsisRegistry::with_providers(vec![Arc::new(AsksForItself)]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        assert_eq!(ctx.compute(&a).and_then(|s| s.ndv()), Some(42));
    }

    /// Supplies correct statistics with the wrong data type.
    #[derive(Debug)]
    struct MislabelledColumn;

    impl SynopsisProvider for MislabelledColumn {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(_) = expr.downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };
            SynopsisResult::Computed(ExprSynopsis::from_column(
                ColumnStatistics {
                    distinct_count: Precision::Exact(7),
                    ..ColumnStatistics::new_unknown()
                },
                DataType::Utf8,
            ))
        }
    }

    #[test]
    fn synopsis_type_follows_the_input_schema() {
        let (stats, schema) = one_col_stats(42, DataType::Float64);
        let registry =
            SynopsisRegistry::with_providers(vec![Arc::new(MislabelledColumn)]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let synopsis = ctx.compute(&a).expect("the provider supplies a synopsis");
        assert_eq!(synopsis.data_type, DataType::Float64);
        assert_eq!(
            synopsis.ndv(),
            Some(7),
            "the provider's statistics are kept"
        );
    }

    /// Sets a selectivity on any column and declares the synopsis Boolean.
    #[derive(Debug)]
    struct SelectivityOnAnyColumn;

    impl SynopsisProvider for SelectivityOnAnyColumn {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(_) = expr.downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };
            SynopsisResult::Computed(ExprSynopsis {
                selectivity: Some(0.5),
                ..ExprSynopsis::unknown(DataType::Boolean)
            })
        }
    }

    #[test]
    fn selectivity_on_non_boolean_expression_is_rejected() {
        let (stats, schema) = one_col_stats(42, DataType::Float64);
        let registry =
            SynopsisRegistry::with_providers(vec![Arc::new(SelectivityOnAnyColumn)]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let synopsis = ctx
            .compute(&a)
            .expect("the built-in rule computes the column");
        assert_eq!(
            synopsis.selectivity, None,
            "the provider's synopsis is rejected"
        );
        assert_eq!(
            synopsis.ndv(),
            Some(42),
            "the built-in column lookup supplies the synopsis instead"
        );
    }

    // A provider's answer replaces the built-in rule's.
    #[test]
    fn provider_answer_replaces_builtin() {
        let (stats, schema) = one_col_stats(42, DataType::Float64);
        let registry =
            SynopsisRegistry::with_providers(vec![Arc::new(OverrideBinary { ndv: 999 })]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let a_plus_1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Plus,
            lit(1.0_f64),
        ));
        assert_eq!(ctx.compute(&a_plus_1).and_then(|s| s.ndv()), Some(999));
    }

    /// Knows the selectivity of a named Boolean column, for example the
    /// result of a predicate evaluated earlier.
    #[derive(Debug)]
    struct ColumnSelectivity {
        column: &'static str,
        selectivity: f64,
    }
    impl SynopsisProvider for ColumnSelectivity {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(col) = expr.downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };
            if col.name() != self.column {
                return SynopsisResult::Delegate;
            }
            SynopsisResult::Computed(ExprSynopsis {
                selectivity: Some(self.selectivity),
                ..ExprSynopsis::unknown(DataType::Boolean)
            })
        }
    }

    /// Combines two predicates under `AND` as the product of their
    /// selectivities. It computes its children through the context, so the
    /// rest of the chain answers for them.
    #[derive(Debug)]
    struct AndCombiner;
    impl SynopsisProvider for AndCombiner {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            if *binary.op() != Operator::And {
                return SynopsisResult::Delegate;
            }
            let Some(left) = ctx.compute(binary.left()).and_then(|s| s.selectivity)
            else {
                return SynopsisResult::Delegate;
            };
            let Some(right) = ctx.compute(binary.right()).and_then(|s| s.selectivity)
            else {
                return SynopsisResult::Delegate;
            };
            SynopsisResult::Computed(ExprSynopsis {
                selectivity: Some(left * right),
                ..ExprSynopsis::unknown(DataType::Boolean)
            })
        }
    }

    #[test]
    fn provider_computes_children_through_chain_for_and() {
        let stats = Statistics {
            num_rows: Precision::Exact(1000),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                ColumnStatistics::new_unknown(),
                ColumnStatistics::new_unknown(),
            ],
        };
        let schema = Schema::new(vec![
            Field::new("p", DataType::Boolean, true),
            Field::new("q", DataType::Boolean, true),
        ]);
        let p: Arc<dyn PhysicalExpr> = Arc::new(Column::new("p", 0));
        let q: Arc<dyn PhysicalExpr> = Arc::new(Column::new("q", 1));
        let pred: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(p, Operator::And, q));

        let registry = SynopsisRegistry::with_providers(vec![
            Arc::new(ColumnSelectivity {
                column: "q",
                selectivity: 0.5,
            }),
            Arc::new(ColumnSelectivity {
                column: "p",
                selectivity: 0.2,
            }),
            Arc::new(AndCombiner),
        ]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);
        let sel = ctx
            .compute(&pred)
            .and_then(|s| s.selectivity)
            .expect("AndCombiner computes both operands through the chain");
        assert!(
            (sel - 0.1).abs() < 1e-9,
            "AND selectivity = 0.2 * 0.5 = {sel}"
        );
    }

    /// A test-local stand-in for cross-column correlation metadata the input
    /// might carry: two columns tend to satisfy their comparisons together
    /// more often (or less often) than independence would predict.
    #[derive(Debug)]
    struct Correlation {
        left: usize,
        right: usize,
        coefficient: f64,
    }

    /// Matches `left_col > c1 AND right_col > c2` and returns a selectivity
    /// that blends the independent product with the correlation coefficient.
    /// This is a simple test stand-in, not a real correlation model.
    #[derive(Debug)]
    struct CorrelatedAnd;

    impl SynopsisProvider for CorrelatedAnd {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            if *binary.op() != Operator::And {
                return SynopsisResult::Delegate;
            }
            let Some(left_cmp) = binary.left().downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            let Some(right_cmp) = binary.right().downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            let Some(left_col) = left_cmp.left().downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };
            let Some(right_col) = right_cmp.left().downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };

            let Some(left_sel) = ctx.compute(binary.left()).and_then(|s| s.selectivity)
            else {
                return SynopsisResult::Delegate;
            };
            let Some(right_sel) = ctx.compute(binary.right()).and_then(|s| s.selectivity)
            else {
                return SynopsisResult::Delegate;
            };
            let Some(correlation) = ctx.get_extension::<Correlation>() else {
                return SynopsisResult::Delegate;
            };
            if correlation.left != left_col.index()
                || correlation.right != right_col.index()
            {
                return SynopsisResult::Delegate;
            }

            // Blend the independent product with the correlation coefficient,
            // clamped to a valid selectivity. A real model would replace this.
            let independent = left_sel * right_sel;
            let blended = (independent
                + correlation.coefficient * left_sel.min(right_sel))
            .clamp(0.0, 1.0);
            SynopsisResult::Computed(ExprSynopsis {
                selectivity: Some(blended),
                ..ExprSynopsis::unknown(DataType::Boolean)
            })
        }
    }

    // A provider that reads node-level correlation metadata through
    // `SynopsisContext::get_extension` returns a selectivity for a
    // conjunction that differs from the built-in estimate without that
    // metadata.
    #[test]
    fn correlated_conjunction_uses_node_level_extension() {
        let stats = Statistics {
            num_rows: Precision::Exact(1000),
            total_byte_size: Precision::Absent,
            column_statistics: vec![cs(10, 0.0, 100.0), cs(10, 0.0, 100.0)],
        };
        let schema = Schema::new(vec![
            Field::new("left", DataType::Float64, true),
            Field::new("right", DataType::Float64, true),
        ]);

        let left: Arc<dyn PhysicalExpr> = Arc::new(Column::new("left", 0));
        let right: Arc<dyn PhysicalExpr> = Arc::new(Column::new("right", 1));
        let left_cmp: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(left, Operator::Gt, lit(5.0_f64)));
        let right_cmp: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(right, Operator::Gt, lit(5.0_f64)));
        let conjunction: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(left_cmp, Operator::And, right_cmp));

        // Without the extension: interval analysis estimates the conjunction
        // under independence.
        let no_extensions_ctx = SynopsisContext::new(&stats, &schema);
        let independent_sel = no_extensions_ctx
            .compute(&conjunction)
            .and_then(|s| s.selectivity)
            .expect("the built-in estimate has a synopsis");

        // With the extension: the provider blends in the correlation.
        let node_extensions = {
            let mut ext = Extensions::new();
            ext.insert(Correlation {
                left: 0,
                right: 1,
                coefficient: 0.3,
            });
            ext
        };
        let column_extensions = HashMap::new();
        let registry = SynopsisRegistry::with_providers(vec![Arc::new(CorrelatedAnd)]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry)
            .with_extensions(&node_extensions, &column_extensions);
        let correlated_sel = ctx
            .compute(&conjunction)
            .and_then(|s| s.selectivity)
            .expect("CorrelatedAnd answers using the correlation extension");

        assert_ne!(
            correlated_sel, independent_sel,
            "the correlation extension changes the selectivity away from the \
             independent product"
        );
    }

    #[derive(Debug, Clone, PartialEq)]
    struct ColumnSketch(u64);

    // A `Column` picks up the per-column extensions that the input carries
    // for its index, with no provider involved, but an expression above it
    // does not. A context built without extensions attaches nothing.
    #[test]
    fn column_carries_its_per_column_extensions() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics::new_unknown()],
        };
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);

        let node_extensions = Extensions::new();
        let mut column_extensions = HashMap::new();
        let mut col0_extensions = Extensions::new();
        col0_extensions.insert(ColumnSketch(42));
        column_extensions.insert(0, col0_extensions);

        let ctx = SynopsisContext::new(&stats, &schema)
            .with_extensions(&node_extensions, &column_extensions);

        let x: Arc<dyn PhysicalExpr> = Arc::new(Column::new("x", 0));
        let synopsis = ctx.compute(&x).expect("Column has a synopsis");
        assert_eq!(
            synopsis.get_extension::<ColumnSketch>(),
            Some(&ColumnSketch(42)),
            "the per-column extension is attached to the Column synopsis"
        );

        let plus_one: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(x, Operator::Plus, lit(1_i64)));
        let plus_one_synopsis = ctx.compute(&plus_one).expect("Plus has a synopsis");
        assert!(
            !plus_one_synopsis.has_extension::<ColumnSketch>(),
            "the built-in Plus rule does not carry the extension forward"
        );

        let no_extensions_ctx = SynopsisContext::new(&stats, &schema);
        let x_again: Arc<dyn PhysicalExpr> = Arc::new(Column::new("x", 0));
        let no_extensions_synopsis = no_extensions_ctx
            .compute(&x_again)
            .expect("Column has a synopsis");
        assert!(
            !no_extensions_synopsis.has_extension::<ColumnSketch>(),
            "a context built without extensions attaches nothing by default"
        );
    }

    // A non-NULL row passes exactly one of `=` and `!=`, And keeps no more
    // rows than either side, Or keeps at least as many as either side and no
    // more than both together, and every selectivity is in [0, 1]. And is
    // unknown when neither side has a selectivity (here, a column that is not
    // a predicate and `Modulo`, which has no rule), so the caller's default
    // applies.
    #[test]
    fn comparison_and_logical_selectivity_rules() {
        let (stats, schema) = one_col_stats(10, DataType::Int64);
        let ctx = SynopsisContext::new(&stats, &schema);
        let selectivity = |expr: &Arc<dyn PhysicalExpr>| {
            let sel = ctx
                .compute(expr)
                .and_then(|s| s.selectivity)
                .expect("the predicate has a selectivity");
            assert!((0.0..=1.0).contains(&sel), "{expr}: {sel}");
            sel
        };
        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("col0", 0));

        let eq: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Eq, lit(5_i64)));
        let not_eq: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::NotEq, lit(5_i64)));
        let (eq_sel, not_eq_sel) = (selectivity(&eq), selectivity(&not_eq));
        assert!(
            (eq_sel + not_eq_sel - 1.0).abs() < 1e-9,
            "with no NULLs, a row passes exactly one of `=` and `!=`"
        );

        let and: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&eq),
            Operator::And,
            Arc::clone(&not_eq),
        ));
        assert!(
            selectivity(&and) <= eq_sel.min(not_eq_sel),
            "And keeps no more rows than either side"
        );

        let or: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(eq, Operator::Or, not_eq));
        let or_sel = selectivity(&or);
        assert!(
            eq_sel.max(not_eq_sel) <= or_sel && or_sel <= eq_sel + not_eq_sel,
            "Or keeps at least as many rows as either side and no more than both"
        );

        let no_rule: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&a),
            Operator::Modulo,
            lit(3_i64),
        ));
        let and_missing_side: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(a, Operator::And, no_rule));
        assert!(
            ctx.compute(&and_missing_side).is_none(),
            "And is unknown when neither side has a selectivity"
        );
    }

    /// `p_size` (Int32, 1 to 50, 50 distinct values) and `p_type` (Utf8,
    /// no statistics), over 1000 rows.
    fn part_stats() -> (Statistics, Schema) {
        let stats = Statistics {
            num_rows: Precision::Exact(1000),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                ColumnStatistics {
                    distinct_count: Precision::Exact(50),
                    min_value: Precision::Exact(ScalarValue::Int32(Some(1))),
                    max_value: Precision::Exact(ScalarValue::Int32(Some(50))),
                    ..ColumnStatistics::new_unknown()
                },
                ColumnStatistics::new_unknown(),
            ],
        };
        let schema = Schema::new(vec![
            Field::new("p_size", DataType::Int32, false),
            Field::new("p_type", DataType::Utf8, false),
        ]);
        (stats, schema)
    }

    // Interval analysis supplies the selectivity of a Boolean expression it
    // supports.
    #[test]
    fn interval_analysis_is_the_builtin_selectivity_of_a_supported_predicate() {
        let (stats, schema) = part_stats();
        let ctx = SynopsisContext::new(&stats, &schema);

        let p_size_gt_40: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("p_size", 0)),
            Operator::Gt,
            lit(40_i32),
        ));
        let selectivity = ctx
            .compute(&p_size_gt_40)
            .and_then(|s| s.selectivity)
            .expect("the predicate has a synopsis");
        let input =
            AnalysisContext::try_from_statistics(&schema, &stats.column_statistics)
                .unwrap();
        let analysis = analyze(&p_size_gt_40, input, &schema).unwrap();
        assert_eq!(Some(selectivity), analysis.selectivity);
    }

    // `p_size > 40 AND p_type LIKE '%BRASS%'`: interval analysis rejects the
    // whole predicate, because `LIKE` is not supported, but supports the
    // `p_size > 40` conjunct, which gets 0.2 from it. `LIKE` has no built-in
    // rule, so the conjunction keeps the supported conjunct's selectivity,
    // combined with the caller's default selectivity when there is one.
    #[test]
    fn and_combines_its_known_conjunct_with_the_default() {
        let (stats, schema) = part_stats();
        let ctx = SynopsisContext::new(&stats, &schema);

        let p_size_gt_40: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("p_size", 0)),
            Operator::Gt,
            lit(40_i32),
        ));
        let p_type_like: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
            false,
            false,
            Arc::new(Column::new("p_type", 1)),
            lit("%BRASS%"),
        ));
        assert!(
            ctx.compute(&p_type_like)
                .and_then(|s| s.selectivity)
                .is_none(),
            "LIKE has no selectivity of its own"
        );
        let and: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(p_size_gt_40, Operator::And, p_type_like));
        let selectivity = ctx
            .compute(&and)
            .and_then(|s| s.selectivity)
            .expect("the conjunction has a synopsis");
        assert!(
            (selectivity - 0.2).abs() < 1e-9,
            "the conjunction keeps the 0.2 of `p_size > 40`, got {selectivity}"
        );

        // With a default selectivity, the unknown conjunct contributes it.
        let ctx = SynopsisContext::new(&stats, &schema).with_default_selectivity(0.5);
        let selectivity = ctx
            .compute(&and)
            .and_then(|s| s.selectivity)
            .expect("the conjunction has a synopsis");
        assert!(
            (selectivity - 0.1).abs() < 1e-9,
            "0.2 for `p_size > 40` times the default 0.5, got {selectivity}"
        );
    }

    // Given `a = 5`, interval analysis narrows `a` to the single value 5, so
    // its conditioned synopsis has a minimum and maximum of 5 and one distinct
    // value. The same column computed without the condition keeps its input
    // statistics, because the conditioned results are cached apart. `b`,
    // which the condition does not constrain, keeps its range as `Inexact`,
    // and its distinct count of 800 is capped at the 20 rows that satisfy the
    // condition (1000 rows * 1 / 50) and then reduced to 13 for the values
    // whose rows the filter removes.
    #[test]
    fn column_under_condition_is_narrowed_by_interval_analysis() {
        let int = |v: i32| ScalarValue::Int32(Some(v));
        let stats = Statistics {
            num_rows: Precision::Exact(1000),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                ColumnStatistics {
                    distinct_count: Precision::Exact(50),
                    min_value: Precision::Exact(int(0)),
                    max_value: Precision::Exact(int(99)),
                    ..ColumnStatistics::new_unknown()
                },
                ColumnStatistics {
                    distinct_count: Precision::Exact(800),
                    min_value: Precision::Exact(int(0)),
                    max_value: Precision::Exact(int(999)),
                    ..ColumnStatistics::new_unknown()
                },
            ],
        };
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let ctx = SynopsisContext::new(&stats, &schema);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let b: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 1));
        let a_eq_5: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Eq, lit(5_i32)));
        let selectivity = ctx
            .compute(&a_eq_5)
            .and_then(|s| s.selectivity)
            .expect("the predicate has a synopsis");

        let filtered = ctx.given(&a_eq_5, selectivity);
        assert!(filtered.args().condition().is_some());
        assert!(ctx.args().condition().is_none());

        let a_given = filtered.compute(&a).expect("a has a synopsis").column;
        assert_eq!(a_given.min_value, Precision::Exact(int(5)));
        assert_eq!(a_given.max_value, Precision::Exact(int(5)));
        assert_eq!(a_given.distinct_count, Precision::Exact(1));
        assert_eq!(
            ctx.compute(&a).expect("a has a synopsis").column.min_value,
            Precision::Exact(int(0)),
            "without the condition, `a` keeps its input minimum"
        );

        let b_given = filtered.compute(&b).expect("b has a synopsis").column;
        assert_eq!(b_given.min_value, Precision::Inexact(int(0)));
        assert_eq!(b_given.max_value, Precision::Inexact(int(999)));
        assert_eq!(b_given.distinct_count, Precision::Inexact(13));
    }

    /// Matches a comparison of the given column with the given operator, so a
    /// test can answer for one conjunct without answering the other.
    #[derive(Debug)]
    struct OverrideComparison {
        column: &'static str,
        op: Operator,
        selectivity: f64,
    }
    impl SynopsisProvider for OverrideComparison {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            if *binary.op() != self.op {
                return SynopsisResult::Delegate;
            }
            let Some(col) = binary.left().downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };
            if col.name() != self.column {
                return SynopsisResult::Delegate;
            }
            SynopsisResult::Computed(ExprSynopsis {
                selectivity: Some(self.selectivity),
                ..ExprSynopsis::unknown(DataType::Boolean)
            })
        }
    }

    // `p_size > 40 AND p_size < 45`: without a provider, one `analyze()`
    // call over the whole conjunction gives its selectivity. With a provider
    // that answers only the left conjunct, the `AND` combines the provider's
    // answer with the right conjunct's own selectivity from interval
    // analysis.
    #[test]
    fn provider_on_one_conjunct_is_not_overridden_by_whole_subtree_analysis() {
        let (stats, schema) = part_stats();
        let p_size_gt_40: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("p_size", 0)),
            Operator::Gt,
            lit(40_i32),
        ));
        let p_size_lt_45: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("p_size", 0)),
            Operator::Lt,
            lit(45_i32),
        ));
        let and: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::clone(&p_size_gt_40),
            Operator::And,
            Arc::clone(&p_size_lt_45),
        ));

        let no_provider_ctx = SynopsisContext::new(&stats, &schema);
        let whole_subtree_selectivity = no_provider_ctx
            .compute(&and)
            .and_then(|s| s.selectivity)
            .expect("the AND is computed from one analyze() call over both conjuncts");

        let overridden_selectivity = 0.9;
        assert!(
            (overridden_selectivity - whole_subtree_selectivity).abs() > 1e-9,
            "the override must differ from the whole-subtree answer for this \
             test to be meaningful"
        );
        let registry =
            SynopsisRegistry::with_providers(vec![Arc::new(OverrideComparison {
                column: "p_size",
                op: Operator::Gt,
                selectivity: overridden_selectivity,
            })]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);
        let right_selectivity = ctx
            .compute(&p_size_lt_45)
            .and_then(|s| s.selectivity)
            .expect("the right conjunct is computed through its own single analyze()");
        let and_selectivity = ctx
            .compute(&and)
            .and_then(|s| s.selectivity)
            .expect("the AND has a synopsis");

        assert_ne!(
            and_selectivity, whole_subtree_selectivity,
            "the provider's answer changes the AND's selectivity away from \
             the whole-subtree analyze() answer"
        );
        assert!(
            (and_selectivity - overridden_selectivity * right_selectivity).abs() < 1e-9,
            "the AND combines the provider's answer for the left conjunct \
             with the right conjunct's own selectivity, got {and_selectivity}"
        );
    }

    // A column with no min or max gives interval analysis nothing to narrow,
    // so interval analysis does not answer for a comparison on it.
    #[test]
    fn interval_analysis_does_not_answer_without_bounds() {
        let stats = Statistics {
            num_rows: Precision::Exact(1000),
            total_byte_size: Precision::Absent,
            column_statistics: vec![
                ColumnStatistics {
                    min_value: Precision::Exact(ScalarValue::Int32(Some(1))),
                    max_value: Precision::Exact(ScalarValue::Int32(Some(50))),
                    ..ColumnStatistics::new_unknown()
                },
                ColumnStatistics::new_unknown(),
            ],
        };
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let ctx = SynopsisContext::new(&stats, &schema).with_default_selectivity(0.5);

        let b_gt_0: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Gt,
            lit(0_i32),
        ));
        assert_eq!(
            ctx.compute(&b_gt_0).and_then(|s| s.selectivity),
            None,
            "no selectivity, not 1.0 from interval analysis"
        );

        let a_gt_40: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Gt,
            lit(40_i32),
        ));
        let and: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(a_gt_40, Operator::And, b_gt_0));
        let selectivity = ctx
            .compute(&and)
            .and_then(|s| s.selectivity)
            .expect("the conjunction has a synopsis");
        assert!(
            (selectivity - 0.1).abs() < 1e-9,
            "0.2 for `a > 40` times the default 0.5, got {selectivity}"
        );
    }

    // A NULL row passes neither `=` nor `!=`, so with half the rows NULL the
    // two keep half the rows between them.
    #[test]
    fn equality_selectivity_excludes_nulls() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics {
                distinct_count: Precision::Exact(10),
                null_count: Precision::Exact(50),
                ..ColumnStatistics::new_unknown()
            }],
        };
        let schema = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
        let ctx = SynopsisContext::new(&stats, &schema);
        let selectivity = |op: Operator| {
            let expr: Arc<dyn PhysicalExpr> =
                Arc::new(BinaryExpr::new(Arc::new(Column::new("s", 0)), op, lit("x")));
            ctx.compute(&expr)
                .and_then(|s| s.selectivity)
                .expect("the comparison has a selectivity")
        };
        let both = selectivity(Operator::Eq) + selectivity(Operator::NotEq);
        assert!((both - 0.5).abs() < 1e-9, "got {both}");
    }

    // `check_support` accepts `col0 > 50.0`, but the column statistics are
    // malformed (`min_value` greater than `max_value`), so interval analysis
    // fails. Computation then falls back to the rules on the children instead
    // of failing, and no rule gives `>` a selectivity.
    #[test]
    fn analyze_error_falls_back_to_compute_from_children() {
        let malformed_stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics {
                distinct_count: Precision::Exact(5),
                min_value: Precision::Exact(ScalarValue::Float64(Some(100.0))),
                max_value: Precision::Exact(ScalarValue::Float64(Some(0.0))),
                ..ColumnStatistics::new_unknown()
            }],
        };
        let schema = Schema::new(vec![Field::new("col0", DataType::Float64, true)]);
        let ctx = SynopsisContext::new(&malformed_stats, &schema);

        let gt: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("col0", 0)),
            Operator::Gt,
            lit(50.0_f64),
        ));

        assert_eq!(
            ctx.compute(&gt).and_then(|s| s.selectivity),
            None,
            "no selectivity from interval analysis, which failed"
        );
    }

    // A cast that is lossless and injective (here, a widening cast between
    // signed integers) carries the child's distinct count over unchanged; a
    // narrowing cast has no NDV rule, so its distinct count is absent, even
    // though its range is still known from evaluate_bounds.
    #[test]
    fn cast_ndv_rule_widening_vs_narrowing() {
        let (stats, schema) = one_col_stats(6, DataType::Int32);
        let ctx = SynopsisContext::new(&stats, &schema);
        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("col0", 0));

        let widen: Arc<dyn PhysicalExpr> =
            Arc::new(CastExpr::new(Arc::clone(&a), DataType::Int64, None));
        assert_eq!(
            ctx.compute(&widen).map(|s| s.column.distinct_count),
            Some(Precision::Exact(6)),
            "a widening cast between signed integers preserves the count exactly"
        );

        let mut stats_with_nulls = stats.clone();
        stats_with_nulls.column_statistics[0].null_count = Precision::Exact(3);
        let ctx_with_nulls = SynopsisContext::new(&stats_with_nulls, &schema);
        assert_eq!(
            ctx_with_nulls.compute(&widen).map(|s| s.column.null_count),
            Some(Precision::Exact(3)),
            "a lossless cast keeps every NULL"
        );

        let narrow: Arc<dyn PhysicalExpr> =
            Arc::new(CastExpr::new(a, DataType::Int16, None));
        assert_eq!(
            ctx.compute(&narrow).map(|s| s.column.distinct_count),
            Some(Precision::Absent),
            "a narrowing cast has no NDV rule"
        );
    }

    // Neither operand of `a + b` is a constant, so no built-in NDV rule
    // covers it and the distinct count is absent. `evaluate_bounds` still
    // supplies a range: the minimum is the sum of the operands' minimums
    // and the maximum is the sum of the operands' maximums, both `Inexact`.
    #[test]
    fn range_from_evaluate_bounds_when_no_ndv_rule_applies() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![cs(50, 0.0, 99.0), cs(50, 200.0, 300.0)],
        };
        let schema = Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
        ]);
        let ctx = SynopsisContext::new(&stats, &schema);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let b: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 1));
        let sum: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(a, Operator::Plus, b));

        let synopsis = ctx
            .compute(&sum)
            .expect("evaluate_bounds supplies a range even with no NDV rule");
        assert_eq!(
            synopsis.column.min_value,
            Precision::Inexact(ScalarValue::Float64(Some(200.0))),
            "the minimum is the sum of the operands' minimums"
        );
        assert_eq!(
            synopsis.column.max_value,
            Precision::Inexact(ScalarValue::Float64(Some(399.0))),
            "the maximum is the sum of the operands' maximums"
        );
        assert_eq!(
            synopsis.column.distinct_count,
            Precision::Absent,
            "no built-in rule covers two non-constant operands"
        );
    }

    /// Supplies a distinct count and a minimum for any binary expression,
    /// but no maximum.
    #[derive(Debug)]
    struct NdvAndMinimum;

    impl SynopsisProvider for NdvAndMinimum {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(_) = expr.downcast_ref::<BinaryExpr>() else {
                return SynopsisResult::Delegate;
            };
            SynopsisResult::Computed(ExprSynopsis::from_column(
                ColumnStatistics {
                    distinct_count: Precision::Exact(5),
                    min_value: Precision::Exact(ScalarValue::Float64(Some(250.0))),
                    ..ColumnStatistics::new_unknown()
                },
                DataType::Float64,
            ))
        }
    }

    // A provider's answer keeps the minimum it sets, and gets the maximum it
    // leaves out from `evaluate_bounds`, as a built-in rule's answer would.
    #[test]
    fn provider_synopsis_gets_a_missing_range_and_keeps_its_own() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![cs(10, 0.0, 99.0), cs(10, 200.0, 300.0)],
        };
        let schema = Schema::new(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
        ]);
        let registry = SynopsisRegistry::with_providers(vec![Arc::new(NdvAndMinimum)]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let b: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 1));
        let sum: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(a, Operator::Plus, b));

        let synopsis = ctx.compute(&sum).expect("the provider supplies a synopsis");
        assert_eq!(synopsis.ndv(), Some(5));
        assert_eq!(
            synopsis.column.min_value,
            Precision::Exact(ScalarValue::Float64(Some(250.0))),
            "the provider's minimum is kept"
        );
        assert_eq!(
            synopsis.column.max_value,
            Precision::Inexact(ScalarValue::Float64(Some(399.0))),
            "the missing maximum comes from evaluate_bounds"
        );
    }

    // A missing distinct count is the number of values the range holds, or
    // the type holds for a Boolean, when that is below the row count.
    #[test]
    fn distinct_count_from_the_size_of_the_range() {
        let int = |v: i32| ScalarValue::Int32(Some(v));
        let range = |min: i32, max: i32| ColumnStatistics {
            min_value: Precision::Exact(int(min)),
            max_value: Precision::Exact(int(max)),
            ..ColumnStatistics::new_unknown()
        };
        let stats = Statistics {
            num_rows: Precision::Exact(10_000),
            total_byte_size: Precision::Absent,
            column_statistics: vec![range(0, 99), range(200, 300), range(0, 1_000_000)],
        };
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]);
        let ctx = SynopsisContext::new(&stats, &schema);
        let ndv = |expr: &Arc<dyn PhysicalExpr>| {
            ctx.compute(expr).map(|s| s.column.distinct_count)
        };
        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let b: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 1));
        let c: Arc<dyn PhysicalExpr> = Arc::new(Column::new("c", 2));

        let sum: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(Arc::clone(&a), Operator::Plus, b));
        assert_eq!(ndv(&sum), Some(Precision::Inexact(200)), "[200, 399]");

        let gt: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(a, Operator::Gt, lit(50_i32)));
        assert_eq!(ndv(&gt), Some(Precision::Inexact(2)), "true and false");

        assert_eq!(
            ndv(&c),
            Some(Precision::Absent),
            "a range larger than the row count says nothing new"
        );
    }

    // A Boolean expression gets its range from `evaluate_bounds` too: `NOT b`,
    // with `b` always false, is always true.
    #[test]
    fn boolean_expression_gets_a_range() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics {
                min_value: Precision::Exact(ScalarValue::Boolean(Some(false))),
                max_value: Precision::Exact(ScalarValue::Boolean(Some(false))),
                ..ColumnStatistics::new_unknown()
            }],
        };
        let schema = Schema::new(vec![Field::new("b", DataType::Boolean, false)]);
        let ctx = SynopsisContext::new(&stats, &schema);

        let not_b: Arc<dyn PhysicalExpr> =
            Arc::new(NotExpr::new(Arc::new(Column::new("b", 0))));
        let synopsis = ctx.compute(&not_b).expect("NOT b has a synopsis");
        let always_true = Precision::Inexact(ScalarValue::Boolean(Some(true)));
        assert_eq!(synopsis.column.min_value, always_true);
        assert_eq!(synopsis.column.max_value, always_true);
    }

    #[derive(Debug, Clone, PartialEq)]
    struct MiniSketch(Vec<i32>);

    /// Attaches a sketch to column `x`, as a catalog that stores sketches
    /// would.
    #[derive(Debug)]
    struct AttachSketch;
    impl SynopsisProvider for AttachSketch {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            _ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(col) = expr.downcast_ref::<Column>() else {
                return SynopsisResult::Delegate;
            };
            if col.name() == "x" {
                let mut s = ExprSynopsis::from_column(
                    ColumnStatistics {
                        distinct_count: Precision::Exact(5),
                        ..ColumnStatistics::new_unknown()
                    },
                    DataType::Int32,
                );
                s.set_extension(MiniSketch(vec![1, 2, 3]));
                return SynopsisResult::Computed(s);
            }
            SynopsisResult::Delegate
        }
    }

    // An extension that a provider attaches to a column stays on that
    // column's synopsis. A widening `CastExpr` above it carries the distinct
    // count through but not the extension, because a built-in rule cannot
    // transform an opaque sketch.
    #[test]
    fn extensions_ride_along_column_and_do_not_survive_cast() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics::new_unknown()],
        };
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let registry = SynopsisRegistry::with_providers(vec![Arc::new(AttachSketch)]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let x: Arc<dyn PhysicalExpr> = Arc::new(Column::new("x", 0));
        let syn = ctx.compute(&x).expect("synopsis for x");
        assert_eq!(
            syn.get_extension::<MiniSketch>(),
            Some(&MiniSketch(vec![1, 2, 3])),
            "sketch rides the column synopsis"
        );

        let cast_x: Arc<dyn PhysicalExpr> =
            Arc::new(CastExpr::new(x, DataType::Int64, None));
        let cast_syn = ctx
            .compute(&cast_x)
            .expect("a widening cast has a rule, so a synopsis exists");
        assert_eq!(
            cast_syn.column.distinct_count,
            Precision::Exact(5),
            "the widening cast carries the distinct count through"
        );
        assert!(
            !cast_syn.has_extension::<MiniSketch>(),
            "the extension does not survive the built-in cast rule"
        );
    }

    /// Understands `CastExpr` over a column carrying a `MiniSketch`: computes
    /// the child through the context, scales the sketch's values, and attaches
    /// the transformed sketch to the cast itself.
    #[derive(Debug)]
    struct ScaleSketchOnCast {
        factor: i32,
    }
    impl SynopsisProvider for ScaleSketchOnCast {
        fn compute_synopsis(
            &self,
            expr: &Arc<dyn PhysicalExpr>,
            ctx: &SynopsisContext,
        ) -> SynopsisResult {
            let Some(cast) = expr.downcast_ref::<CastExpr>() else {
                return SynopsisResult::Delegate;
            };
            let Some(child_synopsis) = ctx.compute(cast.expr()) else {
                return SynopsisResult::Delegate;
            };
            let Some(child_sketch) = child_synopsis.get_extension::<MiniSketch>() else {
                return SynopsisResult::Delegate;
            };
            let scaled =
                MiniSketch(child_sketch.0.iter().map(|v| v * self.factor).collect());
            let mut synopsis = ExprSynopsis::from_column(
                ColumnStatistics::new_unknown(),
                DataType::Int64,
            );
            synopsis.set_extension(scaled);
            SynopsisResult::Computed(synopsis)
        }
    }

    // A provider that understands an opaque sketch can attach a transformed
    // version of the child's sketch to the parent expression.
    #[test]
    fn provider_transforms_sketch_across_cast() {
        let stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Absent,
            column_statistics: vec![ColumnStatistics::new_unknown()],
        };
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        let registry = SynopsisRegistry::with_providers(vec![
            Arc::new(ScaleSketchOnCast { factor: 10 }),
            Arc::new(AttachSketch),
        ]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let x: Arc<dyn PhysicalExpr> = Arc::new(Column::new("x", 0));
        let child_sketch = ctx
            .compute(&x)
            .expect("synopsis for x")
            .get_extension::<MiniSketch>()
            .expect("sketch attached to x")
            .clone();

        let cast_x: Arc<dyn PhysicalExpr> =
            Arc::new(CastExpr::new(x, DataType::Int64, None));
        let cast_sketch = ctx
            .compute(&cast_x)
            .expect("ScaleSketchOnCast supplies a synopsis for the cast")
            .get_extension::<MiniSketch>()
            .expect("transformed sketch attached to the cast")
            .clone();

        assert_eq!(
            cast_sketch,
            MiniSketch(vec![10, 20, 30]),
            "the sketch arriving at the cast is the transformed one"
        );
        assert_ne!(
            cast_sketch, child_sketch,
            "the transform is observable: the cast's sketch differs from the child's"
        );
    }
}
