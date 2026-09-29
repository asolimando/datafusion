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
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use datafusion_common::Statistics;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::synopsis::{ExprSynopsis, SynopsisArgs};

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
/// in the same way as the built-in rules.
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
        }
    }

    /// The arguments every expression in this walk is computed against, the
    /// same ones the built-in rules see.
    pub fn args(&self) -> &SynopsisArgs<'a> {
        &self.args
    }

    /// Computes the synopsis of `expr`, caching it by structural expression
    /// identity.
    ///
    /// The registered providers are asked first, and the first answer is the
    /// result. Otherwise the built-in estimate applies: the expression's own
    /// rule on the synopses of its children, which are computed in the same
    /// way.
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

        let mut result = self.provider_answer(expr, &data_type);
        if result.is_none() {
            result = self.compute_from_children(expr, &data_type);
        }

        self.cache
            .borrow_mut()
            .insert(Arc::clone(expr), result.clone());
        result
    }

    fn provider_answer(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        data_type: &DataType,
    ) -> Option<ExprSynopsis> {
        self.registry.providers.iter().find_map(|p| {
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
        })
    }

    /// Computes `expr` by computing its children through the full chain and
    /// applying `expr`'s own rule to their synopses.
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
        computed.and_then(|computed| {
            expr.synopsis_from_inputs(&self.args, &computed)
                .and_then(|s| conform(s, data_type))
        })
    }
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
    use crate::expressions::{BinaryExpr, Column, lit};
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

    #[test]
    fn provider_answer_replaces_builtin_and_skips_children() {
        let (stats, schema) = one_col_stats(42, DataType::Float64);
        let counter = Arc::new(AtomicUsize::new(0));
        // `OverrideBinary` is first, so it is asked first; `CountComputations`
        // is asked only for an expression that `OverrideBinary` declines.
        let registry = SynopsisRegistry::with_providers(vec![
            Arc::new(OverrideBinary { ndv: 999 }),
            Arc::new(CountComputations(Arc::clone(&counter))),
        ]);
        let ctx = SynopsisContext::new_with_registry(&stats, &schema, &registry);

        let a_plus_1: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Plus,
            lit(1.0_f64),
        ));
        assert_eq!(
            ctx.compute(&a_plus_1).and_then(|s| s.ndv()),
            Some(999),
            "the provider's value is the result"
        );
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "the built-in path never ran, so neither operand was computed"
        );
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
}
