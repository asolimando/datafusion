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

//! Correct a join row estimate with a distinct-value sketch carried as a
//! statistics extension. A distinct count cannot be intersected, but two
//! sketches can.
//!
//! `orders(customer_id)` has 1000 rows over keys `{0..=99}` (10 rows per key).
//! `customers(id)` has 100 rows over keys `{50..=149}` (1 row per key). The
//! two key domains overlap on `{50..=99}`, 50 values.
//!
//! Ground truth for `orders JOIN customers ON orders.customer_id = customers.id`
//! is 50 overlapping keys times 10 `orders` rows times 1 `customers` row: 500.
//!
//! The NDV-based join formula assumes containment (the smaller key domain is
//! a subset of the larger one), which does not hold here:
//! `left_rows * right_rows / max(left_ndv, right_ndv)` = `1000 * 100 / 100` =
//! 1000, wrong by 2x.
//!
//! A sketch carrying the actual distinct values, not just their count, can be
//! intersected. Assuming uniform frequency within each side:
//! `|intersection| * (left_rows / |left_set|) * (right_rows / |right_set|)` =
//! `50 * (1000 / 100) * (100 / 100)` = 500, exact.
//!
//! `KeySketch` stands in for a real distinct-value sketch (e.g. HLL or theta),
//! holding the actual values so it can be intersected like one.
//!
//! # Publish-and-consume chain
//!
//! The sketch is published once, at the scan, and read by column index
//! everywhere above it:
//!
//! 1. `AttachKeySketch` is a `SynopsisProvider` that knows the sketch for a
//!    named join-key column.
//! 2. `ScanSketchProvider` matches the scan nodes, computes each join-key
//!    column through a `SynopsisContext` (which consults `AttachKeySketch`),
//!    takes the sketch off the resulting `ExprSynopsis` with `get_extension`,
//!    and publishes it into the scan's output column slot with
//!    `ExtendedStatistics::set_column_extension`.
//! 3. The `StatisticsContext` walk propagates column extensions upward
//!    through every provider-handled node between the scan and the join.
//! 4. `JoinKeySketchProvider` matches `HashJoinExec` and reads both sketches
//!    directly from its children's column slots with `get_column_extension`,
//!    using each join key's column index within that child's schema. It
//!    builds no `SynopsisContext` of its own. It returns `StatisticsResult::Delegate` unless both sketches are present,
//!    so it never replaces an estimate it has no special knowledge for.
//!
//! Column indices are not remapped through a projection, so this chain only
//! works because nothing between the scan and the join is a `ProjectionExec`
//! that reorders or drops columns.
//!
//! Run with:
//! ```text
//! cargo run --example statistics -- join_sketch_synopsis
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::MemTable;
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, Result, Statistics};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::synopsis_registry::{
    SynopsisContext, SynopsisProvider, SynopsisRegistry, SynopsisResult,
};
use datafusion::physical_expr_common::synopsis::ExprSynopsis;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::operator_statistics::{
    ExtendedStatistics, StatisticsProvider, StatisticsRegistry, StatisticsResult,
};
use datafusion::physical_plan::statistics::StatisticsArgs;
use datafusion::prelude::*;

/// Stand-in for a real distinct-value sketch (e.g. HLL or theta): unlike a
/// bare cardinality estimate, it holds the actual distinct values, so two
/// sketches from independent sides of a join can be intersected.
#[derive(Debug, Clone)]
struct KeySketch(HashSet<i32>);

/// Attaches a `KeySketch` to a join-key `Column`, matched by name, as a
/// catalog that stores sketches would.
#[derive(Debug)]
struct AttachKeySketch {
    column: &'static str,
    sketch: KeySketch,
}

impl SynopsisProvider for AttachKeySketch {
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
        let mut synopsis =
            ExprSynopsis::from_column(ColumnStatistics::new_unknown(), DataType::Int32);
        synopsis.set_extension(self.sketch.clone());
        SynopsisResult::Computed(synopsis)
    }
}

/// Replaces `HashJoinExec`'s row estimate with a sketch-intersection estimate,
/// but only when both sides' join-key sketches are available; otherwise it
/// delegates to the built-in NDV-based estimate. Reads the sketches straight
/// from the children's published column slots: it computes no expression and
/// builds no `SynopsisContext` of its own.
#[derive(Debug)]
struct JoinKeySketchProvider;

impl StatisticsProvider for JoinKeySketchProvider {
    fn matches(&self, plan: &dyn ExecutionPlan) -> bool {
        plan.downcast_ref::<HashJoinExec>().is_some()
    }

    fn compute_statistics_with_args(
        &self,
        plan: &dyn ExecutionPlan,
        child_stats: &[ExtendedStatistics],
        args: &StatisticsArgs,
    ) -> Result<StatisticsResult> {
        if args.partition().is_some() {
            return Ok(StatisticsResult::Delegate);
        }
        let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
            return Ok(StatisticsResult::Delegate);
        };
        let Some((left_key, right_key)) = join.on().first() else {
            return Ok(StatisticsResult::Delegate);
        };
        let (Some(left_column), Some(right_column)) = (
            left_key.downcast_ref::<Column>(),
            right_key.downcast_ref::<Column>(),
        ) else {
            return Ok(StatisticsResult::Delegate);
        };

        let (Some(left_sketch), Some(right_sketch)) = (
            child_stats[0].get_column_extension::<KeySketch>(left_column.index()),
            child_stats[1].get_column_extension::<KeySketch>(right_column.index()),
        ) else {
            return Ok(StatisticsResult::Delegate);
        };

        let left_rows = child_stats[0]
            .base()
            .num_rows
            .get_value()
            .copied()
            .unwrap_or(0);
        let right_rows = child_stats[1]
            .base()
            .num_rows
            .get_value()
            .copied()
            .unwrap_or(0);
        let intersection = left_sketch.0.intersection(&right_sketch.0).count();
        // Assumes uniform frequency within each side: rows per distinct key.
        let estimate = intersection as f64
            * (left_rows as f64 / left_sketch.0.len() as f64)
            * (right_rows as f64 / right_sketch.0.len() as f64);

        let child_base: Vec<Arc<Statistics>> = child_stats
            .iter()
            .map(|c| Arc::clone(c.base_arc()))
            .collect();
        let base = plan.statistics_from_inputs(&child_base, args)?;
        let mut stats = Arc::unwrap_or_clone(base);
        stats.num_rows = Precision::Inexact(estimate.round() as usize);
        Ok(StatisticsResult::Computed(ExtendedStatistics::new(stats)))
    }
}

/// Injects `distinct_count = 100` for both join keys, the value a catalog
/// would supply and an in-memory table lacks, and, when a `SynopsisRegistry`
/// is in scope, computes each join-key column through a `SynopsisContext` and
/// publishes its `KeySketch` into the matching output column slot. Registered
/// in every run so the only variable across scenarios is whether the sketch
/// provider is present.
#[derive(Debug)]
struct ScanSketchProvider;

impl StatisticsProvider for ScanSketchProvider {
    fn matches(&self, plan: &dyn ExecutionPlan) -> bool {
        plan.children().is_empty()
            && (plan.schema().index_of("customer_id").is_ok()
                || plan.schema().index_of("id").is_ok())
    }

    fn compute_statistics_with_args(
        &self,
        plan: &dyn ExecutionPlan,
        _child_stats: &[ExtendedStatistics],
        args: &StatisticsArgs,
    ) -> Result<StatisticsResult> {
        if args.partition().is_some() {
            return Ok(StatisticsResult::Delegate);
        }
        let mut stats = (*plan.statistics_from_inputs(&[], args)?).clone();
        for name in ["customer_id", "id"] {
            if let Ok(idx) = plan.schema().index_of(name) {
                stats.column_statistics[idx].distinct_count = Precision::Inexact(100);
            }
        }

        let mut extended = ExtendedStatistics::new(stats.clone());
        let schema = plan.schema();
        let synopsis_ctx =
            SynopsisContext::new_with_registry(&stats, &schema, args.synopsis_registry());
        for name in ["customer_id", "id"] {
            let Ok(idx) = schema.index_of(name) else {
                continue;
            };
            let column_expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new(name, idx));
            if let Some(sketch) = synopsis_ctx
                .compute(&column_expr)
                .and_then(|synopsis| synopsis.get_extension::<KeySketch>().cloned())
            {
                extended.set_column_extension(idx, sketch);
            }
        }

        Ok(StatisticsResult::Computed(extended))
    }
}

/// `orders(customer_id)`: 1000 rows, `customer_id = i % 100`, keys `{0..=99}`,
/// 10 rows per key.
fn orders_table() -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "customer_id",
        DataType::Int32,
        false,
    )]));
    let customer_ids: Vec<i32> = (0..1000).map(|i| i % 100).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(customer_ids))],
    )?;
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

/// `customers(id, label)`: 100 rows, `id = 50 + i`, keys `{50..=149}`, 1 row
/// per key.
fn customers_table() -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Int32, false),
    ]));
    let ids: Vec<i32> = (0..100).map(|i| 50 + i).collect();
    let labels: Vec<i32> = (0..100).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int32Array::from(labels)),
        ],
    )?;
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

const QUERY: &str = "SELECT o.customer_id, c.label \
     FROM orders o JOIN customers c ON o.customer_id = c.id";

/// Builds a session with the scan-level `ScanSketchProvider` always
/// registered, and, if `with_sketch` is set, also the join-key `SynopsisProvider`s
/// and the `JoinKeySketchProvider`.
fn build_ctx(with_sketch: bool) -> Result<SessionContext> {
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .set_bool("datafusion.explain.physical_plan_only", true)
        .set_bool("datafusion.explain.show_statistics", true);

    let mut registry = StatisticsRegistry::default_with_builtin_providers();
    registry.register(Arc::new(ScanSketchProvider));
    registry.register(Arc::new(JoinKeySketchProvider));

    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_statistics_registry(registry);

    if with_sketch {
        let orders_sketch = KeySketch((0..100).collect());
        let customers_sketch = KeySketch((50..150).collect());
        let synopsis_registry = SynopsisRegistry::with_providers(vec![
            Arc::new(AttachKeySketch {
                column: "id",
                sketch: customers_sketch,
            }),
            Arc::new(AttachKeySketch {
                column: "customer_id",
                sketch: orders_sketch,
            }),
        ]);
        builder = builder.with_synopsis_registry(synopsis_registry);
    }

    let ctx = SessionContext::new_with_state(builder.build());
    ctx.register_table("orders", orders_table()?)?;
    ctx.register_table("customers", customers_table()?)?;
    Ok(ctx)
}

async fn explain(ctx: &SessionContext) -> Result<String> {
    let batches = ctx
        .sql(&format!("EXPLAIN {QUERY}"))
        .await?
        .collect()
        .await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// Extracts the `HashJoinExec` portion of an `EXPLAIN` plan for a focused
/// printout of the join's row estimate and chosen mode, trimming the
/// surrounding table formatting.
fn join_line(explain_output: &str) -> Option<&str> {
    let line = explain_output
        .lines()
        .find(|line| line.contains("HashJoinExec"))?;
    line.find("HashJoinExec").map(|i| line[i..].trim_end())
}

async fn run_scenario(label: &str, with_sketch: bool) -> Result<()> {
    println!("== {label} ==\n");
    let output = explain(&build_ctx(with_sketch)?).await?;
    println!("{output}");
    let line = join_line(&output).unwrap_or("<no HashJoinExec line found>");
    println!("Join node: {line}\n");
    Ok(())
}

pub async fn join_sketch_synopsis() -> Result<()> {
    println!("Query: {QUERY}\n");
    println!(
        "orders: 1000 rows, customer_id in {{0..=99}} (10 rows/key)\n\
         customers: 100 rows, id in {{50..=149}} (1 row/key)\n\
         Overlap: {{50..=99}}, 50 keys. Ground truth: 50 * 10 * 1 = 500 rows.\n\
         NDV estimate: 1000 * 100 / max(100, 100) = 1000 (wrong, assumes containment).\n\
         Sketch estimate: 50 * (1000/100) * (100/100) = 500 (exact).\n"
    );

    println!("-- Without the sketch provider (NDV-based estimate) --");
    run_scenario("without sketch provider", false).await?;

    println!("-- With the sketch provider (sketch-intersection estimate) --");
    run_scenario("with sketch provider", true).await?;

    let ctx = build_ctx(false)?;
    let truth = ctx.sql(QUERY).await?.collect().await?;
    let truth_rows: usize = truth.iter().map(|b| b.num_rows()).sum();
    println!("Ground truth (rows returned): {truth_rows}");

    Ok(())
}
