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

//! Estimate the group count of `GROUP BY date_trunc('month', ts)` with an
//! expression-level statistics provider.
//!
//! Without help, `AggregateExec` knows the distinct count of a bare column
//! group expression only, so for `date_trunc('month', ts)` its row estimate
//! falls back to the input row count.
//!
//! Here a [`SynopsisProvider`] estimates the distinct count of `date_trunc`
//! from the range of its argument. It computes the argument through the
//! [`SynopsisContext`], so the argument's statistics come from the same chain.
//! A scan-level [`StatisticsRegistry`] provider supplies the `ts` range. Both
//! are registered once, on the [`SessionStateBuilder`], and `EXPLAIN` reaches
//! them through ordinary planning.
//!
//! `events` holds 3000 rows over 12 calendar months, so the true group count
//! is 12.
//!
//! With one partition the aggregate has one stage. With several it has two,
//! and the final stage reads back the distinct count that the partial stage
//! publishes, so both plans get the same estimate.
//!
//! Run with:
//! ```text
//! cargo run --example statistics -- aggregate_synopsis_ndv
//! ```

use std::sync::Arc;

use chrono::{DateTime, Datelike, NaiveDate};
use datafusion::arrow::array::{Int64Array, TimestampNanosecondArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::MemTable;
use datafusion::common::Result;
use datafusion::common::stats::Precision;
use datafusion::common::{ColumnStatistics, ScalarValue};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_expr::expressions::Literal;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::operator_statistics::{
    ClosureStatisticsProvider, ExprSynopsis, ExtendedStatistics, StatisticsRegistry,
    StatisticsResult, SynopsisContext, SynopsisProvider, SynopsisRegistry,
    SynopsisResult,
};
use datafusion::physical_plan::statistics::StatisticsArgs;
use datafusion::prelude::*;

/// Estimates the distinct count of `date_trunc('month', ts)` as the number of
/// calendar months spanned by the range of `ts`.
#[derive(Debug)]
struct DateTruncNdv;

impl DateTruncNdv {
    /// Calendar months spanned by `[min, max]`, inclusive of both ends.
    fn months_between(min_nanos: i64, max_nanos: i64) -> Option<usize> {
        let to_date = |nanos: i64| DateTime::from_timestamp_nanos(nanos).date_naive();
        let (lo, hi) = (to_date(min_nanos), to_date(max_nanos));
        let months = (hi.year() - lo.year()) * 12 + hi.month() as i32 - lo.month() as i32;
        usize::try_from(months + 1).ok()
    }

    /// Returns the synopsis of `date_trunc('month', ts)`, or `None` for any
    /// other expression.
    fn date_trunc_synopsis(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        ctx: &SynopsisContext,
    ) -> Option<ExprSynopsis> {
        let func = expr.downcast_ref::<ScalarFunctionExpr>()?;
        if func.name() != "date_trunc" {
            return None;
        }
        let [unit, arg] = func.args() else {
            return None;
        };
        // Only the fixed monthly granularity is modelled here.
        let unit = unit.downcast_ref::<Literal>()?;
        if unit.value() != &ScalarValue::Utf8(Some("month".to_string())) {
            return None;
        }

        // The argument's statistics come from the same chain as any other
        // consumer's.
        let arg_synopsis = ctx.compute(arg)?;
        if !matches!(
            arg_synopsis.data_type,
            DataType::Timestamp(TimeUnit::Nanosecond, _)
        ) {
            return None;
        }
        let (min, max) = (
            arg_synopsis.column.min_value.get_value()?,
            arg_synopsis.column.max_value.get_value()?,
        );
        let (
            ScalarValue::TimestampNanosecond(Some(lo), _),
            ScalarValue::TimestampNanosecond(Some(hi), _),
        ) = (min, max)
        else {
            return None;
        };
        let months = Self::months_between(*lo, *hi)?;

        // `date_trunc` preserves its argument's timestamp type.
        Some(ExprSynopsis::from_column(
            ColumnStatistics {
                distinct_count: Precision::Inexact(months),
                ..ColumnStatistics::new_unknown()
            },
            arg_synopsis.data_type,
        ))
    }
}

impl SynopsisProvider for DateTruncNdv {
    fn compute_synopsis(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        ctx: &SynopsisContext,
    ) -> SynopsisResult {
        match self.date_trunc_synopsis(expr, ctx) {
            Some(synopsis) => SynopsisResult::Computed(synopsis),
            None => SynopsisResult::Delegate,
        }
    }
}

/// Nanoseconds since the epoch for `2025-{month}-{day}T00:00:00Z`.
fn ts_nanos(month: u32, day: u32) -> i64 {
    NaiveDate::from_ymd_opt(2025, month, day)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .unwrap()
}

/// Matches the base `events` scan.
fn scan_matches(plan: &dyn ExecutionPlan) -> bool {
    plan.children().is_empty() && plan.schema().index_of("ts").is_ok()
}

/// Injects the `ts` range a catalog would know and the in-memory table lacks.
fn scan_stats(
    plan: &dyn ExecutionPlan,
    _child_stats: &[ExtendedStatistics],
) -> Result<StatisticsResult> {
    let ts = plan.schema().index_of("ts")?;
    let mut stats = (*plan.statistics_from_inputs(&[], &StatisticsArgs::new())?).clone();
    stats.column_statistics[ts].min_value =
        Precision::Exact(ScalarValue::TimestampNanosecond(Some(ts_nanos(1, 1)), None));
    stats.column_statistics[ts].max_value = Precision::Exact(
        ScalarValue::TimestampNanosecond(Some(ts_nanos(12, 25)), None),
    );
    Ok(StatisticsResult::Computed(ExtendedStatistics::new(stats)))
}

/// `events(ts, amount)`: 3000 rows spread over 12 calendar months of 2025.
fn events_table() -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("amount", DataType::Int64, false),
    ]));

    let day = 86_400_000_000_000i64;
    // 250 rows per month, over the first 25 days of each month.
    let mut timestamps = Vec::with_capacity(3000);
    for month in 0..12i64 {
        let month_start = ts_nanos(1 + month as u32, 1);
        for row in 0..250i64 {
            timestamps.push(month_start + (row % 25) * day);
        }
    }
    let amounts: Vec<i64> = (0..timestamps.len() as i64).collect();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampNanosecondArray::from(timestamps)),
            Arc::new(Int64Array::from(amounts)),
        ],
    )?;
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

const QUERY: &str = "SELECT date_trunc('month', ts) AS month, count(*) AS n \
     FROM events GROUP BY 1";

/// A session with `target_partitions` partitions. With `with_registry`, it
/// registers the `ts` range provider and the `date_trunc` provider. `EXPLAIN`
/// shows the physical plan with statistics.
fn build_ctx(target_partitions: usize, with_registry: bool) -> Result<SessionContext> {
    let config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .set_bool("datafusion.explain.physical_plan_only", true)
        .set_bool("datafusion.explain.show_statistics", true);

    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features();
    if with_registry {
        let mut registry = StatisticsRegistry::default_with_builtin_providers();
        registry.register(Arc::new(ClosureStatisticsProvider::with_matches(
            scan_matches,
            scan_stats,
        )));
        builder = builder
            .with_statistics_registry(registry)
            .with_synopsis_registry(SynopsisRegistry::with_providers(vec![Arc::new(
                DateTruncNdv,
            )]));
    }
    let ctx = SessionContext::new_with_state(builder.build());
    ctx.register_table("events", events_table()?)?;
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

/// Runs the query, prints the row estimate with and without the registry for
/// both scenarios, and prints the ground truth.
async fn run_scenario(label: &str, target_partitions: usize) -> Result<()> {
    println!("== {label} (target_partitions={target_partitions}) ==\n");

    println!("-- Without the registry (date_trunc unestimated) --");
    println!(
        "{}\n",
        explain(&build_ctx(target_partitions, false)?).await?
    );

    println!("-- With the registry (session-registered expression provider) --");
    println!("{}", explain(&build_ctx(target_partitions, true)?).await?);

    let ctx = build_ctx(target_partitions, false)?;
    let truth = ctx.sql(QUERY).await?.collect().await?;
    let truth_rows: usize = truth.iter().map(|b| b.num_rows()).sum();
    println!("Ground truth (rows returned): {truth_rows}\n");
    Ok(())
}

pub async fn aggregate_synopsis_ndv() -> Result<()> {
    println!("Query: {QUERY}\n");
    run_scenario("one-stage aggregate", 1).await?;
    run_scenario("two-phase aggregate", 16).await?;

    Ok(())
}
