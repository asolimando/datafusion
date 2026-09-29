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

//! Plug a per-tenant row count into filter selectivity via the `SynopsisRegistry`.
//!
//! # Background
//!
//! `events(tenant_id INT, user_id INT)` holds 10,000 rows across 1,000
//! tenants, with tenant ids from 12000 to 12999. Tenant 12345 owns 4,000 of
//! those rows; the other 999 tenants share the remaining 6,000 rows in
//! roughly even shares. The application that owns this table already keeps
//! a row count per tenant, gathered from its own bookkeeping rather than
//! from DataFusion.
//!
//! A catalog-like `StatisticsProvider` supplies the minimum, maximum, and
//! distinct count for `tenant_id` (12000, 12999, and 1,000) that a real
//! catalog would already track for this table. With those bounds but no
//! knowledge of the skew, DataFusion's default estimator assumes the 1,000
//! tenants are evenly sized: it estimates `tenant_id = 12345` at about 0.1%
//! of the rows, roughly 10 rows. The application's own count says the true
//! answer is 40%, or 4,000 rows.
//!
//! # How the application's counts reach the plan
//!
//! ```text
//!   application's per-tenant     TenantRowCountSelectivity
//!   row counts (bookkeeping) ───▶ (a SynopsisProvider)
//!                                        │
//!                                        │ registered on the session
//!                                        ▼
//!                              SessionStateBuilder::with_synopsis_registry
//!                                        │
//!                                        ▼
//!                      FilterExec's estimate for `tenant_id = 12345`
//!                                        │
//!                                        ▼
//!                       HashJoinExec's choice of build side
//! ```
//!
//! # Flow
//!
//! `events e JOIN users u ON e.user_id = u.user_id WHERE e.tenant_id = 12345`
//! joins on `user_id`, a column the filter does not touch, so the filter's
//! row estimate only ever reaches the join through statistics, never through
//! a pushed-down predicate on `users`. A partitioned hash join is forced so
//! the row-count statistics alone choose the build side:
//! - Without the provider: the filtered `events` keeps the default
//!   even-distribution estimate of about 10 rows, below `users` (1,000
//!   rows), so `events` builds.
//! - With the provider: it answers `tenant_id = 12345` with selectivity 0.4
//!   (4,000 / 10,000), giving the filtered `events` a row estimate of 4,000,
//!   above `users`, so `users` builds.
//!
//! For a tenant the application does not track, the provider delegates, and
//! the built-in estimate applies as if no provider were registered.
//!
//! The ground-truth query below confirms the true row count for tenant 12345.
//!
//! Run with:
//! ```text
//! cargo run --example statistics -- tenant_skew
//! ```

use std::sync::Arc;

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::MemTable;
use datafusion::common::Result;
use datafusion::common::ScalarValue;
use datafusion::common::stats::Precision;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::operator_statistics::{
    ClosureStatisticsProvider, ExprSynopsis, ExtendedStatistics, StatisticsRegistry,
    StatisticsResult, SynopsisContext, SynopsisProvider, SynopsisRegistry,
    SynopsisResult,
};
use datafusion::physical_plan::statistics::StatisticsArgs;
use datafusion::prelude::*;

const TOTAL_EVENTS: i32 = 10_000;
const TENANT_12345_ROWS: i32 = 4_000;
const TENANT_ID_MIN: i32 = 12_000;
const TENANT_ID_MAX: i32 = 12_999;
const TENANT_ID_COUNT: usize = (TENANT_ID_MAX - TENANT_ID_MIN + 1) as usize;
const OTHER_TENANT_COUNT: i32 = TENANT_ID_COUNT as i32 - 1;
/// Chosen so the default filter estimate (about 10 rows, the even-distribution
/// estimate over 1,000 tenants) is below it and the provider's filter estimate
/// (4,000 rows) is above it: the build side flips between the two runs.
const USERS_ROWS: i32 = 1_000;

/// The application's own per-tenant row counts, gathered outside DataFusion.
/// Answers `tenant_id = <constant>` with the matching fraction of rows, and
/// delegates for a tenant it does not track.
#[derive(Debug)]
struct TenantRowCountSelectivity {
    total_rows: f64,
    tenant_12345_rows: f64,
}

impl SynopsisProvider for TenantRowCountSelectivity {
    fn compute_synopsis(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        _ctx: &SynopsisContext,
    ) -> SynopsisResult {
        let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
            return SynopsisResult::Delegate;
        };
        if *binary.op() != Operator::Eq {
            return SynopsisResult::Delegate;
        }
        let Some(literal) = binary.right().downcast_ref::<Literal>() else {
            return SynopsisResult::Delegate;
        };
        // The only tenant this example's bookkeeping tracks.
        if literal.value() != &ScalarValue::Int32(Some(12345)) {
            return SynopsisResult::Delegate;
        }
        SynopsisResult::Computed(ExprSynopsis {
            selectivity: Some(self.tenant_12345_rows / self.total_rows),
            ..ExprSynopsis::unknown(DataType::Boolean)
        })
    }
}

fn int_col(values: &[i32]) -> Arc<Int32Array> {
    Arc::new(Int32Array::from_iter_values(values.iter().copied()))
}

fn mem_table(fields: &[(&str, Arc<Int32Array>)]) -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(
        fields
            .iter()
            .map(|(name, _)| Field::new(*name, DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));
    let cols = fields.iter().map(|(_, col)| Arc::clone(col) as _).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), cols)?;
    Ok(Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))
}

/// `events.user_id` is a row index, so each event row has a distinct join
/// key unrelated to its tenant. Tenant ids span `TENANT_ID_MIN..=TENANT_ID_MAX`,
/// with 12345 owning `TENANT_12345_ROWS` rows and the other tenants in that
/// range sharing the remainder in roughly even shares.
fn events_table() -> Result<Arc<MemTable>> {
    let tenant_ids: Vec<i32> = (0..TOTAL_EVENTS)
        .map(|i| {
            if i < TENANT_12345_ROWS {
                12345
            } else {
                let other_index = (i - TENANT_12345_ROWS) % OTHER_TENANT_COUNT;
                let tenant_id = TENANT_ID_MIN + other_index;
                if tenant_id == 12345 {
                    TENANT_ID_MAX
                } else {
                    tenant_id
                }
            }
        })
        .collect();
    let user_ids: Vec<i32> = (0..TOTAL_EVENTS).collect();
    mem_table(&[
        ("tenant_id", int_col(&tenant_ids)),
        ("user_id", int_col(&user_ids)),
    ])
}

fn users_table() -> Result<Arc<MemTable>> {
    let user_ids: Vec<i32> = (0..USERS_ROWS).collect();
    mem_table(&[("user_id", int_col(&user_ids))])
}

/// Matches the base `events` scan: a leaf carrying both `tenant_id` and `user_id`.
fn catalog_matches(plan: &dyn ExecutionPlan) -> bool {
    let schema = plan.schema();
    plan.children().is_empty()
        && schema.index_of("tenant_id").is_ok()
        && schema.index_of("user_id").is_ok()
}

/// Injects the catalog-known `tenant_id` range and distinct count that the
/// in-memory table lacks: minimum 12000, maximum 12999, 1,000 distinct values.
fn catalog_stats(
    plan: &dyn ExecutionPlan,
    _child_stats: &[ExtendedStatistics],
) -> Result<StatisticsResult> {
    let schema = plan.schema();
    let tenant_id = schema.index_of("tenant_id")?;
    let mut stats = (*plan.statistics_from_inputs(&[], &StatisticsArgs::new())?).clone();
    stats.column_statistics[tenant_id].min_value =
        Precision::Inexact(ScalarValue::Int32(Some(TENANT_ID_MIN)));
    stats.column_statistics[tenant_id].max_value =
        Precision::Inexact(ScalarValue::Int32(Some(TENANT_ID_MAX)));
    stats.column_statistics[tenant_id].distinct_count =
        Precision::Inexact(TENANT_ID_COUNT);
    Ok(StatisticsResult::Computed(ExtendedStatistics::new(stats)))
}

fn build_ctx(with_provider: bool) -> Result<SessionContext> {
    let config = SessionConfig::new()
        .with_target_partitions(4)
        .set_bool("datafusion.explain.physical_plan_only", true)
        .set_bool("datafusion.explain.show_statistics", true)
        // Force partitioned hash joins so statistics alone drive the build side.
        .set_usize(
            "datafusion.optimizer.hash_join_single_partition_threshold",
            1,
        )
        .set_usize(
            "datafusion.optimizer.hash_join_single_partition_threshold_rows",
            1,
        );

    let mut statistics_registry = StatisticsRegistry::default_with_builtin_providers();
    statistics_registry.register(Arc::new(ClosureStatisticsProvider::with_matches(
        catalog_matches,
        catalog_stats,
    )));
    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_statistics_registry(statistics_registry);
    if with_provider {
        let registry =
            SynopsisRegistry::with_providers(vec![Arc::new(TenantRowCountSelectivity {
                total_rows: TOTAL_EVENTS as f64,
                tenant_12345_rows: TENANT_12345_ROWS as f64,
            })]);
        builder = builder.with_synopsis_registry(registry);
    }
    let ctx = SessionContext::new_with_state(builder.build());

    ctx.register_table("events", events_table()?)?;
    ctx.register_table("users", users_table()?)?;
    Ok(ctx)
}

const QUERY: &str = "SELECT e.user_id, u.user_id AS matched_user_id \
     FROM events e JOIN users u ON e.user_id = u.user_id \
     WHERE e.tenant_id = 12345";

async fn explain(ctx: &SessionContext) -> Result<String> {
    let batches = ctx
        .sql(&format!("EXPLAIN {QUERY}"))
        .await?
        .collect()
        .await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

pub async fn tenant_skew() -> Result<()> {
    let truth_query =
        "SELECT count(*) AS tenant_rows FROM events WHERE tenant_id = 12345";
    println!("-- Ground truth --\n{truth_query}\n");
    let truth = build_ctx(false)?.sql(truth_query).await?.collect().await?;
    println!("{}\n", pretty_format_batches(&truth)?);

    println!("-- Query --\n{QUERY}\n");
    println!(
        "The filter is on `tenant_id`, the join is on `user_id`: the filter's row\n\
         estimate reaches the join only through statistics. Without the provider,\n\
         the default estimate spreads the 10,000 rows evenly over the 1,000\n\
         distinct tenant ids, so the filter keeps about 10 rows, below `users`\n\
         ({USERS_ROWS} rows), and `events` becomes the build side. With the\n\
         provider, its 4,000-row estimate for tenant 12345 is above\n\
         `users`, so `users` becomes the build side instead.\n"
    );
    println!("-- Without the provider (default estimation) --");
    println!("{}\n", explain(&build_ctx(false)?).await?);
    println!("-- With the provider (application's per-tenant row counts) --");
    println!("{}", explain(&build_ctx(true)?).await?);
    Ok(())
}
