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

//! # Pluggable operator statistics (`StatisticsRegistry`)
//!
//! These examples show how a `StatisticsProvider` registered on the session can
//! change a physical plan by supplying statistics the built-in estimators
//! cannot express.
//!
//! ## Usage
//! ```bash
//! cargo run --example statistics -- [all|join_reorder|tenant_skew|aggregate_synopsis_ndv|join_sketch_synopsis]
//! ```
//!
//! Each subcommand runs a corresponding example:
//! - `all`: run all examples included in this module
//!
//! - `join_reorder`
//!   (file: join_reorder.rs, desc: Supply and refine column statistics via a provider to flip a join order)
//!
//! - `tenant_skew`
//!   (file: tenant_skew.rs, desc: Plug a per-tenant row count into filter selectivity via the SynopsisRegistry)
//!
//! - `aggregate_synopsis_ndv`
//!   (file: aggregate_synopsis_ndv.rs, desc: Estimate GROUP BY date_trunc(...) cardinality from an expression-level provider, in one-stage and two-phase plans)
//!
//! - `join_sketch_synopsis`
//!   (file: join_sketch_synopsis.rs, desc: Carry a set-backed sketch through the join key synopsis to correct a join row estimate an NDV formula gets wrong)

mod aggregate_synopsis_ndv;
mod join_reorder;
mod join_sketch_synopsis;
mod tenant_skew;

use datafusion::error::{DataFusionError, Result};
use strum::{IntoEnumIterator, VariantNames};
use strum_macros::{Display, EnumIter, EnumString, VariantNames};

#[derive(EnumIter, EnumString, Display, VariantNames)]
#[strum(serialize_all = "snake_case")]
enum ExampleKind {
    All,
    JoinReorder,
    TenantSkew,
    AggregateSynopsisNdv,
    JoinSketchSynopsis,
}

impl ExampleKind {
    const EXAMPLE_NAME: &str = "statistics";

    fn runnable() -> impl Iterator<Item = ExampleKind> {
        ExampleKind::iter().filter(|v| !matches!(v, ExampleKind::All))
    }

    async fn run(&self) -> Result<()> {
        match self {
            ExampleKind::All => {
                for example in ExampleKind::runnable() {
                    println!("Running example: {example}");
                    Box::pin(example.run()).await?;
                }
                Ok(())
            }
            ExampleKind::JoinReorder => join_reorder::join_reorder().await,
            ExampleKind::TenantSkew => tenant_skew::tenant_skew().await,
            ExampleKind::AggregateSynopsisNdv => {
                aggregate_synopsis_ndv::aggregate_synopsis_ndv().await
            }
            ExampleKind::JoinSketchSynopsis => {
                join_sketch_synopsis::join_sketch_synopsis().await
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let usage = format!(
        "Usage: cargo run --example {} -- [{}]",
        ExampleKind::EXAMPLE_NAME,
        ExampleKind::VARIANTS.join("|")
    );

    let example: ExampleKind = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ExampleKind::All.to_string())
        .parse()
        .map_err(|_| DataFusionError::Execution(format!("Unknown example. {usage}")))?;

    example.run().await
}
