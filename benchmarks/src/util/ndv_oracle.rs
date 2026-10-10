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

//! Temporary harness: plan with true per-column distinct counts at the scans.

use std::{collections::HashMap, path::Path, sync::Arc};

use datafusion::common::{Result, exec_err, stats::Precision};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::operator_statistics::{
    ClosureStatisticsProvider, ExtendedStatistics, StatisticsRegistry, StatisticsResult,
};
use datafusion::physical_plan::statistics::StatisticsArgs;

/// Truth file: {"table.column": ndv} or {"table": {"column": ndv}}. Column names
/// must be unique across tables.
pub fn load_truth(path: &Path) -> Result<HashMap<String, usize>> {
    let text = std::fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))?;
    let mut out = HashMap::new();
    let mut insert = |col: &str, ndv: u64| -> Result<()> {
        if out.insert(col.to_string(), ndv as usize).is_some() {
            return exec_err!("duplicate column name in truth file: {col}");
        }
        Ok(())
    };
    for (key, value) in value.as_object().expect("JSON object") {
        match value {
            serde_json::Value::Number(n) => {
                insert(key.rsplit('.').next().unwrap(), n.as_u64().unwrap())?
            }
            serde_json::Value::Object(cols) => {
                for (c, n) in cols {
                    insert(c, n.as_u64().unwrap())?;
                }
            }
            _ => return exec_err!("unexpected truth entry {key}"),
        }
    }
    Ok(out)
}

/// Overrides `distinct_count` of every scanned column with its true value. Exact
/// counts (declared keys) are kept.
pub fn ndv_oracle_provider(truth: HashMap<String, usize>) -> ClosureStatisticsProvider {
    let truth = Arc::new(truth);
    ClosureStatisticsProvider::with_matches(
        |plan: &dyn ExecutionPlan| plan.downcast_ref::<DataSourceExec>().is_some(),
        move |plan: &dyn ExecutionPlan, _children| {
            let mut stats = Arc::unwrap_or_clone(
                plan.statistics_from_inputs(&[], &StatisticsArgs::new())?,
            );
            for (field, col) in plan
                .schema()
                .fields()
                .iter()
                .zip(stats.column_statistics.iter_mut())
            {
                if let Some(&ndv) = truth.get(field.name())
                    && !matches!(col.distinct_count, Precision::Exact(_))
                {
                    col.distinct_count = Precision::Inexact(ndv);
                }
            }
            Ok(StatisticsResult::Computed(ExtendedStatistics::new(stats)))
        },
    )
}

/// The registry for the harness options.
pub fn harness_registry(
    ndv_oracle: Option<&Path>,
    builtin_providers: bool,
) -> Result<StatisticsRegistry> {
    let mut registry = StatisticsRegistry::new();
    if builtin_providers {
        #[expect(deprecated)]
        let builtin = StatisticsRegistry::default_with_builtin_providers();
        registry = builtin;
    }
    if let Some(path) = ndv_oracle {
        registry.register(Arc::new(ndv_oracle_provider(load_truth(path)?)));
    }
    Ok(registry)
}
