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

//! Planning-time statistics estimates for the output of one expression.
//!
//! [`ExprSynopsis`] is the value that an expression-level statistics walk
//! propagates through an expression tree. It wraps the relation-level
//! [`ColumnStatistics`] and adds the field that `ColumnStatistics` has no
//! place for: the `selectivity` of a Boolean predicate.

use arrow::datatypes::{DataType, Schema};
use datafusion_common::extensions::Extensions;
use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, ScalarValue, Statistics};

/// Statistics estimate for the output values of one expression.
///
/// A synopsis must never drive a correctness decision, such as pruning,
/// because it is an estimate and can be wrong.
#[derive(Debug, Clone)]
pub struct ExprSynopsis {
    /// Statistics of the output values.
    pub column: ColumnStatistics,
    /// For a Boolean predicate, the fraction of input rows for which it is
    /// true, in [0, 1].
    pub selectivity: Option<f64>,
    /// Type-erased custom statistics, for example a sketch.
    pub extensions: Extensions,
    /// Data type of the output values.
    pub data_type: DataType,
}

impl ExprSynopsis {
    /// A synopsis with no known information, for a value of the given type.
    pub fn unknown(data_type: DataType) -> Self {
        Self {
            column: ColumnStatistics::new_unknown(),
            selectivity: None,
            extensions: Extensions::new(),
            data_type,
        }
    }

    /// A synopsis that holds the given column statistics.
    pub fn from_column(column: ColumnStatistics, data_type: DataType) -> Self {
        Self {
            column,
            ..Self::unknown(data_type)
        }
    }

    /// The synopsis of a constant: one distinct value, which is both the
    /// minimum and the maximum. A literal produces one value per input row,
    /// so a NULL constant's `null_count` is `num_rows`; otherwise it is 0.
    pub fn literal(value: ScalarValue, num_rows: Precision<usize>) -> Self {
        let data_type = value.data_type();
        let null_count = if value.is_null() {
            num_rows
        } else {
            Precision::Exact(0)
        };
        let column = ColumnStatistics {
            null_count,
            max_value: Precision::Exact(value.clone()),
            min_value: Precision::Exact(value),
            distinct_count: Precision::Exact(1),
            ..ColumnStatistics::new_unknown()
        };
        Self {
            column,
            ..Self::unknown(data_type)
        }
    }

    /// The number of distinct values, if known.
    pub fn ndv(&self) -> Option<usize> {
        match self.column.distinct_count {
            Precision::Exact(n) | Precision::Inexact(n) => Some(n),
            Precision::Absent => None,
        }
    }
}

/// Per-call input to [`PhysicalExpr::synopsis_from_inputs`]: the
/// relation-level statistics and the schema of the expression's input.
///
/// The fields are private so that a new call parameter can be added without
/// changing the trait method's signature.
///
/// [`PhysicalExpr::synopsis_from_inputs`]: crate::physical_expr::PhysicalExpr::synopsis_from_inputs
#[derive(Debug, Clone, Copy)]
pub struct SynopsisArgs<'a> {
    input_stats: &'a Statistics,
    input_schema: &'a Schema,
}

impl<'a> SynopsisArgs<'a> {
    /// Creates arguments for the given input statistics and schema.
    pub fn new(input_stats: &'a Statistics, input_schema: &'a Schema) -> Self {
        Self {
            input_stats,
            input_schema,
        }
    }

    /// The relation-level statistics of the expression's input.
    pub fn input_stats(&self) -> &'a Statistics {
        self.input_stats
    }

    /// The schema of the expression's input.
    pub fn input_schema(&self) -> &'a Schema {
        self.input_schema
    }
}
