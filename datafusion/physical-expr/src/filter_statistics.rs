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

//! Column statistics of the rows that satisfy a filter predicate.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, ScalarValue};
use datafusion_expr::Operator;

use crate::expressions::{BinaryExpr, Column, IsNotNullExpr, Literal};
use crate::{ExprBoundaries, PhysicalExpr, split_conjunction};

/// Collects column equality information from `col = literal` predicates in a
/// conjunction.
///
/// Returns `(eq_columns, is_infeasible)`:
/// - `eq_columns`: set of column indices constrained to a single literal value.
/// - `is_infeasible`: `true` when the same column is equated to two different
///   non-null literals (e.g. `name = 'alice' AND name = 'bob'`), which is
///   always unsatisfiable.
///
/// Only AND conjunctions are traversed; OR is intentionally skipped
/// since `a = 1 OR a = 2` does not pin NDV to 1.
pub fn collect_equality_columns(
    predicate: &Arc<dyn PhysicalExpr>,
) -> (HashSet<usize>, bool) {
    let mut eq_values: HashMap<usize, ScalarValue> = HashMap::new();
    let mut infeasible = false;

    for expr in split_conjunction(predicate) {
        let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
            continue;
        };
        if *binary.op() != Operator::Eq {
            continue;
        }
        let left = binary.left();
        let right = binary.right();
        let pair = if let Some(col) = left.downcast_ref::<Column>()
            && let Some(lit) = right.downcast_ref::<Literal>()
            && !lit.value().is_null()
        {
            Some((col.index(), lit.value().clone()))
        } else if let Some(col) = right.downcast_ref::<Column>()
            && let Some(lit) = left.downcast_ref::<Literal>()
            && !lit.value().is_null()
        {
            Some((col.index(), lit.value().clone()))
        } else {
            None
        };

        if let Some((idx, value)) = pair {
            match eq_values.entry(idx) {
                Entry::Occupied(prev) => {
                    if *prev.get() != value {
                        infeasible = true;
                        break;
                    }
                }
                Entry::Vacant(slot) => {
                    slot.insert(value);
                }
            }
        }
    }

    (eq_values.into_keys().collect(), infeasible)
}

/// Collects columns that cannot be NULL in any surviving row.
///
/// A filter keeps only rows where the predicate is TRUE, so a column is
/// null-rejecting if some top-level AND conjunct evaluates to NULL or FALSE
/// whenever that column is NULL. Two such conjuncts are recognized:
///
/// - a binary operator that returns NULL on NULL input, applied directly to the
///   column (e.g. `a = 10`, `a < b`);
/// - an `IS NOT NULL` check on the column (e.g. `a IS NOT NULL`).
///
/// This analysis is conservative; for example, OR clauses are not considered
/// null-rejecting, and neither are indirect operands like `a + 1 < 10`.
pub(crate) fn collect_null_rejecting_columns(
    predicate: &Arc<dyn PhysicalExpr>,
) -> HashSet<usize> {
    let mut columns = HashSet::new();

    for expr in split_conjunction(predicate) {
        // `col IS NOT NULL` keeps only rows where `col` is non-null.
        if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
            if let Some(col) = is_not_null.arg().downcast_ref::<Column>() {
                columns.insert(col.index());
            }
            continue;
        }

        // A binary operator that returns NULL on NULL input rejects rows where
        // a direct column operand is NULL.
        if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
            if !binary.op().returns_null_on_null() {
                continue;
            }
            if let Some(col) = binary.left().downcast_ref::<Column>() {
                columns.insert(col.index());
            }
            if let Some(col) = binary.right().downcast_ref::<Column>() {
                columns.insert(col.index());
            }
        }
    }

    columns
}

/// Converts an interval bound to a [`Precision`] value. NULL bounds (which
/// represent "unbounded" in the interval type) map to [`Precision::Absent`].
fn interval_bound_to_precision(
    bound: ScalarValue,
    is_exact: bool,
) -> Precision<ScalarValue> {
    if bound.is_null() {
        Precision::Absent
    } else if is_exact {
        Precision::Exact(bound)
    } else {
        Precision::Inexact(bound)
    }
}

/// Caps a row-bounded column statistic (a null count or distinct count) at the
/// filtered row count, since a column cannot have more nulls or distinct values
/// than it has rows. Known counts are demoted to inexact because a
/// filter-derived row bound is normally an estimate, the exception being an
/// exact zero, which proves the column is empty.
fn cap_at_rows(
    value: Precision<usize>,
    filtered_num_rows: Precision<usize>,
) -> Precision<usize> {
    match filtered_num_rows {
        Precision::Absent => value.to_inexact(),
        Precision::Exact(0) => Precision::Exact(0),
        rows => value.to_inexact().min(&rows),
    }
}

/// Scales a byte size by the filter selectivity. An exact zero row count means
/// the output is exactly empty, so the byte size is an exact zero too.
pub fn scale_byte_size_at_rows(
    byte_size: Precision<usize>,
    selectivity: f64,
    filtered_num_rows: Precision<usize>,
) -> Precision<usize> {
    if filtered_num_rows == Precision::Exact(0) {
        Precision::Exact(0)
    } else {
        byte_size.with_estimated_selectivity(selectivity)
    }
}

/// Returns the NDV for a column constrained to one non-null value (e.g.
/// `column = literal` or a singleton interval), derived from the filtered row
/// estimate: zero rows means zero distinct values, a known positive row count
/// means exactly one, and an unknown row count means an inexact one (the column
/// could still be empty).
///
/// The caller is responsible for proving the singleton domain.
fn distinct_count_for_singleton_domain(
    filtered_num_rows: Precision<usize>,
) -> Precision<usize> {
    match filtered_num_rows {
        Precision::Exact(0) | Precision::Inexact(0) => filtered_num_rows,
        // The row count is unknown, so the column could still be empty (zero
        // distinct values); report an inexact one rather than overstating it.
        Precision::Absent => Precision::Inexact(1),
        _ => Precision::Exact(1),
    }
}

/// Estimate NDV after applying a selectivity factor (filtering).
///
/// When filtering rows, each distinct value has multiple rows. If a value
/// appears `k` times, the probability it survives the filter is `1 - (1-s)^k`
/// where `s` is the selectivity.
///
/// Assuming uniform distribution (each value appears `rows/ndv` times):
/// ```text
/// NDV_after ~ NDV_before * [1 - (1 - selectivity)^(rows/NDV)]
/// ```
pub fn ndv_after_selectivity(
    original_ndv: usize,
    original_rows: usize,
    selectivity: f64,
) -> usize {
    if selectivity <= 0.0 || original_ndv == 0 || original_rows == 0 {
        return 0;
    }
    if selectivity >= 1.0 {
        return original_ndv;
    }

    let ndv = original_ndv as f64;
    let rows = original_rows as f64;

    let rows_per_value = rows / ndv;
    let survival_prob = 1.0 - (1.0 - selectivity).powf(rows_per_value);
    let expected_ndv = ndv * survival_prob;

    (expected_ndv.round() as usize).clamp(1, original_ndv)
}

/// Applies [`ndv_after_selectivity`] to a distinct count that is already
/// capped at the filtered row count, with the selectivity taken as the ratio
/// of the filtered row count to the input row count. A value can appear on
/// several rows, and the filter can remove all of them, so fewer distinct
/// values survive than the cap allows.
///
/// The count is returned unchanged when it is 1 or less (the formula gives
/// the same count), when a row count is unknown, when the input row count is
/// zero, and when the filter keeps every row.
pub(crate) fn distinct_count_after_filter(
    distinct_count: Precision<usize>,
    input_num_rows: Precision<usize>,
    filtered_num_rows: Precision<usize>,
) -> Precision<usize> {
    let (Some(&ndv), Some(&input_rows), Some(&filtered_rows)) = (
        distinct_count.get_value(),
        input_num_rows.get_value(),
        filtered_num_rows.get_value(),
    ) else {
        return distinct_count;
    };
    if ndv <= 1 || input_rows == 0 || filtered_rows >= input_rows {
        return distinct_count;
    }
    let selectivity = filtered_rows as f64 / input_rows as f64;
    Precision::Inexact(ndv_after_selectivity(ndv, input_rows, selectivity))
}

/// Builds one column's statistics after a filter, from that column's
/// interval-analysis boundaries.
///
/// The interval bounds become min/max values, a singleton interval gives a
/// singleton distinct count, and row-bounded counts are kept consistent with
/// the filtered row estimate. `data_type` is the column's type, used for the
/// typed nulls of an empty column.
pub(crate) fn column_statistics_from_boundaries(
    data_type: &DataType,
    input: &ColumnStatistics,
    boundaries: ExprBoundaries,
    selectivity: f64,
    null_rejecting: bool,
    filtered_num_rows: Precision<usize>,
) -> ColumnStatistics {
    let ExprBoundaries {
        interval,
        distinct_count,
        ..
    } = boundaries;
    let Some(interval) = interval else {
        // If the interval is `None`, we can say that there are no rows.
        // Use a typed null to preserve the column's data type, so that
        // downstream interval analysis can still intersect intervals
        // of the same type.
        let typed_null = ScalarValue::try_from(data_type).unwrap_or(ScalarValue::Null);
        return ColumnStatistics {
            null_count: Precision::Exact(0),
            max_value: Precision::Exact(typed_null.clone()),
            min_value: Precision::Exact(typed_null.clone()),
            sum_value: Precision::Exact(typed_null),
            distinct_count: Precision::Exact(0),
            byte_size: Precision::Exact(0),
        };
    };
    let (lower, upper) = interval.into_bounds();
    let is_single_value = !lower.is_null() && !upper.is_null() && lower == upper;
    let min_value = interval_bound_to_precision(lower, is_single_value);
    let max_value = interval_bound_to_precision(upper, is_single_value);

    // Distinct and null counts cannot exceed the number of rows
    // that survive the filter. Singleton intervals and
    // null-rejecting predicates provide tighter bounds.
    let capped_distinct_count = if is_single_value {
        distinct_count_for_singleton_domain(filtered_num_rows)
    } else {
        cap_at_rows(distinct_count, filtered_num_rows)
    };
    let capped_null_count = if null_rejecting {
        Precision::Exact(0)
    } else {
        cap_at_rows(input.null_count, filtered_num_rows)
    };
    let byte_size =
        scale_byte_size_at_rows(input.byte_size, selectivity, filtered_num_rows);
    ColumnStatistics {
        null_count: capped_null_count,
        max_value,
        min_value,
        sum_value: Precision::Absent,
        distinct_count: capped_distinct_count,
        byte_size,
    }
}

/// Builds one column's statistics after a filter that interval analysis does
/// not support, from the selectivity alone.
///
/// Min/max are kept as inexact input values. Row-bounded counts are kept
/// consistent with the filtered row estimate, a column constrained to one
/// value (`single_value`) gets a singleton distinct count, and a
/// null-rejecting column gets a null count of 0.
pub(crate) fn column_statistics_from_selectivity(
    input: &ColumnStatistics,
    selectivity: f64,
    null_rejecting: bool,
    single_value: bool,
    filtered_num_rows: Precision<usize>,
) -> ColumnStatistics {
    let mut column = input.clone().to_inexact();
    column.byte_size =
        scale_byte_size_at_rows(column.byte_size, selectivity, filtered_num_rows);
    column.null_count = if null_rejecting {
        Precision::Exact(0)
    } else {
        cap_at_rows(column.null_count, filtered_num_rows)
    };
    column.distinct_count = if single_value {
        distinct_count_for_singleton_domain(filtered_num_rows)
    } else {
        cap_at_rows(column.distinct_count, filtered_num_rows)
    };
    column
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_equality_columns() {
        // (description, predicate, expected_column_indices, expected_infeasible)
        #[expect(clippy::type_complexity)]
        let cases: Vec<(&str, Arc<dyn PhysicalExpr>, Vec<usize>, bool)> = vec![
            (
                "simple col = literal",
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 0)),
                    Operator::Eq,
                    Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                )),
                vec![0],
                false,
            ),
            (
                "reversed literal = col",
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                    Operator::Eq,
                    Arc::new(Column::new("a", 0)),
                )),
                vec![0],
                false,
            ),
            (
                "AND with two equalities",
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                    )),
                    Operator::And,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b", 1)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Utf8(Some(
                            "hello".to_string(),
                        )))),
                    )),
                )),
                vec![0, 1],
                false,
            ),
            (
                "OR produces empty set",
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                    )),
                    Operator::Or,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(99)))),
                    )),
                )),
                vec![],
                false,
            ),
            (
                "greater-than produces empty set",
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 0)),
                    Operator::Gt,
                    Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                )),
                vec![],
                false,
            ),
            (
                "col = col produces empty set",
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 0)),
                    Operator::Eq,
                    Arc::new(Column::new("b", 1)),
                )),
                vec![],
                false,
            ),
            (
                "nested AND with three equalities",
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("a", 0)),
                            Operator::Eq,
                            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
                        )),
                        Operator::And,
                        Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("b", 1)),
                            Operator::Eq,
                            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
                        )),
                    )),
                    Operator::And,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("c", 2)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
                    )),
                )),
                vec![0, 1, 2],
                false,
            ),
            (
                "AND with mixed equality and non-equality",
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                    )),
                    Operator::And,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("b", 1)),
                        Operator::Gt,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
                    )),
                )),
                vec![0],
                false,
            ),
            (
                "col = NULL is excluded",
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("a", 0)),
                    Operator::Eq,
                    Arc::new(Literal::new(ScalarValue::Int32(None))),
                )),
                vec![],
                false,
            ),
            (
                "NULL = col is excluded",
                Arc::new(BinaryExpr::new(
                    Arc::new(Literal::new(ScalarValue::Utf8(None))),
                    Operator::Eq,
                    Arc::new(Column::new("a", 0)),
                )),
                vec![],
                false,
            ),
            (
                "contradictory: same col, different literals",
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Utf8(Some(
                            "alice".to_string(),
                        )))),
                    )),
                    Operator::And,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Utf8(Some(
                            "bob".to_string(),
                        )))),
                    )),
                )),
                vec![0],
                true,
            ),
            (
                "same col, same literal is not contradictory",
                Arc::new(BinaryExpr::new(
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                    )),
                    Operator::And,
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("a", 0)),
                        Operator::Eq,
                        Arc::new(Literal::new(ScalarValue::Int32(Some(42)))),
                    )),
                )),
                vec![0],
                false,
            ),
        ];

        for (desc, expr, expected_cols, expected_infeasible) in cases {
            let (result, infeasible) = collect_equality_columns(&expr);
            let expected: HashSet<usize> = expected_cols.into_iter().collect();
            if expected_infeasible {
                // The scan stops at the first contradiction, so only the
                // infeasibility flag is asserted. The partial column set is
                // an implementation detail.
                assert!(infeasible, "case '{desc}': expected infeasible");
            } else {
                assert_eq!(result, expected, "case '{desc}': columns mismatch");
                assert!(!infeasible, "case '{desc}': expected feasible");
            }
        }
    }
}
