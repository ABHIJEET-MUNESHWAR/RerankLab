//! Offline evaluation metric containers.
//!
//! [`EvalMetrics`] holds the standard information-retrieval quality measures
//! for a single query at a fixed cutoff `k`. [`MetricsAccumulator`] averages
//! them across a query set to produce mean metrics (e.g. mean NDCG, MAP).

use serde::{Deserialize, Serialize};

/// Relevance-quality metrics for one ranked list at cutoff `k`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// The cutoff these metrics were computed at.
    pub k: usize,
    /// Normalized Discounted Cumulative Gain at `k` (range `0.0..=1.0`).
    pub ndcg: f64,
    /// Mean Reciprocal Rank (reciprocal of the first relevant rank).
    pub mrr: f64,
    /// Recall at `k` (fraction of all relevant docs retrieved in the top `k`).
    pub recall: f64,
    /// Precision at `k` (fraction of the top `k` that are relevant).
    pub precision: f64,
    /// Average Precision (area under the precision-recall curve for this query).
    pub average_precision: f64,
}

impl EvalMetrics {
    /// A zeroed metric set at cutoff `k`.
    #[must_use]
    pub const fn zeroed(k: usize) -> Self {
        Self {
            k,
            ndcg: 0.0,
            mrr: 0.0,
            recall: 0.0,
            precision: 0.0,
            average_precision: 0.0,
        }
    }
}

/// Accumulates per-query [`EvalMetrics`] and reports their mean.
#[derive(Debug, Clone, Default)]
pub struct MetricsAccumulator {
    count: u64,
    sum_ndcg: f64,
    sum_mrr: f64,
    sum_recall: f64,
    sum_precision: f64,
    sum_ap: f64,
    k: usize,
}

impl MetricsAccumulator {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one query's metrics into the running totals.
    pub fn add(&mut self, m: EvalMetrics) {
        self.count += 1;
        self.k = m.k;
        self.sum_ndcg += m.ndcg;
        self.sum_mrr += m.mrr;
        self.sum_recall += m.recall;
        self.sum_precision += m.precision;
        self.sum_ap += m.average_precision;
    }

    /// The number of queries accumulated.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether no queries have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The mean metrics across all accumulated queries.
    ///
    /// With no queries, returns a zeroed set (mean of the empty set is `0`).
    /// The reported `average_precision` field here is the **Mean** Average
    /// Precision (MAP) over the query set.
    #[must_use]
    pub fn mean(&self) -> EvalMetrics {
        if self.count == 0 {
            return EvalMetrics::zeroed(self.k);
        }
        let n = self.count as f64;
        EvalMetrics {
            k: self.k,
            ndcg: self.sum_ndcg / n,
            mrr: self.sum_mrr / n,
            recall: self.sum_recall / n,
            precision: self.sum_precision / n,
            average_precision: self.sum_ap / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_is_all_zero() {
        let m = EvalMetrics::zeroed(10);
        assert_eq!(m.k, 10);
        assert_eq!(m.ndcg, 0.0);
        assert_eq!(m.average_precision, 0.0);
    }

    #[test]
    fn mean_of_empty_is_zero() {
        let acc = MetricsAccumulator::new();
        assert!(acc.is_empty());
        assert_eq!(acc.mean(), EvalMetrics::zeroed(0));
    }

    #[test]
    fn mean_averages_correctly() {
        let mut acc = MetricsAccumulator::new();
        acc.add(EvalMetrics {
            k: 5,
            ndcg: 1.0,
            mrr: 1.0,
            recall: 1.0,
            precision: 1.0,
            average_precision: 1.0,
        });
        acc.add(EvalMetrics {
            k: 5,
            ndcg: 0.0,
            mrr: 0.0,
            recall: 0.0,
            precision: 0.0,
            average_precision: 0.0,
        });
        let mean = acc.mean();
        assert_eq!(acc.count(), 2);
        assert!((mean.ndcg - 0.5).abs() < 1e-12);
        assert!((mean.average_precision - 0.5).abs() < 1e-12);
    }
}
