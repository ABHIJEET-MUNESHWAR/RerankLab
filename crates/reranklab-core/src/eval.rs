//! Offline relevance evaluation.
//!
//! Given a ranked list of document ids and a labeled [`Qrels`] set, this module
//! computes the standard information-retrieval quality metrics at a cutoff `k`:
//!
//! - **DCG / NDCG@k** — Discounted Cumulative Gain, normalized by the ideal
//!   ranking. Uses the standard `gain = 2^rel - 1`, `discount = 1/log2(rank+1)`.
//! - **MRR** — reciprocal of the rank of the first relevant document.
//! - **Recall@k** — relevant docs retrieved in the top `k` over all relevant.
//! - **Precision@k** — relevant docs in the top `k` over `k`.
//! - **Average Precision** — mean of precision values at each relevant hit.

use reranklab_types::{DocId, EvalMetrics, MetricsAccumulator, Qrels, QueryId, RankedList};

use crate::error::CoreError;

/// Discounted Cumulative Gain of a graded-relevance vector.
///
/// `gains[i]` is the relevance grade of the document at rank `i` (0-based).
fn dcg(gains: &[u8]) -> f64 {
    gains
        .iter()
        .enumerate()
        .map(|(i, &rel)| {
            let gain = (2f64.powi(i32::from(rel))) - 1.0;
            let discount = ((i as f64) + 2.0).log2(); // log2(rank+1), rank is 1-based
            gain / discount
        })
        .sum()
}

/// Computes [`EvalMetrics`] for one ranked list at cutoff `k`.
///
/// # Errors
/// Returns [`CoreError::Invalid`] wrapping [`reranklab_types::RerankError::ZeroCutoff`]
/// if `k == 0`.
pub fn evaluate_query(
    query: QueryId,
    ranking: &[DocId],
    qrels: &Qrels,
    k: usize,
) -> Result<EvalMetrics, CoreError> {
    if k == 0 {
        return Err(reranklab_types::RerankError::ZeroCutoff.into());
    }

    let top: Vec<DocId> = ranking.iter().copied().take(k).collect();
    let gains: Vec<u8> = top.iter().map(|&d| qrels.relevance(query, d)).collect();

    // NDCG@k
    let ideal: Vec<u8> = qrels.ideal_gains(query).into_iter().take(k).collect();
    let idcg = dcg(&ideal);
    let ndcg = if idcg > 0.0 { dcg(&gains) / idcg } else { 0.0 };

    // Precision@k and Recall@k
    let relevant_in_top = gains.iter().filter(|&&r| r > 0).count();
    let precision = relevant_in_top as f64 / k as f64;
    let total_relevant = qrels.relevant_count(query);
    let recall = if total_relevant > 0 {
        relevant_in_top as f64 / total_relevant as f64
    } else {
        0.0
    };

    // MRR — reciprocal rank of the first relevant document.
    let mrr = gains
        .iter()
        .position(|&r| r > 0)
        .map_or(0.0, |pos| 1.0 / ((pos as f64) + 1.0));

    // Average Precision — mean of precision@i at each relevant hit.
    let mut hits = 0usize;
    let mut ap_sum = 0.0f64;
    for (i, &rel) in gains.iter().enumerate() {
        if rel > 0 {
            hits += 1;
            ap_sum += hits as f64 / ((i as f64) + 1.0);
        }
    }
    let average_precision = if total_relevant > 0 {
        ap_sum / (total_relevant.min(k) as f64)
    } else {
        0.0
    };

    Ok(EvalMetrics {
        k,
        ndcg,
        mrr,
        recall,
        precision,
        average_precision,
    })
}

/// Evaluates many `(query, ranking)` pairs and returns the mean metrics (MAP,
/// mean NDCG, etc.) across them.
///
/// # Errors
/// Propagates [`CoreError`] from [`evaluate_query`] (e.g. `k == 0`).
pub fn evaluate(
    rankings: &[(QueryId, RankedList)],
    qrels: &Qrels,
    k: usize,
) -> Result<EvalMetrics, CoreError> {
    let mut acc = MetricsAccumulator::new();
    for (query, list) in rankings {
        acc.add(evaluate_query(*query, &list.doc_ids(), qrels, k)?);
    }
    Ok(acc.mean())
}

/// A stateless evaluator bound to a `qrels` set and cutoff, convenient for
/// repeated evaluation with fixed parameters.
#[derive(Debug, Clone)]
pub struct RerankEvaluator {
    qrels: Qrels,
    k: usize,
}

impl RerankEvaluator {
    /// Creates an evaluator.
    #[must_use]
    pub fn new(qrels: Qrels, k: usize) -> Self {
        Self { qrels, k }
    }

    /// Evaluates a single query's ranked list.
    ///
    /// # Errors
    /// Propagates [`CoreError`] from [`evaluate_query`].
    pub fn evaluate_one(
        &self,
        query: QueryId,
        ranking: &RankedList,
    ) -> Result<EvalMetrics, CoreError> {
        evaluate_query(query, &ranking.doc_ids(), &self.qrels, self.k)
    }

    /// Evaluates many queries and returns mean metrics.
    ///
    /// # Errors
    /// Propagates [`CoreError`] from [`evaluate_query`].
    pub fn evaluate_all(
        &self,
        rankings: &[(QueryId, RankedList)],
    ) -> Result<EvalMetrics, CoreError> {
        evaluate(rankings, &self.qrels, self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reranklab_types::{Judgment, ScoredCandidate};

    fn qrels() -> Qrels {
        // Query 1: doc 10 highly relevant (3), doc 11 relevant (1).
        let mut q = Qrels::new();
        q.insert(QueryId(1), Judgment::new(DocId(10), 3));
        q.insert(QueryId(1), Judgment::new(DocId(11), 1));
        q
    }

    fn ranking(ids: &[u64]) -> RankedList {
        let scored = ids
            .iter()
            .rev()
            .enumerate()
            .map(|(i, &id)| ScoredCandidate::new(DocId(id), i as f32, 0.0))
            .collect();
        RankedList::from_scored(scored)
    }

    #[test]
    fn perfect_ranking_scores_one() {
        // Ideal order: 10 (rel 3) then 11 (rel 1).
        let m = evaluate_query(QueryId(1), &[DocId(10), DocId(11)], &qrels(), 2).unwrap();
        assert!((m.ndcg - 1.0).abs() < 1e-9, "ndcg={}", m.ndcg);
        assert!((m.mrr - 1.0).abs() < 1e-9);
        assert!((m.recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn worst_ranking_scores_low_ndcg() {
        // Two irrelevant docs ahead of the relevant ones, cutoff 2.
        let m = evaluate_query(QueryId(1), &[DocId(98), DocId(99)], &qrels(), 2).unwrap();
        assert_eq!(m.ndcg, 0.0);
        assert_eq!(m.mrr, 0.0);
        assert_eq!(m.recall, 0.0);
        assert_eq!(m.precision, 0.0);
    }

    #[test]
    fn reversed_ranking_is_worse_than_ideal() {
        let ideal = evaluate_query(QueryId(1), &[DocId(10), DocId(11)], &qrels(), 2).unwrap();
        let reversed = evaluate_query(QueryId(1), &[DocId(11), DocId(10)], &qrels(), 2).unwrap();
        assert!(reversed.ndcg < ideal.ndcg);
        assert!(reversed.ndcg > 0.0);
    }

    #[test]
    fn precision_and_recall_at_k() {
        // Top-2 of [10, 98]: one relevant of two retrieved; one of two relevant.
        let m = evaluate_query(QueryId(1), &[DocId(10), DocId(98)], &qrels(), 2).unwrap();
        assert!((m.precision - 0.5).abs() < 1e-9);
        assert!((m.recall - 0.5).abs() < 1e-9);
    }

    #[test]
    fn zero_cutoff_errors() {
        let err = evaluate_query(QueryId(1), &[DocId(10)], &qrels(), 0).unwrap_err();
        assert_eq!(err.code(), "zero_cutoff");
    }

    #[test]
    fn average_precision_rewards_early_hits() {
        // Relevant at ranks 1 and 3 vs ranks 2 and 3 — earlier is better.
        let early =
            evaluate_query(QueryId(1), &[DocId(10), DocId(98), DocId(11)], &qrels(), 3).unwrap();
        let late =
            evaluate_query(QueryId(1), &[DocId(98), DocId(10), DocId(11)], &qrels(), 3).unwrap();
        assert!(early.average_precision > late.average_precision);
    }

    #[test]
    fn mean_over_queries() {
        let q = qrels();
        let rankings = vec![
            (QueryId(1), ranking(&[10, 11])), // perfect
            (QueryId(1), ranking(&[98, 99])), // miss
        ];
        let mean = evaluate(&rankings, &q, 2).unwrap();
        assert!(mean.ndcg > 0.0 && mean.ndcg < 1.0);
    }

    #[test]
    fn evaluator_wrapper_matches_free_fn() {
        let q = qrels();
        let ev = RerankEvaluator::new(q.clone(), 2);
        let list = ranking(&[10, 11]);
        let a = ev.evaluate_one(QueryId(1), &list).unwrap();
        let b = evaluate_query(QueryId(1), &list.doc_ids(), &q, 2).unwrap();
        assert_eq!(a, b);
    }
}
