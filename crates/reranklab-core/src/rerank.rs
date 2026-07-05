//! A deterministic, network-free reranker built from lexical features.
//!
//! The [`HeuristicReranker`] blends three interpretable signals — query term
//! coverage, Jaccard overlap, and (log-damped) term frequency — with the
//! original first-stage retrieval score. It has no external dependencies, so it
//! is fully reproducible and serves as the **fallback** whenever a generative
//! reranker is unavailable (see `reranklab-ai`).

use async_trait::async_trait;

use reranklab_types::{Candidate, Query, RankedList, ScoredCandidate};

use crate::error::PortError;
use crate::features::{tokenize, FeatureVector};
use crate::ports::Reranker;

/// Tunable weights for the heuristic scoring function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankConfig {
    /// Weight on query-term coverage (`0.0..=1.0`).
    pub w_coverage: f32,
    /// Weight on Jaccard overlap.
    pub w_jaccard: f32,
    /// Weight on log-damped term frequency.
    pub w_term_freq: f32,
    /// Weight on the original retrieval score (blended in as a prior).
    pub w_retrieval: f32,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            w_coverage: 0.5,
            w_jaccard: 0.3,
            w_term_freq: 0.1,
            w_retrieval: 0.1,
        }
    }
}

/// A reranker that scores candidates by lexical relevance to the query.
#[derive(Debug, Clone, Default)]
pub struct HeuristicReranker {
    config: RerankConfig,
}

impl HeuristicReranker {
    /// Creates a reranker with the given weights.
    #[must_use]
    pub const fn new(config: RerankConfig) -> Self {
        Self { config }
    }

    /// Scores a single candidate against pre-tokenized query terms.
    fn score(&self, query_tokens: &[String], candidate: &Candidate) -> f32 {
        let doc_tokens = tokenize(&candidate.text);
        let f = FeatureVector::extract(query_tokens, &doc_tokens);
        let c = &self.config;
        // Log-damp term frequency so a flood of repeats cannot dominate.
        let tf = (1.0 + f.term_frequency).ln();
        c.w_coverage * f.query_coverage
            + c.w_jaccard * f.jaccard
            + c.w_term_freq * tf
            + c.w_retrieval * candidate.retrieval_score
    }
}

#[async_trait]
impl Reranker for HeuristicReranker {
    async fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
    ) -> Result<RankedList, PortError> {
        let query_tokens = tokenize(query.text());
        let scored = candidates
            .iter()
            .map(|c| ScoredCandidate::new(c.id, self.score(&query_tokens, c), c.retrieval_score))
            .collect();
        Ok(RankedList::from_scored(scored))
    }

    fn name(&self) -> &'static str {
        "heuristic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reranklab_types::{DocId, QueryId};

    fn cand(id: u64, text: &str, retrieval: f32) -> Candidate {
        Candidate::new(DocId(id), text, retrieval).unwrap()
    }

    #[tokio::test]
    async fn ranks_more_relevant_first() {
        let rr = HeuristicReranker::default();
        let q = Query::new(QueryId(1), "rust async runtime").unwrap();
        let candidates = vec![
            cand(1, "a post about python threading", 0.9),
            cand(2, "rust async runtime tokio executor", 0.1),
            cand(3, "rust programming basics", 0.5),
        ];
        let out = rr.rerank(&q, &candidates).await.unwrap();
        // Doc 2 matches all query terms and must rank first despite the lowest
        // retrieval score — the whole point of second-stage reranking.
        assert_eq!(out.doc_ids().first(), Some(&DocId(2)));
    }

    #[tokio::test]
    async fn empty_candidates_yield_empty_list() {
        let rr = HeuristicReranker::default();
        let q = Query::new(QueryId(1), "anything").unwrap();
        let out = rr.rerank(&q, &[]).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn deterministic_across_runs() {
        let rr = HeuristicReranker::default();
        let q = Query::new(QueryId(1), "rust async").unwrap();
        let candidates = vec![
            cand(1, "rust async runtime", 0.2),
            cand(2, "rust basics", 0.3),
        ];
        let a = rr.rerank(&q, &candidates).await.unwrap();
        let b = rr.rerank(&q, &candidates).await.unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn config_default_weights_sum_positive() {
        let c = RerankConfig::default();
        assert!(c.w_coverage + c.w_jaccard + c.w_term_freq + c.w_retrieval > 0.0);
    }

    #[tokio::test]
    async fn name_is_heuristic() {
        assert_eq!(HeuristicReranker::default().name(), "heuristic");
    }
}
