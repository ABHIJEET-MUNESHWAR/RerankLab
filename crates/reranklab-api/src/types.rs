//! GraphQL DTOs — an anti-corruption layer over the domain types.

use async_graphql::{InputObject, SimpleObject};

use reranklab_types::{EvalMetrics, ScoredCandidate};

/// Input for a single first-stage candidate to be reranked.
#[derive(Debug, Clone, InputObject)]
pub struct CandidateInput {
    /// Document id.
    pub id: u64,
    /// Document text (title + body).
    pub text: String,
    /// First-stage retrieval score.
    pub retrieval_score: f32,
}

/// A single reranked result.
#[derive(Debug, Clone, SimpleObject)]
pub struct ScoredCandidateObject {
    /// Document id.
    pub id: u64,
    /// The reranker's score.
    pub score: f32,
    /// The original first-stage retrieval score.
    pub retrieval_score: f32,
}

impl From<ScoredCandidate> for ScoredCandidateObject {
    fn from(c: ScoredCandidate) -> Self {
        Self {
            id: c.id.value(),
            score: c.score,
            retrieval_score: c.retrieval_score,
        }
    }
}

/// Alias kept for schema readability in list positions.
pub type RankedItemObject = ScoredCandidateObject;

/// The result of a rerank request.
#[derive(Debug, Clone, SimpleObject)]
pub struct RerankResultObject {
    /// Name of the reranker that produced the result (`ai` or `heuristic`).
    pub reranker: String,
    /// Reranked candidates, best-first.
    pub ranked: Vec<ScoredCandidateObject>,
}

/// Offline evaluation metrics at a cutoff.
#[derive(Debug, Clone, Copy, SimpleObject)]
pub struct EvalMetricsObject {
    /// The cutoff `k`.
    pub k: u32,
    /// Normalized Discounted Cumulative Gain at `k`.
    pub ndcg: f64,
    /// Mean Reciprocal Rank.
    pub mrr: f64,
    /// Recall at `k`.
    pub recall: f64,
    /// Precision at `k`.
    pub precision: f64,
    /// (Mean) Average Precision.
    pub map: f64,
}

impl From<EvalMetrics> for EvalMetricsObject {
    fn from(m: EvalMetrics) -> Self {
        Self {
            k: m.k as u32,
            ndcg: m.ndcg,
            mrr: m.mrr,
            recall: m.recall,
            precision: m.precision,
            map: m.average_precision,
        }
    }
}
