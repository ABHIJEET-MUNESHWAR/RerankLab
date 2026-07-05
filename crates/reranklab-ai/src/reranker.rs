//! The generative-AI reranker with a deterministic fallback.
//!
//! [`AiReranker`] asks a [`ChatModel`] to score each candidate's relevance to
//! the query, wraps that call in a timeout and bounded retry, and parses the
//! model's JSON reply into scores. If **anything** goes wrong — the model is
//! unavailable, times out, exhausts retries, or returns an unparseable reply —
//! it transparently falls back to the deterministic
//! [`HeuristicReranker`], so the caller always receives a well-formed ranking.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use reranklab_core::error::PortError;
use reranklab_core::ports::Reranker;
use reranklab_core::rerank::HeuristicReranker;
use reranklab_resilience::{retry_if, with_timeout, RetryPolicy};
use reranklab_types::{Candidate, DocId, Query, RankedList, ScoredCandidate};

use crate::client::ChatModel;
use crate::error::AiError;

/// A reranker backed by a generative model, degrading to a heuristic fallback.
pub struct AiReranker {
    model: Arc<dyn ChatModel>,
    fallback: HeuristicReranker,
    timeout: Duration,
    retry: RetryPolicy,
}

/// One `{ "id": <u64>, "score": <f32> }` entry the model is asked to emit.
#[derive(Debug, Deserialize)]
struct ScoreItem {
    id: u64,
    score: f32,
}

impl AiReranker {
    /// Creates an AI reranker over the given model, using default timeout
    /// (5s) and retry (3 attempts) policies and the default heuristic fallback.
    #[must_use]
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self {
            model,
            fallback: HeuristicReranker::default(),
            timeout: Duration::from_secs(5),
            retry: RetryPolicy::default(),
        }
    }

    /// Overrides the per-request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Overrides the retry policy.
    #[must_use]
    pub const fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Builds the scoring prompt listing every candidate.
    fn build_prompt(query: &Query, candidates: &[Candidate]) -> String {
        let mut p = String::with_capacity(256 + candidates.len() * 64);
        p.push_str(
            "You are a search reranker. Score each document's relevance to the query \
             from 0.0 (irrelevant) to 1.0 (perfect). Reply ONLY with a JSON array of \
             objects like [{\"id\": <id>, \"score\": <float>}].\n\nQuery: ",
        );
        p.push_str(query.text());
        p.push_str("\n\nDocuments:\n");
        for c in candidates {
            // Keep prompts bounded; the first 240 chars carry the signal.
            let snippet: String = c.text.chars().take(240).collect();
            p.push_str(&format!("- id {}: {}\n", c.id.value(), snippet));
        }
        p
    }

    /// Parses the model's reply into `(DocId, score)` pairs, keeping only ids
    /// that were actually offered as candidates.
    fn parse_scores(reply: &str, candidates: &[Candidate]) -> Result<Vec<ScoredCandidate>, AiError> {
        let items: Vec<ScoreItem> = serde_json::from_str(reply.trim())
            .map_err(|e| AiError::Parse(e.to_string()))?;

        let retrieval: std::collections::HashMap<u64, f32> = candidates
            .iter()
            .map(|c| (c.id.value(), c.retrieval_score))
            .collect();

        let scored: Vec<ScoredCandidate> = items
            .into_iter()
            .filter_map(|it| {
                retrieval
                    .get(&it.id)
                    .map(|&r| ScoredCandidate::new(DocId(it.id), it.score, r))
            })
            .collect();

        if scored.is_empty() {
            return Err(AiError::Parse("no valid scored candidates".to_string()));
        }
        Ok(scored)
    }

    /// Calls the model with timeout + retry and parses the result.
    async fn score_with_model(
        &self,
        query: &Query,
        candidates: &[Candidate],
    ) -> Result<RankedList, AiError> {
        let prompt = Self::build_prompt(query, candidates);

        let reply = retry_if(
            self.retry,
            AiError::is_retryable,
            || async {
                match with_timeout(self.timeout, self.model.complete(&prompt)).await {
                    Ok(r) => r,
                    Err(_) => Err(AiError::Timeout),
                }
            },
        )
        .await?;

        let scored = Self::parse_scores(&reply, candidates)?;
        Ok(RankedList::from_scored(scored))
    }
}

#[async_trait]
impl Reranker for AiReranker {
    async fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
    ) -> Result<RankedList, PortError> {
        if candidates.is_empty() {
            return Ok(RankedList::default());
        }
        match self.score_with_model(query, candidates).await {
            Ok(list) => {
                metrics::counter!("reranklab_ai_success_total").increment(1);
                Ok(list)
            }
            Err(e) => {
                // Graceful degradation: never fail the request, fall back.
                metrics::counter!("reranklab_ai_fallback_total", "reason" => e.code())
                    .increment(1);
                tracing::warn!(error = %e, "ai reranker failed; using heuristic fallback");
                self.fallback.rerank(query, candidates).await
            }
        }
    }

    fn name(&self) -> &'static str {
        "ai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reranklab_types::QueryId;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn cand(id: u64, text: &str, retrieval: f32) -> Candidate {
        Candidate::new(DocId(id), text, retrieval).unwrap()
    }

    /// A stub model returning a fixed reply.
    struct FixedModel(String);
    #[async_trait]
    impl ChatModel for FixedModel {
        async fn complete(&self, _prompt: &str) -> Result<String, AiError> {
            Ok(self.0.clone())
        }
    }

    /// A stub model that always fails after counting calls.
    struct FailingModel {
        calls: AtomicU32,
        err: fn() -> AiError,
    }
    #[async_trait]
    impl ChatModel for FailingModel {
        async fn complete(&self, _prompt: &str) -> Result<String, AiError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err((self.err)())
        }
    }

    fn query() -> Query {
        Query::new(QueryId(1), "rust async runtime").unwrap()
    }

    fn candidates() -> Vec<Candidate> {
        vec![
            cand(1, "python threads", 0.9),
            cand(2, "rust async runtime tokio", 0.1),
        ]
    }

    #[tokio::test]
    async fn uses_model_scores_when_valid() {
        let model = Arc::new(FixedModel(
            r#"[{"id": 1, "score": 0.1}, {"id": 2, "score": 0.95}]"#.to_string(),
        ));
        let rr = AiReranker::new(model);
        let out = rr.rerank(&query(), &candidates()).await.unwrap();
        // Model ranked doc 2 highest.
        assert_eq!(out.doc_ids().first(), Some(&DocId(2)));
        assert_eq!(rr.name(), "ai");
    }

    #[tokio::test]
    async fn falls_back_on_unparseable_reply() {
        let model = Arc::new(FixedModel("not json at all".to_string()));
        let rr = AiReranker::new(model);
        let out = rr.rerank(&query(), &candidates()).await.unwrap();
        // Heuristic still ranks the lexically-matching doc 2 first.
        assert_eq!(out.doc_ids().first(), Some(&DocId(2)));
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn falls_back_after_retry_exhaustion() {
        let model = Arc::new(FailingModel {
            calls: AtomicU32::new(0),
            err: || AiError::Transport("down".into()),
        });
        let rr = AiReranker::new(model.clone())
            .with_retry(RetryPolicy {
                max_attempts: 2,
                base_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(1),
            });
        let out = rr.rerank(&query(), &candidates()).await.unwrap();
        assert_eq!(out.len(), 2);
        // Retried the configured number of attempts before giving up.
        assert_eq!(model.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable_status() {
        let model = Arc::new(FailingModel {
            calls: AtomicU32::new(0),
            err: || AiError::Status(400),
        });
        let rr = AiReranker::new(model.clone());
        let _ = rr.rerank(&query(), &candidates()).await.unwrap();
        // 400 is not retryable → exactly one attempt, then fallback.
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn empty_candidates_short_circuit() {
        let model = Arc::new(FixedModel("[]".to_string()));
        let rr = AiReranker::new(model);
        let out = rr.rerank(&query(), &[]).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn ignores_unknown_ids_from_model() {
        // Model invents id 999 and omits id 1; only id 2 is valid.
        let model = Arc::new(FixedModel(
            r#"[{"id": 999, "score": 1.0}, {"id": 2, "score": 0.8}]"#.to_string(),
        ));
        let rr = AiReranker::new(model);
        let out = rr.rerank(&query(), &candidates()).await.unwrap();
        assert_eq!(out.doc_ids(), vec![DocId(2)]);
    }
}
