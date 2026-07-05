//! The reranking orchestrator.
//!
//! [`RerankService`] wraps any [`Reranker`] with cross-cutting concerns:
//! ingest rate limiting, metrics, and CQRS event emission. It is generic over a
//! [`Clock`] so the rate limiter can be driven deterministically in tests.

use std::sync::Arc;

use reranklab_resilience::{Clock, RateLimiter, SystemClock};
use reranklab_types::{Candidate, Query, RankedList};

use crate::error::CoreError;
use crate::event::RerankEvent;
use crate::ports::{EventSink, Reranker};

/// The result of a rerank request: the ranked list plus whether the AI path
/// (as opposed to the heuristic fallback) produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankOutcome {
    /// The reranked candidates, best-first.
    pub ranked: RankedList,
    /// The name of the reranker that produced the result.
    pub reranker: &'static str,
}

/// Orchestrates reranking with resilience, metrics, and events.
pub struct RerankService<C: Clock = SystemClock> {
    reranker: Arc<dyn Reranker>,
    events: Arc<dyn EventSink>,
    limiter: RateLimiter<C>,
}

impl<C: Clock> RerankService<C> {
    /// Creates a service from its collaborators.
    #[must_use]
    pub fn new(
        reranker: Arc<dyn Reranker>,
        events: Arc<dyn EventSink>,
        limiter: RateLimiter<C>,
    ) -> Self {
        Self {
            reranker,
            events,
            limiter,
        }
    }

    /// Reranks `candidates` for `query`, applying rate limiting, recording
    /// metrics, and publishing a [`RerankEvent::QueryReranked`].
    ///
    /// # Errors
    /// - [`CoreError::Port`] with `Unavailable` if the rate limit is exceeded.
    /// - [`CoreError::Port`] if the underlying reranker or event sink fails.
    pub async fn rerank(
        &self,
        query: &Query,
        candidates: Vec<Candidate>,
    ) -> Result<RerankOutcome, CoreError> {
        if !self.limiter.try_acquire_n(1.0) {
            metrics::counter!("reranklab_rerank_throttled_total").increment(1);
            return Err(crate::error::PortError::Unavailable("rate limited".into()).into());
        }

        let ranked = self.reranker.rerank(query, &candidates).await?;
        let name = self.reranker.name();
        let used_ai = name != "heuristic";

        metrics::counter!("reranklab_queries_reranked_total").increment(1);
        metrics::histogram!("reranklab_candidates_per_query").record(candidates.len() as f64);
        if used_ai {
            metrics::counter!("reranklab_ai_rerank_total").increment(1);
        } else {
            metrics::counter!("reranklab_heuristic_rerank_total").increment(1);
        }

        self.events
            .publish(RerankEvent::QueryReranked {
                query: query.id,
                candidates: candidates.len(),
                used_ai,
            })
            .await?;

        Ok(RerankOutcome {
            ranked,
            reranker: name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use reranklab_resilience::ManualClock;
    use reranklab_types::{DocId, QueryId, ScoredCandidate};
    use std::time::Duration;

    struct StubReranker;

    #[async_trait]
    impl Reranker for StubReranker {
        async fn rerank(
            &self,
            _q: &Query,
            candidates: &[Candidate],
        ) -> Result<RankedList, crate::error::PortError> {
            let scored = candidates
                .iter()
                .map(|c| ScoredCandidate::new(c.id, c.retrieval_score, c.retrieval_score))
                .collect();
            Ok(RankedList::from_scored(scored))
        }
        fn name(&self) -> &'static str {
            "heuristic"
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        events: Mutex<Vec<RerankEvent>>,
    }

    #[async_trait]
    impl EventSink for RecordingSink {
        async fn publish(&self, event: RerankEvent) -> Result<(), crate::error::PortError> {
            self.events.lock().push(event);
            Ok(())
        }
    }

    fn cand(id: u64, score: f32) -> Candidate {
        Candidate::new(DocId(id), format!("doc {id}"), score).unwrap()
    }

    #[tokio::test]
    async fn reranks_and_emits_event() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::new(10.0, 10.0, clock);
        let sink = Arc::new(RecordingSink::default());
        let svc = RerankService::new(Arc::new(StubReranker), sink.clone(), limiter);

        let q = Query::new(QueryId(1), "hello").unwrap();
        let out = svc
            .rerank(&q, vec![cand(1, 0.2), cand(2, 0.9)])
            .await
            .unwrap();

        assert_eq!(out.reranker, "heuristic");
        assert_eq!(out.ranked.doc_ids(), vec![DocId(2), DocId(1)]);
        assert_eq!(sink.events.lock().len(), 1);
    }

    #[tokio::test]
    async fn rate_limit_rejects() {
        let clock = Arc::new(ManualClock::new());
        // Capacity 1, no refill within the test window.
        let limiter = RateLimiter::new(1.0, 0.0, clock);
        let sink = Arc::new(RecordingSink::default());
        let svc = RerankService::new(Arc::new(StubReranker), sink, limiter);
        let q = Query::new(QueryId(1), "hello").unwrap();

        assert!(svc.rerank(&q, vec![cand(1, 0.1)]).await.is_ok());
        let err = svc.rerank(&q, vec![cand(1, 0.1)]).await.unwrap_err();
        assert_eq!(err.code(), "unavailable");
    }

    #[tokio::test]
    async fn refill_restores_capacity() {
        let clock = Arc::new(ManualClock::new());
        let limiter = RateLimiter::new(1.0, 1.0, clock.clone());
        let sink = Arc::new(RecordingSink::default());
        let svc = RerankService::new(Arc::new(StubReranker), sink, limiter);
        let q = Query::new(QueryId(1), "hello").unwrap();

        assert!(svc.rerank(&q, vec![cand(1, 0.1)]).await.is_ok());
        assert!(svc.rerank(&q, vec![cand(1, 0.1)]).await.is_err());
        clock.advance(Duration::from_secs(1)); // refill one token
        assert!(svc.rerank(&q, vec![cand(1, 0.1)]).await.is_ok());
    }
}
