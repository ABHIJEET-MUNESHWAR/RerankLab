//! Ports: the traits that adapters implement. The core depends only on these,
//! never on concrete storage, models, or transports (dependency inversion).

use async_trait::async_trait;
use futures::stream::BoxStream;

use reranklab_types::{Candidate, Qrels, Query, QueryId, RankedList};

use crate::error::PortError;
use crate::event::RerankEvent;

/// A second-stage reranker: given a query and its candidates, produce a
/// re-scored, re-ordered list.
///
/// Implementations range from a deterministic lexical scorer to a generative
/// model behind an HTTP call. Callers treat them uniformly.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Reranks `candidates` for `query`, returning them best-first.
    async fn rerank(
        &self,
        query: &Query,
        candidates: &[Candidate],
    ) -> Result<RankedList, PortError>;

    /// A short label identifying the reranker (for metrics/logging).
    fn name(&self) -> &'static str;
}

/// Read access to first-stage retrieval candidates for a query.
#[async_trait]
pub trait CandidateStore: Send + Sync {
    /// Fetches the candidate documents retrieved for a query.
    async fn candidates(&self, query: QueryId) -> Result<Vec<Candidate>, PortError>;
}

/// Read access to the relevance judgments used for offline evaluation.
#[async_trait]
pub trait JudgmentStore: Send + Sync {
    /// Loads the full `qrels` set.
    async fn qrels(&self) -> Result<Qrels, PortError>;
}

/// The write side of the event stream — publishes domain events.
#[async_trait]
pub trait EventSink: Send + Sync {
    /// Publishes a reranking event to subscribers.
    async fn publish(&self, event: RerankEvent) -> Result<(), PortError>;
}

/// A boxed stream of reranking events.
pub type EventStream = BoxStream<'static, RerankEvent>;

/// The read side of the event stream — lets consumers subscribe.
pub trait RerankEventStream: Send + Sync {
    /// Subscribes to the live event feed.
    fn subscribe(&self) -> EventStream;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::mock;
    use reranklab_types::{DocId, ScoredCandidate};

    mock! {
        pub Rr {}

        #[async_trait]
        impl Reranker for Rr {
            async fn rerank(
                &self,
                query: &Query,
                candidates: &[Candidate],
            ) -> Result<RankedList, PortError>;
            fn name(&self) -> &'static str;
        }
    }

    #[tokio::test]
    async fn mock_reranker_is_usable() {
        let mut m = MockRr::new();
        m.expect_name().return_const("mock");
        m.expect_rerank().returning(|_, _| {
            Ok(RankedList::from_scored(vec![ScoredCandidate::new(
                DocId(1),
                1.0,
                0.0,
            )]))
        });
        let q = Query::new(QueryId(1), "hello").unwrap();
        let out = m.rerank(&q, &[]).await.unwrap();
        assert_eq!(m.name(), "mock");
        assert_eq!(out.len(), 1);
    }
}
