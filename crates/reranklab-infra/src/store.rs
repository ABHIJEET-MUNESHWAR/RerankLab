//! In-memory candidate and judgment stores.

use async_trait::async_trait;
use dashmap::DashMap;

use reranklab_core::{CandidateStore, JudgmentStore, PortError};
use reranklab_types::{Candidate, Judgment, Qrels, QueryId};

/// An in-memory [`CandidateStore`] backed by a concurrent map from query id to
/// its first-stage candidate list.
#[derive(Debug, Default)]
pub struct InMemoryCandidateStore {
    by_query: DashMap<QueryId, Vec<Candidate>>,
}

impl InMemoryCandidateStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts (or replaces) the candidate list for a query.
    pub fn insert(&self, query: QueryId, candidates: Vec<Candidate>) {
        self.by_query.insert(query, candidates);
    }

    /// The number of queries with stored candidates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_query.len()
    }

    /// Whether the store holds no queries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_query.is_empty()
    }
}

#[async_trait]
impl CandidateStore for InMemoryCandidateStore {
    async fn candidates(&self, query: QueryId) -> Result<Vec<Candidate>, PortError> {
        Ok(self
            .by_query
            .get(&query)
            .map(|v| v.clone())
            .unwrap_or_default())
    }
}

/// An in-memory [`JudgmentStore`] wrapping a [`Qrels`] set behind a lock-free
/// clone.
#[derive(Debug, Default)]
pub struct InMemoryJudgmentStore {
    qrels: parking_lot::RwLock<Qrels>,
}

impl InMemoryJudgmentStore {
    /// Creates a store seeded with the given judgments.
    #[must_use]
    pub fn new(qrels: Qrels) -> Self {
        Self {
            qrels: parking_lot::RwLock::new(qrels),
        }
    }

    /// Adds a judgment for a query.
    pub fn insert(&self, query: QueryId, judgment: Judgment) {
        self.qrels.write().insert(query, judgment);
    }
}

#[async_trait]
impl JudgmentStore for InMemoryJudgmentStore {
    async fn qrels(&self) -> Result<Qrels, PortError> {
        Ok(self.qrels.read().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reranklab_types::DocId;

    #[tokio::test]
    async fn candidate_round_trip() {
        let store = InMemoryCandidateStore::new();
        assert!(store.is_empty());
        store.insert(
            QueryId(1),
            vec![Candidate::new(DocId(1), "hello world", 0.5).unwrap()],
        );
        assert_eq!(store.len(), 1);
        let got = store.candidates(QueryId(1)).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, DocId(1));
    }

    #[tokio::test]
    async fn missing_query_returns_empty() {
        let store = InMemoryCandidateStore::new();
        assert!(store.candidates(QueryId(9)).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn judgment_round_trip() {
        let store = InMemoryJudgmentStore::default();
        store.insert(QueryId(1), Judgment::new(DocId(10), 3));
        let qrels = store.qrels().await.unwrap();
        assert_eq!(qrels.relevance(QueryId(1), DocId(10)), 3);
    }
}
