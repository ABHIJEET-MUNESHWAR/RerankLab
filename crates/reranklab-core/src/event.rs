//! Domain events emitted on the reranking path (the read side of CQRS).

use reranklab_types::QueryId;

/// An event describing something that happened during reranking.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RerankEvent {
    /// A query was reranked.
    QueryReranked {
        /// The query that was reranked.
        query: QueryId,
        /// Number of candidates considered.
        candidates: usize,
        /// Whether the AI reranker was used (`false` = heuristic fallback).
        used_ai: bool,
    },
}

impl RerankEvent {
    /// A short, stable label for the event kind.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::QueryReranked { .. } => "query_reranked",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_is_stable() {
        let e = RerankEvent::QueryReranked {
            query: QueryId(1),
            candidates: 5,
            used_ai: false,
        };
        assert_eq!(e.kind(), "query_reranked");
    }

    #[test]
    fn serializes_with_tag() {
        let e = RerankEvent::QueryReranked {
            query: QueryId(1),
            candidates: 3,
            used_ai: true,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"event\":\"query_reranked\""));
    }
}
