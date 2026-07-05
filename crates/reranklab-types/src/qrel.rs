//! Relevance judgments (`qrels`) — the labeled ground truth for offline
//! evaluation. A judgment maps a `(query, document)` pair to a graded
//! relevance level.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ids::{DocId, QueryId};

/// A single graded relevance judgment for a `(query, document)` pair.
///
/// Grades follow the common TREC convention: `0` = not relevant, and
/// increasing positive integers denote increasing relevance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Judgment {
    /// The judged document.
    pub doc: DocId,
    /// Graded relevance (`0` = irrelevant, higher = more relevant).
    pub relevance: u8,
}

impl Judgment {
    /// Creates a judgment.
    #[must_use]
    pub const fn new(doc: DocId, relevance: u8) -> Self {
        Self { doc, relevance }
    }

    /// Whether this judgment counts the document as relevant (`relevance > 0`).
    #[must_use]
    pub const fn is_relevant(self) -> bool {
        self.relevance > 0
    }
}

/// A collection of relevance judgments, indexed by query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Qrels {
    /// Per-query map from document id to graded relevance.
    by_query: HashMap<QueryId, HashMap<DocId, u8>>,
}

impl Qrels {
    /// Creates an empty judgment set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or overwrites a judgment.
    pub fn insert(&mut self, query: QueryId, judgment: Judgment) {
        self.by_query
            .entry(query)
            .or_default()
            .insert(judgment.doc, judgment.relevance);
    }

    /// The graded relevance of a document for a query (`0` if unjudged).
    #[must_use]
    pub fn relevance(&self, query: QueryId, doc: DocId) -> u8 {
        self.by_query
            .get(&query)
            .and_then(|m| m.get(&doc))
            .copied()
            .unwrap_or(0)
    }

    /// The number of documents judged relevant (`relevance > 0`) for a query.
    #[must_use]
    pub fn relevant_count(&self, query: QueryId) -> usize {
        self.by_query
            .get(&query)
            .map(|m| m.values().filter(|&&r| r > 0).count())
            .unwrap_or(0)
    }

    /// The ideal (descending) relevance vector for a query, used as the
    /// denominator when computing normalized DCG.
    #[must_use]
    pub fn ideal_gains(&self, query: QueryId) -> Vec<u8> {
        let mut gains: Vec<u8> = self
            .by_query
            .get(&query)
            .map(|m| m.values().copied().filter(|&r| r > 0).collect())
            .unwrap_or_default();
        gains.sort_unstable_by(|a, b| b.cmp(a));
        gains
    }

    /// Whether any judgments exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_query.is_empty()
    }

    /// Iterates over every judgment as `(query, doc, relevance)` triples.
    pub fn iter(&self) -> impl Iterator<Item = (QueryId, DocId, u8)> + '_ {
        self.by_query
            .iter()
            .flat_map(|(&q, docs)| docs.iter().map(move |(&doc, &rel)| (q, doc, rel)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Qrels {
        let mut q = Qrels::new();
        q.insert(QueryId(1), Judgment::new(DocId(10), 3));
        q.insert(QueryId(1), Judgment::new(DocId(11), 1));
        q.insert(QueryId(1), Judgment::new(DocId(12), 0));
        q
    }

    #[test]
    fn relevance_lookup() {
        let q = sample();
        assert_eq!(q.relevance(QueryId(1), DocId(10)), 3);
        assert_eq!(q.relevance(QueryId(1), DocId(12)), 0);
        assert_eq!(q.relevance(QueryId(1), DocId(999)), 0);
        assert_eq!(q.relevance(QueryId(9), DocId(10)), 0);
    }

    #[test]
    fn relevant_count_ignores_zero() {
        assert_eq!(sample().relevant_count(QueryId(1)), 2);
    }

    #[test]
    fn ideal_gains_are_descending() {
        assert_eq!(sample().ideal_gains(QueryId(1)), vec![3, 1]);
    }

    #[test]
    fn judgment_relevance_flag() {
        assert!(Judgment::new(DocId(1), 2).is_relevant());
        assert!(!Judgment::new(DocId(1), 0).is_relevant());
    }

    #[test]
    fn iter_yields_all_triples() {
        let q = sample();
        let mut triples: Vec<_> = q.iter().collect();
        triples.sort_by_key(|&(_, d, _)| d.value());
        assert_eq!(
            triples,
            vec![
                (QueryId(1), DocId(10), 3),
                (QueryId(1), DocId(11), 1),
                (QueryId(1), DocId(12), 0),
            ]
        );
    }
}
