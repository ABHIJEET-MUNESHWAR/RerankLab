//! Retrieval candidates and their scored, ranked forms.

use serde::{Deserialize, Serialize};

use crate::error::RerankError;
use crate::ids::DocId;

/// A first-stage retrieval candidate: a document plus the score the retriever
/// assigned it. The reranker consumes these and produces [`ScoredCandidate`]s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// Document identifier.
    pub id: DocId,
    /// The document text (title + body already joined by the retriever).
    pub text: String,
    /// The first-stage retrieval score (e.g. BM25 or fused score).
    pub retrieval_score: f32,
}

impl Candidate {
    /// Creates a candidate, rejecting empty text.
    ///
    /// # Errors
    /// Returns [`RerankError::EmptyCandidate`] if `text` is blank.
    pub fn new(
        id: DocId,
        text: impl Into<String>,
        retrieval_score: f32,
    ) -> Result<Self, RerankError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(RerankError::EmptyCandidate);
        }
        Ok(Self {
            id,
            text,
            retrieval_score,
        })
    }
}

/// A candidate after second-stage scoring. Carries both the original retrieval
/// score and the new reranker score so callers can inspect the delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredCandidate {
    /// Document identifier.
    pub id: DocId,
    /// The reranker's score (higher is more relevant).
    pub score: f32,
    /// The original first-stage retrieval score.
    pub retrieval_score: f32,
}

impl ScoredCandidate {
    /// Creates a scored candidate.
    #[must_use]
    pub const fn new(id: DocId, score: f32, retrieval_score: f32) -> Self {
        Self {
            id,
            score,
            retrieval_score,
        }
    }

    /// Total ordering by score, descending, with `NaN` sunk to the bottom.
    ///
    /// This lets ranked lists be sorted and compared without panicking on
    /// `NaN` and without pulling in extra crates.
    #[must_use]
    pub fn cmp_desc(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so the highest score sorts first; NaN is treated as the
        // smallest value so it sinks to the end.
        match other.score.partial_cmp(&self.score) {
            Some(ord) => ord,
            None => match (self.score.is_nan(), other.score.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => std::cmp::Ordering::Equal,
            },
        }
    }
}

/// An ordered list of scored candidates — the output of a reranker, sorted
/// best-first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RankedList {
    /// Scored candidates in descending score order.
    pub items: Vec<ScoredCandidate>,
}

impl RankedList {
    /// Builds a ranked list from scored candidates, sorting them best-first.
    #[must_use]
    pub fn from_scored(mut items: Vec<ScoredCandidate>) -> Self {
        items.sort_by(ScoredCandidate::cmp_desc);
        Self { items }
    }

    /// The number of candidates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The document ids in rank order — the input to ranking metrics.
    #[must_use]
    pub fn doc_ids(&self) -> Vec<DocId> {
        self.items.iter().map(|c| c.id).collect()
    }

    /// Truncates the list to the top `k` results in place.
    pub fn truncate(&mut self, k: usize) {
        self.items.truncate(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_rejects_blank() {
        assert_eq!(
            Candidate::new(DocId(1), "  ", 1.0).unwrap_err(),
            RerankError::EmptyCandidate
        );
    }

    #[test]
    fn ranked_list_sorts_descending() {
        let list = RankedList::from_scored(vec![
            ScoredCandidate::new(DocId(1), 0.2, 0.0),
            ScoredCandidate::new(DocId(2), 0.9, 0.0),
            ScoredCandidate::new(DocId(3), 0.5, 0.0),
        ]);
        assert_eq!(list.doc_ids(), vec![DocId(2), DocId(3), DocId(1)]);
    }

    #[test]
    fn nan_scores_sink_to_bottom() {
        let list = RankedList::from_scored(vec![
            ScoredCandidate::new(DocId(1), f32::NAN, 0.0),
            ScoredCandidate::new(DocId(2), 0.5, 0.0),
        ]);
        assert_eq!(list.doc_ids(), vec![DocId(2), DocId(1)]);
    }

    #[test]
    fn truncate_keeps_top_k() {
        let mut list = RankedList::from_scored(vec![
            ScoredCandidate::new(DocId(1), 0.9, 0.0),
            ScoredCandidate::new(DocId(2), 0.5, 0.0),
            ScoredCandidate::new(DocId(3), 0.1, 0.0),
        ]);
        list.truncate(2);
        assert_eq!(list.len(), 2);
        assert_eq!(list.doc_ids(), vec![DocId(1), DocId(2)]);
    }

    #[test]
    fn empty_list_reports_empty() {
        assert!(RankedList::default().is_empty());
    }
}
