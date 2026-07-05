//! Text features shared by rerankers.
//!
//! These are cheap, deterministic lexical signals — no embeddings, no network.
//! They power the [`crate::rerank::HeuristicReranker`] and double as the
//! offline fallback when a generative reranker is unavailable.

use std::collections::HashSet;

/// Splits text into lowercase alphanumeric tokens.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The Jaccard overlap between two token sets: `|A ∩ B| / |A ∪ B|`.
///
/// Returns `0.0` when both sets are empty (no signal), which keeps the metric
/// well-defined at the boundary.
#[must_use]
pub fn jaccard_overlap(query_tokens: &[String], doc_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() && doc_tokens.is_empty() {
        return 0.0;
    }
    let q: HashSet<&String> = query_tokens.iter().collect();
    let d: HashSet<&String> = doc_tokens.iter().collect();
    let intersection = q.intersection(&d).count();
    let union = q.union(&d).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// A small bundle of lexical features describing how well a document matches a
/// query. Kept deliberately interpretable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureVector {
    /// Fraction of the query's distinct terms that appear in the document.
    pub query_coverage: f32,
    /// Jaccard overlap between query and document token sets.
    pub jaccard: f32,
    /// Total number of query-term occurrences in the document (term frequency).
    pub term_frequency: f32,
}

impl FeatureVector {
    /// Extracts features for a `(query, document)` pair from their token lists.
    #[must_use]
    pub fn extract(query_tokens: &[String], doc_tokens: &[String]) -> Self {
        let query_set: HashSet<&String> = query_tokens.iter().collect();
        let doc_set: HashSet<&String> = doc_tokens.iter().collect();

        let covered = query_set.intersection(&doc_set).count();
        let query_coverage = if query_set.is_empty() {
            0.0
        } else {
            covered as f32 / query_set.len() as f32
        };

        let tf = doc_tokens.iter().filter(|t| query_set.contains(t)).count() as f32;

        Self {
            query_coverage,
            jaccard: jaccard_overlap(query_tokens, doc_tokens),
            term_frequency: tf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_lowercases_and_splits() {
        assert_eq!(
            tokenize("Rust, Async! runtime"),
            ["rust", "async", "runtime"]
        );
        assert!(tokenize("   ").is_empty());
    }

    #[test]
    fn jaccard_bounds() {
        let a = tokenize("rust async runtime");
        assert!((jaccard_overlap(&a, &a) - 1.0).abs() < 1e-6);
        let b = tokenize("python threads");
        assert_eq!(jaccard_overlap(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_empty_is_zero() {
        assert_eq!(jaccard_overlap(&[], &[]), 0.0);
    }

    #[test]
    fn feature_extraction() {
        let q = tokenize("rust async");
        let d = tokenize("rust rust async runtime concurrency");
        let f = FeatureVector::extract(&q, &d);
        assert!((f.query_coverage - 1.0).abs() < 1e-6);
        assert_eq!(f.term_frequency, 3.0); // rust x2 + async x1
        assert!(f.jaccard > 0.0 && f.jaccard < 1.0);
    }

    #[test]
    fn coverage_partial() {
        let q = tokenize("rust python");
        let d = tokenize("rust runtime");
        let f = FeatureVector::extract(&q, &d);
        assert!((f.query_coverage - 0.5).abs() < 1e-6);
    }
}
