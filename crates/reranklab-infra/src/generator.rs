//! A deterministic synthetic scenario generator.
//!
//! It fabricates a set of queries, each with a pool of candidate documents and
//! a matching `qrels` set. **Relevant** documents are seeded with the query's
//! own terms (so a relevance-aware reranker can find them), while the
//! first-stage retrieval score is deliberately *noisy* — often ranking
//! irrelevant documents highly. This makes the gap between "first-stage order"
//! and "reranked order" observable in demos and benchmarks.

use reranklab_types::{Candidate, DocId, Judgment, Qrels, Query, QueryId};

/// A fully-materialized evaluation scenario.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// The queries to rerank.
    pub queries: Vec<Query>,
    /// Candidate lists, aligned by index with `queries`.
    pub candidates: Vec<(QueryId, Vec<Candidate>)>,
    /// Ground-truth relevance judgments.
    pub qrels: Qrels,
}

/// Generates deterministic [`Scenario`]s from a seed.
#[derive(Debug, Clone, Copy)]
pub struct ScenarioGenerator {
    seed: u64,
}

const TOPICS: [&str; 8] = [
    "rust async runtime",
    "vector database search",
    "distributed consensus raft",
    "graph neural networks",
    "reranking relevance evaluation",
    "tokio scheduler tasks",
    "bm25 inverted index",
    "transformer attention layers",
];

const FILLER: [&str; 12] = [
    "system",
    "design",
    "overview",
    "guide",
    "notes",
    "reference",
    "example",
    "tutorial",
    "deep",
    "dive",
    "practical",
    "advanced",
];

impl ScenarioGenerator {
    /// Creates a generator with the given seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// splitmix64 — a fast, high-quality deterministic PRNG step.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Builds a scenario with `queries` queries and `pool` candidates each, of
    /// which `relevant` are seeded as relevant.
    #[must_use]
    pub fn generate(&self, queries: usize, pool: usize, relevant: usize) -> Scenario {
        let mut state = self.seed;
        let relevant = relevant.min(pool);

        let mut out_queries = Vec::with_capacity(queries);
        let mut out_candidates = Vec::with_capacity(queries);
        let mut qrels = Qrels::new();
        let mut doc_counter = 0u64;

        for qi in 0..queries {
            let topic = TOPICS[qi % TOPICS.len()];
            let qid = QueryId(qi as u64);
            let query = Query::new(qid, topic).expect("topic is non-empty");

            let topic_terms: Vec<&str> = topic.split_whitespace().collect();
            let mut candidates = Vec::with_capacity(pool);

            for slot in 0..pool {
                let doc = DocId(doc_counter);
                doc_counter += 1;

                let is_relevant = slot < relevant;
                let text = if is_relevant {
                    // Seed the document with the query's own terms.
                    let f1 = FILLER[(Self::next(&mut state) as usize) % FILLER.len()];
                    let f2 = FILLER[(Self::next(&mut state) as usize) % FILLER.len()];
                    format!("{} {} {}", topic_terms.join(" "), f1, f2)
                } else {
                    // Irrelevant filler from a different topic.
                    let other = TOPICS[(Self::next(&mut state) as usize) % TOPICS.len()];
                    let f = FILLER[(Self::next(&mut state) as usize) % FILLER.len()];
                    format!("{other} {f}")
                };

                // Noisy first-stage score in [0, 1): decorrelated from relevance.
                let retrieval = (Self::next(&mut state) % 1000) as f32 / 1000.0;
                candidates.push(
                    Candidate::new(doc, text, retrieval).expect("generated text is non-empty"),
                );

                if is_relevant {
                    // Grade 1..=3 so NDCG has graded signal.
                    let grade = 1 + (Self::next(&mut state) % 3) as u8;
                    qrels.insert(qid, Judgment::new(doc, grade));
                }
            }

            out_queries.push(query);
            out_candidates.push((qid, candidates));
        }

        Scenario {
            queries: out_queries,
            candidates: out_candidates,
            qrels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic() {
        let a = ScenarioGenerator::new(42).generate(4, 10, 3);
        let b = ScenarioGenerator::new(42).generate(4, 10, 3);
        assert_eq!(a.queries, b.queries);
        assert_eq!(a.candidates, b.candidates);
    }

    #[test]
    fn shapes_are_correct() {
        let s = ScenarioGenerator::new(1).generate(5, 8, 2);
        assert_eq!(s.queries.len(), 5);
        assert_eq!(s.candidates.len(), 5);
        for (_, cands) in &s.candidates {
            assert_eq!(cands.len(), 8);
        }
    }

    #[test]
    fn relevant_docs_are_judged() {
        let s = ScenarioGenerator::new(7).generate(3, 10, 4);
        for q in &s.queries {
            assert_eq!(s.qrels.relevant_count(q.id), 4);
        }
    }

    #[test]
    fn relevant_docs_contain_query_terms() {
        let s = ScenarioGenerator::new(3).generate(1, 6, 3);
        let query = &s.queries[0];
        let first_term = query.text().split_whitespace().next().unwrap();
        let (_, cands) = &s.candidates[0];
        // At least the relevant candidates should mention the query's terms.
        let mentions = cands.iter().filter(|c| c.text.contains(first_term)).count();
        assert!(mentions >= 3, "expected >=3 mentions, got {mentions}");
    }

    #[test]
    fn relevant_clamped_to_pool() {
        let s = ScenarioGenerator::new(1).generate(1, 3, 99);
        assert_eq!(s.qrels.relevant_count(QueryId(0)), 3);
    }
}
