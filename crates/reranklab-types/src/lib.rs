//! # reranklab-types
//!
//! The pure, framework-free domain vocabulary of RerankLab: identifiers,
//! queries, retrieval candidates, scored documents, relevance labels
//! (`qrels`), and offline evaluation metrics. No I/O, no async — just the
//! nouns of second-stage reranking and relevance evaluation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod candidate;
pub mod error;
pub mod ids;
pub mod metrics;
pub mod qrel;
pub mod query;

pub use candidate::{Candidate, RankedList, ScoredCandidate};
pub use error::RerankError;
pub use ids::{DocId, QueryId};
pub use metrics::{EvalMetrics, MetricsAccumulator};
pub use qrel::{Judgment, Qrels};
pub use query::Query;
