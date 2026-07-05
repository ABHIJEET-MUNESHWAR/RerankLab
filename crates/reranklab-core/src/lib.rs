//! # reranklab-core
//!
//! The framework-free heart of RerankLab. It owns:
//!
//! - **Text features** ([`features`]) — tokenization and lexical overlap
//!   signals shared by scorers.
//! - **Reranking** ([`rerank`]) — the [`ports::Reranker`] trait and a
//!   deterministic [`HeuristicReranker`] that needs no network.
//! - **Evaluation** ([`eval`]) — NDCG@k, MRR, Recall@k, Precision@k, and
//!   Average Precision computed against a labeled `qrels` set.
//! - **Ports** ([`ports`]) — traits adapters implement (candidate store,
//!   judgment store, reranker, event sink).
//! - **Orchestration** ([`service`]) — the [`RerankService`] that ties a
//!   reranker to resilience, metrics, and events.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod eval;
pub mod event;
pub mod features;
pub mod ports;
pub mod rerank;
pub mod service;

pub use error::{CoreError, PortError};
pub use eval::{evaluate, evaluate_query, RerankEvaluator};
pub use event::RerankEvent;
pub use features::{jaccard_overlap, tokenize, FeatureVector};
pub use ports::{CandidateStore, EventSink, JudgmentStore, RerankEventStream, Reranker};
pub use rerank::{HeuristicReranker, RerankConfig};
pub use service::{RerankOutcome, RerankService};
