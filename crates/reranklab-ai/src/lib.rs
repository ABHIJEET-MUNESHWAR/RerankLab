//! # reranklab-ai
//!
//! The generative-AI reranking layer. It provides an [`AiReranker`] that scores
//! candidates with a large language model over HTTP, wrapped in resilience
//! (timeout + bounded retry with backoff), and — crucially — **degrades
//! gracefully** to the deterministic [`reranklab_core::HeuristicReranker`]
//! whenever the model is unavailable, times out, or returns an unusable
//! response. This keeps the system correct and testable with no network.
//!
//! The HTTP boundary is expressed as a [`ChatModel`] port so tests can inject a
//! stub model and exercise both the success and fallback paths deterministically.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod client;
pub mod error;
pub mod reranker;

pub use client::{ChatModel, HttpChatModel, ModelConfig};
pub use error::AiError;
pub use reranker::AiReranker;
