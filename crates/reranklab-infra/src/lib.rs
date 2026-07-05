//! # reranklab-infra
//!
//! In-memory adapters implementing the `reranklab-core` ports: a candidate
//! store, a relevance-judgment store, a broadcast event bus, and a
//! deterministic synthetic corpus + `qrels` generator used by demos, tests,
//! and benchmarks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bus;
pub mod generator;
pub mod store;

pub use bus::{BroadcastEventSink, DEFAULT_CAPACITY};
pub use generator::{Scenario, ScenarioGenerator};
pub use store::{InMemoryCandidateStore, InMemoryJudgmentStore};
