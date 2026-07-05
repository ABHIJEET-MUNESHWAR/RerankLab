//! # reranklab-node
//!
//! The composition root: it wires the reranking service, stores, and event bus,
//! exposes them over a GraphQL/HTTP server, and provides a CLI with `serve`,
//! `demo`, and `bench` subcommands.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod startup;
pub mod telemetry;
