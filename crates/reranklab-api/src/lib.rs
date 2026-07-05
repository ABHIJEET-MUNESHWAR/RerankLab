//! # reranklab-api
//!
//! The GraphQL surface for RerankLab. GraphQL (over REST) is used because the
//! API exposes several related operations — a rerank mutation, an offline
//! evaluation query, statistics, and a live event subscription — behind one
//! typed schema.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod schema;
mod types;

use std::sync::Arc;

use async_graphql::{EmptySubscription, Schema};
use reranklab_core::{CandidateStore, JudgmentStore, RerankEventStream, RerankService};
use reranklab_resilience::SystemClock;

pub use schema::{MutationRoot, QueryRoot, SubscriptionRoot};
pub use types::{
    CandidateInput, EvalMetricsObject, RankedItemObject, RerankResultObject, ScoredCandidateObject,
};

/// Shared context injected into every resolver.
#[derive(Clone)]
pub struct ApiContext {
    /// The reranking orchestrator.
    pub service: Arc<RerankService<SystemClock>>,
    /// Candidate store (first-stage retrieval results).
    pub candidates: Arc<dyn CandidateStore>,
    /// Judgment store (relevance ground truth for evaluation).
    pub judgments: Arc<dyn JudgmentStore>,
    /// The read-side event stream for subscriptions.
    pub events: Arc<dyn RerankEventStream>,
}

impl ApiContext {
    /// Builds a new context from its collaborators.
    #[must_use]
    pub fn new(
        service: Arc<RerankService<SystemClock>>,
        candidates: Arc<dyn CandidateStore>,
        judgments: Arc<dyn JudgmentStore>,
        events: Arc<dyn RerankEventStream>,
    ) -> Self {
        Self {
            service,
            candidates,
            judgments,
            events,
        }
    }
}

/// The concrete schema type.
pub type LabSchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;

/// Builds the GraphQL schema with depth/complexity guards.
#[must_use]
pub fn build_schema(ctx: ApiContext) -> LabSchema {
    Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        .limit_depth(12)
        .limit_complexity(512)
        .data(ctx)
        .finish()
}

/// A schema with an empty subscription, for tests.
#[must_use]
pub fn build_query_schema(ctx: ApiContext) -> Schema<QueryRoot, MutationRoot, EmptySubscription> {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription)
        .limit_depth(12)
        .limit_complexity(512)
        .data(ctx)
        .finish()
}
