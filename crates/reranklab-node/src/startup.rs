//! Wiring and HTTP server assembly.

use std::sync::Arc;

use anyhow::Context as _;
use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse, GraphQLSubscription};
use axum::{
    extract::State,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use metrics_exporter_prometheus::PrometheusHandle;

use reranklab_api::{build_schema, ApiContext, LabSchema};
use reranklab_core::{HeuristicReranker, RerankService};
use reranklab_infra::{
    BroadcastEventSink, InMemoryCandidateStore, InMemoryJudgmentStore, ScenarioGenerator,
};
use reranklab_resilience::{system_rate_limiter, SystemClock};

use crate::config::ServeArgs;

/// Shared HTTP state.
#[derive(Clone)]
pub struct AppState {
    /// The GraphQL schema.
    pub schema: LabSchema,
    /// Prometheus scrape handle.
    pub metrics: PrometheusHandle,
}

/// The wired application components.
pub struct Components {
    /// The reranking service.
    pub service: Arc<RerankService<SystemClock>>,
    /// Candidate store.
    pub candidates: Arc<InMemoryCandidateStore>,
    /// Judgment store.
    pub judgments: Arc<InMemoryJudgmentStore>,
    /// Event bus.
    pub events: Arc<BroadcastEventSink>,
}

/// Builds the reranking service, stores, and event bus.
///
/// Uses the deterministic [`HeuristicReranker`] by default; a real deployment
/// would wrap an [`reranklab_ai::AiReranker`] here.
#[must_use]
pub fn build_components(rate_capacity: f64, rate_refill: f64) -> Components {
    let events = Arc::new(BroadcastEventSink::default());
    let limiter = system_rate_limiter(rate_capacity, rate_refill);
    let service = Arc::new(RerankService::new(
        Arc::new(HeuristicReranker::default()),
        events.clone(),
        limiter,
    ));
    Components {
        service,
        candidates: Arc::new(InMemoryCandidateStore::new()),
        judgments: Arc::new(InMemoryJudgmentStore::default()),
        events,
    }
}

/// Seeds a synthetic scenario (candidates + judgments) into the stores.
///
/// Returns the number of queries seeded.
pub fn seed_scenario(
    components: &Components,
    queries: usize,
    pool: usize,
    relevant: usize,
) -> usize {
    if queries == 0 {
        return 0;
    }
    let scenario = ScenarioGenerator::new(0xC0FF_EE00).generate(queries, pool, relevant);
    for (qid, cands) in scenario.candidates {
        components.candidates.insert(qid, cands);
    }
    for (qid, doc, rel) in scenario.qrels.iter() {
        components
            .judgments
            .insert(qid, reranklab_types::Judgment::new(doc, rel));
    }
    scenario.queries.len()
}

/// Builds the API context and schema.
#[must_use]
pub fn build_schema_for(components: &Components) -> LabSchema {
    build_schema(ApiContext::new(
        components.service.clone(),
        components.candidates.clone(),
        components.judgments.clone(),
        components.events.clone(),
    ))
}

/// Assembles the axum router.
pub fn build_app(schema: LabSchema, metrics: PrometheusHandle) -> Router {
    let state = AppState { schema, metrics };
    Router::new()
        .route("/graphql", get(playground).post(graphql_handler))
        .route_service(
            "/graphql/ws",
            GraphQLSubscription::new(state.schema.clone()),
        )
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

async fn graphql_handler(State(state): State<AppState>, req: GraphQLRequest) -> GraphQLResponse {
    state.schema.execute(req.into_inner()).await.into()
}

async fn playground() -> impl IntoResponse {
    Html(playground_source(
        GraphQLPlaygroundConfig::new("/graphql").subscription_endpoint("/graphql/ws"),
    ))
}

async fn live() -> impl IntoResponse {
    "OK"
}

async fn ready() -> impl IntoResponse {
    "READY"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics.render()
}

/// Runs the HTTP server until a shutdown signal is received.
///
/// # Errors
/// Returns an error if binding or serving fails.
pub async fn run_server(
    args: &ServeArgs,
    schema: LabSchema,
    metrics: PrometheusHandle,
) -> anyhow::Result<()> {
    let app = build_app(schema, metrics);
    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    tracing::info!(bind = %args.bind, "reranklab listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::test_metrics_handle;
    use reranklab_core::JudgmentStore;

    #[test]
    fn app_builds() {
        let components = build_components(10.0, 10.0);
        let schema = build_schema_for(&components);
        let _app = build_app(schema, test_metrics_handle());
    }

    #[tokio::test]
    async fn seeding_populates_candidates() {
        let components = build_components(1e9, 1e9);
        let n = seed_scenario(&components, 5, 20, 4);
        assert_eq!(n, 5);
        assert_eq!(components.candidates.len(), 5);
        let qrels = components.judgments.qrels().await.unwrap();
        assert_eq!(qrels.relevant_count(reranklab_types::QueryId(0)), 4);
    }
}
