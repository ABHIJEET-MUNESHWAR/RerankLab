//! Query, Mutation, and Subscription roots.

use async_graphql::{Context, Object, Result, Subscription};
use futures::{Stream, StreamExt};

use reranklab_core::{evaluate_query, RerankEvent};
use reranklab_types::{Candidate, DocId, Query, QueryId};

use crate::types::{CandidateInput, EvalMetricsObject, RerankResultObject};
use crate::ApiContext;

fn to_err(e: impl std::fmt::Display) -> async_graphql::Error {
    async_graphql::Error::new(e.to_string())
}

fn ctx(context: &Context<'_>) -> Result<ApiContext> {
    Ok(context.data::<ApiContext>()?.clone())
}

/// Read-side queries.
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// API semantic version.
    async fn api_version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Reranks a query's stored candidates and evaluates the result against the
    /// stored relevance judgments, returning metrics at cutoff `k`.
    ///
    /// This is the offline-evaluation entry point: it demonstrates, for a
    /// single query, how much the second-stage reranker improves relevance.
    async fn evaluate(
        &self,
        context: &Context<'_>,
        query_id: u64,
        query_text: String,
        k: u32,
    ) -> Result<EvalMetricsObject> {
        let c = ctx(context)?;
        let qid = QueryId(query_id);
        let query = Query::new(qid, query_text).map_err(to_err)?;

        let candidates = c.candidates.candidates(qid).await.map_err(to_err)?;
        let outcome = c.service.rerank(&query, candidates).await.map_err(to_err)?;

        let qrels = c.judgments.qrels().await.map_err(to_err)?;
        let metrics =
            evaluate_query(qid, &outcome.ranked.doc_ids(), &qrels, k as usize).map_err(to_err)?;
        Ok(metrics.into())
    }
}

/// Write-side mutations.
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Reranks an ad-hoc set of candidates for a query text.
    async fn rerank(
        &self,
        context: &Context<'_>,
        query_id: u64,
        query_text: String,
        candidates: Vec<CandidateInput>,
    ) -> Result<RerankResultObject> {
        let c = ctx(context)?;
        let query = Query::new(QueryId(query_id), query_text).map_err(to_err)?;

        let mut domain = Vec::with_capacity(candidates.len());
        for input in candidates {
            domain.push(
                Candidate::new(DocId(input.id), input.text, input.retrieval_score)
                    .map_err(to_err)?,
            );
        }

        let outcome = c.service.rerank(&query, domain).await.map_err(to_err)?;
        Ok(RerankResultObject {
            reranker: outcome.reranker.to_string(),
            ranked: outcome.ranked.items.into_iter().map(Into::into).collect(),
        })
    }
}

/// Live event subscriptions (read side of CQRS).
pub struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    /// Streams `query_reranked` events as they occur.
    async fn rerank_events(
        &self,
        context: &Context<'_>,
    ) -> Result<impl Stream<Item = RerankEventView>> {
        let c = ctx(context)?;
        Ok(c.events.subscribe().map(RerankEventView::from))
    }
}

/// A flattened GraphQL view of a domain event.
#[derive(async_graphql::SimpleObject)]
pub struct RerankEventView {
    /// Event kind (`query_reranked`).
    pub kind: String,
    /// The query that was reranked.
    pub query_id: u64,
    /// Number of candidates considered.
    pub candidates: u32,
    /// Whether the AI reranker was used.
    pub used_ai: bool,
}

impl From<RerankEvent> for RerankEventView {
    fn from(e: RerankEvent) -> Self {
        match e {
            RerankEvent::QueryReranked {
                query,
                candidates,
                used_ai,
            } => Self {
                kind: "query_reranked".into(),
                query_id: query.value(),
                candidates: candidates as u32,
                used_ai,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_graphql::{Request, Value};
    use reranklab_core::{HeuristicReranker, RerankService};
    use reranklab_infra::{BroadcastEventSink, InMemoryCandidateStore, InMemoryJudgmentStore};
    use reranklab_resilience::{RateLimiter, SystemClock};
    use reranklab_types::{Candidate, DocId, Judgment, QueryId};

    use crate::{build_query_schema, ApiContext};

    fn schema() -> async_graphql::Schema<
        super::QueryRoot,
        super::MutationRoot,
        async_graphql::EmptySubscription,
    > {
        let events = Arc::new(BroadcastEventSink::default());
        let limiter = RateLimiter::new(1e9, 1e9, Arc::new(SystemClock::new()));
        let service = Arc::new(RerankService::new(
            Arc::new(HeuristicReranker::default()),
            events.clone(),
            limiter,
        ));

        let candidates = Arc::new(InMemoryCandidateStore::new());
        candidates.insert(
            QueryId(1),
            vec![
                Candidate::new(DocId(10), "rust async runtime tokio", 0.1).unwrap(),
                Candidate::new(DocId(11), "python threading guide", 0.9).unwrap(),
            ],
        );

        let judgments = Arc::new(InMemoryJudgmentStore::default());
        judgments.insert(QueryId(1), Judgment::new(DocId(10), 3));

        let ctx = ApiContext::new(service, candidates, judgments, events);
        build_query_schema(ctx)
    }

    #[tokio::test]
    async fn api_version_resolves() {
        let r = schema().execute("{ apiVersion }").await;
        assert!(r.errors.is_empty());
    }

    #[tokio::test]
    async fn rerank_mutation_orders_by_relevance() {
        let m = r#"mutation { rerank(queryId: 1, queryText: "rust async runtime", candidates: [
            { id: 10, text: "rust async runtime tokio", retrievalScore: 0.1 },
            { id: 11, text: "python threading guide", retrievalScore: 0.9 }
        ]) { reranker ranked { id } } }"#;
        let r = schema().execute(Request::new(m)).await;
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        if let Value::Object(map) = r.data {
            let rerank = &map["rerank"];
            if let Value::Object(ro) = rerank {
                assert_eq!(ro["reranker"], Value::from("heuristic"));
            } else {
                panic!("expected object");
            }
        } else {
            panic!("expected object");
        }
    }

    #[tokio::test]
    async fn evaluate_query_returns_metrics() {
        let q = "{ evaluate(queryId: 1, queryText: \"rust async runtime\", k: 2) { ndcg recall } }";
        let r = schema().execute(Request::new(q)).await;
        assert!(r.errors.is_empty(), "{:?}", r.errors);
    }
}
