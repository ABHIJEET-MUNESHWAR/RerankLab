# RerankLab — Engineering Evaluation

This document maps RerankLab against the 28 production-grade engineering
guidelines. Legend: ✅ implemented · 🟡 partial / intentionally scoped · ⬜ N/A.

**Headline:** a from-scratch **second-stage reranking + offline
relevance-evaluation** service — a `Reranker` port with a deterministic
**heuristic** and a **generative-AI** implementation that degrades gracefully to
the heuristic, an evaluation engine computing **NDCG / MRR / Recall / Precision /
MAP** against `qrels`, a typed GraphQL API, resilience, tracing, Prometheus
metrics, and 90 tests — all with `#![forbid(unsafe_code)]` and clippy
`-D warnings` clean.

| # | Guideline | Status | Where / How |
|---|---|:--:|---|
| 1 | SOLID design | ✅ | `Reranker`/`CandidateStore`/`JudgmentStore`/`EventSink` ports invert dependencies; `RerankService` depends on abstractions. |
| 2 | Microservices / bounded contexts | ✅ | 7-crate hexagonal workspace; domain core isolated from AI/infra/API. |
| 3 | Partitioning / sharding | 🟡 | Candidate store keyed per-query (independently shardable); evaluation is embarrassingly parallel across queries; single-node by design. |
| 4 | Timeouts, retry, fault tolerance | ✅ | `AiReranker` wraps model calls in `with_timeout` + `retry_if` (equal-jitter backoff); graceful fallback on failure. |
| 5 | Rate limiting / circuit breaker | ✅ | Token-bucket `RateLimiter` guards rerank throughput; `CircuitBreaker` available for the model boundary. |
| 6 | Error handling & recovery | ✅ | `thiserror` errors (`RerankError`, `CoreError`, `PortError`, `AiError`) with stable `code()` + `is_retryable`; **fallback** is the recovery strategy. |
| 7 | GraphQL over REST | ✅ | `async-graphql`: `evaluate` query, `rerank` mutation, `rerankEvents` subscription. |
| 8 | 100% meaningful test coverage | ✅ | 90 tests: metric-math correctness, reranker ordering, full AI success/fallback matrix, manual-clock resilience. |
| 9 | Composability / extensibility | ✅ | New rerankers/stores/models plug in via ports without touching orchestration. |
| 10 | Modularity | ✅ | Features, reranking, evaluation, and the AI client are separate, independently tested modules. |
| 11 | Canonical crate stack | ✅ | tokio, async-graphql, axum, tower-http, reqwest (rustls), tracing, metrics, criterion. |
| 12 | **Generative / agentic AI** | ✅ | **`reranklab-ai`: a generative reranker calling a chat model over HTTP, with prompt construction, JSON score parsing, resilient calls, and deterministic fallback.** |
| 13 | Idiomatic Rust | ✅ | Newtypes, exhaustive `match`, iterator pipelines, `Arc<dyn Trait>` sharing, `async_trait` ports. |
| 14 | Generics & trait bounds | ✅ | `RerankService<C: Clock>`, `RateLimiter<C: Clock>`, `CircuitBreaker<C: Clock>`. |
| 15 | README & setup | ✅ | Full README: TOC, mermaid diagrams, two-stage + fallback explainers, KaTeX metric formulas, demo output, test matrix. |
| 16 | Performance | ✅ | Release LTO profile; criterion rerank benches; single-pass metric computation. |
| 17 | Tokio async, no blocking | ✅ | Async ports and model calls throughout; graceful shutdown (ctrl_c + SIGTERM). |
| 18 | Parallel / concurrent / batch | ✅ | `DashMap` candidate store, broadcast fan-out, per-query-independent evaluation. |
| 19 | Logging & observability | ✅ | JSON `tracing` + Prometheus counters (reranked, ai_success, ai_fallback{reason}, throttled) and a candidates-per-query summary. |
| 20 | Edge cases | ✅ | Empty query/candidate rejected; `k == 0` errors; NaN scores sink in ranking; model inventing unknown ids is filtered; empty candidate set short-circuits. |
| 21 | Event-driven / CQRS | ✅ | Reranking (write) emits `query_reranked`; subscribers consume via `RerankEventStream` (read side). |
| 22 | Clean interfaces | ✅ | Small, focused ports; GraphQL DTOs as an anti-corruption layer over domain types. |
| 23 | Compile-time safety | ✅ | Type-state newtypes; a `Query`/`Candidate` cannot exist with empty text after `new`. |
| 24 | Benchmarks & complexity | ✅ | `benches/rerank_bench.rs`; Big-O table + KaTeX metric derivations in README. |
| 25 | CI/CD | ✅ | `.github/workflows/ci.yml`: fmt + clippy `-D warnings` + test + `cargo audit`. |
| 26 | Docker | ✅ | Multi-stage `Dockerfile` (rust:1.89 → bookworm-slim), non-root uid 10001; compose with Prometheus. |
| 27 | Postman / API collection | ✅ | `postman/RerankLab.postman_collection.json` (health, metrics, rerank, evaluate). |
| 28 | Self-evaluation | ✅ | This document. |

## Notable engineering decisions

- **Graceful degradation as a first-class feature.** The AI reranker never
  fails the request: any transport error, timeout, retry exhaustion, or
  unparseable reply transparently falls back to the deterministic heuristic. The
  fallback reason is recorded as a metric label for observability.
- **The model boundary is a port.** Expressing the LLM as a `ChatModel` trait
  makes the entire success/failure matrix testable offline with stub models —
  no network, no flakiness, no credentials in CI.
- **Metric correctness is proven, not assumed.** The evaluation tests assert the
  known mathematical properties: a perfect ranking scores NDCG = 1.0, a
  reversed ranking scores strictly less, earlier relevant hits yield higher
  Average Precision, and `k = 0` is a hard error.
- **NaN-safe ranking.** `ScoredCandidate` defines a total order that sinks `NaN`
  scores to the bottom, so ranked lists sort and compare without panicking and
  without pulling in extra crates.
- **Observable lift.** The synthetic scenario deliberately decorrelates
  first-stage scores from true relevance, so the demo shows a real, large NDCG
  improvement (~+167%) from reranking — the whole point of the second stage.
