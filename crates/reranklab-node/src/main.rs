//! RerankLab binary entry point.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use reranklab_core::{evaluate_query, HeuristicReranker, Reranker};
use reranklab_infra::ScenarioGenerator;
use reranklab_node::config::{Cli, Command, DemoArgs};
use reranklab_node::startup::{build_components, build_schema_for, run_server, seed_scenario};
use reranklab_node::telemetry::{init_tracing, install_metrics, test_metrics_handle};
use reranklab_types::{MetricsAccumulator, RankedList, ScoredCandidate};

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    init_tracing();
    let cli = Cli::parse();

    match cli.command {
        Command::Serve(args) => {
            let metrics = install_metrics()?;
            let components = build_components(args.rate_capacity, args.rate_refill);
            let seeded = seed_scenario(&components, args.seed_queries, args.pool, args.relevant);
            if seeded > 0 {
                tracing::info!(seeded, "seeded synthetic scenario");
            }
            let schema = build_schema_for(&components);
            run_server(&args, schema, metrics).await?;
        }
        Command::Demo(args) => run_demo(args).await?,
        Command::Bench(args) => run_bench(args).await?,
    }
    Ok(())
}

/// The first-stage ranking: candidates ordered by their retrieval score alone.
fn baseline_ranking(candidates: &[reranklab_types::Candidate]) -> RankedList {
    let scored = candidates
        .iter()
        .map(|c| ScoredCandidate::new(c.id, c.retrieval_score, c.retrieval_score))
        .collect();
    RankedList::from_scored(scored)
}

async fn run_demo(args: DemoArgs) -> Result<()> {
    let _ = test_metrics_handle();
    let scenario =
        ScenarioGenerator::new(0xC0FF_EE00).generate(args.queries, args.pool, args.relevant);
    let reranker = HeuristicReranker::default();

    let mut base_acc = MetricsAccumulator::new();
    let mut rr_acc = MetricsAccumulator::new();

    for (qid, candidates) in &scenario.candidates {
        let query = scenario
            .queries
            .iter()
            .find(|q| q.id == *qid)
            .expect("query exists");

        // Baseline: first-stage retrieval order.
        let base = baseline_ranking(candidates);
        base_acc.add(evaluate_query(
            *qid,
            &base.doc_ids(),
            &scenario.qrels,
            args.k,
        )?);

        // Reranked: second-stage order.
        let ranked = reranker
            .rerank(query, candidates)
            .await
            .expect("infallible");
        rr_acc.add(evaluate_query(
            *qid,
            &ranked.doc_ids(),
            &scenario.qrels,
            args.k,
        )?);
    }

    let base = base_acc.mean();
    let rr = rr_acc.mean();

    println!(
        "Reranked {} queries ({} candidates each, {} relevant) at k={}\n",
        args.queries, args.pool, args.relevant, args.k
    );
    print_header();
    print_metric("NDCG", base.ndcg, rr.ndcg);
    print_metric("MRR", base.mrr, rr.mrr);
    print_metric("Recall", base.recall, rr.recall);
    print_metric("Precision", base.precision, rr.precision);
    print_metric("MAP", base.average_precision, rr.average_precision);

    let lift = if base.ndcg > 0.0 {
        100.0 * (rr.ndcg - base.ndcg) / base.ndcg
    } else {
        0.0
    };
    println!("\nNDCG@{} lift from reranking: {:+.1}%", args.k, lift);
    Ok(())
}

fn print_header() {
    println!(
        "  {:<10} {:>12} {:>12} {:>10}",
        "metric", "first-stage", "reranked", "delta"
    );
    println!("  {:-<10} {:->12} {:->12} {:->10}", "", "", "", "");
}

fn print_metric(name: &str, base: f64, reranked: f64) {
    let delta = format!("{:+.3}", reranked - base);
    println!("  {name:<10} {base:>12.4} {reranked:>12.4} {delta:>10}");
}

async fn run_bench(args: DemoArgs) -> Result<()> {
    let scenario =
        ScenarioGenerator::new(0xC0FF_EE00).generate(args.queries, args.pool, args.relevant);
    let reranker: Arc<dyn Reranker> = Arc::new(HeuristicReranker::default());

    let start = Instant::now();
    for (qid, candidates) in &scenario.candidates {
        let query = scenario
            .queries
            .iter()
            .find(|q| q.id == *qid)
            .expect("query exists");
        let _ = reranker
            .rerank(query, candidates)
            .await
            .expect("infallible");
    }
    let elapsed = start.elapsed();
    println!(
        "Reranked {} queries ({} candidates each) in {:.2?} ({:.0} queries/s)",
        args.queries,
        args.pool,
        elapsed,
        args.queries as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    Ok(())
}
