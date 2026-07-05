//! Criterion micro-benchmarks for the reranking path.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use reranklab_core::{HeuristicReranker, Reranker};
use reranklab_infra::ScenarioGenerator;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

fn bench_rerank(c: &mut Criterion) {
    let rt = runtime();
    let reranker: Arc<dyn Reranker> = Arc::new(HeuristicReranker::default());
    let mut group = c.benchmark_group("rerank");

    for &pool in &[20usize, 100] {
        let scenario = ScenarioGenerator::new(0xC0FF_EE00).generate(1, pool, pool / 4);
        let (qid, candidates) = scenario.candidates[0].clone();
        let query = scenario
            .queries
            .iter()
            .find(|q| q.id == qid)
            .expect("query")
            .clone();

        group.throughput(Throughput::Elements(pool as u64));
        group.bench_with_input(BenchmarkId::from_parameter(pool), &pool, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    reranker.rerank(&query, &candidates).await.unwrap();
                });
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rerank);
criterion_main!(benches);
