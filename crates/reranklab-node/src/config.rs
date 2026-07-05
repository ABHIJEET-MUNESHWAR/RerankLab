//! CLI definition and runtime configuration.

use clap::{Args, Parser, Subcommand};

/// RerankLab — second-stage reranking with offline relevance evaluation.
#[derive(Parser, Debug)]
#[command(name = "reranklab", version, about)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run the GraphQL/HTTP server.
    Serve(ServeArgs),
    /// Rerank a synthetic scenario and print before/after evaluation metrics.
    Demo(DemoArgs),
    /// Micro-benchmark reranking throughput.
    Bench(DemoArgs),
}

/// Arguments for `serve`.
#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Bind address, e.g. `0.0.0.0:8080`.
    #[arg(long, env = "RERANKLAB_BIND", default_value = "0.0.0.0:8080")]
    pub bind: String,

    /// Rerank rate-limit token-bucket capacity (queries).
    #[arg(long, env = "RERANKLAB_RATE_CAPACITY", default_value_t = 8192.0)]
    pub rate_capacity: f64,

    /// Rerank rate-limit refill (queries/second).
    #[arg(long, env = "RERANKLAB_RATE_REFILL", default_value_t = 8192.0)]
    pub rate_refill: f64,

    /// Seed a synthetic scenario of this many queries on startup (0 disables).
    #[arg(long, env = "RERANKLAB_SEED", default_value_t = 0)]
    pub seed_queries: usize,

    /// Candidates per seeded query.
    #[arg(long, default_value_t = 50)]
    pub pool: usize,

    /// Relevant documents per seeded query.
    #[arg(long, default_value_t = 8)]
    pub relevant: usize,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8080".to_string(),
            rate_capacity: 8192.0,
            rate_refill: 8192.0,
            seed_queries: 0,
            pool: 50,
            relevant: 8,
        }
    }
}

/// Arguments for `demo` / `bench`.
#[derive(Args, Debug, Clone)]
pub struct DemoArgs {
    /// Number of synthetic queries.
    #[arg(long, default_value_t = 200)]
    pub queries: usize,

    /// Candidates per query.
    #[arg(long, default_value_t = 50)]
    pub pool: usize,

    /// Relevant documents per query.
    #[arg(long, default_value_t = 8)]
    pub relevant: usize,

    /// Evaluation cutoff `k`.
    #[arg(long, default_value_t = 10)]
    pub k: usize,
}

impl Default for DemoArgs {
    fn default() -> Self {
        Self {
            queries: 200,
            pool: 50,
            relevant: 8,
            k: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_verifies() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_serve() {
        let cli = Cli::try_parse_from(["reranklab", "serve", "--bind", "127.0.0.1:9000"]).unwrap();
        match cli.command {
            Command::Serve(a) => assert_eq!(a.bind, "127.0.0.1:9000"),
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn defaults_are_sane() {
        assert!(ServeArgs::default().rate_capacity > 0.0);
        assert!(DemoArgs::default().queries > 0);
        assert!(DemoArgs::default().k > 0);
    }
}
