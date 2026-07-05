//! Telemetry: structured JSON tracing and a Prometheus metrics recorder.

use anyhow::Context;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::Once;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

static TRACING: Once = Once::new();

/// Initializes JSON structured logging once. Idempotent.
pub fn init_tracing() {
    TRACING.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,reranklab=debug"));
        let fmt_layer = fmt::layer().json().with_target(true);
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .try_init();
    });
}

/// Installs the global Prometheus recorder and returns a scrape handle.
///
/// # Errors
/// Returns an error if a global recorder was already installed.
pub fn install_metrics() -> anyhow::Result<PrometheusHandle> {
    PrometheusBuilder::new()
        .install_recorder()
        .context("installing Prometheus recorder")
}

/// Builds a recorder handle without touching global state — used by tests.
#[must_use]
pub fn test_metrics_handle() -> PrometheusHandle {
    PrometheusBuilder::new().build_recorder().handle()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_init_is_idempotent() {
        init_tracing();
        init_tracing();
    }

    #[test]
    fn test_handle_renders() {
        let handle = test_metrics_handle();
        let _ = handle.render();
    }
}
