//! Error types for the core reranking and evaluation layer.

use thiserror::Error;

use reranklab_types::RerankError;

/// Errors returned by ports (adapters) — storage, rerankers, event sinks.
#[derive(Debug, Error)]
pub enum PortError {
    /// The backing resource is temporarily unavailable.
    #[error("port unavailable: {0}")]
    Unavailable(String),

    /// The operation exceeded its deadline.
    #[error("port timed out")]
    Timeout,

    /// An unexpected internal failure.
    #[error("port internal error: {0}")]
    Internal(String),
}

impl PortError {
    /// Whether retrying the operation could plausibly succeed.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Unavailable(_) | Self::Timeout)
    }

    /// A short, stable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Unavailable(_) => "unavailable",
            Self::Timeout => "timeout",
            Self::Internal(_) => "internal",
        }
    }
}

/// The top-level error for core operations.
#[derive(Debug, Error)]
pub enum CoreError {
    /// A domain value was invalid (empty query, bad cutoff, ...).
    #[error(transparent)]
    Invalid(#[from] RerankError),

    /// A downstream port failed.
    #[error(transparent)]
    Port(#[from] PortError),
}

impl CoreError {
    /// A short, stable error code spanning both variants.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Invalid(e) => e.code(),
            Self::Port(e) => e.code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_error_retryability() {
        assert!(PortError::Timeout.is_retryable());
        assert!(PortError::Unavailable("x".into()).is_retryable());
        assert!(!PortError::Internal("x".into()).is_retryable());
    }

    #[test]
    fn core_error_codes() {
        assert_eq!(
            CoreError::from(RerankError::ZeroCutoff).code(),
            "zero_cutoff"
        );
        assert_eq!(CoreError::from(PortError::Timeout).code(), "timeout");
    }
}
