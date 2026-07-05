//! Error type for the AI reranking layer.

use thiserror::Error;

/// Failures that can occur when calling a generative model.
#[derive(Debug, Error)]
pub enum AiError {
    /// The HTTP transport failed (connection, DNS, TLS, ...).
    #[error("model transport error: {0}")]
    Transport(String),

    /// The model did not respond within the deadline.
    #[error("model request timed out")]
    Timeout,

    /// The model returned a non-success HTTP status.
    #[error("model returned status {0}")]
    Status(u16),

    /// The model's response could not be parsed into scores.
    #[error("model returned an unparseable response: {0}")]
    Parse(String),
}

impl AiError {
    /// Whether the failure is transient and worth retrying.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Timeout | Self::Transport(_) => true,
            // 429 and 5xx are retryable; other statuses are not.
            Self::Status(code) => *code == 429 || *code >= 500,
            Self::Parse(_) => false,
        }
    }

    /// A short, stable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Transport(_) => "transport",
            Self::Timeout => "timeout",
            Self::Status(_) => "status",
            Self::Parse(_) => "parse",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryability_rules() {
        assert!(AiError::Timeout.is_retryable());
        assert!(AiError::Transport("x".into()).is_retryable());
        assert!(AiError::Status(503).is_retryable());
        assert!(AiError::Status(429).is_retryable());
        assert!(!AiError::Status(400).is_retryable());
        assert!(!AiError::Parse("x".into()).is_retryable());
    }

    #[test]
    fn codes_are_stable() {
        assert_eq!(AiError::Timeout.code(), "timeout");
        assert_eq!(AiError::Status(500).code(), "status");
    }
}
