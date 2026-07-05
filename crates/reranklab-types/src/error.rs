//! Domain error type for the pure reranking layer.

use thiserror::Error;

/// Errors that arise from constructing or manipulating domain values.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RerankError {
    /// A query was created with empty text.
    #[error("query text must not be empty")]
    EmptyQuery,

    /// A candidate was created with empty text.
    #[error("candidate text must not be empty")]
    EmptyCandidate,

    /// A cutoff `k` of zero was supplied to a metric.
    #[error("metric cutoff k must be greater than zero")]
    ZeroCutoff,
}

impl RerankError {
    /// A short, stable, machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::EmptyQuery => "empty_query",
            Self::EmptyCandidate => "empty_candidate",
            Self::ZeroCutoff => "zero_cutoff",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable() {
        assert_eq!(RerankError::EmptyQuery.code(), "empty_query");
        assert_eq!(RerankError::EmptyCandidate.code(), "empty_candidate");
        assert_eq!(RerankError::ZeroCutoff.code(), "zero_cutoff");
    }

    #[test]
    fn display_is_human_readable() {
        assert!(RerankError::EmptyQuery.to_string().contains("empty"));
    }
}
