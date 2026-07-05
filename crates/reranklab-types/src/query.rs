//! The query a reranker scores candidates against.

use serde::{Deserialize, Serialize};

use crate::error::RerankError;
use crate::ids::QueryId;

/// A search query: an identifier plus its natural-language text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Query {
    /// Stable query identifier.
    pub id: QueryId,
    /// The natural-language query text.
    pub text: String,
}

impl Query {
    /// Creates a query, rejecting empty (or whitespace-only) text.
    ///
    /// # Errors
    /// Returns [`RerankError::EmptyQuery`] if `text` is blank.
    pub fn new(id: QueryId, text: impl Into<String>) -> Result<Self, RerankError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(RerankError::EmptyQuery);
        }
        Ok(Self { id, text })
    }

    /// Returns the query text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_blank_text() {
        assert_eq!(
            Query::new(QueryId(1), "   ").unwrap_err(),
            RerankError::EmptyQuery
        );
    }

    #[test]
    fn accepts_valid_query() {
        let q = Query::new(QueryId(1), "rust async runtime").unwrap();
        assert_eq!(q.text(), "rust async runtime");
        assert_eq!(q.id, QueryId(1));
    }
}
