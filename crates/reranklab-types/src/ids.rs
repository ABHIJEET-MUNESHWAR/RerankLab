//! Strongly-typed identifiers for queries and documents.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A document identifier. Newtype over `u64` so a document id can never be
/// confused with any other numeric quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocId(pub u64);

impl DocId {
    /// Returns the underlying numeric value.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "doc:{}", self.0)
    }
}

impl From<u64> for DocId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// A query identifier. Newtype over `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QueryId(pub u64);

impl QueryId {
    /// Returns the underlying numeric value.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q:{}", self.0)
    }
}

impl From<u64> for QueryId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_prefixed() {
        assert_eq!(DocId(7).to_string(), "doc:7");
        assert_eq!(QueryId(3).to_string(), "q:3");
    }

    #[test]
    fn value_round_trips() {
        assert_eq!(DocId::from(42).value(), 42);
        assert_eq!(QueryId::from(9).value(), 9);
    }

    #[test]
    fn ordering_is_numeric() {
        let mut v = [DocId(3), DocId(1), DocId(2)];
        v.sort();
        assert_eq!(v, [DocId(1), DocId(2), DocId(3)]);
    }

    #[test]
    fn serde_is_transparent() {
        let json = serde_json::to_string(&DocId(5)).unwrap();
        assert_eq!(json, "5");
        let back: DocId = serde_json::from_str("5").unwrap();
        assert_eq!(back, DocId(5));
    }
}
