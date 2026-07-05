//! # reranklab-resilience
//!
//! Reusable, framework-agnostic resilience primitives shared across IndexForge
//! crates: a pluggable [`Clock`], [`with_timeout`], bounded [`retry_if`] with
//! backoff, a [`CircuitBreaker`], and a token-bucket [`RateLimiter`].
//!
//! All time-dependent components are generic over [`Clock`] so they can be
//! driven by a [`ManualClock`] in tests — no sleeping, fully deterministic.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod breaker;
pub mod clock;
pub mod rate_limit;
pub mod retry;
pub mod timeout;

pub use breaker::{BreakerConfig, BreakerState, CircuitBreaker};
pub use clock::{Clock, ManualClock, SystemClock};
pub use rate_limit::{system_rate_limiter, RateLimiter, DEFAULT_COOLDOWN};
pub use retry::{retry_if, RetryPolicy};
pub use timeout::{with_timeout, TimeoutError};
