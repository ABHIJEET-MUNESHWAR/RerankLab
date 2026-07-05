//! A three-state circuit breaker (Closed / Open / HalfOpen) generic over a
//! [`Clock`] so state transitions are testable without real time.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::clock::{Clock, SystemClock};

/// Breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Calls flow through; failures are counted.
    Closed,
    /// Calls are rejected until the cooldown elapses.
    Open,
    /// A single trial call is allowed to probe recovery.
    HalfOpen,
}

impl BreakerState {
    /// Stable name for metrics/labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

/// Configuration for a [`CircuitBreaker`].
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Consecutive failures that trip the breaker open.
    pub failure_threshold: u32,
    /// Cooldown before a tripped breaker allows a probe.
    pub cooldown: Duration,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            cooldown: Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
struct Inner {
    state: BreakerState,
    consecutive_failures: u32,
    opened_at: Duration,
}

/// A circuit breaker guarding a fallible dependency.
#[derive(Clone)]
pub struct CircuitBreaker<C: Clock = SystemClock> {
    config: BreakerConfig,
    clock: Arc<C>,
    inner: Arc<Mutex<Inner>>,
}

impl<C: Clock> CircuitBreaker<C> {
    /// Creates a breaker with the given config and clock.
    pub fn new(config: BreakerConfig, clock: Arc<C>) -> Self {
        Self {
            config,
            clock,
            inner: Arc::new(Mutex::new(Inner {
                state: BreakerState::Closed,
                consecutive_failures: 0,
                opened_at: Duration::ZERO,
            })),
        }
    }

    /// Returns the current logical state, transitioning `Open -> HalfOpen` if
    /// the cooldown has elapsed.
    pub fn state(&self) -> BreakerState {
        let mut inner = self.inner.lock();
        if inner.state == BreakerState::Open
            && self.clock.now().saturating_sub(inner.opened_at) >= self.config.cooldown
        {
            inner.state = BreakerState::HalfOpen;
        }
        inner.state
    }

    /// Returns `true` if a call should be allowed through right now.
    pub fn allow(&self) -> bool {
        !matches!(self.state(), BreakerState::Open)
    }

    /// Records a successful call, closing the breaker.
    pub fn on_success(&self) {
        let mut inner = self.inner.lock();
        inner.consecutive_failures = 0;
        inner.state = BreakerState::Closed;
    }

    /// Records a failed call, tripping the breaker if the threshold is reached.
    pub fn on_failure(&self) {
        let mut inner = self.inner.lock();
        inner.consecutive_failures += 1;
        if inner.consecutive_failures >= self.config.failure_threshold {
            inner.state = BreakerState::Open;
            inner.opened_at = self.clock.now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;

    fn breaker() -> (CircuitBreaker<ManualClock>, ManualClock) {
        let clock = ManualClock::new();
        let b = CircuitBreaker::new(
            BreakerConfig {
                failure_threshold: 3,
                cooldown: Duration::from_secs(10),
            },
            Arc::new(clock.clone()),
        );
        (b, clock)
    }

    #[test]
    fn trips_after_threshold() {
        let (b, _) = breaker();
        assert!(b.allow());
        b.on_failure();
        b.on_failure();
        assert_eq!(b.state(), BreakerState::Closed);
        b.on_failure();
        assert_eq!(b.state(), BreakerState::Open);
        assert!(!b.allow());
    }

    #[test]
    fn recovers_via_half_open() {
        let (b, clock) = breaker();
        for _ in 0..3 {
            b.on_failure();
        }
        assert_eq!(b.state(), BreakerState::Open);
        clock.advance(Duration::from_secs(10));
        assert_eq!(b.state(), BreakerState::HalfOpen);
        assert!(b.allow());
        b.on_success();
        assert_eq!(b.state(), BreakerState::Closed);
    }

    #[test]
    fn state_names() {
        assert_eq!(BreakerState::Closed.name(), "closed");
        assert_eq!(BreakerState::Open.name(), "open");
        assert_eq!(BreakerState::HalfOpen.name(), "half_open");
    }
}
