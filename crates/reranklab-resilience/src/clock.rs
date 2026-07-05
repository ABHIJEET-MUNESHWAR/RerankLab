//! A pluggable monotonic clock so time-dependent resilience logic (breakers,
//! rate limiters) is deterministically testable without sleeping.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

/// A source of monotonic elapsed time.
pub trait Clock: Send + Sync {
    /// Returns elapsed time since an arbitrary fixed epoch.
    fn now(&self) -> Duration;
}

/// A real clock backed by [`std::time::Instant`].
#[derive(Debug, Clone)]
pub struct SystemClock {
    epoch: std::time::Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            epoch: std::time::Instant::now(),
        }
    }
}

impl SystemClock {
    /// Creates a new system clock.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.epoch.elapsed()
    }
}

/// A manually-advanced clock for deterministic tests.
#[derive(Debug, Clone, Default)]
pub struct ManualClock {
    offset: Arc<Mutex<Duration>>,
}

impl ManualClock {
    /// Creates a manual clock at time zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances the clock by `d`.
    pub fn advance(&self, d: Duration) {
        *self.offset.lock() += d;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        *self.offset.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_advances() {
        let c = ManualClock::new();
        assert_eq!(c.now(), Duration::ZERO);
        c.advance(Duration::from_secs(5));
        assert_eq!(c.now(), Duration::from_secs(5));
    }

    #[test]
    fn system_clock_is_monotonic() {
        let c = SystemClock::new();
        let a = c.now();
        let b = c.now();
        assert!(b >= a);
    }
}
