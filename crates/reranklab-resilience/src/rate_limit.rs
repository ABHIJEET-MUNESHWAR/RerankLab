//! A token-bucket rate limiter generic over a [`Clock`].

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::clock::{Clock, SystemClock};

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    last_refill: Duration,
}

/// A token-bucket limiter. `capacity` tokens accrue at `refill_per_sec`.
#[derive(Clone)]
pub struct RateLimiter<C: Clock = SystemClock> {
    capacity: f64,
    refill_per_sec: f64,
    clock: Arc<C>,
    bucket: Arc<Mutex<Bucket>>,
}

impl<C: Clock> RateLimiter<C> {
    /// Creates a limiter starting full.
    pub fn new(capacity: f64, refill_per_sec: f64, clock: Arc<C>) -> Self {
        let now = clock.now();
        Self {
            capacity,
            refill_per_sec,
            clock,
            bucket: Arc::new(Mutex::new(Bucket {
                tokens: capacity,
                last_refill: now,
            })),
        }
    }

    /// Attempts to consume a single token. Returns `true` if allowed.
    pub fn try_acquire(&self) -> bool {
        self.try_acquire_n(1.0)
    }

    /// Attempts to consume `n` tokens. Returns `true` if allowed.
    pub fn try_acquire_n(&self, n: f64) -> bool {
        let mut b = self.bucket.lock();
        let now = self.clock.now();
        let elapsed = now.saturating_sub(b.last_refill).as_secs_f64();
        b.tokens = (b.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        b.last_refill = now;
        if b.tokens >= n {
            b.tokens -= n;
            true
        } else {
            false
        }
    }
}

/// Convenience alias for building a limiter over the [`SystemClock`].
#[must_use]
pub fn system_rate_limiter(capacity: f64, refill_per_sec: f64) -> RateLimiter<SystemClock> {
    RateLimiter::new(capacity, refill_per_sec, Arc::new(SystemClock::new()))
}

/// A default cooldown span used by callers that want a simple pause helper.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;

    #[test]
    fn enforces_capacity_then_refills() {
        let clock = ManualClock::new();
        let rl = RateLimiter::new(2.0, 1.0, Arc::new(clock.clone()));
        assert!(rl.try_acquire());
        assert!(rl.try_acquire());
        assert!(!rl.try_acquire());
        clock.advance(Duration::from_secs(1));
        assert!(rl.try_acquire());
        assert!(!rl.try_acquire());
    }

    #[test]
    fn burst_acquire_n() {
        let clock = ManualClock::new();
        let rl = RateLimiter::new(10.0, 1.0, Arc::new(clock));
        assert!(rl.try_acquire_n(10.0));
        assert!(!rl.try_acquire_n(1.0));
    }

    #[test]
    fn system_helper_starts_full() {
        let rl = system_rate_limiter(1.0, 1.0);
        assert!(rl.try_acquire());
    }
}
