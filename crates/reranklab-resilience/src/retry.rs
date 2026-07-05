//! Bounded retry with exponential backoff and deterministic equal jitter.
//!
//! Jitter is derived from the system clock's nanoseconds (no `rand` dependency),
//! keeping the crate dependency-light while still de-correlating retries.

use std::future::Future;
use std::time::{Duration, SystemTime};

/// Policy controlling retry attempts and backoff growth.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Base delay for the first backoff.
    pub base_delay: Duration,
    /// Maximum delay any single backoff may reach.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
        }
    }
}

impl RetryPolicy {
    /// Computes the (jittered) backoff delay for a given zero-based attempt.
    #[must_use]
    pub fn backoff(&self, attempt: u32) -> Duration {
        let exp = self.base_delay.saturating_mul(2u32.saturating_pow(attempt));
        let capped = exp.min(self.max_delay);
        // Equal jitter: half fixed + half random-in-[0, half].
        let half = capped / 2;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let jitter_span = half.as_nanos().max(1) as u64;
        let jitter = Duration::from_nanos(u64::from(nanos) % jitter_span);
        half + jitter
    }
}

/// Runs `op` up to `policy.max_attempts` times, retrying while `retryable`
/// returns `true` for the error. Sleeps `policy.backoff(attempt)` between tries.
///
/// # Errors
/// Returns the last error if all attempts fail or the error is not retryable.
pub async fn retry_if<F, Fut, T, E, R>(policy: RetryPolicy, retryable: R, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    R: Fn(&E) -> bool,
{
    let mut attempt = 0;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                attempt += 1;
                if attempt >= policy.max_attempts || !retryable(&e) {
                    return Err(e);
                }
                tokio::time::sleep(policy.backoff(attempt - 1)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test(start_paused = true)]
    async fn succeeds_after_transient_failures() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let r: Result<u32, &str> = retry_if(
            RetryPolicy::default(),
            |_| true,
            move || {
                let c = c.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err("boom")
                    } else {
                        Ok(n)
                    }
                }
            },
        )
        .await;
        assert_eq!(r, Ok(2));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn stops_on_non_retryable() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let r: Result<u32, &str> = retry_if(
            RetryPolicy::default(),
            |_| false,
            move || {
                let c = c.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err("fatal")
                }
            },
        )
        .await;
        assert_eq!(r, Err("fatal"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn backoff_is_capped() {
        let p = RetryPolicy {
            max_attempts: 10,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(4),
        };
        // High attempt would explode without the cap; must stay <= max_delay.
        assert!(p.backoff(20) <= Duration::from_secs(4));
    }
}
