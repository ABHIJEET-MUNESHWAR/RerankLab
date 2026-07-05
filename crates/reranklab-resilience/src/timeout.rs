//! Timeout wrapper for any async operation.

use std::future::Future;
use std::time::Duration;

use thiserror::Error;

/// Returned when an operation does not complete within its deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("operation timed out after {0:?}")]
pub struct TimeoutError(pub Duration);

/// Runs `fut`, failing with [`TimeoutError`] if it does not complete within `dur`.
///
/// # Errors
/// Returns `Err(TimeoutError)` if the future does not resolve in time.
pub async fn with_timeout<F, T>(dur: Duration, fut: F) -> Result<T, TimeoutError>
where
    F: Future<Output = T>,
{
    match tokio::time::timeout(dur, fut).await {
        Ok(v) => Ok(v),
        Err(_) => Err(TimeoutError(dur)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn completes_before_deadline() {
        let r = with_timeout(Duration::from_secs(1), async { 7 }).await;
        assert_eq!(r, Ok(7));
    }

    #[tokio::test(start_paused = true)]
    async fn times_out() {
        let r = with_timeout(Duration::from_millis(10), async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            1
        })
        .await;
        assert_eq!(r, Err(TimeoutError(Duration::from_millis(10))));
    }
}
