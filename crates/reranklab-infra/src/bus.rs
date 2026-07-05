//! A broadcast event bus implementing both the write-side [`EventSink`] and the
//! read-side [`RerankEventStream`] ports.

use async_trait::async_trait;
use futures::StreamExt;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use reranklab_core::ports::EventStream;
use reranklab_core::{EventSink, PortError, RerankEvent, RerankEventStream};

/// Default broadcast channel capacity.
pub const DEFAULT_CAPACITY: usize = 1024;

/// A `tokio::sync::broadcast`-backed event bus.
#[derive(Debug, Clone)]
pub struct BroadcastEventSink {
    sender: broadcast::Sender<RerankEvent>,
}

impl Default for BroadcastEventSink {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl BroadcastEventSink {
    /// Creates a bus with the given channel capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self { sender }
    }

    /// Current number of active subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

#[async_trait]
impl EventSink for BroadcastEventSink {
    async fn publish(&self, event: RerankEvent) -> Result<(), PortError> {
        // A send error only means there are no subscribers — not a failure.
        let _ = self.sender.send(event);
        Ok(())
    }
}

impl RerankEventStream for BroadcastEventSink {
    fn subscribe(&self) -> EventStream {
        let rx = self.sender.subscribe();
        BroadcastStream::new(rx)
            .filter_map(|r| async move { r.ok() })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reranklab_types::QueryId;

    #[tokio::test]
    async fn delivers_events_to_subscribers() {
        let bus = BroadcastEventSink::default();
        let mut stream = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);

        bus.publish(RerankEvent::QueryReranked {
            query: QueryId(1),
            candidates: 3,
            used_ai: true,
        })
        .await
        .unwrap();

        let event = stream.next().await.unwrap();
        assert_eq!(event.kind(), "query_reranked");
    }

    #[tokio::test]
    async fn publish_without_subscribers_is_ok() {
        let bus = BroadcastEventSink::default();
        assert!(bus
            .publish(RerankEvent::QueryReranked {
                query: QueryId(1),
                candidates: 0,
                used_ai: false,
            })
            .await
            .is_ok());
    }
}
