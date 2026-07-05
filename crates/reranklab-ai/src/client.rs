//! The generative-model HTTP boundary.
//!
//! [`ChatModel`] is the port: given a prompt, return the model's raw text
//! completion. Expressing it as a trait lets the [`crate::reranker::AiReranker`]
//! be tested with a deterministic stub instead of a live network call.
//! [`HttpChatModel`] is the production adapter, a thin `reqwest` client speaking
//! an OpenAI-style chat-completions JSON contract.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AiError;

/// A port over a chat/completions model: prompt in, completion text out.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Sends `prompt` to the model and returns its text completion.
    async fn complete(&self, prompt: &str) -> Result<String, AiError>;
}

/// Configuration for [`HttpChatModel`].
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Full URL of the chat-completions endpoint.
    pub endpoint: String,
    /// Bearer API key (sent as `Authorization: Bearer ...`).
    pub api_key: String,
    /// Model identifier (e.g. `gpt-4o-mini`).
    pub model: String,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            timeout: Duration::from_secs(10),
        }
    }
}

// --- Wire types for the OpenAI-style chat contract ---

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

/// A `reqwest`-backed [`ChatModel`] adapter.
pub struct HttpChatModel {
    client: reqwest::Client,
    config: ModelConfig,
}

impl HttpChatModel {
    /// Builds an HTTP model client from configuration.
    ///
    /// # Errors
    /// Returns [`AiError::Transport`] if the underlying client cannot be built.
    pub fn new(config: ModelConfig) -> Result<Self, AiError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AiError::Transport(e.to_string()))?;
        Ok(Self { client, config })
    }
}

#[async_trait]
impl ChatModel for HttpChatModel {
    async fn complete(&self, prompt: &str) -> Result<String, AiError> {
        let body = ChatRequest {
            model: &self.config.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            temperature: 0.0,
        };

        let resp = self
            .client
            .post(&self.config.endpoint)
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AiError::Timeout
                } else {
                    AiError::Transport(e.to_string())
                }
            })?;

        if !resp.status().is_success() {
            return Err(AiError::Status(resp.status().as_u16()));
        }

        let parsed: ChatResponse = resp
            .json()
            .await
            .map_err(|e| AiError::Parse(e.to_string()))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| AiError::Parse("no choices in response".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let c = ModelConfig::default();
        assert!(c.endpoint.starts_with("https://"));
        assert!(c.timeout > Duration::ZERO);
    }

    #[test]
    fn http_model_builds() {
        assert!(HttpChatModel::new(ModelConfig::default()).is_ok());
    }
}
