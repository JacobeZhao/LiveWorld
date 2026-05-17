// Unified async LLM adapter trait.
// Implementors: MockLlm (testing), OpenAiClient, AnthropicClient, OllamaClient.
// All calls are async and non-blocking. The decision loop spawns these on a
// separate task pool so tick thread is never blocked.

use crate::types::LlmModel;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// A single LLM request.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub model: LlmModel,
    pub system_prompt: String,
    pub user_prompt: String,
    pub max_tokens: u32,
}

/// A single LLM response.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub model: LlmModel,
    pub tokens_used: u32,
}

/// Unified async interface for all LLM backends.
#[async_trait]
pub trait LlmAdapter: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse>;
    fn model(&self) -> &LlmModel;
}

// ── Mock implementation ───────────────────────────────────────────────────────

/// Deterministic mock for testing: returns a canned response instantly.
pub struct MockLlm {
    model: LlmModel,
    response_template: String,
    /// Optional artificial delay to simulate API latency.
    delay: Option<Duration>,
}

impl MockLlm {
    pub fn new() -> Self {
        Self {
            model: LlmModel::Mock,
            response_template: "I observe the world around me and decide to act.".to_string(),
            delay: None,
        }
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    pub fn with_response(mut self, response: impl Into<String>) -> Self {
        self.response_template = response.into();
        self
    }
}

impl Default for MockLlm {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmAdapter for MockLlm {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        if let Some(d) = self.delay {
            tokio::time::sleep(d).await;
        }
        Ok(LlmResponse {
            text: format!("[{}] {}", req.model, self.response_template),
            model: self.model.clone(),
            tokens_used: 20,
        })
    }

    fn model(&self) -> &LlmModel {
        &self.model
    }
}

// ── OpenAI client ─────────────────────────────────────────────────────────────

pub struct OpenAiClient {
    model: LlmModel,
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            model: LlmModel::Gpt4o,
            api_key: api_key.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }
}

#[async_trait]
impl LlmAdapter for OpenAiClient {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        use serde_json::{json, Value};

        let body = json!({
            "model": req.model.to_string(),
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user",   "content": req.user_prompt}
            ],
            "max_tokens": req.max_tokens
        });

        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        let text = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let tokens = resp["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LlmResponse {
            text,
            model: req.model,
            tokens_used: tokens,
        })
    }

    fn model(&self) -> &LlmModel {
        &self.model
    }
}

// ── Anthropic client ──────────────────────────────────────────────────────────

pub struct AnthropicClient {
    model: LlmModel,
    api_key: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>, model: LlmModel) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }
}

#[async_trait]
impl LlmAdapter for AnthropicClient {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        use serde_json::{json, Value};

        let body = json!({
            "model": self.model.to_string(),
            "max_tokens": req.max_tokens,
            "system": req.system_prompt,
            "messages": [
                {"role": "user", "content": req.user_prompt}
            ]
        });

        let resp = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        let text = resp["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let tokens = resp["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(LlmResponse {
            text,
            model: self.model.clone(),
            tokens_used: tokens,
        })
    }

    fn model(&self) -> &LlmModel {
        &self.model
    }
}

// ── Ollama client (local) ─────────────────────────────────────────────────────

pub struct OllamaClient {
    model: LlmModel,
    base_url: String,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(model_name: impl Into<String>, base_url: impl Into<String>) -> Self {
        let model_name = model_name.into();
        Self {
            model: LlmModel::Ollama(model_name),
            base_url: base_url.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }
}

#[async_trait]
impl LlmAdapter for OllamaClient {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        use serde_json::{json, Value};

        let model_name = match &self.model {
            LlmModel::Ollama(n) => n.clone(),
            _ => unreachable!(),
        };

        let body = json!({
            "model": model_name,
            "prompt": format!("{}\n\n{}", req.system_prompt, req.user_prompt),
            "stream": false
        });

        let resp = self
            .http
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        let text = resp["response"].as_str().unwrap_or("").to_string();

        Ok(LlmResponse {
            text,
            model: self.model.clone(),
            tokens_used: 0,
        })
    }

    fn model(&self) -> &LlmModel {
        &self.model
    }
}

/// Factory: create the appropriate adapter from model and env vars.
pub fn create_adapter(model: &LlmModel) -> Arc<dyn LlmAdapter> {
    match model {
        LlmModel::Mock => Arc::new(MockLlm::new()),
        LlmModel::Gpt4o => {
            let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            Arc::new(OpenAiClient::new(key))
        }
        LlmModel::ClaudeSonnet | LlmModel::ClaudeOpus => {
            let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
            Arc::new(AnthropicClient::new(key, model.clone()))
        }
        LlmModel::Ollama(name) => {
            let url = std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            Arc::new(OllamaClient::new(name.clone(), url))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn mock_req() -> LlmRequest {
        LlmRequest {
            model: LlmModel::Mock,
            system_prompt: "You are a curious agent.".to_string(),
            user_prompt: "What do you see?".to_string(),
            max_tokens: 64,
        }
    }

    #[tokio::test]
    async fn mock_llm_returns_response() {
        let llm = MockLlm::new();
        let resp = llm.complete(mock_req()).await.unwrap();
        assert!(!resp.text.is_empty());
        assert_eq!(resp.model, LlmModel::Mock);
    }

    #[tokio::test]
    async fn mock_llm_with_delay() {
        let llm = MockLlm::new().with_delay(Duration::from_millis(50));
        let start = std::time::Instant::now();
        llm.complete(mock_req()).await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(40));
    }

    #[tokio::test]
    async fn factory_creates_mock() {
        let adapter = create_adapter(&LlmModel::Mock);
        let resp = adapter.complete(mock_req()).await.unwrap();
        assert!(!resp.text.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn openai_integration() {
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");
        let client = OpenAiClient::new(key);
        let resp = client.complete(mock_req()).await.unwrap();
        assert!(!resp.text.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn anthropic_integration() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");
        let client = AnthropicClient::new(key, LlmModel::ClaudeSonnet);
        let resp = client.complete(mock_req()).await.unwrap();
        assert!(!resp.text.is_empty());
    }
}
