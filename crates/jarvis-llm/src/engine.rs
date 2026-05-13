//! LLM backend trait + a stub implementation usable until `llama-cpp-2` is
//! wired through a feature gate.

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use crate::chat::ChatHistory;
use crate::grammar::ToolSpec;

#[async_trait]
pub trait LlmEngine: Send + Sync {
    /// Run a constrained generation given the chat history and the available
    /// tools. Returns the raw JSON string the model produced (the caller
    /// parses it via the grammar).
    async fn generate(
        &self,
        history: &ChatHistory,
        tools: &[ToolSpec],
    ) -> anyhow::Result<String>;

    /// Token-streaming variant. Default impl buffers `generate()` and yields a
    /// single chunk so callers can write streaming-first code that still works
    /// on backends without a real token stream. Backends that *do* support
    /// streaming (LlamaEngine with `llama-cpp-2`) should override this to emit
    /// tokens as they arrive — this is what makes time-to-first-audio drop
    /// from ~5s to ~1s on long replies.
    async fn generate_stream<'a>(
        &'a self,
        history: &'a ChatHistory,
        tools: &'a [ToolSpec],
    ) -> anyhow::Result<BoxStream<'a, anyhow::Result<String>>> {
        let full = self.generate(history, tools).await?;
        Ok(stream::iter(std::iter::once(Ok(full))).boxed())
    }
}

/// Placeholder that always replies in JSON form so the orchestrator can be
/// exercised end-to-end without a real model on disk.
pub struct StubEngine {
    canned_reply: String,
}

impl StubEngine {
    pub fn new() -> Self {
        Self {
            canned_reply: "Modèle LLM non chargé — installer le GGUF et activer le feature.".into(),
        }
    }
    pub fn with_reply(reply: impl Into<String>) -> Self {
        Self {
            canned_reply: reply.into(),
        }
    }
}

impl Default for StubEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmEngine for StubEngine {
    async fn generate(
        &self,
        _history: &ChatHistory,
        _tools: &[ToolSpec],
    ) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&serde_json::json!({
            "response": self.canned_reply,
        }))?)
    }
}

/// Engine adapter useful for tests and demos: emit a pre-scripted sequence of
/// tokens through `generate_stream`. The buffered `generate()` returns the
/// concatenation, so it also drives non-streaming callers.
pub struct ScriptedStreamEngine {
    tokens: Vec<String>,
}

impl ScriptedStreamEngine {
    pub fn new<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            tokens: tokens.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl LlmEngine for ScriptedStreamEngine {
    async fn generate(
        &self,
        _history: &ChatHistory,
        _tools: &[ToolSpec],
    ) -> anyhow::Result<String> {
        Ok(self.tokens.join(""))
    }

    async fn generate_stream<'a>(
        &'a self,
        _history: &'a ChatHistory,
        _tools: &'a [ToolSpec],
    ) -> anyhow::Result<BoxStream<'a, anyhow::Result<String>>> {
        let toks: Vec<anyhow::Result<String>> =
            self.tokens.iter().cloned().map(Ok).collect();
        Ok(stream::iter(toks).boxed())
    }
}
