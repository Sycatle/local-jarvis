//! Local LLM with tool-calling.
//!
//! - [`chat`]: message types and the Qwen2.5 chat template.
//! - [`grammar`]: GBNF builder that constrains output to
//!   `{"response": ...}` or `{"tool_call": {"name": ..., "arguments": ...}}`.
//! - [`tools`]: orchestration of the tool-call loop, dispatching through a
//!   pluggable [`ToolRegistry`].
//! - [`engine`]: backend trait + a stub implementation. `llama-cpp-2` lands
//!   behind a feature flag once the GGUF is available.

pub mod chat;
pub mod engine;
pub mod grammar;
pub mod sanitize;
pub mod streaming;
pub mod tools;
#[cfg(feature = "llama")]
pub mod llama;

pub use chat::{ChatHistory, ChatMessage, Role};
pub use engine::{LlmEngine, ScriptedStreamEngine, StubEngine};
pub use grammar::ToolSpec;
pub use sanitize::for_tts as sanitize_for_tts;
pub use streaming::sentence_stream;
pub use tools::{ToolRegistry, run_tool_loop, ToolLoopOutcome, ToolStep, StepCallback};
#[cfg(feature = "llama")]
pub use llama::{LlamaConfig, LlamaEngine};
