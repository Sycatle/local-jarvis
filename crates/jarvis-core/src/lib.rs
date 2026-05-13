//! Shared types and configuration for Jarvis.

pub mod config;
pub mod dirs;
pub mod events;
pub mod state;

pub use config::Config;
pub use events::{LlmReply, SkillResult, ToolCall, Transcript, WakeEvent, WakeSource};
pub use state::State;
