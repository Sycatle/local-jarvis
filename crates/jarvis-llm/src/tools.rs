//! Tool-call orchestration: parse the grammar-constrained JSON, dispatch to a
//! [`ToolRegistry`], re-feed the result, and iterate up to `max_iterations`.
//!
//! Three output formats from the LLM are accepted, in priority order:
//!
//! 1. **Qwen2.5 native**: `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`.
//!    This is what Qwen2.5-Instruct emits when handed an OpenAI-style tools
//!    list in its system prompt — the format we should prefer because the
//!    GBNF wrapper currently breaks the Qwen tokenizer.
//! 2. **Legacy wrapped JSON**: `{"tool_call": {"name": "...", "arguments": {...}}}`
//!    or `{"response": "..."}`. Kept for the StubEngine and for fallback when
//!    a future GBNF grammar is re-enabled.
//! 3. **Embedded legacy JSON**: a `{"tool_call": ...}` object found anywhere
//!    inside otherwise-free-form text.
//!
//! Each iteration is reported through an optional [`StepCallback`] so the
//! orchestrator can emit a D-Bus `StepTaken` signal for UIs.

use std::sync::Arc;

use async_trait::async_trait;
use jarvis_core::events::ToolCall;
use serde::Deserialize;

use crate::chat::{ChatHistory, ChatMessage};
use crate::engine::LlmEngine;
use crate::grammar::ToolSpec;
use crate::sanitize;

#[async_trait]
pub trait ToolRegistry: Send + Sync {
    fn specs(&self) -> Vec<ToolSpec>;
    async fn invoke(&self, call: &ToolCall) -> serde_json::Value;
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LlmOutput {
    Response { response: String },
    ToolCall { tool_call: ToolCallJson },
}

#[derive(Debug, Deserialize)]
struct ToolCallJson {
    name: String,
    arguments: serde_json::Value,
}

/// Qwen2.5 native tool-call payload (the JSON inside `<tool_call>...</tool_call>`).
#[derive(Debug, Deserialize)]
struct QwenToolCall {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

/// One pass of the ReAct loop. `action` is `None` for the final reply turn;
/// `observation` is `None` until the tool returns. The callback fires twice
/// per tool-using iteration (once at dispatch, once after the result is
/// available) so a watching UI can show the model thinking before the tool
/// actually runs.
#[derive(Debug, Clone)]
pub struct ToolStep {
    pub iteration: u32,
    pub thought: String,
    pub action: Option<ToolCall>,
    pub observation: Option<String>,
}

pub type StepCallback = Arc<dyn Fn(ToolStep) + Send + Sync>;

#[derive(Debug)]
pub enum ToolLoopOutcome {
    /// Plain user-facing reply.
    Reply(String),
    /// Loop bailed at max iterations.
    MaxIterations,
    /// Engine returned text that didn't match any known format.
    Unparseable(String),
}

/// Drive the LLM through the ReAct loop. See module docs for accepted formats.
///
/// `step_cb`, if set, is invoked at each iteration with the model's raw
/// "thought" output and any action it decided to take.
pub async fn run_tool_loop(
    engine: &dyn LlmEngine,
    registry: &dyn ToolRegistry,
    history: &mut ChatHistory,
    user_input: &str,
    max_iterations: u32,
    step_cb: Option<StepCallback>,
) -> anyhow::Result<ToolLoopOutcome> {
    history.push(ChatMessage::user(user_input));
    let tools = registry.specs();

    for iter in 0..max_iterations {
        let raw = engine.generate(history, &tools).await?;
        tracing::debug!(iter, raw = %raw, "llm raw output");

        // 1. Qwen2.5 native <tool_call> tag — preferred path.
        if let Some(call) = extract_qwen_tool_call(&raw) {
            emit_step(&step_cb, iter, &raw, Some(&call), None);
            let result = registry.invoke(&call).await;
            let obs = result.to_string();
            emit_step(&step_cb, iter, &raw, Some(&call), Some(&obs));
            history.push(ChatMessage::assistant(raw));
            history.push(ChatMessage::tool(obs));
            continue;
        }

        // 2. Legacy strict-JSON output (StubEngine, GBNF when re-enabled).
        if let Ok(parsed) = serde_json::from_str::<LlmOutput>(raw.trim()) {
            match parsed {
                LlmOutput::Response { response } => {
                    let clean = sanitize::for_tts(&response);
                    emit_step(&step_cb, iter, &clean, None, None);
                    history.push(ChatMessage::assistant(clean.clone()));
                    return Ok(ToolLoopOutcome::Reply(clean));
                }
                LlmOutput::ToolCall { tool_call } => {
                    let call = ToolCall {
                        name: tool_call.name.clone(),
                        arguments: tool_call.arguments.clone(),
                    };
                    emit_step(&step_cb, iter, &raw, Some(&call), None);
                    let result = registry.invoke(&call).await;
                    let obs = result.to_string();
                    emit_step(&step_cb, iter, &raw, Some(&call), Some(&obs));
                    history.push(ChatMessage::assistant(raw));
                    history.push(ChatMessage::tool(obs));
                    continue;
                }
            }
        }

        // 3. Legacy embedded JSON tool call (model wrote prose around it).
        if let Some(call) = extract_legacy_tool_call(&raw) {
            emit_step(&step_cb, iter, &raw, Some(&call), None);
            let result = registry.invoke(&call).await;
            let obs = result.to_string();
            emit_step(&step_cb, iter, &raw, Some(&call), Some(&obs));
            history.push(ChatMessage::assistant(raw));
            history.push(ChatMessage::tool(obs));
            continue;
        }

        // Otherwise treat as a plain spoken reply; sanitize for TTS.
        let clean = sanitize::for_tts(&raw);
        if clean.is_empty() {
            return Ok(ToolLoopOutcome::Unparseable(raw));
        }
        emit_step(&step_cb, iter, &clean, None, None);
        history.push(ChatMessage::assistant(clean.clone()));
        return Ok(ToolLoopOutcome::Reply(clean));
    }
    Ok(ToolLoopOutcome::MaxIterations)
}

fn emit_step(
    cb: &Option<StepCallback>,
    iteration: u32,
    thought: &str,
    action: Option<&ToolCall>,
    observation: Option<&str>,
) {
    if let Some(cb) = cb {
        cb(ToolStep {
            iteration,
            thought: thought.to_string(),
            action: action.cloned(),
            observation: observation.map(str::to_string),
        });
    }
}

/// Parse a Qwen2.5-native `<tool_call>...</tool_call>` block.
///
/// Tolerant of whitespace and of multiple tool calls in one response (returns
/// the first; the loop will pick up subsequent calls on the next iteration).
fn extract_qwen_tool_call(raw: &str) -> Option<ToolCall> {
    let start = raw.find("<tool_call>")? + "<tool_call>".len();
    let end_rel = raw[start..].find("</tool_call>")?;
    let inner = raw[start..start + end_rel].trim();
    let parsed: QwenToolCall = serde_json::from_str(inner).ok()?;
    Some(ToolCall {
        name: parsed.name,
        arguments: if parsed.arguments.is_null() {
            serde_json::json!({})
        } else {
            parsed.arguments
        },
    })
}

/// Find a `{"tool_call": {...}}` JSON object anywhere in the raw output.
fn extract_legacy_tool_call(raw: &str) -> Option<ToolCall> {
    let start = raw
        .find("{\"tool_call\"")
        .or_else(|| raw.find("{ \"tool_call\""))?;
    let bytes = raw.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        let c = b as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let slice = &raw[start..=i];
                    if let Ok(LlmOutput::ToolCall { tool_call }) =
                        serde_json::from_str::<LlmOutput>(slice)
                    {
                        return Some(ToolCall {
                            name: tool_call.name,
                            arguments: tool_call.arguments,
                        });
                    }
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_native_tool_call() {
        let raw = "<tool_call>\n{\"name\": \"volume\", \"arguments\": {\"action\": \"up\"}}\n</tool_call>";
        let call = extract_qwen_tool_call(raw).expect("parsed");
        assert_eq!(call.name, "volume");
        assert_eq!(call.arguments["action"], "up");
    }

    #[test]
    fn qwen_native_with_prose_before() {
        let raw = "Let me adjust that.\n<tool_call>{\"name\":\"volume\",\"arguments\":{\"action\":\"set\",\"value\":40}}</tool_call>";
        let call = extract_qwen_tool_call(raw).expect("parsed");
        assert_eq!(call.name, "volume");
        assert_eq!(call.arguments["value"], 40);
    }

    #[test]
    fn qwen_native_missing_arguments_defaults_to_empty_object() {
        let raw = "<tool_call>{\"name\":\"media_play_pause\"}</tool_call>";
        let call = extract_qwen_tool_call(raw).expect("parsed");
        assert_eq!(call.name, "media_play_pause");
        assert!(call.arguments.is_object());
    }

    #[test]
    fn legacy_strict_json_still_parses() {
        let raw = r#"{"tool_call":{"name":"volume","arguments":{"action":"mute"}}}"#;
        let parsed: LlmOutput = serde_json::from_str(raw).expect("legacy json");
        match parsed {
            LlmOutput::ToolCall { tool_call } => assert_eq!(tool_call.name, "volume"),
            _ => panic!("expected tool_call"),
        }
    }

    #[test]
    fn legacy_embedded_json() {
        let raw = r#"Sure, calling: {"tool_call":{"name":"brightness","arguments":{"action":"up"}}} done."#;
        let call = extract_legacy_tool_call(raw).expect("parsed");
        assert_eq!(call.name, "brightness");
    }

    #[test]
    fn no_tool_call_returns_none() {
        assert!(extract_qwen_tool_call("hello world").is_none());
        assert!(extract_legacy_tool_call("hello world").is_none());
    }

    #[test]
    fn malformed_qwen_tag_returns_none() {
        // Missing closing tag.
        assert!(extract_qwen_tool_call("<tool_call>{\"name\":\"x\"}").is_none());
        // Invalid JSON inside.
        assert!(extract_qwen_tool_call("<tool_call>not json</tool_call>").is_none());
    }
}
