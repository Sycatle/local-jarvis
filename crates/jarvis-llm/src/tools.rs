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
use futures::stream::StreamExt;
use jarvis_core::events::ToolCall;
use serde::Deserialize;

use crate::chat::{ChatHistory, ChatMessage};
use crate::engine::LlmEngine;
use crate::grammar::ToolSpec;
use crate::sanitize;

/// Sentence terminators we flush on, mirroring [`crate::streaming`]. Kept
/// inline because `sentence_stream` requires a `'static` upstream and the
/// LLM token stream borrows from `ChatHistory`.
const SENTENCE_TERMINATORS: &[char] = &['.', '!', '?', ';', '\n'];
/// Soft flush threshold (chars) to bound latency on run-on sentences.
const SOFT_FLUSH_AT: usize = 160;

/// How many characters of LLM output to peek before deciding the stream can
/// safely commit to TTS. Qwen2.5 always opens with `<tool_call>` when calling
/// a tool, so 80 characters of tool-free prose is plenty to rule it out.
pub const STREAM_COMMIT_CHARS: usize = 80;

/// Sink for streamed sentences. Receives already-sanitised text, one
/// sentence per call. Implementations typically push into an mpsc that a
/// concurrent TTS task consumes.
pub type SentenceSink = Arc<dyn Fn(String) + Send + Sync>;

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

/// Same as [`run_tool_loop`] but, when `sentence_sink` is set, attempts to
/// stream each LLM iteration token-by-token. If the prefix turns out to
/// contain a tool-call indicator, the iteration falls back to the buffered
/// parse path (no sentences emitted). Otherwise sanitised sentences are
/// pushed to the sink as they form — the caller pipes them into TTS for a
/// markedly lower time-to-first-audio on plain replies.
pub async fn run_tool_loop_streamed(
    engine: &dyn LlmEngine,
    registry: &dyn ToolRegistry,
    history: &mut ChatHistory,
    user_input: &str,
    max_iterations: u32,
    step_cb: Option<StepCallback>,
    sentence_sink: Option<SentenceSink>,
) -> anyhow::Result<ToolLoopOutcome> {
    history.push(ChatMessage::user(user_input));
    let tools = registry.specs();

    for iter in 0..max_iterations {
        let raw = if let Some(sink) = sentence_sink.as_ref() {
            match stream_or_buffer(engine, history, &tools, sink.clone(), STREAM_COMMIT_CHARS)
                .await?
            {
                StreamFirstOutcome::Streamed(text) => {
                    emit_step(&step_cb, iter, &text, None, None);
                    history.push(ChatMessage::assistant(text.clone()));
                    return Ok(ToolLoopOutcome::Reply(text));
                }
                StreamFirstOutcome::Buffered(raw) => raw,
            }
        } else {
            engine.generate(history, &tools).await?
        };
        tracing::debug!(iter, raw = %raw, "llm raw output");

        if let Some(call) = extract_qwen_tool_call(&raw) {
            emit_step(&step_cb, iter, &raw, Some(&call), None);
            let result = registry.invoke(&call).await;
            let obs = result.to_string();
            emit_step(&step_cb, iter, &raw, Some(&call), Some(&obs));
            history.push(ChatMessage::assistant(raw));
            history.push(ChatMessage::tool(obs));
            continue;
        }

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

        if let Some(call) = extract_legacy_tool_call(&raw) {
            emit_step(&step_cb, iter, &raw, Some(&call), None);
            let result = registry.invoke(&call).await;
            let obs = result.to_string();
            emit_step(&step_cb, iter, &raw, Some(&call), Some(&obs));
            history.push(ChatMessage::assistant(raw));
            history.push(ChatMessage::tool(obs));
            continue;
        }

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

/// Decide whether the rolling LLM output buffer should commit to streaming
/// mode (no tool call ahead) or buffered mode (a tool-call marker is being
/// emitted).
///
/// - `Some(true)` — a tool-call marker is present (even partial); switch to
///   buffered.
/// - `Some(false)` — buffer reached `commit_threshold` characters with no
///   marker; safe to start piping sentences to the sink.
/// - `None` — undecided; keep peeking.
///
/// The substring check is intentionally permissive: Qwen2.5 always opens its
/// tool-call output with `<tool_call>`, so seeing `<tool` mid-stream (e.g.
/// after only the first three tokens have arrived) is a near-certain signal.
/// Plain French replies essentially never contain `<tool` or `{"tool_call`.
pub fn tool_call_committed(buf: &str, commit_threshold: usize) -> Option<bool> {
    if buf.contains("<tool") || buf.contains("{\"tool_call") || buf.contains("{ \"tool_call") {
        return Some(true);
    }
    if buf.chars().count() >= commit_threshold {
        Some(false)
    } else {
        None
    }
}

enum StreamFirstOutcome {
    /// All sentences from this iteration were already pushed to the sink.
    /// The string is the concatenated sanitised reply (for history + return).
    Streamed(String),
    /// A tool-call marker appeared; the raw text is returned untouched so the
    /// caller can run it through the regular tool-call parsers.
    Buffered(String),
}

async fn stream_or_buffer(
    engine: &dyn LlmEngine,
    history: &ChatHistory,
    tools: &[ToolSpec],
    sink: SentenceSink,
    commit_threshold: usize,
) -> anyhow::Result<StreamFirstOutcome> {
    let token_stream = engine.generate_stream(history, tools).await?;
    let mut token_stream = Box::pin(token_stream);

    let mut buf = String::new();
    let mut decision: Option<bool> = None;

    while let Some(item) = token_stream.next().await {
        let tok = item?;
        buf.push_str(&tok);
        if let Some(b) = tool_call_committed(&buf, commit_threshold) {
            decision = Some(b);
            break;
        }
    }

    match decision {
        Some(true) => {
            // Drain remaining tokens into the buffer so the caller can parse.
            while let Some(item) = token_stream.next().await {
                buf.push_str(&item?);
            }
            Ok(StreamFirstOutcome::Buffered(buf))
        }
        Some(false) | None => {
            // Either committed to streaming (Some(false)) or the stream ended
            // before commit threshold (None). The peeked tokens are still in
            // `buf` — keep chunking on terminators / soft flush as more tokens
            // arrive, sanitise each sentence, push to the sink, accumulate the
            // full reply for history.
            let mut full = String::new();
            // Drain whatever sentences already fit in the peek buffer.
            flush_chunks(&mut buf, &sink, &mut full, false);
            while let Some(item) = token_stream.next().await {
                let tok = item?;
                buf.push_str(&tok);
                flush_chunks(&mut buf, &sink, &mut full, false);
            }
            // Final residual flush.
            flush_chunks(&mut buf, &sink, &mut full, true);
            Ok(StreamFirstOutcome::Streamed(full))
        }
    }
}

/// Emit complete sentences from `buf` to `sink`, keeping any residual partial
/// sentence in `buf` for the next call. When `flush_residual` is true, the
/// remaining text is flushed unconditionally (end-of-stream).
fn flush_chunks(buf: &mut String, sink: &SentenceSink, full: &mut String, flush_residual: bool) {
    while let Some(cut) = next_chunk_boundary(buf) {
        let chunk: String = buf[..cut].trim().to_string();
        buf.drain(..cut);
        emit_sentence(&chunk, sink, full);
    }
    if flush_residual {
        let tail = std::mem::take(buf).trim().to_string();
        emit_sentence(&tail, sink, full);
    }
}

fn next_chunk_boundary(buf: &str) -> Option<usize> {
    // Hard terminator: cut just after the punctuation.
    for (i, ch) in buf.char_indices() {
        if SENTENCE_TERMINATORS.contains(&ch) {
            return Some(i + ch.len_utf8());
        }
    }
    // Soft flush: at SOFT_FLUSH_AT chars, cut at the most recent whitespace.
    if buf.chars().count() >= SOFT_FLUSH_AT {
        let last_ws = buf.char_indices().rfind(|(_, c)| c.is_whitespace())?;
        return Some(last_ws.0 + last_ws.1.len_utf8());
    }
    None
}

fn emit_sentence(chunk: &str, sink: &SentenceSink, full: &mut String) {
    if chunk.is_empty() {
        return;
    }
    let clean = sanitize::for_tts(chunk);
    if clean.is_empty() {
        return;
    }
    sink(clean.clone());
    if !full.is_empty() {
        full.push(' ');
    }
    full.push_str(&clean);
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
    use crate::engine::ScriptedStreamEngine;
    use std::sync::Mutex;

    #[test]
    fn tool_call_committed_detects_qwen_marker_early() {
        // After three tokens, the buffer already contains `<tool`.
        let buf = "<tool";
        assert_eq!(tool_call_committed(buf, 80), Some(true));
    }

    #[test]
    fn tool_call_committed_detects_full_qwen_tag() {
        let buf = "<tool_call>\n{\"name";
        assert_eq!(tool_call_committed(buf, 80), Some(true));
    }

    #[test]
    fn tool_call_committed_detects_legacy_json_marker() {
        assert_eq!(
            tool_call_committed("{\"tool_call\":", 80),
            Some(true)
        );
        assert_eq!(
            tool_call_committed("{ \"tool_call\":", 80),
            Some(true)
        );
    }

    #[test]
    fn tool_call_committed_waits_under_threshold() {
        // 50 chars of plain prose without any marker is still undecided.
        let buf = "Bonjour Sir, je suis prêt à vous aider.";
        assert!(buf.chars().count() < 80);
        assert_eq!(tool_call_committed(buf, 80), None);
    }

    #[test]
    fn tool_call_committed_commits_to_stream_past_threshold() {
        let buf = "a".repeat(120);
        assert_eq!(tool_call_committed(&buf, 80), Some(false));
    }

    #[tokio::test]
    async fn stream_or_buffer_streams_plain_reply_through_sink() {
        let engine = ScriptedStreamEngine::new([
            "Bonjour Sir. ",
            "Le volume est ",
            "réglé. ",
            "Autre chose ?",
        ]);
        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink: SentenceSink = {
            let c = Arc::clone(&collected);
            Arc::new(move |s| c.lock().unwrap().push(s))
        };
        let history = ChatHistory::new();
        let tools: Vec<ToolSpec> = vec![];
        let out = stream_or_buffer(&engine, &history, &tools, sink, 80)
            .await
            .unwrap();
        match out {
            StreamFirstOutcome::Streamed(text) => {
                assert!(text.contains("Bonjour Sir"));
                let chunks = collected.lock().unwrap().clone();
                assert!(!chunks.is_empty(), "expected sentences in the sink");
                // First chunk must arrive before the model finishes — proxied
                // here by the fact that we got more than one sentence.
                assert!(chunks.iter().any(|s| s.contains("Bonjour Sir")));
            }
            StreamFirstOutcome::Buffered(_) => panic!("plain reply should stream"),
        }
    }

    #[tokio::test]
    async fn stream_or_buffer_falls_back_to_buffer_on_tool_call() {
        let engine = ScriptedStreamEngine::new([
            "<tool_call>",
            "{\"name\":\"volume\",\"arguments\":{\"action\":\"up\"}}",
            "</tool_call>",
        ]);
        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink: SentenceSink = {
            let c = Arc::clone(&collected);
            Arc::new(move |s| c.lock().unwrap().push(s))
        };
        let history = ChatHistory::new();
        let tools: Vec<ToolSpec> = vec![];
        let out = stream_or_buffer(&engine, &history, &tools, sink, 80)
            .await
            .unwrap();
        match out {
            StreamFirstOutcome::Buffered(raw) => {
                assert!(raw.contains("<tool_call>"));
                assert!(raw.contains("volume"));
                let chunks = collected.lock().unwrap().clone();
                assert!(chunks.is_empty(), "no sentences should have been sunk");
            }
            StreamFirstOutcome::Streamed(t) => {
                panic!("tool call should buffer, got streamed: {t}")
            }
        }
    }

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
