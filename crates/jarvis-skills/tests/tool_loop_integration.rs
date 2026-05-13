//! Integration tests for the LLM ↔ skills handoff.
//!
//! These don't exercise the orchestrator's audio state machine (which is
//! wired to concrete `Capture` / `PcmPlayer` types and would need a traits
//! refactor to mock cleanly). They do exercise the critical seam we just
//! touched in v1.1: the Qwen `<tool_call>` parser, the `SkillRegistry`
//! dispatch, and the `Skill` impl emitted by the `#[skill]` proc-macro.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use jarvis_llm::{
    chat::ChatHistory,
    engine::LlmEngine,
    grammar::ToolSpec,
    tools::{run_tool_loop, ToolLoopOutcome},
};
use jarvis_skills::{Capabilities, Skill, SkillRegistry};
use jarvis_skills_macros::skill;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

/// Scripted LLM: returns the next response in the queue on each `generate` call.
/// Panics if the queue is exhausted, which makes "ran too many iterations"
/// failures loud rather than silent.
struct ScriptedEngine {
    queue: Mutex<Vec<String>>,
}

impl ScriptedEngine {
    fn new<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut v: Vec<String> = replies.into_iter().map(Into::into).collect();
        v.reverse();
        Self {
            queue: Mutex::new(v),
        }
    }
}

#[async_trait]
impl LlmEngine for ScriptedEngine {
    async fn generate(
        &self,
        _history: &ChatHistory,
        _tools: &[ToolSpec],
    ) -> anyhow::Result<String> {
        let mut q = self.queue.lock().unwrap();
        Ok(q.pop()
            .expect("ScriptedEngine: queue exhausted (loop ran more turns than scripted)"))
    }
}

#[derive(Deserialize, JsonSchema)]
struct EchoArgs {
    value: String,
}

#[derive(Default, Clone)]
struct EchoSkill {
    calls: Arc<Mutex<Vec<String>>>,
}

#[skill(name = "echo", description = "Echo back a string — testing skill.")]
impl EchoSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::none()
    }
    async fn run(&self, args: EchoArgs) -> Value {
        self.calls.lock().unwrap().push(args.value.clone());
        json!({"ok": true, "echoed": args.value})
    }
}

fn build_registry(skill: Arc<dyn Skill>) -> SkillRegistry {
    let mut r = SkillRegistry::new();
    r.register(skill);
    r
}

#[tokio::test]
async fn qwen_native_tool_call_flows_through_registry() {
    let echo = EchoSkill::default();
    let recorder = Arc::clone(&echo.calls);
    let registry = build_registry(Arc::new(echo));

    // Turn 1: model emits a Qwen-native tool call.
    // Turn 2: after observing the tool result, model emits a plain reply.
    let engine = ScriptedEngine::new([
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"value\":\"hello\"}}</tool_call>",
        "Done.",
    ]);
    let mut history = ChatHistory::with_system("test");

    let outcome = run_tool_loop(&engine, &registry, &mut history, "say hello", 4, None)
        .await
        .expect("loop ok");

    match outcome {
        ToolLoopOutcome::Reply(r) => assert_eq!(r, "Done."),
        other => panic!("expected Reply, got {other:?}"),
    }
    let calls = recorder.lock().unwrap();
    assert_eq!(&*calls, &["hello".to_string()]);
}

#[tokio::test]
async fn legacy_json_tool_call_still_dispatches() {
    let echo = EchoSkill::default();
    let recorder = Arc::clone(&echo.calls);
    let registry = build_registry(Arc::new(echo));

    let engine = ScriptedEngine::new([
        r#"{"tool_call":{"name":"echo","arguments":{"value":"legacy"}}}"#,
        r#"{"response":"All done."}"#,
    ]);
    let mut history = ChatHistory::new();

    let outcome = run_tool_loop(&engine, &registry, &mut history, "trigger", 4, None)
        .await
        .expect("loop ok");

    match outcome {
        ToolLoopOutcome::Reply(r) => assert_eq!(r, "All done."),
        other => panic!("expected Reply, got {other:?}"),
    }
    assert_eq!(&*recorder.lock().unwrap(), &["legacy".to_string()]);
}

#[tokio::test]
async fn plain_text_reply_short_circuits_loop() {
    let registry = build_registry(Arc::new(EchoSkill::default()));
    let engine = ScriptedEngine::new(["Bonjour."]);
    let mut history = ChatHistory::new();

    let outcome = run_tool_loop(&engine, &registry, &mut history, "salut", 4, None)
        .await
        .expect("loop ok");
    match outcome {
        ToolLoopOutcome::Reply(r) => assert_eq!(r, "Bonjour."),
        other => panic!("expected Reply, got {other:?}"),
    }
}

#[tokio::test]
async fn max_iterations_terminates() {
    let echo = EchoSkill::default();
    let registry = build_registry(Arc::new(echo));

    // Five tool calls in a row, never converging.
    let call = "<tool_call>{\"name\":\"echo\",\"arguments\":{\"value\":\"x\"}}</tool_call>";
    let engine = ScriptedEngine::new([call, call, call, call, call]);
    let mut history = ChatHistory::new();

    let outcome = run_tool_loop(&engine, &registry, &mut history, "loop", 3, None)
        .await
        .expect("loop ok");
    assert!(matches!(outcome, ToolLoopOutcome::MaxIterations));
}

#[tokio::test]
async fn unknown_skill_yields_error_payload_but_loop_continues() {
    let registry = build_registry(Arc::new(EchoSkill::default()));
    let engine = ScriptedEngine::new([
        "<tool_call>{\"name\":\"nope\",\"arguments\":{}}</tool_call>",
        "Recovered.",
    ]);
    let mut history = ChatHistory::new();

    let outcome = run_tool_loop(&engine, &registry, &mut history, "try", 4, None)
        .await
        .expect("loop ok");
    match outcome {
        ToolLoopOutcome::Reply(r) => assert_eq!(r, "Recovered."),
        other => panic!("expected Reply, got {other:?}"),
    }
    // The tool error message should have been pushed back into history for
    // the next LLM turn.
    let last_tool_msg = history
        .messages()
        .iter()
        .rev()
        .find(|m| matches!(m.role, jarvis_llm::Role::Tool))
        .expect("expected a tool message in history");
    assert!(last_tool_msg.content.contains("unknown skill"));
}

#[test]
fn macro_generated_skill_has_expected_metadata() {
    let echo = EchoSkill::default();
    assert_eq!(echo.name(), "echo");
    assert!(echo.description().contains("Echo"));
    let params = echo.parameters();
    // schemars emits a `properties.value` entry for the Args struct.
    assert!(params["properties"]["value"].is_object());
}

#[test]
fn registry_audit_lists_skills_with_caps() {
    let mut r = SkillRegistry::new();
    r.register(Arc::new(EchoSkill::default()));
    let audit = r.audit();
    assert_eq!(audit.len(), 1);
    let (name, caps) = &audit[0];
    assert_eq!(*name, "echo");
    assert!(caps.exec.is_empty());
    assert!(!caps.net);
}
