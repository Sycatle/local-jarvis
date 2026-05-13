//! Skill trait + registry.
//!
//! A [`Skill`] is an async function exposed to the LLM as a JSON-schema tool.
//! The [`SkillRegistry`] holds dyn-dispatched skills and implements
//! [`jarvis_llm::ToolRegistry`] so the tool-call loop dispatches through it
//! without further glue.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use jarvis_core::events::ToolCall;
use jarvis_llm::tools::ToolRegistry;
use jarvis_llm::ToolSpec;
use serde_json::Value;

use crate::capabilities::Capabilities;

#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters(&self) -> Value;
    /// Capability manifest. Default is empty (skill claims no privileged
    /// access). Override per-skill to declare exec/dbus/fs/net requirements.
    fn capabilities(&self) -> Capabilities {
        Capabilities::none()
    }
    async fn invoke(&self, args: Value) -> Value;
}

#[derive(Default)]
pub struct SkillRegistry {
    skills: HashMap<&'static str, Arc<dyn Skill>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, skill: Arc<dyn Skill>) {
        self.skills.insert(skill.name(), skill);
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.skills.keys().copied().collect();
        v.sort();
        v
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Capability snapshot for every registered skill — handy for `jarvis
    /// skills` audit output and tests.
    pub fn audit(&self) -> Vec<(&'static str, Capabilities)> {
        let mut v: Vec<_> = self
            .skills
            .values()
            .map(|s| (s.name(), s.capabilities()))
            .collect();
        v.sort_by_key(|(name, _)| *name);
        v
    }
}

#[async_trait]
impl ToolRegistry for SkillRegistry {
    fn specs(&self) -> Vec<ToolSpec> {
        self.skills
            .values()
            .map(|s| ToolSpec {
                name: s.name().to_string(),
                description: s.description().to_string(),
                parameters: s.parameters(),
            })
            .collect()
    }

    async fn invoke(&self, call: &ToolCall) -> Value {
        match self.skills.get(call.name.as_str()) {
            Some(s) => {
                let caps = s.capabilities();
                // Structured audit log: this line is the canonical record of
                // what the assistant actually invoked on the user's behalf.
                tracing::info!(
                    target: "jarvis::audit",
                    skill = s.name(),
                    args = %call.arguments,
                    exec = ?caps.exec,
                    dbus = ?caps.dbus,
                    net = caps.net,
                    "skill invoke",
                );
                s.invoke(call.arguments.clone()).await
            }
            None => serde_json::json!({
                "ok": false,
                "error": format!("unknown skill: {}", call.name),
            }),
        }
    }
}
