//! Debug CLI: invoke a registered skill directly, without going through the
//! wake → STT → LLM → tool-call loop.
//!
//! Used to smoke-test system control end-to-end before any ML feature is
//! activated. Mirrors the skill registration done in [`crate::runner`].

use anyhow::{anyhow, Result};
use jarvis_core::events::ToolCall;
use jarvis_llm::tools::ToolRegistry;
use jarvis_skills::{MediaSkills, SkillRegistry, SystemSkills};

async fn build_registry() -> Result<SkillRegistry> {
    let desktop = jarvis_desktop::make_desktop().await?;
    let mut registry = SkillRegistry::new();
    for s in SystemSkills::new(desktop).skills() {
        registry.register(s);
    }
    for s in MediaSkills::skills() {
        registry.register(s);
    }
    Ok(registry)
}

pub async fn list() -> Result<()> {
    let r = build_registry().await?;
    for n in r.names() {
        println!("{n}");
    }
    Ok(())
}

pub async fn invoke(name: &str, args_json: &str) -> Result<()> {
    let arguments: serde_json::Value = serde_json::from_str(args_json)
        .map_err(|e| anyhow!("--args must be valid JSON: {e}"))?;
    let r = build_registry().await?;
    let call = ToolCall {
        name: name.to_string(),
        arguments,
    };
    let result = r.invoke(&call).await;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
