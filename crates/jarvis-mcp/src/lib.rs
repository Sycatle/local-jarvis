//! MCP (Model Context Protocol) client adapter.
//!
//! Goal: every MCP server declared in `~/.config/jarvis/config.toml`
//! `[mcp.servers.<name>]` becomes a tool the LLM can invoke, alongside the
//! native Rust skills. Each server runs in its own subprocess (stdio
//! transport), which gives us the OpenClaw-alternative property "isolation
//! per agent" for free — the host process never executes plugin code in its
//! own address space.
//!
//! This file currently ships the **scaffolding**: config types, the
//! `McpRegistry` shell, the `CompositeToolRegistry` that merges multiple
//! `ToolRegistry`s, and an `is_empty()` placeholder. Wiring the rmcp SDK for
//! actual stdio protocol handshake is the next focused task in v1.1.

use std::collections::HashMap;

use async_trait::async_trait;
use jarvis_core::events::ToolCall;
use jarvis_llm::tools::ToolRegistry;
use jarvis_llm::ToolSpec;
use serde::{Deserialize, Serialize};

/// One MCP server entry in config.
///
/// Example:
/// ```toml
/// [mcp.servers.fs]
/// command = "npx"
/// args = ["@modelcontextprotocol/server-filesystem", "/home/sycatle/Documents"]
/// env = { READ_ONLY = "1" }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerCfg {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Optional tool-name prefix to disambiguate when several servers expose
    /// the same tool name. Defaults to the server's config key.
    #[serde(default)]
    pub prefix: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpCfg {
    /// Keyed by short name → server spec.
    pub servers: HashMap<String, McpServerCfg>,
}

/// Registry of remote tools exposed by configured MCP servers.
///
/// Today this is inert: it holds the configs and reports an empty tool list.
/// Once the rmcp transport lands, `connect()` will spawn each subprocess,
/// perform the handshake, and populate `tools`.
pub struct McpRegistry {
    cfg: McpCfg,
    tools: Vec<ToolSpec>,
}

impl McpRegistry {
    pub fn new(cfg: McpCfg) -> Self {
        Self {
            cfg,
            tools: Vec::new(),
        }
    }

    pub fn server_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.cfg.servers.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn is_empty(&self) -> bool {
        self.cfg.servers.is_empty()
    }

    /// Placeholder: real implementation will spawn each subprocess, run the
    /// MCP `initialize` handshake, call `tools/list`, and populate
    /// `self.tools` with the discovered specs.
    pub async fn connect(&mut self) -> anyhow::Result<()> {
        if self.cfg.servers.is_empty() {
            return Ok(());
        }
        tracing::warn!(
            servers = ?self.server_names(),
            "MCP config loaded but rmcp transport not wired yet; servers will not be reachable",
        );
        Ok(())
    }
}

#[async_trait]
impl ToolRegistry for McpRegistry {
    fn specs(&self) -> Vec<ToolSpec> {
        self.tools.clone()
    }

    async fn invoke(&self, call: &ToolCall) -> serde_json::Value {
        serde_json::json!({
            "ok": false,
            "error": format!(
                "MCP tool '{}' is configured but not yet connected (rmcp transport pending)",
                call.name
            ),
        })
    }
}

/// Combine multiple `ToolRegistry`s into one. Tool specs are unioned; on name
/// collision, earlier registries win. Dispatch tries each registry in order
/// until one recognises the tool.
pub struct CompositeToolRegistry {
    registries: Vec<Box<dyn ToolRegistry>>,
}

impl CompositeToolRegistry {
    pub fn new() -> Self {
        Self {
            registries: Vec::new(),
        }
    }

    pub fn push(&mut self, registry: Box<dyn ToolRegistry>) -> &mut Self {
        self.registries.push(registry);
        self
    }

    pub fn with(mut self, registry: Box<dyn ToolRegistry>) -> Self {
        self.registries.push(registry);
        self
    }
}

impl Default for CompositeToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolRegistry for CompositeToolRegistry {
    fn specs(&self) -> Vec<ToolSpec> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in &self.registries {
            for spec in r.specs() {
                if seen.insert(spec.name.clone()) {
                    out.push(spec);
                }
            }
        }
        out
    }

    async fn invoke(&self, call: &ToolCall) -> serde_json::Value {
        for r in &self.registries {
            if r.specs().iter().any(|s| s.name == call.name) {
                return r.invoke(call).await;
            }
        }
        serde_json::json!({
            "ok": false,
            "error": format!("no registry handles tool: {}", call.name),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticRegistry {
        specs: Vec<ToolSpec>,
    }

    #[async_trait]
    impl ToolRegistry for StaticRegistry {
        fn specs(&self) -> Vec<ToolSpec> {
            self.specs.clone()
        }
        async fn invoke(&self, call: &ToolCall) -> serde_json::Value {
            serde_json::json!({"ok": true, "from": "static", "name": call.name})
        }
    }

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: format!("{name} tool"),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn composite_unions_specs_first_wins_on_collision() {
        let a = Box::new(StaticRegistry {
            specs: vec![spec("alpha"), spec("shared")],
        });
        let b = Box::new(StaticRegistry {
            specs: vec![spec("shared"), spec("beta")],
        });
        let comp = CompositeToolRegistry::new().with(a).with(b);
        let names: Vec<_> = comp.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["alpha", "shared", "beta"]);
    }

    #[tokio::test]
    async fn composite_dispatches_to_first_match() {
        let a = Box::new(StaticRegistry {
            specs: vec![spec("alpha")],
        });
        let b = Box::new(StaticRegistry {
            specs: vec![spec("beta")],
        });
        let comp = CompositeToolRegistry::new().with(a).with(b);

        let v = comp
            .invoke(&ToolCall {
                name: "beta".into(),
                arguments: serde_json::json!({}),
            })
            .await;
        assert_eq!(v["from"], "static");
        assert_eq!(v["name"], "beta");

        let miss = comp
            .invoke(&ToolCall {
                name: "gamma".into(),
                arguments: serde_json::json!({}),
            })
            .await;
        assert_eq!(miss["ok"], false);
    }

    #[test]
    fn mcp_registry_is_empty_when_no_servers() {
        let r = McpRegistry::new(McpCfg::default());
        assert!(r.is_empty());
        assert!(r.specs().is_empty());
    }

    #[test]
    fn mcp_cfg_roundtrips_through_toml() {
        let toml_src = r#"
[servers.fs]
command = "npx"
args = ["@modelcontextprotocol/server-filesystem", "/tmp"]

[servers.gh]
command = "gh-mcp"
prefix = "github"
"#;
        let cfg: McpCfg = toml::from_str(toml_src).expect("parse");
        assert_eq!(cfg.servers.len(), 2);
        assert_eq!(cfg.servers["fs"].command, "npx");
        assert_eq!(cfg.servers["gh"].prefix.as_deref(), Some("github"));
    }
}
