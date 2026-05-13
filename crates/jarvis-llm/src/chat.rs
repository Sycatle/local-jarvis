//! Chat history with the Qwen2.5 ChatML template.

use serde::{Deserialize, Serialize};

use crate::grammar::ToolSpec;

/// Augment `base_prompt` with the Qwen2.5 native tool-calling preamble. The
/// model expects function signatures inside `<tools>...</tools>` followed by
/// instructions to emit `<tool_call>{...}</tool_call>` blocks. The signatures
/// follow the OpenAI function-calling JSON shape Qwen was trained on.
///
/// The block is kept in English because Qwen2.5 was post-trained with this
/// exact template; the user-facing reply still follows whatever language the
/// caller specified earlier in `base_prompt`.
pub fn format_system_with_tools(base_prompt: &str, tools: &[ToolSpec]) -> String {
    if tools.is_empty() {
        return base_prompt.to_string();
    }
    let mut out = String::with_capacity(base_prompt.len() + 1024);
    out.push_str(base_prompt);
    out.push_str(
        "\n\n# Tools\n\n\
        You have access to the following functions. Function signatures are \
        inside <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        let entry = serde_json::json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        });
        // Compact single-line JSON per Qwen template convention.
        out.push_str(&entry.to_string());
        out.push('\n');
    }
    out.push_str(
        "</tools>\n\n\
        For each function call, return a JSON object with the function name \
        and arguments inside <tool_call></tool_call> XML tags:\n\
        <tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n",
    );
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(s: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: s.into(),
        }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: s.into(),
        }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: s.into(),
        }
    }
    pub fn tool(s: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: s.into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatHistory {
    messages: Vec<ChatMessage>,
}

impl ChatHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system(system_prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::system(system_prompt)],
        }
    }

    pub fn push(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Drop the oldest non-system turns so the rendered prompt stays bounded.
    /// Keeps the first message (assumed to be `system`) and the most recent
    /// `keep_pairs * 2` user/assistant/tool messages. A `keep_pairs` of 0 is
    /// treated as 1 to avoid wiping the entire conversation right before a
    /// generate call.
    pub fn truncate_keeping_system(&mut self, keep_pairs: usize) {
        let max_tail = keep_pairs.max(1) * 2;
        if self.messages.len() <= 1 + max_tail {
            return;
        }
        let tail_start = self.messages.len() - max_tail;
        let tail: Vec<ChatMessage> = self.messages.drain(tail_start..).collect();
        // Truncate everything except the first (system) message, then re-append
        // the tail. A `tool` message at the head of the tail would orphan its
        // assistant call — drop until we see a `user` to anchor the window.
        self.messages.truncate(1);
        let mut anchored = false;
        for m in tail {
            if !anchored {
                if matches!(m.role, Role::User) {
                    anchored = true;
                } else {
                    continue;
                }
            }
            self.messages.push(m);
        }
    }

    /// Render the conversation in Qwen2.5 ChatML format, ending with the
    /// `<|im_start|>assistant` opener so the model continues from there.
    pub fn render_chatml(&self) -> String {
        let mut out = String::new();
        for m in &self.messages {
            out.push_str("<|im_start|>");
            out.push_str(m.role.as_str());
            out.push('\n');
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_round_trip_shape() {
        let mut h = ChatHistory::with_system("You are Jarvis.");
        h.push(ChatMessage::user("Bonjour"));
        let s = h.render_chatml();
        assert!(s.starts_with("<|im_start|>system\nYou are Jarvis.<|im_end|>\n"));
        assert!(s.contains("<|im_start|>user\nBonjour<|im_end|>\n"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn truncate_keeps_system_and_recent_tail() {
        let mut h = ChatHistory::with_system("sys");
        for i in 0..10 {
            h.push(ChatMessage::user(format!("u{i}")));
            h.push(ChatMessage::assistant(format!("a{i}")));
        }
        h.truncate_keeping_system(3);
        let msgs = h.messages();
        assert_eq!(msgs.len(), 1 + 6);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].content, "sys");
        assert_eq!(msgs[1].content, "u7");
        assert_eq!(msgs.last().unwrap().content, "a9");
    }

    #[test]
    fn truncate_skips_orphan_tool_at_head() {
        // If the truncation window starts on a `tool` message, that tool result
        // has lost its preceding assistant tool-call and would confuse the
        // model. The dangling tool message must be dropped until a user anchor.
        let mut h = ChatHistory::with_system("sys");
        h.push(ChatMessage::user("u0"));
        h.push(ChatMessage::assistant("a0"));
        h.push(ChatMessage::tool("orphan-tool"));
        h.push(ChatMessage::user("u1"));
        h.push(ChatMessage::assistant("a1"));
        h.truncate_keeping_system(1);
        let msgs = h.messages();
        // System + u1 + a1 — orphan tool dropped, oldest pair dropped.
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].content, "u1");
        assert_eq!(msgs[2].content, "a1");
    }

    #[test]
    fn format_system_with_tools_injects_block() {
        let tools = vec![
            ToolSpec {
                name: "volume".into(),
                description: "Set sound volume".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolSpec {
                name: "notify".into(),
                description: "Send desktop notification".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let out = format_system_with_tools("Base prompt.", &tools);
        assert!(out.starts_with("Base prompt.\n\n# Tools"));
        assert!(out.contains("<tools>"));
        assert!(out.contains("</tools>"));
        assert!(out.contains("\"name\":\"volume\""));
        assert!(out.contains("\"name\":\"notify\""));
        assert!(out.contains("<tool_call>"));
        assert!(out.contains("</tool_call>"));
    }

    #[test]
    fn format_system_with_tools_passthrough_when_empty() {
        let out = format_system_with_tools("Just system.", &[]);
        assert_eq!(out, "Just system.");
    }

    #[test]
    fn truncate_no_op_when_already_small() {
        let mut h = ChatHistory::with_system("sys");
        h.push(ChatMessage::user("u0"));
        h.push(ChatMessage::assistant("a0"));
        h.truncate_keeping_system(8);
        assert_eq!(h.messages().len(), 3);
    }
}
