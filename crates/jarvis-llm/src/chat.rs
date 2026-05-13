//! Chat history with the Qwen2.5 ChatML template.

use serde::{Deserialize, Serialize};

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
}
