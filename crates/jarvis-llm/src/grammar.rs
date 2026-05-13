//! GBNF generator constraining LLM output to a known JSON shape.
//!
//! Two top-level alternatives:
//! - `{"response": "<text>"}`
//! - `{"tool_call": {"name": "<one-of-known-skills>", "arguments": { ... }}}`
//!
//! The arguments shape is given by the JSON schema attached to each
//! [`ToolSpec`]. We do not deeply validate the schema in grammar; we constrain
//! to a generic JSON object and rely on a serde validation step after parsing.

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema object
}

pub fn build_gbnf(tools: &[ToolSpec]) -> String {
    let mut s = String::new();
    s.push_str("root ::= response-obj | tool-call-obj\n");
    s.push_str("response-obj ::= \"{\\\"response\\\": \" json-string \"}\"\n");
    if tools.is_empty() {
        // No tools — fall back to a never-matching alternative so the grammar
        // still parses while constraining the LLM to JSON responses only.
        s.push_str("tool-call-obj ::= \"<<no-tool>>\"\n");
    } else {
        s.push_str("tool-call-obj ::= \"{\\\"tool_call\\\": {\\\"name\\\": \" tool-name \", \\\"arguments\\\": \" json-object \"}}\"\n");
        s.push_str("tool-name ::= ");
        for (i, t) in tools.iter().enumerate() {
            if i > 0 {
                s.push_str(" | ");
            }
            s.push('\"');
            s.push('\\');
            s.push('\"');
            s.push_str(&t.name);
            s.push('\\');
            s.push('\"');
            s.push('\"');
        }
        s.push('\n');
    }
    // Generic JSON primitives — small grammar lifted from the llama.cpp examples.
    s.push_str(
        r##"
json-value ::= json-object | json-array | json-string | json-number | "true" | "false" | "null"
json-object ::= "{" ws ( json-string ws ":" ws json-value ( "," ws json-string ws ":" ws json-value )* )? ws "}"
json-array ::= "[" ws ( json-value ( "," ws json-value )* )? ws "]"
json-string ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\""
json-number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
ws ::= [ \t\n]*
"##,
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_contains_root_and_tool_names() {
        let tools = vec![ToolSpec {
            name: "volume".into(),
            description: "Adjust output volume.".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let g = build_gbnf(&tools);
        assert!(g.contains("root ::="));
        assert!(g.contains("volume"));
    }

    #[test]
    fn empty_tools_falls_back_safely() {
        let g = build_gbnf(&[]);
        assert!(g.contains("response-obj"));
        assert!(g.contains("tool-call-obj"));
    }
}
