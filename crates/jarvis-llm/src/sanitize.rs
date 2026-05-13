//! Make LLM output safe for a text-to-speech engine to read aloud.
//!
//! Local Qwen-style models still emit markdown emphasis (`*word*`, `**word**`),
//! stage directions (`*Souffle*`, `*Porte qui claque*`), code fences and
//! occasional JSON wrappers even when the system prompt forbids them. A
//! synthesiser reads these characters literally ("astérisque, astérisque,
//! souffle, astérisque"), so we strip them just before speaking.

use std::sync::OnceLock;

use regex::Regex;

fn re_emphasis() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // **bold** / __bold__ / *italic* / _italic_ → inner text only.
    R.get_or_init(|| Regex::new(r"(?:\*\*|__)(.+?)(?:\*\*|__)|(?:\*|_)([^*_\n]+?)(?:\*|_)").unwrap())
}

fn re_code_block() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?s)```[a-zA-Z0-9_-]*\n?(.*?)```").unwrap())
}

fn re_inline_code() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap())
}

fn re_header() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?m)^#{1,6}\s+").unwrap())
}

fn re_bullet() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?m)^\s*[-*+]\s+").unwrap())
}

fn re_numbered() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?m)^\s*\d+\.\s+").unwrap())
}

fn re_json_wrapper() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // {"response": "..."} → inner string. Tolerant of whitespace + quotes.
    R.get_or_init(|| {
        Regex::new(r#"^\s*\{\s*"response"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}\s*$"#).unwrap()
    })
}

fn re_residual_stars() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[*_]+").unwrap())
}

fn re_emoji_misc() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // Strip common pictographs / dingbats. Not exhaustive, but cheap.
    R.get_or_init(|| {
        Regex::new(
            r"[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2700}-\u{27BF}]",
        )
        .unwrap()
    })
}

fn re_collapse_ws() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[ \t]+").unwrap())
}

/// Strip markdown, stage directions and JSON wrappers so the result reads
/// cleanly when passed to a TTS engine. Always returns a non-empty string when
/// the input contained any speakable content; otherwise the input is returned
/// trimmed (callers should still guard against an empty TTS payload).
pub fn for_tts(input: &str) -> String {
    let mut s = input.trim().to_string();

    // 1. Unwrap `{"response": "..."}` if the LLM wrapped its answer in JSON.
    if let Some(cap) = re_json_wrapper().captures(&s) {
        if let Some(inner) = cap.get(1) {
            let unescaped = inner
                .as_str()
                .replace("\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\t", " ");
            s = unescaped;
        }
    }

    // 2. Drop fenced code blocks entirely (TTS shouldn't read code).
    s = re_code_block().replace_all(&s, " ").to_string();

    // 3. Inline `code` → code.
    s = re_inline_code().replace_all(&s, "$1").to_string();

    // 4. Markdown emphasis → inner text.
    //    Replaced twice to catch nested cases (`***foo***` etc).
    for _ in 0..3 {
        let replaced = re_emphasis()
            .replace_all(&s, |caps: &regex::Captures<'_>| {
                caps.get(1)
                    .or_else(|| caps.get(2))
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default()
            })
            .to_string();
        if replaced == s {
            break;
        }
        s = replaced;
    }

    // 5. Headers / bullets / numbered list markers.
    s = re_header().replace_all(&s, "").to_string();
    s = re_bullet().replace_all(&s, "").to_string();
    s = re_numbered().replace_all(&s, "").to_string();

    // 6. Residual stand-alone stars/underscores from malformed markdown.
    s = re_residual_stars().replace_all(&s, "").to_string();

    // 7. Emoji / dingbats.
    s = re_emoji_misc().replace_all(&s, "").to_string();

    // 8. Collapse whitespace.
    s = re_collapse_ws().replace_all(&s, " ").to_string();
    s = s.replace(" \n", "\n").replace("\n ", "\n");

    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_stage_directions() {
        assert_eq!(for_tts("*Porte ouverte*"), "Porte ouverte");
        assert_eq!(for_tts("*Souffle*"), "Souffle");
        assert_eq!(for_tts("Bonjour *Sir*."), "Bonjour Sir.");
    }

    #[test]
    fn strips_bold_italic() {
        assert_eq!(for_tts("**Bonjour** Sir."), "Bonjour Sir.");
        assert_eq!(for_tts("Voici un mot _important_."), "Voici un mot important.");
    }

    #[test]
    fn unwraps_json_response() {
        assert_eq!(
            for_tts(r#"{"response": "Il est 14h00."}"#),
            "Il est 14h00."
        );
    }

    #[test]
    fn handles_plain_text() {
        assert_eq!(for_tts("Bonjour Sir."), "Bonjour Sir.");
    }

    #[test]
    fn strips_code_fences() {
        let out = for_tts("Voici :\n```rust\nfn main() {}\n```\nFini.");
        assert!(!out.contains("```"));
        assert!(!out.contains("fn main"));
    }

    #[test]
    fn strips_bullets_and_numbers() {
        assert_eq!(for_tts("- premier\n- deuxième"), "premier\ndeuxième");
        assert_eq!(for_tts("1. un\n2. deux"), "un\ndeux");
    }

    #[test]
    fn strips_emoji() {
        assert_eq!(for_tts("Compris 👍"), "Compris");
    }
}
