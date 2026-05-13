//! Sentence-level chunker over an LLM token stream.
//!
//! Bridges between the LLM (which emits arbitrary text fragments) and the TTS
//! (which synthesises whole utterances). The goal is to release the *first*
//! speakable chunk as soon as the model has produced a complete clause —
//! typically after one sentence — so the user hears audio while the rest of
//! the answer is still being generated. On a 200-token reply, this trims
//! time-to-first-audio from ~5 s (buffer-then-speak) to ~1 s.
//!
//! The chunker is purely textual; the actual TTS dispatch and audio playback
//! happen in `jarvis::streaming` (which lives in the binary crate so it has
//! access to the player).

use futures::stream::{Stream, StreamExt};

/// Break on these characters when the buffer has at least one full clause.
/// `;` is included because it routinely marks the end of an independent
/// breath group in spoken French.
const SENTENCE_TERMINATORS: &[char] = &['.', '!', '?', ';', '\n'];

/// Force a flush once the buffer grows past this many characters even without
/// a terminator — keeps latency bounded for run-on sentences.
const SOFT_FLUSH_AT: usize = 160;

/// Adapts a `Stream<Item = anyhow::Result<String>>` (token fragments) into a
/// `Stream<Item = anyhow::Result<String>>` of complete sentences.
///
/// - Whitespace at the boundary of a chunk is collapsed; empty chunks are
///   skipped.
/// - On a terminating punctuation, the buffer up to and including the
///   terminator is emitted.
/// - On `SOFT_FLUSH_AT` chars without a terminator, the next whitespace
///   triggers an emission to avoid unbounded buffering.
/// - When the upstream stream ends, any residual non-empty buffer is flushed.
/// - Errors from upstream are propagated as `Err` items in order.
pub fn sentence_stream<S>(input: S) -> impl Stream<Item = anyhow::Result<String>>
where
    S: Stream<Item = anyhow::Result<String>> + Send + 'static,
{
    async_stream::stream! {
        let mut buf = String::new();
        let mut input = Box::pin(input);
        while let Some(item) = input.next().await {
            match item {
                Err(e) => yield Err(e),
                Ok(token) => {
                    for ch in token.chars() {
                        buf.push(ch);
                        let hit_terminator = SENTENCE_TERMINATORS.contains(&ch);
                        let soft_flush = buf.len() >= SOFT_FLUSH_AT && ch.is_whitespace();
                        if hit_terminator || soft_flush {
                            let chunk = buf.trim().to_string();
                            buf.clear();
                            if !chunk.is_empty() {
                                yield Ok(chunk);
                            }
                        }
                    }
                }
            }
        }
        let tail = buf.trim().to_string();
        if !tail.is_empty() {
            yield Ok(tail);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    async fn collect_chunks<S>(s: S) -> Vec<String>
    where
        S: Stream<Item = anyhow::Result<String>>,
    {
        let mut s = Box::pin(s);
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.unwrap());
        }
        out
    }

    fn ok_tokens(tokens: &[&str]) -> impl Stream<Item = anyhow::Result<String>> {
        let v: Vec<anyhow::Result<String>> = tokens.iter().map(|s| Ok(s.to_string())).collect();
        stream::iter(v)
    }

    #[tokio::test]
    async fn splits_on_period() {
        let tokens = ok_tokens(&["Bonjour", " monde", ". Comment", " allez", "-vous", " ?"]);
        let out = collect_chunks(sentence_stream(tokens)).await;
        assert_eq!(out, vec!["Bonjour monde.", "Comment allez-vous ?"]);
    }

    #[tokio::test]
    async fn splits_on_question_and_exclamation() {
        let tokens = ok_tokens(&["Quoi ?! ", "Mais oui ! ", "Vraiment."]);
        let out = collect_chunks(sentence_stream(tokens)).await;
        assert_eq!(out, vec!["Quoi ?", "!", "Mais oui !", "Vraiment."]);
    }

    #[tokio::test]
    async fn flushes_residual_buffer_at_end() {
        let tokens = ok_tokens(&["sans", " point"]);
        let out = collect_chunks(sentence_stream(tokens)).await;
        assert_eq!(out, vec!["sans point"]);
    }

    #[tokio::test]
    async fn empty_stream_yields_nothing() {
        let out = collect_chunks(sentence_stream(ok_tokens(&[]))).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn soft_flush_breaks_run_on_text() {
        // 200-char run-on with spaces but no terminator.
        let long = "a ".repeat(120);
        let tokens = ok_tokens(&[&long]);
        let out = collect_chunks(sentence_stream(tokens)).await;
        // At least one flush should have occurred before the residual flush.
        assert!(
            out.len() >= 2,
            "expected ≥2 chunks for run-on input, got {out:?}"
        );
    }

    #[tokio::test]
    async fn newline_is_a_terminator() {
        let tokens = ok_tokens(&["Premier item\n", "Second item\n"]);
        let out = collect_chunks(sentence_stream(tokens)).await;
        assert_eq!(out, vec!["Premier item", "Second item"]);
    }
}
