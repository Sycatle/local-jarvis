//! Streaming speak pipeline.
//!
//! Bridges the LLM token stream (chunked into sentences by
//! [`jarvis_llm::sentence_stream`]) to the TTS backend, with two simultaneous
//! syntheses in flight at most. The first chunk's PCM hits the player as soon
//! as it's ready, so the user hears audio while later sentences are still
//! being synthesised.
//!
//! `tracing` spans on `target = "jarvis::latency"` mark the time-to-first-
//! audio boundary so we can measure the win against the buffered baseline.

use std::sync::Arc;
use std::time::Instant;

use futures::stream::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use jarvis_tts::piper::TtsBackend;

/// Sink for synthesised PCM. Production uses `PcmPlayer`; tests use a
/// recording sink.
pub trait PcmSink: Send + Sync {
    fn write(&self, pcm: &[i16]);
}

impl PcmSink for jarvis_audio::PcmPlayer {
    fn write(&self, pcm: &[i16]) {
        jarvis_audio::PcmPlayer::write(self, pcm)
    }
}

/// Drive a stream of sentence chunks through the TTS and the player.
///
/// - Up to two syntheses run concurrently (`buffered(2)`): one playing, one
///   being prepared. Raising this further didn't help in practice — the
///   bottleneck is single-threaded ONNX inference.
/// - The first successful chunk emits a `tts_first_audio` info-level log
///   carrying `elapsed_ms`. Pair it with a `llm_first_token` span at the
///   caller to read end-to-end latency from journald.
/// - Cancellation is honoured at chunk boundaries — current synthesis runs to
///   completion (the cancel token is also forwarded into each `synthesise`
///   call, so backends that respect it can bail earlier).
///
/// Returns the number of audio chunks that reached the player.
pub async fn speak_stream<S, P>(
    chunks: S,
    tts: Arc<dyn TtsBackend>,
    sink: Arc<P>,
    cancel: CancellationToken,
) -> usize
where
    S: Stream<Item = anyhow::Result<String>> + Send + 'static,
    P: PcmSink + ?Sized + 'static,
{
    let start = Instant::now();
    let mut first_audio_logged = false;
    let mut played = 0usize;

    let synth_stream = chunks
        .filter_map(|res| async move { res.ok() })
        .map(|chunk| {
            let tts = Arc::clone(&tts);
            let cancel = cancel.clone();
            async move {
                let out = tts.synthesise(&chunk, cancel).await;
                (chunk, out)
            }
        })
        .buffered(2);

    futures::pin_mut!(synth_stream);
    while let Some((text, res)) = synth_stream.next().await {
        if cancel.is_cancelled() {
            tracing::debug!(target: "jarvis::latency", "speak_stream cancelled before chunk {played}");
            break;
        }
        match res {
            Ok((pcm, _sr)) => {
                if !first_audio_logged {
                    tracing::info!(
                        target: "jarvis::latency",
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        chunk_chars = text.len(),
                        "tts_first_audio",
                    );
                    first_audio_logged = true;
                }
                sink.write(&pcm);
                played += 1;
            }
            Err(e) => {
                tracing::warn!(chunk = %text, "tts chunk error: {e}");
            }
        }
    }
    played
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::stream;
    use jarvis_tts::piper::{PiperTtsError, TtsBackend};
    use std::sync::Mutex;

    struct RecordingSink {
        chunks: Mutex<Vec<Vec<i16>>>,
    }
    impl RecordingSink {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                chunks: Mutex::new(Vec::new()),
            })
        }
        fn snapshot(&self) -> Vec<Vec<i16>> {
            self.chunks.lock().unwrap().clone()
        }
    }
    impl PcmSink for RecordingSink {
        fn write(&self, pcm: &[i16]) {
            self.chunks.lock().unwrap().push(pcm.to_vec());
        }
    }

    struct CountingTts {
        calls: Mutex<Vec<String>>,
    }
    impl CountingTts {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
            })
        }
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }
    #[async_trait]
    impl TtsBackend for CountingTts {
        async fn synthesise(
            &self,
            text: &str,
            _cancel: CancellationToken,
        ) -> Result<(Vec<i16>, u32), PiperTtsError> {
            self.calls.lock().unwrap().push(text.to_string());
            // Encode the length of the text into the PCM payload so the test
            // can prove ordering preserved across the buffered() pipeline.
            let n = text.len() as i16;
            Ok((vec![n, n, n], 22_050))
        }
    }

    fn text_stream(items: &[&str]) -> impl Stream<Item = anyhow::Result<String>> {
        let v: Vec<anyhow::Result<String>> = items.iter().map(|s| Ok(s.to_string())).collect();
        stream::iter(v)
    }

    #[tokio::test]
    async fn forwards_chunks_in_order() {
        let tts = CountingTts::new();
        let sink = RecordingSink::new();
        let chunks = text_stream(&["alpha.", "beta.", "gamma."]);
        let n = speak_stream(
            chunks,
            Arc::clone(&tts) as Arc<dyn TtsBackend>,
            Arc::clone(&sink),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(n, 3);
        // Sink PCM tagged with chunk length: alpha. = 6, beta. = 5, gamma. = 6.
        let recorded = sink.snapshot();
        assert_eq!(recorded.len(), 3);
        assert_eq!(recorded[0][0], 6);
        assert_eq!(recorded[1][0], 5);
        assert_eq!(recorded[2][0], 6);
        // And TTS saw the texts in the right order.
        assert_eq!(tts.calls(), vec!["alpha.", "beta.", "gamma."]);
    }

    #[tokio::test]
    async fn empty_stream_plays_nothing() {
        let tts = CountingTts::new();
        let sink = RecordingSink::new();
        let n = speak_stream(
            text_stream(&[]),
            Arc::clone(&tts) as Arc<dyn TtsBackend>,
            Arc::clone(&sink),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(n, 0);
        assert!(sink.snapshot().is_empty());
    }

    #[tokio::test]
    async fn cancellation_stops_dispatching() {
        let tts = CountingTts::new();
        let sink = RecordingSink::new();
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancel — first chunk still flushes, loop bails after
        let n = speak_stream(
            text_stream(&["one.", "two.", "three."]),
            Arc::clone(&tts) as Arc<dyn TtsBackend>,
            Arc::clone(&sink),
            cancel,
        )
        .await;
        // With pre-cancelled token, the loop checks cancel before writing —
        // no chunks should reach the sink.
        assert_eq!(n, 0);
        assert!(sink.snapshot().is_empty());
    }
}
