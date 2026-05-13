//! `espeak-ng` subprocess backend.
//!
//! Used as a fallback while the Piper ONNX path isn't yet wired. Speech goes
//! straight to the default PulseAudio/PipeWire sink — we don't return PCM.
//! The trait still expects a `(Vec<i16>, u32)`, so we return an empty buffer
//! (the orchestrator's `player.write(&[])` is a no-op).

use async_trait::async_trait;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::piper::{PiperTtsError, TtsBackend};

pub struct EspeakNgBackend {
    voice: String,
    speed_wpm: u32,
}

impl EspeakNgBackend {
    pub fn new() -> Self {
        Self {
            voice: "fr".into(),
            speed_wpm: 170,
        }
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    pub fn with_speed(mut self, wpm: u32) -> Self {
        self.speed_wpm = wpm;
        self
    }
}

impl Default for EspeakNgBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsBackend for EspeakNgBackend {
    async fn synthesise(
        &self,
        text: &str,
        cancel: CancellationToken,
    ) -> Result<(Vec<i16>, u32), PiperTtsError> {
        let mut child = Command::new("espeak-ng")
            .args([
                "-v",
                &self.voice,
                "-s",
                &self.speed_wpm.to_string(),
                "--",
                text,
            ])
            .spawn()
            .map_err(|e| PiperTtsError::Synthesis(format!("spawn espeak-ng: {e}")))?;

        tokio::select! {
            res = child.wait() => {
                match res {
                    Ok(status) if status.success() => Ok((Vec::new(), 22_050)),
                    Ok(status) => Err(PiperTtsError::Synthesis(format!(
                        "espeak-ng exit {:?}", status.code()
                    ))),
                    Err(e) => Err(PiperTtsError::Synthesis(e.to_string())),
                }
            }
            _ = cancel.cancelled() => {
                let _ = child.kill().await;
                Ok((Vec::new(), 22_050))
            }
        }
    }
}
