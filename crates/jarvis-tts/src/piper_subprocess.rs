//! Piper subprocess backend.
//!
//! Wraps the prebuilt `piper` binary (https://github.com/rhasspy/piper):
//! - feed `text\n` on stdin
//! - read raw 16-bit signed-LE mono PCM on stdout (sample rate from the
//!   voice's `.onnx.json` config — we read it once at construction)
//!
//! Returns the full PCM buffer to the orchestrator, which pushes it into the
//! cpal-backed [`jarvis_audio::PcmPlayer`] at the matching sample rate.
//! That gives us a native Rust playback path with proper barge-in (the
//! player stops immediately on `cancel`), without bundling ONNX Runtime
//! into the binary.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::piper::{PiperTtsError, TtsBackend};

#[derive(Debug, Deserialize)]
struct VoiceConfig {
    audio: AudioCfg,
}
#[derive(Debug, Deserialize)]
struct AudioCfg {
    sample_rate: u32,
}

pub struct PiperSubprocess {
    bin: PathBuf,
    model: PathBuf,
    sample_rate: u32,
}

impl PiperSubprocess {
    pub fn try_new(bin: PathBuf, model: PathBuf, config: PathBuf) -> Option<Self> {
        if !bin.is_file() || !model.is_file() || !config.is_file() {
            return None;
        }
        let raw = std::fs::read_to_string(&config).ok()?;
        let parsed: VoiceConfig = serde_json::from_str(&raw).ok()?;
        Some(Self {
            bin,
            model,
            sample_rate: parsed.audio.sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Common system locations for the piper binary.
    pub fn discover_bin() -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let candidates = [
            PathBuf::from(format!("{home}/.local/share/jarvis/piper_bin/piper/piper")),
            PathBuf::from(format!("{home}/.local/bin/piper")),
            PathBuf::from("/usr/local/bin/piper"),
            PathBuf::from("/usr/bin/piper"),
        ];
        candidates.into_iter().find(|p| p.is_file())
    }

    pub fn discover(model: &Path, config: &Path) -> Option<Self> {
        let bin = Self::discover_bin()?;
        Self::try_new(bin, model.to_path_buf(), config.to_path_buf())
    }
}

#[async_trait]
impl TtsBackend for PiperSubprocess {
    async fn synthesise(
        &self,
        text: &str,
        cancel: CancellationToken,
    ) -> Result<(Vec<i16>, u32), PiperTtsError> {
        let mut child = Command::new(&self.bin)
            .args(["--model"])
            .arg(&self.model)
            .arg("--output-raw")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| PiperTtsError::Synthesis(format!("spawn piper: {e}")))?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| PiperTtsError::Synthesis("no piper stdin".into()))?;
        let text_owned = format!("{text}\n");
        let write_task = tokio::spawn(async move {
            let _ = stdin.write_all(text_owned.as_bytes()).await;
            drop(stdin);
        });

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| PiperTtsError::Synthesis("no piper stdout".into()))?;

        let mut bytes = Vec::with_capacity(64 * 1024);
        let read_loop = async {
            stdout.read_to_end(&mut bytes).await
        };

        tokio::select! {
            res = read_loop => {
                res.map_err(|e| PiperTtsError::Synthesis(e.to_string()))?;
            }
            _ = cancel.cancelled() => {
                let _ = child.kill().await;
                let _ = write_task.await;
                return Ok((Vec::new(), self.sample_rate));
            }
        }

        let _ = child.wait().await;
        let _ = write_task.await;

        let pcm: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok((pcm, self.sample_rate))
    }
}
