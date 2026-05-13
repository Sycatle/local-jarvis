use std::path::PathBuf;

use async_trait::async_trait;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Error)]
pub enum PiperTtsError {
    #[error("voice model not found at {0:?}")]
    ModelMissing(PathBuf),
    #[error("TTS backend not yet wired (Piper ONNX feature pending)")]
    NotImplemented,
    #[error("synthesis failed: {0}")]
    Synthesis(String),
}

#[derive(Clone, Debug)]
pub struct PiperConfig {
    pub voice_model: PathBuf,
    pub voice_config: PathBuf,
    pub language: String,
    pub length_scale: f32,
    pub noise_scale: f32,
    pub noise_w: f32,
}

impl Default for PiperConfig {
    fn default() -> Self {
        Self {
            voice_model: PathBuf::new(),
            voice_config: PathBuf::new(),
            language: "fr".into(),
            length_scale: 1.0,
            noise_scale: 0.667,
            noise_w: 0.8,
        }
    }
}

#[async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesise `text` into a single PCM mono i16 buffer (no streaming yet).
    /// `cancel` allows callers to interrupt long syntheses.
    async fn synthesise(
        &self,
        text: &str,
        cancel: CancellationToken,
    ) -> Result<(Vec<i16>, u32), PiperTtsError>;
}

pub struct PiperTts {
    cfg: PiperConfig,
    available: bool,
}

impl PiperTts {
    pub fn new(cfg: PiperConfig) -> Self {
        let available = cfg.voice_model.is_file() && cfg.voice_config.is_file();
        if !available {
            tracing::warn!(
                "Piper voice not found at {:?} — TTS disabled",
                cfg.voice_model
            );
        }
        Self { cfg, available }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn config(&self) -> &PiperConfig {
        &self.cfg
    }
}

#[async_trait]
impl TtsBackend for PiperTts {
    async fn synthesise(
        &self,
        _text: &str,
        _cancel: CancellationToken,
    ) -> Result<(Vec<i16>, u32), PiperTtsError> {
        if !self.available {
            return Err(PiperTtsError::ModelMissing(self.cfg.voice_model.clone()));
        }
        Err(PiperTtsError::NotImplemented)
    }
}
