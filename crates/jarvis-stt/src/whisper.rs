//! whisper.cpp wrapper, behind the `whisper` cargo feature.
//!
//! When the feature is off (or the GGML model is missing), `transcribe`
//! returns a clear error and the orchestrator keeps the state machine alive.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WhisperSttError {
    #[error("model not found at {0:?}")]
    ModelMissing(PathBuf),
    #[error("whisper backend not yet wired (feature gate pending)")]
    NotImplemented,
    #[error("transcription failed: {0}")]
    Inference(String),
}

#[derive(Clone, Debug)]
pub struct WhisperConfig {
    pub model: PathBuf,
    pub language: String,
    pub n_threads: i32,
    pub translate: bool,
    /// Bias decoding away from silence-hallucination phrases. Defaults to a
    /// short French sentence so the model is primed for our domain.
    pub initial_prompt: String,
    /// Reject segments where the model assigns this probability or more to
    /// the "no-speech" token. 0.0 disables.
    pub no_speech_threshold: f32,
    /// Strip non-speech markers (`*Bip*`, `(rires)`, etc.) at decode time.
    pub suppress_non_speech: bool,
    /// >0 → request GPU offload from whisper.cpp (effective only when the
    /// `cuda` cargo feature is enabled at build time). 0 = CPU-only.
    pub n_gpu_layers: i32,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            language: "fr".into(),
            n_threads: 4,
            translate: false,
            initial_prompt: "Assistant vocal Jarvis en français. \
                Questions courtes et ordres système."
                .to_string(),
            no_speech_threshold: 0.6,
            suppress_non_speech: true,
            n_gpu_layers: 99,
        }
    }
}

#[cfg(feature = "whisper")]
mod backend {
    use super::{WhisperConfig, WhisperSttError};
    use std::sync::Mutex;
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    pub struct Backend {
        ctx: Mutex<WhisperContext>,
        cfg: WhisperConfig,
    }

    impl Backend {
        pub fn new(cfg: WhisperConfig) -> Result<Self, WhisperSttError> {
            let mut params = WhisperContextParameters::default();
            // whisper-rs 0.13 exposes a `use_gpu` flag; setting it true only
            // has effect when the `cuda` (or `metal`) feature was compiled in.
            params.use_gpu = cfg.n_gpu_layers > 0;
            let ctx = WhisperContext::new_with_params(cfg.model.to_string_lossy().as_ref(), params)
                .map_err(|e| WhisperSttError::Inference(format!("ctx init: {e}")))?;
            Ok(Self {
                ctx: Mutex::new(ctx),
                cfg,
            })
        }

        pub fn transcribe(&self, samples: &[f32]) -> Result<String, WhisperSttError> {
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_translate(self.cfg.translate);
            params.set_language(Some(self.cfg.language.as_str()));
            if self.cfg.n_threads > 0 {
                params.set_n_threads(self.cfg.n_threads);
            }
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_single_segment(true);
            // Anti-hallucination hardening: at decode time, suppress the
            // non-speech tokens that produce `*Bip*` / `(rires)`, drop blank
            // segments, lock to deterministic sampling, and reject low-
            // confidence detections.
            params.set_suppress_blank(true);
            params.set_suppress_non_speech_tokens(self.cfg.suppress_non_speech);
            params.set_no_speech_thold(self.cfg.no_speech_threshold);
            params.set_temperature(0.0);
            params.set_no_context(true);
            if !self.cfg.initial_prompt.is_empty() {
                params.set_initial_prompt(self.cfg.initial_prompt.as_str());
            }

            let ctx = self
                .ctx
                .lock()
                .map_err(|_| WhisperSttError::Inference("ctx poisoned".into()))?;
            let mut state = ctx
                .create_state()
                .map_err(|e| WhisperSttError::Inference(format!("state: {e}")))?;
            state
                .full(params, samples)
                .map_err(|e| WhisperSttError::Inference(format!("full: {e}")))?;

            let n_segments = state
                .full_n_segments()
                .map_err(|e| WhisperSttError::Inference(format!("n_segments: {e}")))?;
            let mut out = String::new();
            for i in 0..n_segments {
                let seg = state
                    .full_get_segment_text(i)
                    .map_err(|e| WhisperSttError::Inference(format!("seg {i}: {e}")))?;
                out.push_str(seg.trim());
                out.push(' ');
            }
            Ok(out.trim().to_string())
        }
    }
}

pub struct WhisperStt {
    cfg: WhisperConfig,
    available: bool,
    #[cfg(feature = "whisper")]
    backend: Option<backend::Backend>,
}

impl WhisperStt {
    pub fn new(cfg: WhisperConfig) -> Self {
        let available = cfg.model.is_file();
        if !available {
            tracing::warn!("whisper model not found at {:?} — STT disabled", cfg.model);
        }
        #[cfg(feature = "whisper")]
        let backend = if available {
            match backend::Backend::new(cfg.clone()) {
                Ok(b) => {
                    tracing::info!("whisper.cpp loaded ({:?})", cfg.model);
                    Some(b)
                }
                Err(e) => {
                    tracing::warn!("whisper backend init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        Self {
            cfg,
            available,
            #[cfg(feature = "whisper")]
            backend,
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    /// Transcribe a 16 kHz mono f32 utterance. Blocks on the calling thread;
    /// the orchestrator runs this inside `tokio::task::spawn_blocking`.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String, WhisperSttError> {
        if !self.available {
            return Err(WhisperSttError::ModelMissing(self.cfg.model.clone()));
        }
        #[cfg(feature = "whisper")]
        {
            match self.backend.as_ref() {
                Some(b) => b.transcribe(samples),
                None => Err(WhisperSttError::NotImplemented),
            }
        }
        #[cfg(not(feature = "whisper"))]
        {
            let _ = samples;
            Err(WhisperSttError::NotImplemented)
        }
    }
}
