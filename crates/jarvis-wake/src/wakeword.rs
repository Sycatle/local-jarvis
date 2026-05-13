//! Voice wake-word detector — openWakeWord pipeline.
//!
//! Three ONNX models chained per 80 ms (1280-sample @ 16 kHz) step:
//!   raw audio → melspectrogram → embedding → wake score.
//! Activated by the `openwakeword` cargo feature. When the feature is off
//! (or any of the three model files are missing), the detector returns
//! `None` for every chunk so the wake manager keeps running.

use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct WakeWordConfig {
    /// Path to the wake-word model (e.g. `hey_jarvis_v0.1.onnx`). The
    /// melspectrogram and embedding models are loaded from sibling files
    /// `melspectrogram.onnx` / `embedding_model.onnx` in the same dir.
    pub model: PathBuf,
    pub threshold: f32,
    pub cooldown_s: f32,
    pub sample_rate: u32,
    pub frame_samples: usize,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            threshold: 0.5,
            cooldown_s: 1.5,
            sample_rate: 16_000,
            frame_samples: 1_280,
        }
    }
}

pub struct WakeWordDetector {
    cfg: WakeWordConfig,
    last_fire: Option<Instant>,
    #[cfg(feature = "openwakeword")]
    state: Option<backend::State>,
}

impl WakeWordDetector {
    pub fn new(cfg: WakeWordConfig) -> Self {
        #[cfg(feature = "openwakeword")]
        let state = backend::State::try_new(&cfg);
        Self {
            #[cfg(feature = "openwakeword")]
            state,
            cfg,
            last_fire: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        #[cfg(feature = "openwakeword")]
        {
            self.state.is_some()
        }
        #[cfg(not(feature = "openwakeword"))]
        {
            false
        }
    }

    pub fn config(&self) -> &WakeWordConfig {
        &self.cfg
    }

    /// Feed 16 kHz mono f32 samples. Returns the wake confidence the first
    /// time a chunk's score crosses `threshold` (cooldown applies).
    pub fn process(&mut self, _samples: &[f32]) -> Option<f32> {
        #[cfg(feature = "openwakeword")]
        {
            let state = self.state.as_mut()?;
            if let Some(t) = self.last_fire {
                if t.elapsed() < Duration::from_secs_f32(self.cfg.cooldown_s) {
                    // Keep the pipeline ticking so rolling buffers don't
                    // freeze; just suppress the event.
                    let _ = state.step(_samples);
                    return None;
                }
            }
            let score = state.step(_samples)?;
            if score >= self.cfg.threshold {
                self.last_fire = Some(Instant::now());
                tracing::info!("wake-word fired (score={:.3})", score);
                Some(score)
            } else {
                None
            }
        }
        #[cfg(not(feature = "openwakeword"))]
        {
            None
        }
    }
}

#[cfg(feature = "openwakeword")]
mod backend {
    use super::WakeWordConfig;
    use ndarray::Array;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;
    use std::path::Path;

    const CHUNK_SAMPLES: usize = 1_280; // 80 ms @ 16 kHz
    const MEL_FEATURES: usize = 32;
    const MEL_WINDOW: usize = 76; // openWakeWord embedding input length
    const EMBED_DIM: usize = 96;
    const WAKE_WINDOW: usize = 16; // wake-model input length
    // Empirical: the mel model embeds an internal log-mel scaling that's
    // ~10× the value openWakeWord's embedding model expects. The reference
    // python pipeline divides by 10 before feeding the embedding model.
    const MEL_DIVISOR: f32 = 10.0;

    pub(super) struct State {
        mel: Session,
        embed: Session,
        wake: Session,
        mel_input: String,
        embed_input: String,
        wake_input: String,
        raw_buf: Vec<f32>,
        mel_buf: Vec<f32>, // flat row-major [n_frames, 32]
        emb_buf: Vec<f32>, // flat row-major [n_emb, 96]
    }

    impl State {
        pub(super) fn try_new(cfg: &WakeWordConfig) -> Option<Self> {
            if !cfg.model.is_file() {
                tracing::warn!(
                    "wake-word model not found at {:?} — voice wake-word disabled",
                    cfg.model
                );
                return None;
            }
            let dir = cfg.model.parent()?;
            let mel_path = dir.join("melspectrogram.onnx");
            let embed_path = dir.join("embedding_model.onnx");
            if !mel_path.is_file() || !embed_path.is_file() {
                tracing::warn!(
                    "openWakeWord support models missing in {:?} \
                     (need melspectrogram.onnx + embedding_model.onnx) — \
                     voice wake-word disabled",
                    dir
                );
                return None;
            }
            let mel = build_session(&mel_path).ok()?;
            let embed = build_session(&embed_path).ok()?;
            let wake = build_session(&cfg.model).ok()?;

            let mel_input = mel.inputs.first()?.name.clone();
            let embed_input = embed.inputs.first()?.name.clone();
            let wake_input = wake.inputs.first()?.name.clone();

            tracing::info!(
                "openWakeWord loaded: wake={:?} mel_in={:?} embed_in={:?} wake_in={:?}",
                cfg.model, mel_input, embed_input, wake_input
            );

            Some(Self {
                mel,
                embed,
                wake,
                mel_input,
                embed_input,
                wake_input,
                raw_buf: Vec::with_capacity(CHUNK_SAMPLES * 4),
                mel_buf: Vec::with_capacity(MEL_FEATURES * MEL_WINDOW * 2),
                emb_buf: Vec::with_capacity(EMBED_DIM * WAKE_WINDOW * 2),
            })
        }

        pub(super) fn step(&mut self, samples: &[f32]) -> Option<f32> {
            self.raw_buf.extend_from_slice(samples);
            let mut last_score: Option<f32> = None;

            while self.raw_buf.len() >= CHUNK_SAMPLES {
                let chunk: Vec<f32> = self.raw_buf.drain(..CHUNK_SAMPLES).collect();
                let new_mel = match run_mel(&mut self.mel, &self.mel_input, &chunk) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("mel run failed: {e}");
                        return last_score;
                    }
                };
                self.mel_buf.extend_from_slice(&new_mel);

                let cap_frames = MEL_WINDOW + 32;
                let cur_frames = self.mel_buf.len() / MEL_FEATURES;
                if cur_frames > cap_frames {
                    let drop = (cur_frames - cap_frames) * MEL_FEATURES;
                    self.mel_buf.drain(..drop);
                }

                if self.mel_buf.len() >= MEL_WINDOW * MEL_FEATURES {
                    let start = self.mel_buf.len() - MEL_WINDOW * MEL_FEATURES;
                    let win = self.mel_buf[start..].to_vec();
                    let emb = match run_embed(&mut self.embed, &self.embed_input, &win) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("embed run failed: {e}");
                            return last_score;
                        }
                    };
                    self.emb_buf.extend_from_slice(&emb);

                    let cap_emb = (WAKE_WINDOW + 4) * EMBED_DIM;
                    if self.emb_buf.len() > cap_emb {
                        let drop = self.emb_buf.len() - cap_emb;
                        self.emb_buf.drain(..drop);
                    }

                    if self.emb_buf.len() >= WAKE_WINDOW * EMBED_DIM {
                        let start = self.emb_buf.len() - WAKE_WINDOW * EMBED_DIM;
                        let win = self.emb_buf[start..].to_vec();
                        match run_wake(&mut self.wake, &self.wake_input, &win) {
                            Ok(s) => last_score = Some(s),
                            Err(e) => tracing::warn!("wake run failed: {e}"),
                        }
                    }
                }
            }
            last_score
        }
    }

    fn build_session(path: &Path) -> Result<Session, ort::Error> {
        // Wake-word models are tiny (~few MB); a single intra-op thread is
        // enough and leaves cores free for STT / LLM. Without this cap, ORT
        // spawns one worker per logical CPU and the wake loop spikes CPU
        // every 80 ms.
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(path)
    }

    fn run_mel(
        session: &mut Session,
        input_name: &str,
        chunk: &[f32],
    ) -> Result<Vec<f32>, ort::Error> {
        let arr = Array::from_shape_vec((1, chunk.len()), chunk.to_vec())
            .map_err(|e| ort::Error::new(format!("mel shape: {e}")))?;
        let tensor = Tensor::from_array(arr)?;
        let outputs = session.run(ort::inputs![input_name => tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v /= MEL_DIVISOR;
        }
        Ok(out)
    }

    fn run_embed(
        session: &mut Session,
        input_name: &str,
        mel_window: &[f32],
    ) -> Result<Vec<f32>, ort::Error> {
        let arr = Array::from_shape_vec(
            (1, MEL_WINDOW, MEL_FEATURES, 1),
            mel_window.to_vec(),
        )
        .map_err(|e| ort::Error::new(format!("embed shape: {e}")))?;
        let tensor = Tensor::from_array(arr)?;
        let outputs = session.run(ort::inputs![input_name => tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    fn run_wake(
        session: &mut Session,
        input_name: &str,
        embed_window: &[f32],
    ) -> Result<f32, ort::Error> {
        let arr = Array::from_shape_vec(
            (1, WAKE_WINDOW, EMBED_DIM),
            embed_window.to_vec(),
        )
        .map_err(|e| ort::Error::new(format!("wake shape: {e}")))?;
        let tensor = Tensor::from_array(arr)?;
        let outputs = session.run(ort::inputs![input_name => tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(*data.first().unwrap_or(&0.0))
    }
}
