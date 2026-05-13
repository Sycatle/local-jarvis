//! Kokoro-82M TTS backend via ONNX Runtime.
//!
//! Pipeline: text → IPA phonemes (espeak-ng) → tokens (vocab) → ONNX
//! (`input_ids`, `style`, `speed`) → 24 kHz f32 waveform → i16 PCM.
//!
//! Long text (>510 tokens after phonemisation) is split on strong punctuation
//! (`.`, `!`, `?`, `;`) and the resulting waveforms are concatenated with a
//! 50 ms silence between segments.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::phonemize::{PhonemiseError, Phonemiser};
use crate::piper::{PiperTtsError, TtsBackend};

#[cfg(feature = "kokoro")]
use ort::session::{builder::GraphOptimizationLevel, Session};

const SAMPLE_RATE: u32 = 24_000;
const STYLE_DIM: usize = 256;
const MAX_VOICE_LEN: usize = 510;

#[derive(Clone, Debug)]
pub struct KokoroConfig {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub voices_dir: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub language: String,
    /// Request the CUDA execution provider for ONNX Runtime. Effective only
    /// when the `cuda` cargo feature is enabled; CPU fallback otherwise.
    pub gpu: bool,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            tokenizer: PathBuf::new(),
            voices_dir: PathBuf::new(),
            voice: "im_nicola".into(),
            speed: 1.0,
            language: "fr".into(),
            gpu: true,
        }
    }
}

#[derive(Debug, Error)]
pub enum KokoroError {
    #[error("kokoro model not found at {0:?}")]
    ModelMissing(PathBuf),
    #[error("voice file not found at {0:?}")]
    VoiceMissing(PathBuf),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] crate::kokoro_tokens::TokenError),
    #[error("phonemise error: {0}")]
    Phonemise(#[from] PhonemiseError),
    #[error("ort error: {0}")]
    Ort(String),
    #[error("voice file has wrong size {actual} (expected multiple of {STYLE_DIM} * 4)")]
    BadVoiceSize { actual: usize },
    #[error("empty input")]
    EmptyInput,
}

impl From<KokoroError> for PiperTtsError {
    fn from(e: KokoroError) -> Self {
        match e {
            KokoroError::ModelMissing(p) => PiperTtsError::ModelMissing(p),
            KokoroError::VoiceMissing(p) => PiperTtsError::ModelMissing(p),
            other => PiperTtsError::Synthesis(other.to_string()),
        }
    }
}

#[cfg(feature = "kokoro")]
pub struct KokoroTts {
    session: Arc<Mutex<Session>>,
    tokenizer: crate::kokoro_tokens::KokoroTokenizer,
    voice: Arc<Vec<Vec<f32>>>, // voice[k] = style vector for token-count k, len STYLE_DIM
    speed: f32,
    language: String,
    phonemiser: Arc<dyn Phonemiser>,
}

#[cfg(feature = "kokoro")]
impl KokoroTts {
    pub fn load(cfg: &KokoroConfig, phonemiser: Arc<dyn Phonemiser>) -> Result<Self, KokoroError> {
        if !cfg.model.is_file() {
            return Err(KokoroError::ModelMissing(cfg.model.clone()));
        }
        let voice_path = cfg.voices_dir.join(format!("{}.bin", cfg.voice));
        if !voice_path.is_file() {
            return Err(KokoroError::VoiceMissing(voice_path));
        }

        let mut builder = Session::builder()
            .map_err(|e| KokoroError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| KokoroError::Ort(e.to_string()))?;

        // Keep ORT's thread pools tight: Kokoro is small (~82M params), and
        // we don't want it stealing cores from llama.cpp during streaming.
        builder = builder
            .with_intra_threads(2)
            .map_err(|e| KokoroError::Ort(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| KokoroError::Ort(e.to_string()))?;

        // Execution providers: try CUDA first (only compiled in with the
        // `cuda` feature), then fall back to CPU. ORT silently skips any
        // EP whose shared library is missing at runtime.
        #[cfg(feature = "cuda")]
        if cfg.gpu {
            use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
            builder = builder
                .with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ])
                .map_err(|e| KokoroError::Ort(e.to_string()))?;
        }

        let session = builder
            .commit_from_file(&cfg.model)
            .map_err(|e| KokoroError::Ort(e.to_string()))?;

        let tokenizer = crate::kokoro_tokens::KokoroTokenizer::load(&cfg.tokenizer)?;

        let voice = load_voice_bin(&voice_path)?;

        tracing::info!(
            "Kokoro loaded: model={:?} voice={} ({} length-slots)",
            cfg.model,
            cfg.voice,
            voice.len(),
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            voice: Arc::new(voice),
            speed: cfg.speed.clamp(0.5, 2.0),
            language: cfg.language.clone(),
            phonemiser,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    async fn synth_segment(&self, phonemes: &str) -> Result<Vec<f32>, KokoroError> {
        let raw_tokens = self.tokenizer.encode(phonemes);
        if raw_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let n = raw_tokens.len().min(MAX_VOICE_LEN);
        let raw_tokens = &raw_tokens[..n];

        // [0, ...tokens..., 0]
        let mut padded = Vec::with_capacity(n + 2);
        padded.push(0i64);
        padded.extend_from_slice(raw_tokens);
        padded.push(0);

        let style_idx = n.min(self.voice.len().saturating_sub(1));
        let style_vec = self.voice[style_idx].clone();
        let speed = self.speed;

        let session = Arc::clone(&self.session);
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, KokoroError> {
            let mut guard = session
                .lock()
                .map_err(|e| KokoroError::Ort(format!("session poisoned: {e}")))?;
            run_session(&mut guard, &padded, &style_vec, speed)
        })
        .await
        .map_err(|e| KokoroError::Ort(format!("join: {e}")))??;

        Ok(result)
    }
}

#[cfg(feature = "kokoro")]
#[async_trait]
impl TtsBackend for KokoroTts {
    async fn synthesise(
        &self,
        text: &str,
        cancel: CancellationToken,
    ) -> Result<(Vec<i16>, u32), PiperTtsError> {
        if text.trim().is_empty() {
            return Ok((Vec::new(), SAMPLE_RATE));
        }
        let mut all = Vec::<f32>::new();
        let silence: Vec<f32> = vec![0.0; (SAMPLE_RATE as usize) / 20]; // 50 ms

        for segment in split_segments(text) {
            if cancel.is_cancelled() {
                break;
            }
            let ipa = self
                .phonemiser
                .phonemise(segment, &self.language)
                .await
                .map_err(KokoroError::from)
                .map_err(PiperTtsError::from)?;
            if ipa.is_empty() {
                continue;
            }
            let audio = self
                .synth_segment(&ipa)
                .await
                .map_err(PiperTtsError::from)?;
            if !all.is_empty() && !audio.is_empty() {
                all.extend_from_slice(&silence);
            }
            all.extend(audio);
        }

        // f32 [-1, 1] → i16 PCM with clamping.
        let pcm: Vec<i16> = all
            .into_iter()
            .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        Ok((pcm, SAMPLE_RATE))
    }
}

#[cfg(feature = "kokoro")]
fn run_session(
    session: &mut Session,
    tokens: &[i64],
    style: &[f32],
    speed: f32,
) -> Result<Vec<f32>, KokoroError> {
    use ndarray::Array;
    use ort::value::Tensor;

    let n = tokens.len();
    let input_ids_arr = Array::from_shape_vec((1, n), tokens.to_vec())
        .map_err(|e| KokoroError::Ort(format!("input_ids shape: {e}")))?;
    let style_arr = Array::from_shape_vec((1, STYLE_DIM), style.to_vec())
        .map_err(|e| KokoroError::Ort(format!("style shape: {e}")))?;
    let speed_arr = Array::from_vec(vec![speed]);

    let input_ids = Tensor::from_array(input_ids_arr)
        .map_err(|e| KokoroError::Ort(format!("input_ids tensor: {e}")))?;
    let style_t = Tensor::from_array(style_arr)
        .map_err(|e| KokoroError::Ort(format!("style tensor: {e}")))?;
    let speed_t = Tensor::from_array(speed_arr)
        .map_err(|e| KokoroError::Ort(format!("speed tensor: {e}")))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => input_ids,
            "style" => style_t,
            "speed" => speed_t,
        ])
        .map_err(|e| KokoroError::Ort(format!("run: {e}")))?;

    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| KokoroError::Ort(format!("extract: {e}")))?;
    Ok(data.to_vec())
}

#[cfg(feature = "kokoro")]
fn load_voice_bin(path: &Path) -> Result<Vec<Vec<f32>>, KokoroError> {
    let bytes = std::fs::read(path).map_err(|e| KokoroError::Ort(format!("read voice: {e}")))?;
    if bytes.len() % (STYLE_DIM * 4) != 0 {
        return Err(KokoroError::BadVoiceSize {
            actual: bytes.len(),
        });
    }
    let n_slots = bytes.len() / (STYLE_DIM * 4);
    let mut out = Vec::with_capacity(n_slots);
    for k in 0..n_slots {
        let mut v = Vec::with_capacity(STYLE_DIM);
        let base = k * STYLE_DIM * 4;
        for i in 0..STYLE_DIM {
            let off = base + i * 4;
            v.push(f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
        }
        out.push(v);
    }
    Ok(out)
}

/// Split a text on strong terminal punctuation. Keeps the terminator with the
/// preceding segment so eSpeak phonemises it as part of the intonation.
fn split_segments(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if matches!(c, '.' | '!' | '?' | ';') {
            let seg = text[start..=i].trim();
            if !seg.is_empty() {
                out.push(seg);
            }
            start = i + 1;
        }
        i += 1;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    if out.is_empty() {
        out.push(text.trim());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_terminal_punct() {
        let segs = split_segments("Bonjour Sir. Comment allez-vous ? Très bien !");
        assert_eq!(segs.len(), 3);
        assert!(segs[0].ends_with('.'));
        assert!(segs[1].ends_with('?'));
        assert!(segs[2].ends_with('!'));
    }

    #[test]
    fn split_handles_no_punct() {
        let segs = split_segments("simple texte");
        assert_eq!(segs, vec!["simple texte"]);
    }
}
