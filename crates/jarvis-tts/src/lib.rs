//! Text-to-speech backends.
//!
//! - **Kokoro-82M** (default, feature `kokoro`): ONNX via `ort`, IPA tokens via
//!   `espeak-ng` subprocess, 24 kHz output. Voice masculine FR par défaut
//!   (`im_nicola` + phonèmes français).
//! - **Piper** (feature `piper`, stub): kept as fallback for low-resource use.
//! - **espeak-ng subprocess** (`EspeakNgBackend`): degraded mode, plays via
//!   default sink without returning PCM.

pub mod espeak;
pub mod kokoro;
pub mod kokoro_tokens;
pub mod phonemize;
pub mod piper;
pub mod piper_subprocess;

pub use espeak::EspeakNgBackend;
#[cfg(feature = "kokoro")]
pub use kokoro::{KokoroConfig, KokoroError, KokoroTts};
pub use kokoro_tokens::{KokoroTokenizer, TokenError};
pub use phonemize::{EspeakNgPhonemiser, PhonemiseError, Phonemiser, StubPhonemiser};
pub use piper::{PiperConfig, PiperTts, PiperTtsError, TtsBackend};
pub use piper_subprocess::PiperSubprocess;
