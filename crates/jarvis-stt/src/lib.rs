//! Speech-to-text pipeline.
//!
//! - [`vad`]: utterance boundary detector. Ships an energy-based fallback
//!   today; Silero ONNX wiring lands when `ort` is plumbed.
//! - [`whisper`]: transcription via `whisper-rs`. Stubbed until the model is
//!   present on disk; the API is fully specified so the orchestrator can be
//!   built against it.

pub mod vad;
pub mod whisper;

pub use vad::{EnergyVad, UtteranceCollector, VadEvent};
pub use whisper::{WhisperConfig, WhisperStt, WhisperSttError};
