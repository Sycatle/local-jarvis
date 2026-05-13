//! Wake detection: double-clap (DSP-only) and voice wake-word ("Jarvis").
//!
//! - [`clap`]: real-time double-clap detector — band-passed energy onsets
//!   within a configurable window, mirroring the parameters of the original
//!   Python POC (44.1 kHz / 1024-sample blocks / 1.5–6 kHz band).
//! - [`wakeword`]: openWakeWord ONNX wrapper running on `ort` behind the
//!   `openwakeword` cargo feature. The manager still handles the disabled or
//!   misconfigured case gracefully (feature off, model files missing, or
//!   capture sample rate ≠ wake-word rate).
//! - [`manager`]: multiplexes both, owns the audio subscription, exposes
//!   start/pause/resume/stop and a `WakeEvent` stream.

pub mod clap;
pub mod manager;
pub mod wakeword;

pub use clap::{ClapConfig, ClapDetector};
pub use manager::WakeManager;
pub use wakeword::{WakeWordConfig, WakeWordDetector};
