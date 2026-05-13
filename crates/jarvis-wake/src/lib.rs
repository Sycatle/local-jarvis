//! Wake detection: double-clap (DSP-only) and voice wake-word ("Jarvis").
//!
//! - [`clap`]: real-time double-clap detector — band-passed energy onsets
//!   within a configurable window, mirroring the parameters of the original
//!   Python POC (44.1 kHz / 1024-sample blocks / 1.5–6 kHz band).
//! - [`wakeword`]: openWakeWord ONNX wrapper. Stubbed until the runtime is
//!   wired through `ort` in a later phase — the manager handles its absence
//!   gracefully.
//! - [`manager`]: multiplexes both, owns the audio subscription, exposes
//!   start/pause/resume/stop and a `WakeEvent` stream.

pub mod clap;
pub mod manager;
pub mod wakeword;

pub use clap::{ClapConfig, ClapDetector};
pub use manager::WakeManager;
pub use wakeword::{WakeWordConfig, WakeWordDetector};
