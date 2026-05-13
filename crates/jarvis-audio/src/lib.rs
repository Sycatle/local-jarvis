//! Audio capture, playback, and notification sounds.
//!
//! - [`ding`]: short 880 Hz attention chime
//! - [`Player`]: blocking PCM playback at a given sample rate
//! - [`PcmPlayer`]: streaming PCM playback with interrupt (barge-in ready)
//! - [`Capture`]: microphone capture into a `tokio::sync::broadcast` channel
//!
//! Backend: `cpal` (PortAudio-style host abstraction). On Pop!_OS this talks
//! to PipeWire via its ALSA / PulseAudio compat layer.

pub mod capture;
pub mod ding;
pub mod player;

pub use capture::{Capture, CaptureError, CaptureFrame};
pub use ding::play_ding;
pub use player::{PcmPlayer, PlaybackError, Player};
