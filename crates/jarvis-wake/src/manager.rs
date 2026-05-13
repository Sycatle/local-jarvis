//! Wake manager: routes microphone frames into both detectors and emits
//! [`WakeEvent`]s on a `tokio::sync::mpsc` channel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use jarvis_audio::Capture;
use jarvis_core::events::{WakeEvent, WakeSource};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::clap::{ClapConfig, ClapDetector, ClapOutcome};
use crate::wakeword::{WakeWordConfig, WakeWordDetector};

pub struct WakeManager {
    paused: Arc<AtomicBool>,
    task: Option<JoinHandle<()>>,
    rx: mpsc::Receiver<WakeEvent>,
}

/// Decide whether the openWakeWord detector can safely consume frames at the
/// given capture sample rate. The ONNX pipeline expects 16 kHz mono; feeding
/// any other rate silently corrupts the mel/embedding output and produces
/// false negatives or positives that are nearly impossible to diagnose at
/// runtime. Returning `false` lets the manager skip the detector instead.
pub(crate) fn wakeword_accepts_rate(capture_rate: u32, cfg_rate: u32) -> bool {
    capture_rate == cfg_rate
}

impl WakeManager {
    /// Start the manager. Subscribes to `capture` at its native sample rate;
    /// the clap detector operates on that rate directly. If `capture` is not
    /// 44.1 kHz, clap detection thresholds may need tuning.
    ///
    /// The wake-word detector requires its configured sample rate (16 kHz by
    /// default) to match `capture.sample_rate()` exactly. On mismatch it is
    /// disabled at runtime and an error is logged — the manager keeps running
    /// with the clap detector alone rather than silently producing wrong
    /// scores.
    pub fn start(
        capture: &Capture,
        clap_cfg: Option<ClapConfig>,
        wakeword_cfg: WakeWordConfig,
    ) -> Self {
        let mut rx_audio = capture.subscribe();
        let (tx, rx) = mpsc::channel::<WakeEvent>(16);
        let paused = Arc::new(AtomicBool::new(false));
        let paused_t = Arc::clone(&paused);

        let mut clap_det = clap_cfg.map(ClapDetector::new);

        let capture_rate = capture.sample_rate();
        let ww_rate = wakeword_cfg.sample_rate;
        let mut ww_det = if wakeword_accepts_rate(capture_rate, ww_rate) {
            Some(WakeWordDetector::new(wakeword_cfg))
        } else {
            tracing::error!(
                capture_rate,
                wakeword_rate = ww_rate,
                "wake-word detector disabled: capture sample rate does not match \
                 the wake-word model's required rate; resample upstream or align \
                 [audio].sample_rate_capture with [wake.wakeword].sample_rate"
            );
            None
        };

        let task = tokio::spawn(async move {
            while let Ok(frame) = rx_audio.recv().await {
                if paused_t.load(Ordering::Acquire) {
                    continue;
                }
                if let Some(det) = clap_det.as_mut() {
                    if let ClapOutcome::DoubleClap = det.process_block(&frame) {
                        let _ = tx
                            .send(WakeEvent {
                                source: WakeSource::Clap,
                                confidence: 1.0,
                            })
                            .await;
                    }
                }
                if let Some(det) = ww_det.as_mut() {
                    if let Some(c) = det.process(&frame) {
                        let _ = tx
                            .send(WakeEvent {
                                source: WakeSource::WakeWord,
                                confidence: c,
                            })
                            .await;
                    }
                }
            }
        });

        Self {
            paused,
            task: Some(task),
            rx,
        }
    }

    pub fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }

    pub async fn recv(&mut self) -> Option<WakeEvent> {
        self.rx.recv().await
    }

    pub fn stop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

impl Drop for WakeManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wakeword_accepts_matching_rate() {
        assert!(wakeword_accepts_rate(16_000, 16_000));
    }

    #[test]
    fn wakeword_rejects_capture_at_clap_rate() {
        // Most likely real-world misconfiguration: a single capture stream
        // tuned to 44.1 kHz for clap detection silently feeding the 16 kHz
        // openWakeWord pipeline.
        assert!(!wakeword_accepts_rate(44_100, 16_000));
    }

    #[test]
    fn wakeword_rejects_any_non_matching_rate() {
        for rate in [8_000_u32, 22_050, 32_000, 48_000] {
            assert!(!wakeword_accepts_rate(rate, 16_000));
        }
    }
}
