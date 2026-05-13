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

impl WakeManager {
    /// Start the manager. Subscribes to `capture` at its native sample rate;
    /// the clap detector operates on that rate directly. If `capture` is not
    /// 44.1 kHz, clap detection thresholds may need tuning.
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
        let mut ww_det = WakeWordDetector::new(wakeword_cfg);

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
                if let Some(c) = ww_det.process(&frame) {
                    let _ = tx
                        .send(WakeEvent {
                            source: WakeSource::WakeWord,
                            confidence: c,
                        })
                        .await;
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
