//! Voice-activity detection and utterance buffering.
//!
//! Today's implementation is energy-based with hysteresis — enough to drive
//! the wake-to-STT capture loop without needing Silero ONNX bundled. The
//! [`UtteranceCollector`] API matches what the orchestrator expects, so the
//! Silero swap-in (phase 5+) is a constructor change only.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Nothing significant — still inside (or outside) speech, no transition.
    Idle,
    /// Speech just started.
    SpeechStart,
    /// Speech is ongoing.
    SpeechContinue,
    /// Speech just ended (silence ≥ `silence_ms`).
    SpeechEnd,
}

#[derive(Debug, Clone, Copy)]
pub struct EnergyVadConfig {
    pub sample_rate: u32,
    pub frame_ms: u32,
    /// RMS threshold to enter speech.
    pub start_rms: f32,
    /// RMS threshold to remain in speech (hysteresis).
    pub continue_rms: f32,
    pub min_speech_ms: u32,
    pub silence_ms: u32,
    pub max_utterance_s: u32,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_ms: 30,
            start_rms: 0.04,
            continue_rms: 0.015,
            min_speech_ms: 250,
            silence_ms: 600,
            max_utterance_s: 20,
        }
    }
}

pub struct EnergyVad {
    cfg: EnergyVadConfig,
    in_speech: bool,
    speech_started_at: Option<Instant>,
    last_voiced_at: Option<Instant>,
}

impl EnergyVad {
    pub fn new(cfg: EnergyVadConfig) -> Self {
        Self {
            cfg,
            in_speech: false,
            speech_started_at: None,
            last_voiced_at: None,
        }
    }

    pub fn reset(&mut self) {
        self.in_speech = false;
        self.speech_started_at = None;
        self.last_voiced_at = None;
    }

    pub fn process_frame(&mut self, samples: &[f32]) -> VadEvent {
        let rms = rms_of(samples);
        let now = Instant::now();
        let threshold = if self.in_speech {
            self.cfg.continue_rms
        } else {
            self.cfg.start_rms
        };
        let voiced = rms >= threshold;

        if voiced {
            self.last_voiced_at = Some(now);
        }

        if !self.in_speech {
            if voiced {
                self.in_speech = true;
                self.speech_started_at = Some(now);
                return VadEvent::SpeechStart;
            }
            return VadEvent::Idle;
        }

        // Inside speech.
        let started_at = self.speech_started_at.unwrap_or(now);
        let last_voiced = self.last_voiced_at.unwrap_or(started_at);
        let silence_dur = now.duration_since(last_voiced);
        let speech_dur = now.duration_since(started_at);

        let silence_long_enough = silence_dur
            >= Duration::from_millis(self.cfg.silence_ms as u64)
            && speech_dur >= Duration::from_millis(self.cfg.min_speech_ms as u64);
        let max_reached =
            speech_dur >= Duration::from_secs(self.cfg.max_utterance_s as u64);
        if silence_long_enough || max_reached {
            self.in_speech = false;
            self.speech_started_at = None;
            self.last_voiced_at = None;
            VadEvent::SpeechEnd
        } else {
            VadEvent::SpeechContinue
        }
    }
}

fn rms_of(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Buffers audio across VAD events into one utterance returned at `SpeechEnd`.
pub struct UtteranceCollector {
    vad: EnergyVad,
    buffer: Vec<f32>,
    capacity_samples: usize,
}

impl UtteranceCollector {
    pub fn new(vad: EnergyVad) -> Self {
        let capacity_samples =
            (vad.cfg.sample_rate as u64 * vad.cfg.max_utterance_s as u64) as usize;
        Self {
            vad,
            buffer: Vec::with_capacity(capacity_samples),
            capacity_samples,
        }
    }

    /// Feed a frame. Returns `Some(utterance)` at speech-end.
    pub fn push(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let event = self.vad.process_frame(samples);
        match event {
            VadEvent::SpeechStart => {
                self.buffer.clear();
                self.buffer.extend_from_slice(samples);
                None
            }
            VadEvent::SpeechContinue => {
                if self.buffer.len() + samples.len() <= self.capacity_samples {
                    self.buffer.extend_from_slice(samples);
                }
                None
            }
            VadEvent::SpeechEnd => {
                self.buffer.extend_from_slice(samples);
                let take = std::mem::take(&mut self.buffer);
                Some(take)
            }
            VadEvent::Idle => None,
        }
    }

    pub fn reset(&mut self) {
        self.vad.reset();
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * ((i as f32) * 0.3).sin()).collect()
    }

    #[test]
    fn rms_basics() {
        assert_eq!(rms_of(&silence(100)), 0.0);
        assert!(rms_of(&tone(100, 0.5)) > 0.1);
    }

    #[test]
    fn vad_transitions() {
        let cfg = EnergyVadConfig {
            min_speech_ms: 0,
            silence_ms: 50,
            ..Default::default()
        };
        let mut v = EnergyVad::new(cfg);
        assert_eq!(v.process_frame(&silence(480)), VadEvent::Idle);
        assert_eq!(v.process_frame(&tone(480, 0.6)), VadEvent::SpeechStart);
        assert_eq!(v.process_frame(&tone(480, 0.6)), VadEvent::SpeechContinue);
        std::thread::sleep(Duration::from_millis(70));
        assert_eq!(v.process_frame(&silence(480)), VadEvent::SpeechEnd);
    }
}
