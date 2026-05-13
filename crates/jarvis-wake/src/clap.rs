//! Double-clap detector.
//!
//! Algorithm (mirrors the original Python POC):
//! 1. Band-pass the input to 1.5–6 kHz (where claps concentrate).
//! 2. Compute per-block RMS energy.
//! 3. Maintain a slow-moving baseline (EMA) of the band-passed energy.
//! 4. A "clap onset" fires when the block energy exceeds
//!    `onset_factor * baseline` AND a minimum absolute threshold.
//! 5. Two onsets within `[min_gap_ms, max_gap_ms]` yield a `Clap` event.
//! 6. A refractory period after a detection prevents triple-fires.

use std::time::{Duration, Instant};

/// 4th-order Butterworth band-pass parameters at 44.1 kHz, 1.5–6 kHz.
///
/// Implemented as a cascade of two biquads (low-pass at 6 kHz then high-pass
/// at 1.5 kHz). Coefficients computed offline (cookbook formulae) and kept
/// here as constants so the runtime path stays allocation-free.
#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        // Direct Form II transposed.
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// 2nd-order Butterworth high-pass at `fc` for sample rate `sr`.
    fn highpass(fc: f32, sr: f32) -> Self {
        let w0 = std::f32::consts::TAU * fc / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2.recip());
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// 2nd-order Butterworth low-pass at `fc` for sample rate `sr`.
    fn lowpass(fc: f32, sr: f32) -> Self {
        let w0 = std::f32::consts::TAU * fc / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2.recip());
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClapConfig {
    pub sample_rate: u32,
    pub band_low_hz: f32,
    pub band_high_hz: f32,
    /// Energy must rise to this fraction of recent baseline to count as onset.
    pub onset_factor: f32,
    /// Absolute floor on band-passed RMS for an onset.
    pub onset_floor: f32,
    /// EMA smoothing of the baseline. Closer to 1.0 = slower.
    pub baseline_smoothing: f32,
    pub min_gap_ms: u64,
    pub max_gap_ms: u64,
    pub refractory_ms: u64,
}

impl Default for ClapConfig {
    fn default() -> Self {
        // Hardened against keyboard clicks and other transients: higher
        // onset_factor + floor, faster baseline adaptation (key bursts
        // raise it and self-suppress), wider min gap (rapid keystrokes
        // are usually <180 ms apart).
        Self {
            sample_rate: 44_100,
            band_low_hz: 1_500.0,
            band_high_hz: 6_000.0,
            onset_factor: 6.0,
            onset_floor: 0.05,
            baseline_smoothing: 0.985,
            min_gap_ms: 180,
            max_gap_ms: 600,
            refractory_ms: 800,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ClapOutcome {
    /// No event for this block.
    Silence,
    /// First onset registered, waiting for the second.
    SingleClap,
    /// Two onsets within the allowed gap.
    DoubleClap,
}

pub struct ClapDetector {
    cfg: ClapConfig,
    hp: Biquad,
    lp: Biquad,
    baseline: f32,
    last_onset: Option<Instant>,
    last_double: Option<Instant>,
}

impl ClapDetector {
    pub fn new(cfg: ClapConfig) -> Self {
        let sr = cfg.sample_rate as f32;
        Self {
            hp: Biquad::highpass(cfg.band_low_hz, sr),
            lp: Biquad::lowpass(cfg.band_high_hz, sr),
            cfg,
            baseline: 0.01,
            last_onset: None,
            last_double: None,
        }
    }

    pub fn reset(&mut self) {
        self.hp.reset();
        self.lp.reset();
        self.baseline = 0.01;
        self.last_onset = None;
        self.last_double = None;
    }

    /// Feed one block of mono f32 samples (any size). Returns the event for
    /// this block. Internal state evolves regardless of block size.
    pub fn process_block(&mut self, samples: &[f32]) -> ClapOutcome {
        // 1. Band-pass + accumulate squared energy.
        let mut sum_sq = 0.0f32;
        for &x in samples {
            let y = self.lp.process(self.hp.process(x));
            sum_sq += y * y;
        }
        let rms = (sum_sq / samples.len().max(1) as f32).sqrt();

        // 2. Slow-moving baseline.
        let a = self.cfg.baseline_smoothing;
        self.baseline = a * self.baseline + (1.0 - a) * rms;

        // 3. Onset detection.
        let onset = rms > self.cfg.onset_floor && rms > self.cfg.onset_factor * self.baseline;

        if !onset {
            return ClapOutcome::Silence;
        }

        let now = Instant::now();
        // Suppress if still in refractory after a recent double-clap.
        if let Some(last) = self.last_double {
            if now.duration_since(last) < Duration::from_millis(self.cfg.refractory_ms) {
                return ClapOutcome::Silence;
            }
        }

        match self.last_onset {
            Some(prev) => {
                let gap = now.duration_since(prev);
                if gap >= Duration::from_millis(self.cfg.min_gap_ms)
                    && gap <= Duration::from_millis(self.cfg.max_gap_ms)
                {
                    self.last_onset = None;
                    self.last_double = Some(now);
                    ClapOutcome::DoubleClap
                } else if gap > Duration::from_millis(self.cfg.max_gap_ms) {
                    // Too late: start a fresh single-clap window from now.
                    self.last_onset = Some(now);
                    ClapOutcome::SingleClap
                } else {
                    // Too close (debounce): ignore.
                    ClapOutcome::Silence
                }
            }
            None => {
                self.last_onset = Some(now);
                ClapOutcome::SingleClap
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn burst(n: usize, amp: f32) -> Vec<f32> {
        // Synthesise a brief 3 kHz tone — squarely in the band.
        let sr = 44_100.0;
        let f = 3_000.0;
        (0..n)
            .map(|i| amp * (std::f32::consts::TAU * f * (i as f32) / sr).sin())
            .collect()
    }

    #[test]
    fn silence_yields_no_event() {
        let mut d = ClapDetector::new(ClapConfig::default());
        for _ in 0..50 {
            assert!(matches!(
                d.process_block(&silence(1024)),
                ClapOutcome::Silence
            ));
        }
    }

    #[test]
    fn double_clap_within_window_is_detected() {
        let mut d = ClapDetector::new(ClapConfig::default());
        // Warm up baseline at near-zero.
        for _ in 0..50 {
            let _ = d.process_block(&silence(1024));
        }
        // First clap.
        let first = d.process_block(&burst(1024, 0.6));
        assert!(matches!(
            first,
            ClapOutcome::SingleClap | ClapOutcome::Silence
        ));
        // Wait > min_gap.
        std::thread::sleep(Duration::from_millis(200));
        // Second clap.
        let second = d.process_block(&burst(1024, 0.6));
        assert!(matches!(second, ClapOutcome::DoubleClap));
    }
}
