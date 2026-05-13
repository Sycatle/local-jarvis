use crate::player::{PlaybackError, Player};

/// Play a short 880 Hz attention chime (~200 ms, exponential decay).
pub fn play_ding() -> Result<(), PlaybackError> {
    play_ding_with(880.0, 0.2, 22_050)
}

pub fn play_ding_with(freq: f32, duration_s: f32, sample_rate: u32) -> Result<(), PlaybackError> {
    let n = (duration_s * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n);
    let two_pi = std::f32::consts::TAU;
    let decay = 8.0; // Higher = shorter tail.
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let env = (-decay * t).exp();
        let v = env * (two_pi * freq * t).sin();
        let s = (v.clamp(-1.0, 1.0) * 0.7 * i16::MAX as f32) as i16;
        samples.push(s);
    }
    Player::play_blocking(&samples, sample_rate)
}
