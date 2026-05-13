//! `jarvis tts-preview` — synthesise a phrase with Kokoro and play it locally.
//!
//! Useful for A/B-testing voices (`im_nicola`, `bm_george`, `ff_siwis`, ...)
//! without restarting the daemon. Bypasses D-Bus and the orchestrator.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use jarvis_audio::PcmPlayer;
use jarvis_core::config::Config;
use jarvis_tts::{EspeakNgPhonemiser, KokoroConfig, KokoroTts, Phonemiser, TtsBackend};
use tokio_util::sync::CancellationToken;

pub async fn run(voice_override: Option<String>, text: &str) -> Result<()> {
    let cfg = Config::load()?;
    let voice = voice_override.unwrap_or_else(|| cfg.tts.kokoro.voice.clone());

    let kcfg = KokoroConfig {
        model: cfg.tts.kokoro.model.clone(),
        tokenizer: cfg.tts.kokoro.tokenizer.clone(),
        voices_dir: cfg.tts.kokoro.voices_dir.clone(),
        voice: voice.clone(),
        speed: cfg.tts.kokoro.speed,
        language: cfg.tts.kokoro.language.clone(),
        gpu: cfg.tts.kokoro.gpu,
    };

    let phon: Arc<dyn Phonemiser> = Arc::new(EspeakNgPhonemiser::new());
    let tts = KokoroTts::load(&kcfg, phon).map_err(|e| anyhow!("Kokoro load: {e}"))?;

    println!("voice={voice}  lang={}  text={text:?}", kcfg.language);
    let t0 = Instant::now();
    let (pcm, sr) = tts
        .synthesise(text, CancellationToken::new())
        .await
        .map_err(|e| anyhow!("synth: {e}"))?;
    let synth_ms = t0.elapsed().as_millis();
    let audio_ms = (pcm.len() as f64 / sr as f64 * 1000.0) as u64;
    let rtf = synth_ms as f64 / audio_ms.max(1) as f64;
    println!(
        "synth={synth_ms} ms  audio={audio_ms} ms  rtf={rtf:.2}  samples={}  sr={sr}",
        pcm.len()
    );

    let player = PcmPlayer::new(sr)?;
    player.write(&pcm);
    // Wait for the queue to drain.
    let est_ms = (pcm.len() as u64 * 1000 / sr.max(1) as u64) + 200;
    tokio::time::sleep(std::time::Duration::from_millis(est_ms)).await;
    Ok(())
}
