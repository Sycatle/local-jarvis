//! Boot the orchestrator: load config, init every subsystem, hand off.

use std::sync::Arc;

use anyhow::Result;
use jarvis_audio::{Capture, PcmPlayer};
use jarvis_core::config::Config;
use jarvis_core::config::TtsEngine;
use jarvis_llm::engine::StubEngine;
use jarvis_llm::{ChatHistory, ChatMessage};
use jarvis_mcp::{
    CompositeToolRegistry, McpCfg as McpAdapterCfg, McpRegistry,
    McpServerCfg as McpAdapterServerCfg,
};
use jarvis_memory::Memory;
use jarvis_service::ServiceHandle;
use jarvis_skills::{MediaSkills, Skill, SkillRegistry, SystemSkills};
use jarvis_stt::whisper::WhisperConfig;
use jarvis_stt::WhisperStt;
use jarvis_tts::piper::{PiperConfig, PiperTts};
use jarvis_tts::EspeakNgPhonemiser;
use jarvis_wake::clap::ClapConfig;
use jarvis_wake::wakeword::WakeWordConfig;
use jarvis_wake::WakeManager;
use tokio::sync::{mpsc, watch};

use crate::orchestrator::Orchestrator;

pub async fn run() -> Result<()> {
    let config = Config::load()?;
    tracing::info!(
        "loaded config from {}",
        jarvis_core::dirs::config_file().display()
    );

    check_mic_gain();

    // Audio.
    let capture = Capture::start(config.audio.sample_rate_clap)?;
    // Player sample rate is selected to match the chosen TTS engine to avoid
    // having to resample at runtime: Kokoro=24k, Piper=22.05k.
    let tts_sample_rate = match config.tts.engine {
        TtsEngine::Kokoro => 24_000,
        TtsEngine::Piper | TtsEngine::Espeak => 22_050,
    };
    let player = Arc::new(PcmPlayer::new(tts_sample_rate)?);

    // Wake. The clap detector is off by default — it fires on keyboard
    // clicks and other broadband transients. Users opt back in via
    // `[wake.clap] enabled = true` once they've tuned thresholds for their
    // room.
    let clap_cfg = if config.wake.clap.enabled {
        Some(ClapConfig {
            sample_rate: config.audio.sample_rate_clap,
            ..Default::default()
        })
    } else {
        tracing::info!("clap wake disabled (config); use the wake word or Listen() to activate");
        None
    };
    let ww_cfg = WakeWordConfig {
        model: config.wake.wakeword.model.clone(),
        threshold: config.wake.wakeword.threshold,
        cooldown_s: config.wake.wakeword.cooldown_s,
        ..Default::default()
    };
    let wake = WakeManager::start(&capture, clap_cfg, ww_cfg);

    // Persistent memory store (opened before skills so `index_docs` can hold
    // an Arc<Memory>). Hydration of ChatHistory happens further down once the
    // system prompt is in place.
    let memory: Option<Arc<Memory>> = if config.memory.enabled {
        if let Some(parent) = config.memory.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match Memory::open(&config.memory.path) {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => {
                tracing::warn!(
                    "memory disabled — open {:?} failed: {e:#}",
                    config.memory.path
                );
                None
            }
        }
    } else {
        None
    };

    // Optional embedder for RAG. Stays `None` when the `embed` feature is off
    // or the model files are absent — RAG simply self-disables.
    let embedder: Option<Arc<dyn jarvis_memory::Embedder>> = load_embedder(&config.rag);

    // Desktop + skills. Honour the persistent disable list so users can hide
    // individual skills without recompiling.
    let desktop = jarvis_desktop::make_desktop().await?;
    let disabled = crate::skills_state::load().disabled;
    let mut registry = SkillRegistry::new();
    for s in SystemSkills::new(desktop).skills() {
        if disabled.contains(s.name()) {
            tracing::info!("skill disabled by state: {}", s.name());
            continue;
        }
        registry.register(s);
    }
    for s in MediaSkills::skills() {
        if disabled.contains(s.name()) {
            tracing::info!("skill disabled by state: {}", s.name());
            continue;
        }
        registry.register(s);
    }
    if let (Some(mem), Some(emb)) = (memory.as_ref(), embedder.as_ref()) {
        let idx = Arc::new(jarvis_skills::IndexDocsSkill::new(
            Arc::clone(mem),
            Arc::clone(emb),
            config.rag.corpus_dir.clone(),
            config.rag.chunk_tokens,
        ));
        if !disabled.contains(idx.name()) {
            registry.register(idx);
            tracing::info!(
                "rag: index_docs registered (corpus={:?}, top_k={})",
                config.rag.corpus_dir,
                config.rag.top_k
            );
        }
    }
    tracing::info!(
        "native skills registered: {:?}",
        registry
            .audit()
            .into_iter()
            .map(|(n, _)| n)
            .collect::<Vec<_>>()
    );
    let skills = Arc::new(registry);

    // MCP servers (config-only today; protocol wiring lands in the next pass).
    // We still build the composite registry so the orchestrator path is
    // identical whether MCP is in use or not.
    let mcp_cfg = config_to_mcp(&config);
    let mut mcp = McpRegistry::new(mcp_cfg);
    if let Err(e) = mcp.connect().await {
        tracing::warn!("MCP connect failed: {e:#}");
    }
    let tools: Arc<dyn jarvis_llm::tools::ToolRegistry> = if mcp.is_empty() {
        Arc::clone(&skills) as Arc<dyn jarvis_llm::tools::ToolRegistry>
    } else {
        // CompositeToolRegistry takes Box<dyn>, but our SkillRegistry is in an
        // Arc shared with the orchestrator for audit. Wrap a shared handle.
        let skills_handle = SkillsHandle(Arc::clone(&skills));
        let composite = CompositeToolRegistry::new()
            .with(Box::new(skills_handle))
            .with(Box::new(mcp));
        Arc::new(composite)
    };

    // LLM **avant** STT : whisper-rs embarque sa propre copie statique de
    // ggml, et llama-cpp-2/dynamic-link charge la sienne via libllama.so.
    // Charger whisper d'abord pousse les symboles `ggml_*` statiques en
    // global et fait segfault llama au moment de l'alloc des tenseurs.
    // Initialiser llama en premier garantit que libggml.so dynamique gagne
    // la résolution de symboles, et whisper utilise ensuite sa copie privée.
    let llm: Arc<dyn jarvis_llm::engine::LlmEngine> = {
        #[cfg(feature = "llama")]
        {
            if config.llm.model.is_file() {
                let cfg = jarvis_llm::LlamaConfig {
                    model: config.llm.model.clone(),
                    n_ctx: config.llm.n_ctx,
                    n_threads: config.llm.n_threads,
                    n_gpu_layers: config.llm.n_gpu_layers,
                    temperature: config.llm.temperature,
                    ..Default::default()
                };
                match jarvis_llm::LlamaEngine::load(cfg) {
                    Ok(e) => {
                        tracing::info!("LLM: llama.cpp ({} GPU layers)", config.llm.n_gpu_layers);
                        Arc::new(e) as Arc<dyn jarvis_llm::engine::LlmEngine>
                    }
                    Err(e) => {
                        tracing::warn!("llama.cpp failed to load ({e:#}) — using stub");
                        Arc::new(StubEngine::new())
                    }
                }
            } else {
                tracing::warn!(
                    "llama model not found at {:?} — using stub",
                    config.llm.model
                );
                Arc::new(StubEngine::new())
            }
        }
        #[cfg(not(feature = "llama"))]
        {
            Arc::new(StubEngine::new())
        }
    };

    // STT — chargé après llama pour éviter le conflit ggml.
    let stt_cfg = WhisperConfig {
        model: config.stt.model.clone(),
        language: config.stt.language.clone(),
        initial_prompt: config.stt.initial_prompt.clone(),
        no_speech_threshold: config.stt.no_speech_threshold,
        suppress_non_speech: config.stt.suppress_non_speech,
        n_threads: config.stt.n_threads,
        n_gpu_layers: config.stt.n_gpu_layers,
        ..Default::default()
    };
    let stt = Arc::new(WhisperStt::new(stt_cfg));

    // TTS dispatch: Kokoro (default v1.1) → Piper subprocess → espeak-ng.
    let tts: Arc<dyn jarvis_tts::piper::TtsBackend> = match config.tts.engine {
        TtsEngine::Kokoro => build_kokoro_or_fallback(&config).await,
        TtsEngine::Piper => build_piper_or_fallback(&config),
        TtsEngine::Espeak => {
            tracing::info!("TTS: espeak-ng (forced by config)");
            Arc::new(jarvis_tts::EspeakNgBackend::new())
        }
    };

    // D-Bus.
    let (cmd_tx, bus_rx) = mpsc::channel(16);
    let (state_tx, state_rx) = watch::channel("idle".to_string());
    let service = ServiceHandle::start(cmd_tx, state_rx).await?;

    let mut history = ChatHistory::with_system(&config.llm.system_prompt);
    if let Some(mem) = memory.as_ref() {
        if config.memory.hydrate_n > 0 {
            match mem.recent(config.memory.hydrate_n) {
                Ok(rows) => {
                    tracing::info!("memory: hydrating {} prior turns", rows.len());
                    for r in rows {
                        history.push(ChatMessage::user(r.user_text));
                        history.push(ChatMessage::assistant(r.assistant_text));
                    }
                }
                Err(e) => tracing::warn!("memory.recent failed: {e:#}"),
            }
        }
    }

    let orchestrator = Orchestrator {
        config,
        capture,
        player,
        wake,
        stt,
        llm,
        tts,
        skills,
        tools,
        state_tx,
        bus_rx,
        service,
        history,
        memory,
        embedder,
    };
    orchestrator.run().await
}

#[cfg(feature = "embed")]
fn load_embedder(rag: &jarvis_core::config::RagCfg) -> Option<Arc<dyn jarvis_memory::Embedder>> {
    use jarvis_memory::Embedder as _;
    if !rag.enabled {
        return None;
    }
    match jarvis_memory::bge::BgeSmallEmbedder::load(&rag.model, &rag.tokenizer) {
        Ok(e) => {
            tracing::info!(
                "rag: bge-small embedder loaded ({}d, model={:?})",
                e.dim(),
                rag.model
            );
            Some(Arc::new(e))
        }
        Err(e) => {
            tracing::warn!("rag: embedder unavailable ({e:#}) — RAG disabled");
            None
        }
    }
}

#[cfg(not(feature = "embed"))]
fn load_embedder(rag: &jarvis_core::config::RagCfg) -> Option<Arc<dyn jarvis_memory::Embedder>> {
    if rag.enabled {
        tracing::warn!("rag: enabled in config but `embed` feature not compiled — disabled");
    }
    None
}

/// Wrapper that exposes a shared `Arc<SkillRegistry>` as a `ToolRegistry`
/// without giving up the Arc — the orchestrator keeps the original for audit
/// listings while the composite registry borrows through this handle.
struct SkillsHandle(Arc<SkillRegistry>);

#[async_trait::async_trait]
impl jarvis_llm::tools::ToolRegistry for SkillsHandle {
    fn specs(&self) -> Vec<jarvis_llm::ToolSpec> {
        jarvis_llm::tools::ToolRegistry::specs(&*self.0)
    }
    async fn invoke(&self, call: &jarvis_core::events::ToolCall) -> serde_json::Value {
        jarvis_llm::tools::ToolRegistry::invoke(&*self.0, call).await
    }
}

fn config_to_mcp(cfg: &Config) -> McpAdapterCfg {
    let servers = cfg
        .mcp
        .servers
        .iter()
        .map(|(name, s)| {
            (
                name.clone(),
                McpAdapterServerCfg {
                    command: s.command.clone(),
                    args: s.args.clone(),
                    env: s.env.clone(),
                    prefix: s.prefix.clone(),
                },
            )
        })
        .collect();
    McpAdapterCfg { servers }
}

async fn build_kokoro_or_fallback(config: &Config) -> Arc<dyn jarvis_tts::piper::TtsBackend> {
    let phon = Arc::new(EspeakNgPhonemiser::new()) as Arc<dyn jarvis_tts::Phonemiser>;
    let kcfg = jarvis_tts::KokoroConfig {
        model: config.tts.kokoro.model.clone(),
        tokenizer: config.tts.kokoro.tokenizer.clone(),
        voices_dir: config.tts.kokoro.voices_dir.clone(),
        voice: config.tts.kokoro.voice.clone(),
        speed: config.tts.kokoro.speed,
        language: config.tts.kokoro.language.clone(),
        gpu: config.tts.kokoro.gpu,
    };
    match jarvis_tts::KokoroTts::load(&kcfg, phon) {
        Ok(k) => {
            tracing::info!(
                "TTS: Kokoro-82M @ 24 kHz (voice {}, lang {})",
                kcfg.voice,
                kcfg.language
            );
            Arc::new(k)
        }
        Err(e) => {
            tracing::warn!("Kokoro unavailable ({e}) — falling back to Piper/espeak");
            build_piper_or_fallback(config)
        }
    }
}

fn build_piper_or_fallback(config: &Config) -> Arc<dyn jarvis_tts::piper::TtsBackend> {
    if let Some(p) =
        jarvis_tts::PiperSubprocess::discover(&config.tts.voice_model, &config.tts.voice_config)
    {
        tracing::info!(
            "TTS: piper subprocess @ {} Hz (model {:?})",
            p.sample_rate(),
            config.tts.voice_model
        );
        return Arc::new(p);
    }
    let piper = PiperTts::new(PiperConfig {
        voice_model: config.tts.voice_model.clone(),
        voice_config: config.tts.voice_config.clone(),
        ..Default::default()
    });
    if piper.is_available() {
        Arc::new(piper)
    } else {
        tracing::info!("Piper missing — falling back to espeak-ng");
        Arc::new(jarvis_tts::EspeakNgBackend::new())
    }
}

/// Run `wpctl get-volume @DEFAULT_AUDIO_SOURCE@` and warn loudly if the
/// microphone is set quieter than 70 %. Whisper hallucinates aggressively on
/// faint audio, so this is the cheapest single fix for "Jarvis doesn't hear me".
fn check_mic_gain() {
    let out = match std::process::Command::new("wpctl")
        .args(["get-volume", "@DEFAULT_AUDIO_SOURCE@"])
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        Ok(o) => {
            tracing::debug!("wpctl get-volume failed: {:?}", o.status);
            return;
        }
        Err(e) => {
            tracing::debug!("wpctl not callable: {e}");
            return;
        }
    };
    let text = String::from_utf8_lossy(&out);
    // Output looks like: "Volume: 0.36" or "Volume: 0.36 [MUTED]".
    let Some(v) = text.split_whitespace().find_map(|t| t.parse::<f32>().ok()) else {
        return;
    };
    if text.contains("[MUTED]") {
        tracing::warn!(
            "default mic is MUTED — STT will only hear silence; \
             run `wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0` then \
             `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.85`"
        );
    } else if v < 0.70 {
        tracing::warn!(
            "default mic volume is {:.0} % — Whisper hallucinates on faint \
             audio; consider `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.85`",
            v * 100.0
        );
    } else {
        tracing::info!("mic volume {:.0} %", v * 100.0);
    }
}
