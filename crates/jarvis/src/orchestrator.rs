//! Asyncio-style state machine wiring all subsystems together.
//!
//! Today's flow uses real audio I/O + a working clap detector + the
//! energy-based VAD + the LLM stub + the TTS stub. Once the ML feature gates
//! are turned on (whisper-rs / llama-cpp-2 / piper-ort), the same wiring runs
//! production inference without code changes.

use std::sync::Arc;
use std::time::Duration;

use jarvis_audio::{Capture, PcmPlayer};
use jarvis_core::config::Config;
use jarvis_core::events::WakeEvent;
use jarvis_core::State;
use jarvis_llm::engine::LlmEngine;
use jarvis_llm::tools::{run_tool_loop, ToolLoopOutcome, ToolRegistry};
use jarvis_llm::ChatHistory;
use jarvis_service::{BusCommand, ServiceHandle};
use jarvis_skills::SkillRegistry;
use jarvis_stt::{EnergyVad, UtteranceCollector, WhisperStt};
use jarvis_tts::piper::TtsBackend;
use jarvis_wake::WakeManager;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

pub struct Orchestrator {
    pub config: Config,
    pub capture: Capture,
    pub player: Arc<PcmPlayer>,
    pub wake: WakeManager,
    pub stt: Arc<WhisperStt>,
    pub llm: Arc<dyn LlmEngine>,
    pub tts: Arc<dyn TtsBackend>,
    /// Native Rust skills (system, media). Kept around for audit/listing.
    pub skills: Arc<SkillRegistry>,
    /// Effective tool registry seen by the LLM — typically a composite of
    /// `skills` and any configured MCP servers. May equal `skills` if no MCP
    /// servers are configured.
    pub tools: Arc<dyn ToolRegistry>,
    pub state_tx: watch::Sender<String>,
    pub bus_rx: mpsc::Receiver<BusCommand>,
    pub service: Arc<ServiceHandle>,
    pub history: ChatHistory,
    /// Optional cross-session persistence. `None` when `[memory] enabled =
    /// false` or when opening the SQLite file failed at boot.
    pub memory: Option<Arc<jarvis_memory::Memory>>,
    /// Optional sentence embedder used for RAG retrieval. Loaded only when
    /// `[rag] enabled = true` and the ONNX + tokenizer files are on disk.
    pub embedder: Option<Arc<dyn jarvis_memory::Embedder>>,
}

impl Orchestrator {
    fn set_state(&self, state: State) {
        let s = state.as_str();
        let _ = self.state_tx.send(s.to_string());
        let svc = Arc::clone(&self.service);
        let s = s.to_string();
        tokio::spawn(async move {
            let _ = svc.emit_state_changed(&s).await;
        });
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        self.set_state(State::Idle);
        tracing::info!(
            "jarvis online — skills: {:?}, ww: {}, stt: {}",
            self.skills.names(),
            "stub",
            self.stt.is_available()
        );

        let mut tts_cancel = CancellationToken::new();

        loop {
            tokio::select! {
                Some(evt) = self.wake.recv() => {
                    self.handle_wake(evt, &mut tts_cancel).await;
                }
                Some(cmd) = self.bus_rx.recv() => {
                    match cmd {
                        BusCommand::Speak(text) => {
                            self.speak(&text, tts_cancel.clone()).await;
                        }
                        BusCommand::Ask(prompt, reply_tx) => {
                            let reply = self.run_prompt(&prompt, tts_cancel.clone()).await;
                            let _ = reply_tx.send(reply);
                        }
                        BusCommand::Listen(reply_tx) => {
                            let transcript = self.listen_once().await.unwrap_or_default();
                            let _ = reply_tx.send(transcript);
                        }
                        BusCommand::Cancel => {
                            tts_cancel.cancel();
                            tts_cancel = CancellationToken::new();
                            self.player.stop();
                            self.set_state(State::Idle);
                        }
                    }
                }
                else => break,
            }
        }
        Ok(())
    }

    async fn handle_wake(&mut self, _evt: WakeEvent, tts_cancel: &mut CancellationToken) {
        // Anchor the whole turn so each downstream stage can log its elapsed
        // time on `target = jarvis::latency`. Pair with `tts_first_audio` (in
        // streaming.rs) to read end-to-end first-audio latency from journald.
        let turn_start = std::time::Instant::now();
        tracing::info!(target: "jarvis::latency", "wake_received");

        tts_cancel.cancel();
        *tts_cancel = CancellationToken::new();
        self.player.stop();

        if let Err(e) = jarvis_audio::play_ding() {
            tracing::warn!("ding failed: {e}");
        }

        let Some(transcript) = self.listen_once().await else {
            self.set_state(State::Idle);
            return;
        };
        tracing::info!(
            target: "jarvis::latency",
            elapsed_ms = turn_start.elapsed().as_millis() as u64,
            chars = transcript.len(),
            "stt_done",
        );
        if transcript.trim().is_empty() {
            self.set_state(State::Idle);
            return;
        }
        if is_whisper_hallucination(&transcript) {
            tracing::info!("ignoring likely Whisper hallucination: {transcript}");
            self.set_state(State::Idle);
            return;
        }
        self.run_prompt(&transcript, tts_cancel.clone()).await;
        tracing::info!(
            target: "jarvis::latency",
            elapsed_ms = turn_start.elapsed().as_millis() as u64,
            "turn_done",
        );
    }

    /// Run a text prompt through the LLM tool-loop, speak the reply, return it.
    /// Shared between the voice wake path and the D-Bus `Ask` method.
    async fn run_prompt(&mut self, prompt: &str, tts_cancel: CancellationToken) -> String {
        self.set_state(State::Thinking);

        // Cap history before the prompt enters the LLM so the rendered ChatML
        // stays well under `n_batch`. Without this the model crashes after
        // ~17 turns on a 512-batch default.
        self.history
            .truncate_keeping_system(self.config.llm.history_keep_pairs);

        // Optional RAG: embed the user utterance, look up the closest stored
        // chunks, and prepend them to the prompt as context. The raw `prompt`
        // is still what gets persisted to memory so hydration stays clean.
        let augmented = self.maybe_retrieve(prompt);
        let prompt_for_llm: &str = augmented.as_deref().unwrap_or(prompt);

        let llm_start = std::time::Instant::now();
        let svc = Arc::clone(&self.service);
        let tool_log: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let tool_log_cb = Arc::clone(&tool_log);
        let step_cb: jarvis_llm::StepCallback = Arc::new(move |step: jarvis_llm::ToolStep| {
            if let (Some(action), Some(obs)) = (step.action.as_ref(), step.observation.as_ref()) {
                tool_log_cb.lock().unwrap().push(serde_json::json!({
                    "name": action.name,
                    "arguments": action.arguments,
                    "result": obs,
                }));
            }
            let svc = Arc::clone(&svc);
            let action_name = step
                .action
                .as_ref()
                .map(|a| a.name.clone())
                .unwrap_or_default();
            let observation = step.observation.clone().unwrap_or_default();
            tokio::spawn(async move {
                let _ = svc
                    .emit_step_taken(step.iteration, &step.thought, &action_name, &observation)
                    .await;
            });
        });
        let reply = match run_tool_loop(
            self.llm.as_ref(),
            self.tools.as_ref(),
            &mut self.history,
            prompt_for_llm,
            self.config.llm.max_tool_iterations,
            Some(step_cb),
        )
        .await
        {
            Ok(ToolLoopOutcome::Reply(t)) => t,
            Ok(ToolLoopOutcome::MaxIterations) => "Trop d'étapes d'outils.".into(),
            Ok(ToolLoopOutcome::Unparseable(raw)) => {
                tracing::warn!("LLM output didn't parse and sanitised to empty: {raw}");
                "Je n'ai pas compris.".into()
            }
            Err(e) => format!("Erreur LLM : {e}"),
        };
        tracing::info!(
            target: "jarvis::latency",
            llm_ms = llm_start.elapsed().as_millis() as u64,
            reply_chars = reply.len(),
            "llm_done",
        );
        let reply = jarvis_llm::sanitize_for_tts(&reply);
        if let Some(mem) = &self.memory {
            let calls = std::mem::take(&mut *tool_log.lock().unwrap());
            let calls_json = serde_json::Value::Array(calls);
            if let Err(e) = mem.record(prompt, &reply, &calls_json) {
                tracing::warn!("memory.record failed: {e:#}");
            }
        }
        self.speak(&reply, tts_cancel).await;
        self.set_state(State::Idle);
        reply
    }

    /// If RAG is enabled and the embedder + memory are loaded, retrieve the
    /// top-k matching chunks for `prompt` and return a string that prepends
    /// them as context. Returns `None` when retrieval should be skipped (RAG
    /// disabled, missing components, or no hits).
    fn maybe_retrieve(&self, prompt: &str) -> Option<String> {
        if !self.config.rag.enabled || self.config.rag.top_k == 0 {
            return None;
        }
        let embedder = self.embedder.as_ref()?;
        let memory = self.memory.as_ref()?;
        let q = match embedder.embed(prompt) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("rag: embed failed: {e:#}");
                return None;
            }
        };
        let hits = match memory.search_facts(&q, self.config.rag.top_k) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("rag: search failed: {e:#}");
                return None;
            }
        };
        if hits.is_empty() {
            return None;
        }
        let mut buf = String::from("Contexte récupéré:\n");
        for (i, h) in hits.iter().enumerate() {
            let src = h.source.as_deref().unwrap_or("(inconnu)");
            buf.push_str(&format!("[{}] ({src}) {}\n", i + 1, h.content));
        }
        buf.push_str("\nQuestion: ");
        buf.push_str(prompt);
        tracing::info!(
            target: "jarvis::rag",
            hits = hits.len(),
            top_score = hits.first().map(|h| h.score).unwrap_or(0.0),
            "retrieval injected",
        );
        Some(buf)
    }

    async fn listen_once(&mut self) -> Option<String> {
        self.set_state(State::Listening);
        self.wake.pause();
        let vad_cfg = jarvis_stt::vad::EnergyVadConfig {
            sample_rate: self.capture.sample_rate(),
            silence_ms: self.config.stt.silence_ms,
            min_speech_ms: self.config.stt.min_speech_ms,
            max_utterance_s: self.config.stt.max_utterance_s,
            ..Default::default()
        };
        let mut collector = UtteranceCollector::new(EnergyVad::new(vad_cfg));
        let mut rx = self.capture.subscribe();
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs(self.config.stt.max_utterance_s as u64 + 5);

        let utterance = loop {
            if tokio::time::Instant::now() > deadline {
                break None;
            }
            match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
                Ok(Ok(frame)) => {
                    if let Some(u) = collector.push(&frame) {
                        break Some(u);
                    }
                }
                Ok(Err(_)) => break None,
                Err(_) => continue,
            }
        };
        self.wake.resume();

        let text = match utterance {
            None => None,
            Some(samples) => {
                let stt = Arc::clone(&self.stt);
                tokio::task::spawn_blocking(move || stt.transcribe(&samples))
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }
        };
        if let Some(ref t) = text {
            let _ = self.service.emit_transcribed(t).await;
            tracing::info!("transcript: {t}");
        }
        text
    }

    async fn speak(&mut self, text: &str, cancel: CancellationToken) {
        self.set_state(State::Speaking);
        // Pipeline: feed the reply text as a single "token" into the sentence
        // chunker, then dispatch each clause through speak_stream so synthesis
        // and playback overlap (up to 2 chunks in flight). For one-sentence
        // replies this is no worse than the buffered path; for multi-sentence
        // replies it cuts time-to-first-audio roughly in half.
        let tokens = futures::stream::iter(vec![Ok::<String, anyhow::Error>(text.to_string())]);
        let chunks = jarvis_llm::sentence_stream(tokens);
        let played = crate::streaming::speak_stream(
            chunks,
            Arc::clone(&self.tts),
            Arc::clone(&self.player),
            cancel.clone(),
        )
        .await;
        if played == 0 {
            tracing::warn!("TTS produced no audio for: {text}");
        }
        let _ = self.service.emit_spoken(text).await;
        self.set_state(State::Idle);
    }
}

/// Cheap heuristic that catches the recurring strings whisper.cpp produces
/// when fed silence or near-silence (especially with the multilingual ggml).
/// Triggered if the transcript matches a known phrase, is dominated by
/// non-letters, or is too short to be a real query.
fn is_whisper_hallucination(t: &str) -> bool {
    let trimmed = t.trim();
    let s = trimmed.to_lowercase();

    // Empty / very short / no alphabetic content at all.
    let alpha_count = s.chars().filter(|c| c.is_alphabetic()).count();
    if alpha_count < 4 {
        return true;
    }

    // Stage direction wrapper: `*Bip*`, `*chuchote*`, `*Bruit de la porte*`.
    // Whisper produces these when the audio is silence / breath / noise.
    if trimmed.starts_with('*') && trimmed.ends_with('*') && trimmed.len() >= 2 {
        return true;
    }
    // Parenthetical stage directions: `(rire)`, `(applaudissements)`.
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        return true;
    }

    // Music glyphs: whisper.cpp emits ♪/♫ on silence with the multilingual ggml.
    if trimmed.chars().any(|c| matches!(c, '♪' | '♫' | '♬' | '♩')) {
        return true;
    }

    // Trailing copyright credits ("Copyright WDR 2021", "© 2024 ...").
    if s.starts_with("copyright ") || s.contains(" copyright ") || trimmed.contains('©') {
        return true;
    }

    // Punctuation soup: less than 40% alphabetic across the whole string.
    let total = s.chars().count();
    if total > 0 && alpha_count * 100 / total < 40 {
        return true;
    }

    // Average word length — gibberish hallucinations are typically a string
    // of 1-2 letter "words" ("Moj pa v o r s h a.").
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() >= 3 {
        let letters: usize = words
            .iter()
            .map(|w| w.chars().filter(|c| c.is_alphabetic()).count())
            .sum();
        let avg = letters as f32 / words.len() as f32;
        if avg < 2.5 {
            return true;
        }
    }

    // Single token repeated to make up the whole transcript ("... ... ...").
    if words.len() >= 3 && words.iter().all(|w| *w == words[0]) {
        return true;
    }

    // N-gram repetition: bigram or trigram covers >= 70% of the transcript
    // ("you you you you", "merci merci merci", "thank you thank you thank you").
    if words.len() >= 4 {
        for n in [1usize, 2, 3] {
            if words.len() < n * 2 {
                continue;
            }
            let head = &words[..n];
            let mut hits = 0usize;
            let mut i = 0;
            while i + n <= words.len() {
                if &words[i..i + n] == head {
                    hits += 1;
                    i += n;
                } else {
                    i += 1;
                }
            }
            if (hits * n) as f32 / words.len() as f32 >= 0.7 {
                return true;
            }
        }
    }

    const PATTERNS: &[&str] = &[
        // FR — credits / Amara / Radio-Canada
        "sous-titres",
        "sous titres",
        "sous-titrage",
        "soustitreur.com",
        "st'501",
        "radio-canada",
        "amara.org",
        "para la communauté",
        // FR — call-to-action YouTube
        "merci d'avoir regardé",
        "merci de votre attention",
        "merci à tous",
        "abonnez-vous",
        "n'oubliez pas de",
        "j'espère que vous avez",
        "je vous remercie de vous abonner",
        "like et abonne",
        "laissez un commentaire",
        // EN — language fallback
        "thanks for watching",
        "subtitles by",
        "transcript by",
        "transcript emily beynon",
        "please subscribe",
        "don't forget to",
        // DE/ES/IT/PT — multilingual ggml drift
        "untertitel",
        "copyright wdr",
        "copyright zdf",
        "subtítulos",
        "comunidad de amara",
        "legendas pela",
        "sottotitoli",
        "a cura di",
        // Stage directions / silence artefacts (FR)
        "rassurez",
        "rassurez-vous",
        "pour votre famille",
        "pour votre femme",
        "musique de fond",
        "musique entraînante",
        "musique douce",
        "[music]",
        "[musique]",
        "applaudissements",
        "bruit de",
        "porte ouverte",
        "porte qui",
        "bip",
        "souffle",
        "chuchote",
        "rire",
        "rires",
        "magie frtrans",
        "larousse",
        "frtrans",
        "il faut s'en aller à l'heure",
    ];
    PATTERNS.iter().any(|p| s.contains(p))
}

#[cfg(test)]
mod hallucination_tests {
    use super::is_whisper_hallucination;

    #[test]
    fn catches_known_phrases() {
        assert!(is_whisper_hallucination(
            "...vous vous rassurez... ...pour votre femme."
        ));
        assert!(is_whisper_hallucination(
            "Sous-titres réalisés par la communauté."
        ));
        assert!(is_whisper_hallucination(
            "Merci d'avoir regardé cette vidéo."
        ));
        assert!(is_whisper_hallucination("Magie Frtrans larousse"));
        assert!(is_whisper_hallucination(
            "Le jour où il faut s'en aller à l'heure."
        ));
    }

    #[test]
    fn catches_letter_soup() {
        assert!(is_whisper_hallucination("Moj pa v o r s h a."));
        assert!(is_whisper_hallucination("A A A U S H T A G I A T H A L."));
    }

    #[test]
    fn catches_stage_directions() {
        assert!(is_whisper_hallucination("*Bip*"));
        assert!(is_whisper_hallucination("*chuchote*"));
        assert!(is_whisper_hallucination("*signal*"));
        assert!(is_whisper_hallucination("*Bruit de la porte*"));
        assert!(is_whisper_hallucination("*Souffle*"));
        assert!(is_whisper_hallucination("(rires)"));
        assert!(is_whisper_hallucination("(applaudissements)"));
    }

    #[test]
    fn catches_punctuation_soup() {
        assert!(is_whisper_hallucination("... ... ... ... ... ... ..."));
        assert!(is_whisper_hallucination("..."));
        assert!(is_whisper_hallucination("!?!?!"));
    }

    #[test]
    fn allows_real_queries() {
        assert!(!is_whisper_hallucination(
            "Bonjour Jarvis quelle heure est-il"
        ));
        assert!(!is_whisper_hallucination(
            "Mets la musique en pause s'il te plaît"
        ));
        assert!(!is_whisper_hallucination("Baisse le volume"));
        assert!(!is_whisper_hallucination("Prends une capture d'écran"));
        assert!(!is_whisper_hallucination("Merci Jarvis"));
        assert!(!is_whisper_hallucination("Oui non peut-être je sais pas"));
    }

    #[test]
    fn catches_amara_variants() {
        assert!(is_whisper_hallucination("Sous-titrage ST'501"));
        assert!(is_whisper_hallucination("❤️ par SousTitreur.com"));
        assert!(is_whisper_hallucination(
            "Sous-titres réalisés para la communauté d'Amara.org"
        ));
        assert!(is_whisper_hallucination(
            "Sous-titrage Société Radio-Canada"
        ));
    }

    #[test]
    fn catches_call_to_action() {
        assert!(is_whisper_hallucination("Abonnez-vous à la chaîne"));
        assert!(is_whisper_hallucination(
            "J'espère que vous avez apprécié la vidéo"
        ));
        assert!(is_whisper_hallucination("Please subscribe to my channel"));
        assert!(is_whisper_hallucination("Je vous remercie de vous abonner"));
    }

    #[test]
    fn catches_multilingual_drift() {
        assert!(is_whisper_hallucination(
            "Untertitel der Amara.org-Community"
        ));
        assert!(is_whisper_hallucination(
            "Subtítulos realizados por la comunidad de Amara"
        ));
        assert!(is_whisper_hallucination(
            "Legendas pela comunidade Amara.org"
        ));
        assert!(is_whisper_hallucination(
            "Sottotitoli e revisione a cura di QTSS"
        ));
    }

    #[test]
    fn catches_music_glyphs() {
        assert!(is_whisper_hallucination("♪♪♪"));
        assert!(is_whisper_hallucination("♫ musique ♫"));
        assert!(is_whisper_hallucination("[Music] some text [Music]"));
    }

    #[test]
    fn catches_ngram_repetition() {
        assert!(is_whisper_hallucination("you you you you you"));
        assert!(is_whisper_hallucination(
            "thank you thank you thank you thank you"
        ));
        assert!(is_whisper_hallucination("merci merci merci merci"));
    }

    #[test]
    fn catches_copyright_trailer() {
        assert!(is_whisper_hallucination("Copyright WDR 2021"));
        assert!(is_whisper_hallucination("© 2024 France Télévisions"));
    }
}
