use std::path::PathBuf;

use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;
use serde::{Deserialize, Serialize};

use crate::dirs;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioCfg {
    pub sample_rate_capture: u32,
    pub sample_rate_clap: u32,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
}

impl Default for AudioCfg {
    fn default() -> Self {
        Self {
            sample_rate_capture: 16_000,
            // 16 kHz is what whisper.cpp and openWakeWord expect natively.
            // Keeping a higher rate forced us to do (untracked) resampling
            // before STT, which caused massive whisper hallucinations.
            sample_rate_clap: 16_000,
            input_device: None,
            output_device: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ClapCfg {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WakeWordCfg {
    pub enabled: bool,
    pub model: PathBuf,
    pub threshold: f32,
    pub cooldown_s: f32,
}

impl Default for WakeWordCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            model: dirs::models_dir().join("hey_jarvis_v0.1.onnx"),
            threshold: 0.6,
            cooldown_s: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WakeCfg {
    pub clap: ClapCfg,
    pub wakeword: WakeWordCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SttCfg {
    pub model: PathBuf,
    pub language: String,
    pub silence_ms: u32,
    pub min_speech_ms: u32,
    pub max_utterance_s: u32,
    /// Whisper anti-hallucination prompt; primes the decoder toward our
    /// domain (short French queries, system commands).
    pub initial_prompt: String,
    /// Reject segments below this no-speech confidence. 0.6 is a sane
    /// default; raise to 0.8 if you still hear hallucinations.
    pub no_speech_threshold: f32,
    /// Strip `*Bip*`, `(rires)` and other non-speech markers at decode time.
    pub suppress_non_speech: bool,
    /// Decoding threads. Only used when GPU offload is disabled (CUDA build
    /// puts the heavy work on the GPU). 4 is a safe default; -1 = auto.
    pub n_threads: i32,
    /// GPU layers offloaded to CUDA when whisper.cpp is built with the
    /// `cuda` feature. Set 0 to force CPU even on a CUDA build. The whisper
    /// backend treats any value > 0 as "use GPU".
    pub n_gpu_layers: i32,
}

impl Default for SttCfg {
    fn default() -> Self {
        Self {
            // Medium gives noticeably better FR than small at the cost of
            // ~1.5 GB on disk + GPU. The model is downloaded by setup.sh.
            model: dirs::models_dir().join("ggml-medium.bin"),
            language: "fr".to_string(),
            silence_ms: 600,
            min_speech_ms: 250,
            max_utterance_s: 20,
            initial_prompt: "Assistant vocal Jarvis en français. \
                Questions courtes et ordres système."
                .to_string(),
            no_speech_threshold: 0.6,
            suppress_non_speech: true,
            n_threads: 4,
            n_gpu_layers: 99,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmCfg {
    pub model: PathBuf,
    pub n_ctx: u32,
    pub n_threads: i32,
    pub n_gpu_layers: i32,
    pub temperature: f32,
    pub system_prompt: String,
    pub max_tool_iterations: u32,
}

impl Default for LlmCfg {
    fn default() -> Self {
        Self {
            model: dirs::models_dir().join("qwen2.5-3b-instruct-q4_k_m.gguf"),
            n_ctx: 4096,
            n_threads: 4,
            n_gpu_layers: 99,
            temperature: 0.4,
            system_prompt: default_system_prompt(),
            max_tool_iterations: 8,
        }
    }
}

fn default_system_prompt() -> String {
    "Tu es Jarvis, assistant vocal personnel. \
Tu réponds en français, en vouvoiement, factuel et concis (1 à 2 phrases). \
\
RÈGLES DE SORTIE STRICTES — ta réponse sera lue à voix haute par un \
synthétiseur vocal, donc :\n\
- Écris uniquement du texte parlable, comme à l'oral.\n\
- Aucun markdown : pas d'astérisques, pas de gras, pas d'italique, pas de \
  titres, pas de listes à puces, pas de backticks.\n\
- Pas de didascalies entre astérisques (jamais '*Souffle*', '*Compris*', \
  '*porte qui s'ouvre*', etc.).\n\
- Pas de balises, pas d'émojis, pas de code.\n\
- Pas de numérotation '1.' '2.' — utilise « premièrement, deuxièmement ».\n\
- Les nombres et abréviations doivent être prononçables (« deux heures » \
  plutôt que « 2h »).\n\
- Si tu n'as pas la réponse, dis-le brièvement.\n\
\n\
Tu disposes d'outils pour contrôler le PC et la musique. Utilise-les sans \
demander confirmation pour les actions réversibles. Pour appeler un outil, \
émets un bloc `<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>` \
(format Qwen natif). Le résultat te sera renvoyé pour que tu formules la \
réponse parlée. Sinon, réponds en texte simple, directement."
        .to_string()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TtsEngine {
    Kokoro,
    Piper,
    Espeak,
}

impl Default for TtsEngine {
    fn default() -> Self {
        Self::Kokoro
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KokoroCfg {
    pub model: PathBuf,
    pub tokenizer: PathBuf,
    pub voices_dir: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub language: String,
    /// Attempt to use the CUDA execution provider for ONNX Runtime. Requires
    /// the `cuda` cargo feature on `jarvis-tts`. Falls back to CPU silently
    /// if CUDA is unavailable at runtime.
    pub gpu: bool,
}

impl Default for KokoroCfg {
    fn default() -> Self {
        let kokoro = dirs::data_dir().join("kokoro");
        Self {
            model: kokoro.join("kokoro-v1.0.onnx"),
            tokenizer: kokoro.join("tokenizer.json"),
            voices_dir: kokoro.join("voices"),
            voice: "am_onyx".to_string(),
            speed: 1.0,
            language: "fr".to_string(),
            gpu: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsCfg {
    pub engine: TtsEngine,
    pub kokoro: KokoroCfg,
    // Piper fields kept flat for backward compat with existing config.toml.
    pub voice_model: PathBuf,
    pub voice_config: PathBuf,
    pub length_scale: f32,
    pub noise_scale: f32,
    pub noise_w: f32,
}

impl Default for TtsCfg {
    fn default() -> Self {
        let piper = dirs::data_dir().join("piper");
        Self {
            engine: TtsEngine::default(),
            kokoro: KokoroCfg::default(),
            voice_model: piper.join("fr_FR-siwis-medium.onnx"),
            voice_config: piper.join("fr_FR-siwis-medium.onnx.json"),
            length_scale: 1.0,
            noise_scale: 0.667,
            noise_w: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiCfg {
    pub log_state_changes: bool,
    pub log_transcripts: bool,
}

impl Default for UiCfg {
    fn default() -> Self {
        Self {
            log_state_changes: true,
            log_transcripts: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServerCfg {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryCfg {
    /// Enable on-disk SQLite persistence of conversations.
    pub enabled: bool,
    /// Path to the SQLite file. Defaults to `<data_dir>/memory.sqlite`.
    pub path: PathBuf,
    /// Number of recent turns to replay into ChatHistory at startup. 0
    /// disables hydration but still persists new turns.
    pub hydrate_n: usize,
}

impl Default for MemoryCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            path: dirs::data_dir().join("memory.sqlite"),
            hydrate_n: 6,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpCfg {
    pub servers: std::collections::HashMap<String, McpServerCfg>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RagCfg {
    /// Enable retrieval-augmented generation. When true, the orchestrator
    /// embeds the user utterance and prepends top-k matching chunks from the
    /// `facts` table to the system prompt before calling the LLM.
    pub enabled: bool,
    /// Path to a BGE-small-en-v1.5 ONNX export.
    pub model: PathBuf,
    /// Path to the matching HuggingFace `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// Directory scanned by the `index_docs` skill.
    pub corpus_dir: PathBuf,
    /// Number of chunks injected into the prompt per turn.
    pub top_k: usize,
    /// Maximum whitespace-tokens per chunk emitted by `index_docs`.
    /// The embedder truncates beyond 512 wordpieces regardless.
    pub chunk_tokens: usize,
}

impl Default for RagCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            model: dirs::models_dir().join("bge-small-en-v1.5.onnx"),
            tokenizer: dirs::models_dir().join("bge-small-en-v1.5.tokenizer.json"),
            corpus_dir: dirs::data_dir().join("corpus"),
            top_k: 3,
            chunk_tokens: 220,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioCfg,
    pub wake: WakeCfg,
    pub stt: SttCfg,
    pub llm: LlmCfg,
    pub tts: TtsCfg,
    pub ui: UiCfg,
    pub mcp: McpCfg,
    pub memory: MemoryCfg,
    pub rag: RagCfg,
}

impl Config {
    /// Load from `~/.config/jarvis/config.toml`, layered over defaults,
    /// with env overrides (`JARVIS_<SECTION>__<KEY>`).
    pub fn load() -> anyhow::Result<Self> {
        Self::load_from(&dirs::config_file())
    }

    pub fn load_from(path: &std::path::Path) -> anyhow::Result<Self> {
        let mut fig = Figment::from(Serialized::defaults(Self::default()));
        if path.exists() {
            fig = fig.merge(Toml::file(path));
        }
        fig = fig.merge(Env::prefixed("JARVIS_").split("__"));
        Ok(fig.extract()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_serializable() {
        let cfg = Config::default();
        let s = toml::to_string(&cfg).expect("serialize defaults");
        let back: Config = toml::from_str(&s).expect("deserialize defaults");
        assert_eq!(back.stt.language, "fr");
        assert_eq!(back.llm.max_tool_iterations, 8);
    }

    #[test]
    fn loads_from_toml_layered_over_defaults() {
        let dir = std::env::temp_dir().join("jarvis-core-test-toml");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "[stt]\nlanguage = \"en\"\n").unwrap();
        let cfg = Config::load_from(&path).expect("loads from TOML");
        assert_eq!(cfg.stt.language, "en");
        // Other defaults remain.
        assert_eq!(cfg.llm.max_tool_iterations, 8);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let cfg = Config::load_from(std::path::Path::new("/nonexistent/jarvis.toml"))
            .expect("loads with defaults");
        assert_eq!(cfg.stt.language, "fr");
    }
}
