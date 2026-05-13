//! Real LLM backend via `llama-cpp-2` (bindings to llama.cpp).
//!
//! Loaded behind the `llama` cargo feature. Add `cuda` on top to compile
//! llama.cpp with CUDA offload (requires `nvcc` + libcudart at build time).
//!
//! Concurrency: the llama context and sampler are constructed inside the
//! `spawn_blocking` closure (neither is `Send`). A `Mutex` serialises calls
//! through the engine so we only ever build one context at a time.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::Mutex;

use crate::chat::ChatHistory;
use crate::engine::LlmEngine;
use crate::grammar::ToolSpec;

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub model: PathBuf,
    pub n_ctx: u32,
    pub n_threads: i32,
    pub n_gpu_layers: i32,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            n_ctx: 4096,
            n_threads: -1,
            n_gpu_layers: 0,
            temperature: 0.4,
            max_tokens: 512,
        }
    }
}

pub struct LlamaEngine {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    cfg: LlamaConfig,
    // The llama context is not Send; serialise generate() calls. Wrapped in
    // Arc so streaming can hand an OwnedMutexGuard to the blocking worker.
    lock: Arc<Mutex<()>>,
}

impl LlamaEngine {
    pub fn load(cfg: LlamaConfig) -> anyhow::Result<Self> {
        let backend = Arc::new(LlamaBackend::init()?);

        let mut model_params = LlamaModelParams::default();
        if cfg.n_gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(cfg.n_gpu_layers as u32);
        }

        tracing::info!(
            "loading llama model from {:?} (n_gpu_layers={})",
            cfg.model,
            cfg.n_gpu_layers
        );
        let model = LlamaModel::load_from_file(&backend, cfg.model.clone(), &model_params)?;
        tracing::info!("llama model loaded");

        Ok(Self {
            backend,
            model: Arc::new(model),
            cfg,
            lock: Arc::new(Mutex::new(())),
        })
    }
}

#[async_trait]
impl LlmEngine for LlamaEngine {
    async fn generate(&self, history: &ChatHistory, tools: &[ToolSpec]) -> anyhow::Result<String> {
        let _g = self.lock.lock().await;

        // GBNF currently breaks Qwen's tokenizer (llama-grammar.cpp:940
        // assertion). Until the grammar is rewritten in JSON-Schema form and
        // converted via llama.cpp's json-schema-to-grammar, we rely on Qwen's
        // strong instruction-following + system prompt to emit valid JSON.
        let _ = tools;
        let prompt = history.render_chatml();
        let n_max = self.cfg.max_tokens as i32;
        let backend = self.backend.clone();
        let model = self.model.clone();
        let cfg = self.cfg.clone();

        // Everything llama-cpp-2 is blocking & not Send. Build context + sampler
        // *inside* the worker thread.
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let n_ctx = NonZeroU32::new(cfg.n_ctx).unwrap_or(NonZeroU32::new(4096).unwrap());
            let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
            if cfg.n_threads > 0 {
                ctx_params = ctx_params
                    .with_n_threads(cfg.n_threads)
                    .with_n_threads_batch(cfg.n_threads);
            }
            let mut ctx: LlamaContext = model.new_context(&backend, ctx_params)?;

            // Tokenize the rendered ChatML prompt (no extra BOS — the template
            // already contains the role markers).
            let tokens: Vec<LlamaToken> = model.str_to_token(&prompt, AddBos::Never)?;

            let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
            let last_idx = tokens.len() as i32 - 1;
            for (i, &tok) in tokens.iter().enumerate() {
                let is_last = i as i32 == last_idx;
                batch.add(tok, i as i32, &[0], is_last)?;
            }
            ctx.decode(&mut batch)?;

            // Build sampler inside the worker. Grammar disabled (see top of
            // generate()).
            let temp = LlamaSampler::temp(cfg.temperature);
            let dist = LlamaSampler::dist(0);
            let mut sampler = LlamaSampler::chain_simple([temp, dist]);

            let mut out = String::new();
            let mut n_decoded = tokens.len() as i32;
            for _ in 0..n_max {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                if model.is_eog_token(token) {
                    break;
                }

                // token_to_piece in 0.1.146 needs an encoding_rs decoder; the
                // older token_to_str API is deprecated but still works and is
                // sufficient for plain UTF-8 prompts.
                #[allow(deprecated)]
                let piece = model.token_to_str(token, Special::Tokenize)?;
                out.push_str(&piece);

                batch.clear();
                batch.add(token, n_decoded, &[0], true)?;
                ctx.decode(&mut batch)?;
                n_decoded += 1;
            }

            Ok(out)
        })
        .await?
    }

    async fn generate_stream<'a>(
        &'a self,
        history: &'a ChatHistory,
        tools: &'a [ToolSpec],
    ) -> anyhow::Result<BoxStream<'a, anyhow::Result<String>>> {
        let _ = tools;
        let guard = self.lock.clone().lock_owned().await;

        let prompt = history.render_chatml();
        let n_max = self.cfg.max_tokens as i32;
        let backend = self.backend.clone();
        let model = self.model.clone();
        let cfg = self.cfg.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<String>>();

        tokio::task::spawn_blocking(move || {
            let _hold = guard;
            let work = (|| -> anyhow::Result<()> {
                let n_ctx = NonZeroU32::new(cfg.n_ctx).unwrap_or(NonZeroU32::new(4096).unwrap());
                let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
                if cfg.n_threads > 0 {
                    ctx_params = ctx_params
                        .with_n_threads(cfg.n_threads)
                        .with_n_threads_batch(cfg.n_threads);
                }
                let mut ctx: LlamaContext = model.new_context(&backend, ctx_params)?;

                let tokens: Vec<LlamaToken> = model.str_to_token(&prompt, AddBos::Never)?;
                let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
                let last_idx = tokens.len() as i32 - 1;
                for (i, &tok) in tokens.iter().enumerate() {
                    let is_last = i as i32 == last_idx;
                    batch.add(tok, i as i32, &[0], is_last)?;
                }
                ctx.decode(&mut batch)?;

                let temp = LlamaSampler::temp(cfg.temperature);
                let dist = LlamaSampler::dist(0);
                let mut sampler = LlamaSampler::chain_simple([temp, dist]);

                let mut n_decoded = tokens.len() as i32;
                for _ in 0..n_max {
                    let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                    sampler.accept(token);
                    if model.is_eog_token(token) {
                        break;
                    }
                    #[allow(deprecated)]
                    let piece = model.token_to_str(token, Special::Tokenize)?;
                    if tx.send(Ok(piece)).is_err() {
                        // Consumer dropped the stream — stop generating.
                        break;
                    }
                    batch.clear();
                    batch.add(token, n_decoded, &[0], true)?;
                    ctx.decode(&mut batch)?;
                    n_decoded += 1;
                }
                Ok(())
            })();
            if let Err(e) = work {
                let _ = tx.send(Err(e));
            }
        });

        let stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(stream.boxed())
    }
}
