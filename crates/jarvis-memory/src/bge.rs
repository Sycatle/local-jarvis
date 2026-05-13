//! BGE-small-en-v1.5 sentence embedder via ONNX Runtime.
//!
//! Compiled when the `embed` feature is on. Loads a HuggingFace ONNX export of
//! `BAAI/bge-small-en-v1.5` (384-dim) plus the matching `tokenizer.json`. The
//! ONNX file and tokenizer must be present on disk; the embedder doesn't
//! download anything.
//!
//! Output convention: `[CLS]` token from `last_hidden_state`, L2-normalised.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::embedder::Embedder;

const DIM: usize = 384;
const MAX_TOKENS: usize = 512;

pub struct BgeSmallEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    has_token_type_ids: bool,
}

impl BgeSmallEmbedder {
    pub fn load(model: &Path, tokenizer: &Path) -> Result<Self> {
        if !model.is_file() {
            return Err(anyhow!("bge model not found at {model:?}"));
        }
        if !tokenizer.is_file() {
            return Err(anyhow!("bge tokenizer not found at {tokenizer:?}"));
        }

        let session = Session::builder()
            .context("ort session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("ort optimisation level")?
            .with_intra_threads(2)
            .context("ort intra threads")?
            .with_inter_threads(1)
            .context("ort inter threads")?
            .commit_from_file(model)
            .with_context(|| format!("loading bge ONNX from {model:?}"))?;

        let has_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

        let tk = Tokenizer::from_file(tokenizer)
            .map_err(|e| anyhow!("loading bge tokenizer at {tokenizer:?}: {e}"))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: tk,
            has_token_type_ids,
        })
    }

    /// Convenience for [`Config`] paths.
    pub fn load_paths(model: PathBuf, tokenizer: PathBuf) -> Result<Self> {
        Self::load(&model, &tokenizer)
    }
}

impl Embedder for BgeSmallEmbedder {
    fn dim(&self) -> usize {
        DIM
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_batch(&[text])?;
        v.into_iter()
            .next()
            .ok_or_else(|| anyhow!("empty embedding result"))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenising batch: {e}"))?;

        // Pad/truncate to the longest sequence in the batch (capped at
        // MAX_TOKENS) so the ONNX inputs are a clean 2D tensor.
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_TOKENS);
        let batch = encodings.len();

        let mut input_ids = Array2::<i64>::zeros((batch, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch, max_len));
        for (b, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let tt = enc.get_type_ids();
            let n = ids.len().min(max_len);
            for i in 0..n {
                input_ids[(b, i)] = ids[i] as i64;
                attention_mask[(b, i)] = mask[i] as i64;
                if i < tt.len() {
                    token_type_ids[(b, i)] = tt[i] as i64;
                }
            }
        }

        let ids_tensor =
            Tensor::from_array(input_ids).map_err(|e| anyhow!("input_ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(attention_mask)
            .map_err(|e| anyhow!("attention_mask tensor: {e}"))?;
        let tt_tensor = Tensor::from_array(token_type_ids)
            .map_err(|e| anyhow!("token_type_ids tensor: {e}"))?;

        let mut session = self.session.lock().expect("bge session mutex poisoned");
        let outputs = if self.has_token_type_ids {
            session
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                    "token_type_ids" => tt_tensor,
                ])
                .map_err(|e| anyhow!("bge run: {e}"))?
        } else {
            session
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                ])
                .map_err(|e| anyhow!("bge run: {e}"))?
        };

        // last_hidden_state: [batch, seq, hidden]. Take [:, 0, :] (CLS) then
        // L2-normalise each row.
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extract bge output: {e}"))?;
        if shape.len() != 3 {
            return Err(anyhow!(
                "unexpected bge output rank {} (shape {:?})",
                shape.len(),
                shape
            ));
        }
        let hidden = shape[2] as usize;
        if hidden != DIM {
            return Err(anyhow!("bge hidden dim {hidden} != expected {DIM}"));
        }
        let seq = shape[1] as usize;

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let start = b * seq * hidden;
            let cls = &data[start..start + hidden];
            let norm = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
            let v: Vec<f32> = if norm == 0.0 {
                cls.to_vec()
            } else {
                cls.iter().map(|x| x / norm).collect()
            };
            out.push(v);
        }
        Ok(out)
    }
}
