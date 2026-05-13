//! Trait for text → vector embedding backends. Implementations live in
//! sibling modules (e.g. [`crate::bge`]) and are feature-gated so the bare
//! `jarvis-memory` crate stays free of ML deps.

use anyhow::Result;

pub trait Embedder: Send + Sync {
    /// Embedding dimensionality. Used to validate persisted vectors at load
    /// time and to allocate buffers.
    fn dim(&self) -> usize;

    /// Embed a single passage. Implementations should L2-normalise the
    /// returned vector — the SQLite cosine search assumes unit-norm vectors
    /// are not required but consistent normalisation speeds up comparison.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batched variant. Default impl falls back to repeated `embed()` so
    /// backends that don't benefit from batching get a free implementation;
    /// backends with real batch support (BgeSmallEmbedder) should override.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
