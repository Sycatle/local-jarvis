//! `index_docs` skill: walk a directory, chunk text files, embed, and store in
//! the `facts` table. Re-indexing is idempotent — existing rows tagged with
//! the file path are dropped before insert.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use jarvis_memory::{Embedder, Memory};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::capabilities::Capabilities;
use crate::registry::Skill;

const ALLOWED_EXTS: &[&str] = &["md", "markdown", "txt", "rst", "org"];

pub struct IndexDocsSkill {
    memory: Arc<Memory>,
    embedder: Arc<dyn Embedder>,
    /// Default root used when the LLM omits the `path` argument.
    default_root: PathBuf,
    chunk_tokens: usize,
}

impl IndexDocsSkill {
    pub fn new(
        memory: Arc<Memory>,
        embedder: Arc<dyn Embedder>,
        default_root: PathBuf,
        chunk_tokens: usize,
    ) -> Self {
        Self {
            memory,
            embedder,
            default_root,
            chunk_tokens,
        }
    }
}

#[derive(Deserialize)]
pub struct IndexArgs {
    #[serde(default)]
    pub path: Option<String>,
}

#[async_trait]
impl Skill for IndexDocsSkill {
    fn name(&self) -> &'static str {
        "index_docs"
    }
    fn description(&self) -> &'static str {
        "Index plain-text documents under a directory for retrieval. Walks .md/.txt/.rst/.org files, chunks them, embeds, and stores the vectors. Re-indexing a file replaces its previous chunks."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory or single file to index. Defaults to the configured corpus_dir."
                }
            },
            "additionalProperties": false
        })
    }
    fn capabilities(&self) -> Capabilities {
        // Reads files only, no exec/dbus/net. Path is supplied by the LLM but
        // restricted by the OS-level user permissions.
        Capabilities::none()
    }
    async fn invoke(&self, args: Value) -> Value {
        let parsed: IndexArgs = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return json!({"ok": false, "error": format!("bad args: {e}")}),
        };
        let root = parsed
            .path
            .map(PathBuf::from)
            .unwrap_or_else(|| self.default_root.clone());
        if !root.exists() {
            return json!({"ok": false, "error": format!("path not found: {root:?}")});
        }

        let chunk_tokens = self.chunk_tokens.max(40);
        let memory = Arc::clone(&self.memory);
        let embedder = Arc::clone(&self.embedder);

        let result = tokio::task::spawn_blocking(move || {
            index_path(&root, chunk_tokens, memory.as_ref(), embedder.as_ref())
        })
        .await;
        match result {
            Ok(Ok(stats)) => json!({
                "ok": true,
                "files_indexed": stats.files,
                "chunks": stats.chunks,
                "skipped": stats.skipped,
            }),
            Ok(Err(e)) => json!({"ok": false, "error": format!("{e:#}")}),
            Err(e) => json!({"ok": false, "error": format!("join: {e}")}),
        }
    }
}

struct Stats {
    files: usize,
    chunks: usize,
    skipped: usize,
}

fn index_path(
    root: &Path,
    chunk_tokens: usize,
    memory: &Memory,
    embedder: &dyn Embedder,
) -> anyhow::Result<Stats> {
    let mut stats = Stats {
        files: 0,
        chunks: 0,
        skipped: 0,
    };
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        if p.is_dir() {
            for entry in std::fs::read_dir(&p)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    // Skip hidden + heavyweight tooling dirs.
                    let name = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default();
                    if name.starts_with('.') || matches!(name, "node_modules" | "target") {
                        continue;
                    }
                    stack.push(path);
                } else {
                    index_one(&path, chunk_tokens, memory, embedder, &mut stats)?;
                }
            }
        } else {
            index_one(&p, chunk_tokens, memory, embedder, &mut stats)?;
        }
    }
    Ok(stats)
}

fn index_one(
    path: &Path,
    chunk_tokens: usize,
    memory: &Memory,
    embedder: &dyn Embedder,
    stats: &mut Stats,
) -> anyhow::Result<()> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !ALLOWED_EXTS.contains(&ext.as_str()) {
        stats.skipped += 1;
        return Ok(());
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => {
            stats.skipped += 1;
            return Ok(());
        }
    };
    let source = path.to_string_lossy().into_owned();
    memory.forget_source(&source)?;
    let chunks = chunk_text(&text, chunk_tokens);
    if chunks.is_empty() {
        stats.files += 1;
        return Ok(());
    }
    let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
    let vecs = embedder.embed_batch(&refs)?;
    for (chunk, vec) in chunks.iter().zip(vecs.iter()) {
        memory.add_fact("doc", Some(&source), chunk, vec)?;
        stats.chunks += 1;
    }
    stats.files += 1;
    Ok(())
}

/// Naive whitespace chunker with a 10% overlap. Good enough for personal
/// markdown notes; swap for a sentence-aware splitter when we add Mistral-7B
/// or move to chunks > 1 paragraph.
fn chunk_text(text: &str, chunk_tokens: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let stride = chunk_tokens.saturating_sub(chunk_tokens / 10).max(1);
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_tokens).min(words.len());
        let chunk = words[i..end].join(" ");
        if !chunk.is_empty() {
            out.push(chunk);
        }
        if end == words.len() {
            break;
        }
        i += stride;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_overlaps_at_10_pct() {
        let words = (0..50)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_text(&words, 20);
        assert!(chunks.len() >= 2);
        // First chunk: w0..w19. Stride = 20 - 2 = 18, so second starts at w18.
        assert!(chunks[0].starts_with("w0 "));
        assert!(chunks[1].starts_with("w18 "));
    }

    #[test]
    fn chunker_handles_short_text() {
        let chunks = chunk_text("only three words here", 20);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "only three words here");
    }
}
