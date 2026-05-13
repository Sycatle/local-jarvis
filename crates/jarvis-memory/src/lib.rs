//! Persistent memory store for the local Jarvis voice assistant.
//!
//! Stores conversation turns and retrievable facts (with optional vector
//! embeddings) in a local SQLite database so that subsequent sessions can be
//! hydrated with prior context and grounded in indexed documents.

#[cfg(feature = "embed")]
pub mod bge;
pub mod embedder;

pub use embedder::Embedder;

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde_json::Value;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS interactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    user_text      TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    tool_calls     TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);

CREATE TABLE IF NOT EXISTS facts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    kind      TEXT NOT NULL,
    source    TEXT,
    content   TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source);
"#;

/// Pre-existing databases may have an older `facts` table without the `source`
/// column. Run a one-shot migration that adds it if missing; ignore the error
/// when it already exists.
fn migrate_facts_source(conn: &Connection) -> Result<()> {
    let already_has: bool = conn
        .prepare("SELECT 1 FROM pragma_table_info('facts') WHERE name = 'source'")?
        .exists([])?;
    if !already_has {
        conn.execute("ALTER TABLE facts ADD COLUMN source TEXT", [])?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct FactHit {
    pub id: i64,
    pub kind: String,
    pub source: Option<String>,
    pub content: String,
    /// Cosine similarity in [-1, 1]; higher is more similar.
    pub score: f32,
}

fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    let n = b.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&b[i * 4..i * 4 + 4]);
        out.push(f32::from_le_bytes(buf));
    }
    out
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[derive(Debug, Clone)]
pub struct Interaction {
    pub id: i64,
    pub ts: DateTime<Utc>,
    pub user_text: String,
    pub assistant_text: String,
    pub tool_calls: Value,
}

pub struct Memory {
    conn: Mutex<Connection>,
}

impl Memory {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("opening sqlite db at {:?}", path.as_ref()))?;
        Self::init(conn)
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("opening in-memory sqlite db")?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(SCHEMA).context("running migrations")?;
        migrate_facts_source(&conn).context("migrating facts.source")?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn record(&self, user_text: &str, assistant_text: &str, tool_calls: &Value) -> Result<i64> {
        let ts = Utc::now().to_rfc3339();
        let tool_calls_str =
            serde_json::to_string(tool_calls).context("serialising tool_calls to JSON")?;
        let conn = self.conn.lock().expect("memory mutex poisoned");
        conn.execute(
            "INSERT INTO interactions (ts, user_text, assistant_text, tool_calls)
             VALUES (?1, ?2, ?3, ?4)",
            params![ts, user_text, assistant_text, tool_calls_str],
        )
        .context("inserting interaction")?;
        Ok(conn.last_insert_rowid())
    }

    pub fn recent(&self, limit: usize) -> Result<Vec<Interaction>> {
        let conn = self.conn.lock().expect("memory mutex poisoned");
        // Pull the most recent `limit` rows, then flip to chronological order.
        let mut stmt = conn.prepare(
            "SELECT id, ts, user_text, assistant_text, tool_calls
             FROM interactions
             ORDER BY id DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let id: i64 = row.get(0)?;
            let ts_str: String = row.get(1)?;
            let user_text: String = row.get(2)?;
            let assistant_text: String = row.get(3)?;
            let tool_calls_str: String = row.get(4)?;
            Ok((id, ts_str, user_text, assistant_text, tool_calls_str))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (id, ts_str, user_text, assistant_text, tool_calls_str) = row?;
            let ts = DateTime::parse_from_rfc3339(&ts_str)
                .with_context(|| format!("parsing ts {ts_str}"))?
                .with_timezone(&Utc);
            let tool_calls: Value = serde_json::from_str(&tool_calls_str)
                .with_context(|| format!("parsing tool_calls JSON for interaction {id}"))?;
            out.push(Interaction {
                id,
                ts,
                user_text,
                assistant_text,
                tool_calls,
            });
        }
        out.reverse();
        Ok(out)
    }

    pub fn forget_all(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("memory mutex poisoned");
        let n = conn
            .execute("DELETE FROM interactions", [])
            .context("deleting interactions")?;
        Ok(n)
    }

    /// Insert a retrievable chunk into the `facts` table. `source` is an
    /// arbitrary tag (typically a file path) used by `forget_source` to
    /// re-index without duplicates.
    pub fn add_fact(
        &self,
        kind: &str,
        source: Option<&str>,
        content: &str,
        embedding: &[f32],
    ) -> Result<i64> {
        let ts = Utc::now().to_rfc3339();
        let bytes = f32_slice_to_bytes(embedding);
        let conn = self.conn.lock().expect("memory mutex poisoned");
        conn.execute(
            "INSERT INTO facts (ts, kind, source, content, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![ts, kind, source, content, bytes],
        )
        .context("inserting fact")?;
        Ok(conn.last_insert_rowid())
    }

    /// Drop every fact tagged with `source`. Returns the number of rows
    /// deleted. Useful before re-indexing a file.
    pub fn forget_source(&self, source: &str) -> Result<usize> {
        let conn = self.conn.lock().expect("memory mutex poisoned");
        let n = conn
            .execute("DELETE FROM facts WHERE source = ?1", params![source])
            .context("deleting facts by source")?;
        Ok(n)
    }

    pub fn facts_count(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("memory mutex poisoned");
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))?;
        Ok(n as usize)
    }

    /// Brute-force cosine-similarity search across all stored facts. For the
    /// expected corpus size (a few thousand chunks of personal documents) the
    /// linear scan in Rust is faster than the round-trip cost of a vector
    /// extension; if `facts_count() > ~50k` consider plugging `sqlite-vec`.
    pub fn search_facts(&self, query: &[f32], top_k: usize) -> Result<Vec<FactHit>> {
        if top_k == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        let q_norm = l2_norm(query);
        if q_norm == 0.0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("memory mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, kind, source, content, embedding
             FROM facts
             WHERE embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let kind: String = row.get(1)?;
            let source: Option<String> = row.get(2)?;
            let content: String = row.get(3)?;
            let blob: Vec<u8> = row.get(4)?;
            Ok((id, kind, source, content, blob))
        })?;

        let mut heap: Vec<FactHit> = Vec::new();
        for row in rows {
            let (id, kind, source, content, blob) = row?;
            let emb = bytes_to_f32_vec(&blob);
            if emb.len() != query.len() {
                continue;
            }
            let n = l2_norm(&emb);
            if n == 0.0 {
                continue;
            }
            let dot: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            let score = dot / (q_norm * n);
            heap.push(FactHit {
                id,
                kind,
                source,
                content,
                score,
            });
        }
        heap.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        heap.truncate(top_k);
        Ok(heap)
    }

    pub fn count(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("memory mutex poisoned");
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM interactions", [], |row| row.get(0))?;
        Ok(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_record_and_recent() {
        let mem = Memory::in_memory().unwrap();
        mem.record("hi 1", "hello 1", &json!([])).unwrap();
        mem.record("hi 2", "hello 2", &json!([])).unwrap();
        mem.record("hi 3", "hello 3", &json!([])).unwrap();

        let got = mem.recent(10).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].user_text, "hi 1");
        assert_eq!(got[1].user_text, "hi 2");
        assert_eq!(got[2].user_text, "hi 3");
        assert_eq!(got[2].assistant_text, "hello 3");
    }

    #[test]
    fn recent_respects_limit() {
        let mem = Memory::in_memory().unwrap();
        for i in 0..5 {
            mem.record(&format!("u{i}"), &format!("a{i}"), &json!([]))
                .unwrap();
        }
        let got = mem.recent(2).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].user_text, "u3");
        assert_eq!(got[1].user_text, "u4");
    }

    #[test]
    fn forget_all_clears() {
        let mem = Memory::in_memory().unwrap();
        mem.record("a", "b", &json!([])).unwrap();
        mem.record("c", "d", &json!([])).unwrap();
        let deleted = mem.forget_all().unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(mem.count().unwrap(), 0);
        assert!(mem.recent(10).unwrap().is_empty());
    }

    #[test]
    fn facts_cosine_search_ranks_by_similarity() {
        let mem = Memory::in_memory().unwrap();
        // Hand-rolled unit vectors so the expected ordering is unambiguous.
        mem.add_fact("doc", Some("a.md"), "alpha", &[1.0, 0.0, 0.0])
            .unwrap();
        mem.add_fact("doc", Some("b.md"), "beta", &[0.0, 1.0, 0.0])
            .unwrap();
        mem.add_fact("doc", Some("c.md"), "gamma", &[0.7, 0.7, 0.0])
            .unwrap();

        let hits = mem.search_facts(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].content, "alpha");
        assert_eq!(hits[1].content, "gamma");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn forget_source_targets_one_file() {
        let mem = Memory::in_memory().unwrap();
        mem.add_fact("doc", Some("a.md"), "x", &[1.0, 0.0]).unwrap();
        mem.add_fact("doc", Some("a.md"), "y", &[0.0, 1.0]).unwrap();
        mem.add_fact("doc", Some("b.md"), "z", &[1.0, 1.0]).unwrap();
        let dropped = mem.forget_source("a.md").unwrap();
        assert_eq!(dropped, 2);
        assert_eq!(mem.facts_count().unwrap(), 1);
    }

    #[test]
    fn tool_calls_json_preserved() {
        let mem = Memory::in_memory().unwrap();
        let tc = json!([
            {
                "name": "search_web",
                "arguments": { "q": "weather paris", "limit": 3 },
                "result": { "ok": true, "hits": ["a", "b", "c"] }
            },
            {
                "name": "get_time",
                "arguments": {},
                "result": "2026-05-13T10:00:00Z"
            }
        ]);
        mem.record("what's up?", "here you go", &tc).unwrap();
        let got = mem.recent(1).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].tool_calls, tc);
    }
}
