//! CLI for the persistent memory store: `jarvis memory show|forget`.
//!
//! These commands open the SQLite file directly (no D-Bus round-trip), so they
//! work even when the daemon is stopped — useful for auditing what the
//! assistant has remembered across sessions.

use anyhow::Result;
use jarvis_core::config::Config;
use jarvis_memory::Memory;

fn open() -> Result<Memory> {
    let cfg = Config::load()?;
    if !cfg.memory.enabled {
        anyhow::bail!("memory is disabled in config ([memory] enabled = false)");
    }
    Memory::open(&cfg.memory.path)
}

pub fn show(limit: usize) -> Result<()> {
    let m = open()?;
    let rows = m.recent(limit)?;
    if rows.is_empty() {
        println!("(no interactions recorded)");
        return Ok(());
    }
    for r in rows {
        println!("[{}] #{}", r.ts.to_rfc3339(), r.id);
        println!("  user      : {}", r.user_text);
        println!("  assistant : {}", r.assistant_text);
        if !r.tool_calls.is_null()
            && !matches!(&r.tool_calls, serde_json::Value::Array(a) if a.is_empty())
        {
            println!(
                "  tools     : {}",
                serde_json::to_string(&r.tool_calls).unwrap_or_default()
            );
        }
    }
    Ok(())
}

pub fn forget() -> Result<()> {
    let m = open()?;
    let n = m.forget_all()?;
    println!("forgot {n} interactions");
    Ok(())
}
