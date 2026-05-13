//! Persistent on/off state for skills, stored alongside the user's other
//! state at `~/.local/state/jarvis/skills.toml`.
//!
//! Format:
//! ```toml
//! disabled = ["volume", "brightness_up"]
//! ```
//!
//! Read at boot by [`crate::runner`] to drop disabled skills from the
//! registry. Edited via `jarvis skills enable/disable <name>`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SkillsState {
    #[serde(default)]
    pub disabled: BTreeSet<String>,
}

fn path() -> PathBuf {
    // Mirror jarvis_core::dirs::data_dir() but for XDG state, which we don't
    // have a helper for; fall back to data_dir if XDG_STATE_HOME is unset.
    if let Some(xdg) = std::env::var_os("XDG_STATE_HOME") {
        PathBuf::from(xdg).join("jarvis").join("skills.toml")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("jarvis")
            .join("skills.toml")
    } else {
        PathBuf::from("/tmp/jarvis-skills.toml")
    }
}

pub fn load() -> SkillsState {
    let p = path();
    if !p.exists() {
        return SkillsState::default();
    }
    match std::fs::read_to_string(&p) {
        Ok(s) => toml::from_str(&s).unwrap_or_default(),
        Err(_) => SkillsState::default(),
    }
}

fn save(state: &SkillsState) -> Result<()> {
    let p = path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = toml::to_string_pretty(state)?;
    std::fs::write(&p, s)?;
    Ok(())
}

pub fn enable(name: &str) -> Result<()> {
    let mut s = load();
    if s.disabled.remove(name) {
        save(&s)?;
        println!("enabled: {name}");
    } else {
        println!("{name} is already enabled");
    }
    Ok(())
}

pub fn disable(name: &str) -> Result<()> {
    let mut s = load();
    if s.disabled.insert(name.to_string()) {
        save(&s)?;
        println!("disabled: {name}");
    } else {
        println!("{name} is already disabled");
    }
    Ok(())
}
