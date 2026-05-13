//! Media skills: MPRIS via `playerctl` subprocess.

use std::sync::Arc;

use jarvis_skills_macros::skill;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;

use crate::capabilities::Capabilities;
use crate::registry::Skill;

async fn playerctl(args: &[&str]) -> Value {
    match Command::new("playerctl").args(args).output().await {
        Ok(o) if o.status.success() => json!({
            "ok": true,
            "stdout": String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        }),
        Ok(o) => json!({
            "ok": false,
            "error": format!("playerctl exit {:?}", o.status.code()),
            "stderr": String::from_utf8_lossy(&o.stderr).into_owned(),
        }),
        Err(e) => json!({"ok": false, "error": format!("spawn playerctl: {e}")}),
    }
}

#[derive(Deserialize, JsonSchema, Default)]
pub struct EmptyArgs {}

pub struct PlayPauseSkill;
#[skill(
    name = "media_play_pause",
    description = "Toggle play/pause on the active media player."
)]
impl PlayPauseSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["playerctl"])
    }
    async fn run(&self, _args: EmptyArgs) -> Value {
        playerctl(&["play-pause"]).await
    }
}

pub struct NextSkill;
#[skill(name = "media_next", description = "Skip to the next track.")]
impl NextSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["playerctl"])
    }
    async fn run(&self, _args: EmptyArgs) -> Value {
        playerctl(&["next"]).await
    }
}

pub struct PreviousSkill;
#[skill(name = "media_previous", description = "Go to the previous track.")]
impl PreviousSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["playerctl"])
    }
    async fn run(&self, _args: EmptyArgs) -> Value {
        playerctl(&["previous"]).await
    }
}

pub struct StopSkill;
#[skill(name = "media_stop", description = "Stop the active media player.")]
impl StopSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["playerctl"])
    }
    async fn run(&self, _args: EmptyArgs) -> Value {
        playerctl(&["stop"]).await
    }
}

pub struct NowPlayingSkill;
#[skill(
    name = "media_now_playing",
    description = "Return the currently playing title and artist."
)]
impl NowPlayingSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["playerctl"])
    }
    async fn run(&self, _args: EmptyArgs) -> Value {
        playerctl(&["metadata", "--format", "{{ artist }} — {{ title }}"]).await
    }
}

pub struct MediaSkills;

impl MediaSkills {
    pub fn skills() -> Vec<Arc<dyn Skill>> {
        vec![
            Arc::new(PlayPauseSkill),
            Arc::new(NextSkill),
            Arc::new(PreviousSkill),
            Arc::new(StopSkill),
            Arc::new(NowPlayingSkill),
        ]
    }
}
