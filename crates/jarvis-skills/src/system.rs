//! System control skills: volume, brightness, launch app, focus window,
//! notify, screenshot. Each declares its capability manifest inline so the
//! registry's audit log records what the assistant actually invoked.

use std::sync::Arc;

use jarvis_desktop::{backend::NotificationPriority, Desktop, NotificationOptions};
use jarvis_skills_macros::skill;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;

use crate::capabilities::Capabilities;
use crate::registry::Skill;

pub struct SystemSkills {
    desktop: Arc<dyn Desktop>,
}

impl SystemSkills {
    pub fn new(desktop: Arc<dyn Desktop>) -> Self {
        Self { desktop }
    }
}

async fn run_cmd(cmd: &str, args: &[&str]) -> Value {
    match Command::new(cmd).args(args).output().await {
        Ok(o) if o.status.success() => json!({"ok": true}),
        Ok(o) => json!({
            "ok": false,
            "error": format!("{cmd} exit {:?}", o.status.code()),
            "stderr": String::from_utf8_lossy(&o.stderr).into_owned(),
        }),
        Err(e) => json!({"ok": false, "error": format!("spawn {cmd}: {e}")}),
    }
}

// ---------------- Volume ----------------

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum VolumeAction {
    Up,
    Down,
    Set,
    Mute,
}

#[derive(Deserialize, JsonSchema)]
pub struct VolumeArgs {
    pub action: VolumeAction,
    #[serde(default)]
    pub value: Option<u32>,
}

pub struct VolumeSkill;

#[skill(
    name = "volume",
    description = "Adjust the default audio sink: 'up'/'down' tweaks by step, 'set' uses value (0-100), 'mute' toggles."
)]
impl VolumeSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["wpctl"])
    }
    async fn run(&self, args: VolumeArgs) -> Value {
        match args.action {
            VolumeAction::Up => {
                run_cmd("wpctl", &["set-volume", "@DEFAULT_AUDIO_SINK@", "5%+"]).await
            }
            VolumeAction::Down => {
                run_cmd("wpctl", &["set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"]).await
            }
            VolumeAction::Mute => {
                run_cmd("wpctl", &["set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"]).await
            }
            VolumeAction::Set => {
                let v = args.value.unwrap_or(50).min(100);
                let pct = format!("{}%", v);
                run_cmd("wpctl", &["set-volume", "@DEFAULT_AUDIO_SINK@", &pct]).await
            }
        }
    }
}

// ---------------- Brightness ----------------

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BrightnessAction {
    Up,
    Down,
    Set,
}

#[derive(Deserialize, JsonSchema)]
pub struct BrightnessArgs {
    pub action: BrightnessAction,
    #[serde(default)]
    pub value: Option<u32>,
}

pub struct BrightnessSkill;

#[skill(
    name = "brightness",
    description = "Adjust display brightness via brightnessctl."
)]
impl BrightnessSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["brightnessctl"])
    }
    async fn run(&self, args: BrightnessArgs) -> Value {
        match args.action {
            BrightnessAction::Up => run_cmd("brightnessctl", &["set", "+10%"]).await,
            BrightnessAction::Down => run_cmd("brightnessctl", &["set", "10%-"]).await,
            BrightnessAction::Set => {
                let v = args.value.unwrap_or(50).clamp(1, 100);
                let pct = format!("{}%", v);
                run_cmd("brightnessctl", &["set", &pct]).await
            }
        }
    }
}

// ---------------- Launch app ----------------

#[derive(Deserialize, JsonSchema)]
pub struct LaunchAppArgs {
    pub name: String,
}

pub struct LaunchAppSkill;

#[skill(
    name = "launch_app",
    description = "Launch an application by its desktop name or executable."
)]
impl LaunchAppSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["gtk-launch", "xdg-open"])
    }
    async fn run(&self, args: LaunchAppArgs) -> Value {
        // Try gtk-launch first (resolves .desktop files), fall back to xdg-open.
        if let Ok(s) = Command::new("gtk-launch").arg(&args.name).status().await {
            if s.success() {
                return json!({"ok": true, "via": "gtk-launch"});
            }
        }
        match Command::new("xdg-open").arg(&args.name).status().await {
            Ok(s) if s.success() => json!({"ok": true, "via": "xdg-open"}),
            Ok(s) => json!({"ok": false, "error": format!("xdg-open exit {:?}", s.code())}),
            Err(e) => json!({"ok": false, "error": format!("spawn xdg-open: {e}")}),
        }
    }
}

// ---------------- Focus window ----------------

#[derive(Deserialize, JsonSchema)]
pub struct FocusWindowArgs {
    pub pattern: String,
}

pub struct FocusWindowSkill {
    desktop: Arc<dyn Desktop>,
}

impl FocusWindowSkill {
    pub fn new(desktop: Arc<dyn Desktop>) -> Self {
        Self { desktop }
    }
}

#[skill(
    name = "focus_window",
    description = "Bring a window matching a substring pattern to the foreground."
)]
impl FocusWindowSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::exec(["wmctrl"])
    }
    async fn run(&self, args: FocusWindowArgs) -> Value {
        match self.desktop.focus_window(&args.pattern).await {
            Ok(()) => json!({"ok": true}),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        }
    }
}

// ---------------- Notify ----------------

#[derive(Deserialize, JsonSchema)]
pub struct NotifyArgs {
    pub title: String,
    pub body: String,
}

pub struct NotifySkill {
    desktop: Arc<dyn Desktop>,
}

impl NotifySkill {
    pub fn new(desktop: Arc<dyn Desktop>) -> Self {
        Self { desktop }
    }
}

#[skill(
    name = "notify",
    description = "Show a desktop notification."
)]
impl NotifySkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::dbus(["org.freedesktop.portal.Notification"])
    }
    async fn run(&self, args: NotifyArgs) -> Value {
        let res = self
            .desktop
            .notify(NotificationOptions {
                title: args.title,
                body: args.body,
                priority: NotificationPriority::default(),
            })
            .await;
        match res {
            Ok(()) => json!({"ok": true}),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        }
    }
}

// ---------------- Screenshot ----------------

#[derive(Deserialize, JsonSchema, Default)]
pub struct ScreenshotArgs {}

pub struct ScreenshotSkill {
    desktop: Arc<dyn Desktop>,
}

impl ScreenshotSkill {
    pub fn new(desktop: Arc<dyn Desktop>) -> Self {
        Self { desktop }
    }
}

#[skill(
    name = "screenshot",
    description = "Take a screenshot via the XDG Portal."
)]
impl ScreenshotSkill {
    fn capabilities(&self) -> Capabilities {
        Capabilities::dbus(["org.freedesktop.portal.Screenshot"])
    }
    async fn run(&self, _args: ScreenshotArgs) -> Value {
        match self.desktop.screenshot(None).await {
            Ok(p) => json!({"ok": true, "path": p.to_string_lossy()}),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        }
    }
}

impl SystemSkills {
    /// Hand out an `Arc`'d list of every built-in system skill.
    pub fn skills(self) -> Vec<Arc<dyn Skill>> {
        let desktop = self.desktop;
        vec![
            Arc::new(VolumeSkill),
            Arc::new(BrightnessSkill),
            Arc::new(LaunchAppSkill),
            Arc::new(FocusWindowSkill::new(Arc::clone(&desktop))),
            Arc::new(NotifySkill::new(Arc::clone(&desktop))),
            Arc::new(ScreenshotSkill::new(desktop)),
        ]
    }
}
