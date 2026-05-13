//! X11 helpers via `wmctrl` / `xdotool`. Wayland equivalents will land here
//! (or in a sibling `wayland.rs`) once the COSMIC migration is on the table.

use tokio::process::Command;

use crate::backend::DesktopError;

pub struct X11Backend;

impl X11Backend {
    pub fn new() -> Self {
        Self
    }

    pub async fn focus_window(&self, pattern: &str) -> Result<(), DesktopError> {
        let out = Command::new("wmctrl")
            .args(["-a", pattern])
            .status()
            .await
            .map_err(|e| DesktopError::Subprocess(format!("spawn wmctrl: {e}")))?;
        if !out.success() {
            return Err(DesktopError::Subprocess(format!(
                "wmctrl exit {:?}",
                out.code()
            )));
        }
        Ok(())
    }
}

impl Default for X11Backend {
    fn default() -> Self {
        Self::new()
    }
}
