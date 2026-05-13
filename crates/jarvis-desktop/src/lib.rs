//! Desktop integration: a small abstraction over the operations Jarvis
//! performs on the user's session.
//!
//! Backends:
//! - [`portals`]: XDG Portals (`org.freedesktop.portal.*`) via `zbus`.
//!   Preferred for any visible UI action (screenshot, notification, openuri,
//!   idle inhibitor) — works across X11, Wayland, and Flatpak.
//! - [`x11`]: subprocess wrappers around `wmctrl` / `xdotool` for things the
//!   portals do not cover (window focus). Future Wayland equivalent uses
//!   `ydotool`.
//!
//! [`make_desktop`] selects the right composite backend based on
//! `$XDG_SESSION_TYPE`.

pub mod backend;
pub mod portals;
pub mod x11;

pub use backend::{Desktop, DesktopError, NotificationOptions};
pub use portals::Portals;
pub use x11::X11Backend;

use async_trait::async_trait;
use std::sync::Arc;

/// Composite backend used by the orchestrator.
pub struct DesktopFacade {
    portals: Arc<Portals>,
    x11: Arc<X11Backend>,
}

impl DesktopFacade {
    pub async fn new() -> Result<Self, DesktopError> {
        let portals = Arc::new(Portals::connect().await?);
        let x11 = Arc::new(X11Backend::new());
        Ok(Self { portals, x11 })
    }
}

#[async_trait]
impl Desktop for DesktopFacade {
    async fn notify(&self, opts: NotificationOptions) -> Result<(), DesktopError> {
        self.portals.notify(opts).await
    }

    async fn screenshot(&self, path: Option<&std::path::Path>) -> Result<std::path::PathBuf, DesktopError> {
        self.portals.screenshot(path).await
    }

    async fn open_uri(&self, uri: &str) -> Result<(), DesktopError> {
        self.portals.open_uri(uri).await
    }

    async fn focus_window(&self, pattern: &str) -> Result<(), DesktopError> {
        self.x11.focus_window(pattern).await
    }
}

pub async fn make_desktop() -> Result<Arc<dyn Desktop>, DesktopError> {
    let facade = DesktopFacade::new().await?;
    Ok(Arc::new(facade))
}
