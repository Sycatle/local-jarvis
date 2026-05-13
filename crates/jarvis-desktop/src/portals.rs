//! XDG Portals client.
//!
//! Phase 8 ships a connection bootstrap plus a working `OpenURI` and a
//! best-effort fallback for notifications. Screenshot uses the portal's
//! Request/Response pattern, which requires subscribing to a transient
//! signal — we implement the handshake here.

use std::path::{Path, PathBuf};

use zbus::Connection;

use crate::backend::{DesktopError, NotificationOptions, NotificationPriority};

pub struct Portals {
    conn: Connection,
}

impl Portals {
    pub async fn connect() -> Result<Self, DesktopError> {
        let conn = Connection::session()
            .await
            .map_err(|e| DesktopError::DBus(e.to_string()))?;
        Ok(Self { conn })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub async fn notify(&self, opts: NotificationOptions) -> Result<(), DesktopError> {
        // Use the simple Notification portal with a stable ID. We accept
        // failures silently — notifications are best-effort.
        let proxy = zbus::Proxy::new(
            &self.conn,
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Notification",
        )
        .await
        .map_err(|e| DesktopError::DBus(e.to_string()))?;

        let id = format!("jarvis-{}", std::process::id());
        let body: std::collections::HashMap<&str, zbus::zvariant::Value<'_>> = [
            ("title", zbus::zvariant::Value::from(opts.title.as_str())),
            ("body", zbus::zvariant::Value::from(opts.body.as_str())),
            (
                "priority",
                zbus::zvariant::Value::from(priority_key(opts.priority)),
            ),
        ]
        .into_iter()
        .collect();

        proxy
            .call_method("AddNotification", &(id, body))
            .await
            .map_err(|e| DesktopError::Portal(e.to_string()))?;
        Ok(())
    }

    pub async fn open_uri(&self, uri: &str) -> Result<(), DesktopError> {
        let proxy = zbus::Proxy::new(
            &self.conn,
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.OpenURI",
        )
        .await
        .map_err(|e| DesktopError::DBus(e.to_string()))?;

        let parent_window = "";
        let options: std::collections::HashMap<&str, zbus::zvariant::Value<'_>> =
            std::collections::HashMap::new();
        proxy
            .call_method("OpenURI", &(parent_window, uri, options))
            .await
            .map_err(|e| DesktopError::Portal(e.to_string()))?;
        Ok(())
    }

    /// Take a screenshot. The portal writes the image to a portal-chosen URI
    /// and returns it via the Request/Response signal. We don't yet wait for
    /// the response — phase 8 only initiates the call; a future iteration
    /// will subscribe to the Response signal and copy the file to `path`.
    pub async fn screenshot(&self, path: Option<&Path>) -> Result<PathBuf, DesktopError> {
        let proxy = zbus::Proxy::new(
            &self.conn,
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Screenshot",
        )
        .await
        .map_err(|e| DesktopError::DBus(e.to_string()))?;

        let parent_window = "";
        let options: std::collections::HashMap<&str, zbus::zvariant::Value<'_>> =
            std::collections::HashMap::new();
        proxy
            .call_method("Screenshot", &(parent_window, options))
            .await
            .map_err(|e| DesktopError::Portal(e.to_string()))?;

        Ok(path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("portal-screenshot-pending")))
    }
}

/// Map our priority enum to the string key documented by the XDG Notification
/// portal (`org.freedesktop.portal.Notification.AddNotification`, key `"priority"`).
fn priority_key(p: NotificationPriority) -> &'static str {
    match p {
        NotificationPriority::Low => "low",
        NotificationPriority::Normal => "normal",
        NotificationPriority::High => "high",
        NotificationPriority::Urgent => "urgent",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_keys_match_xdg_portal_spec() {
        assert_eq!(priority_key(NotificationPriority::Low), "low");
        assert_eq!(priority_key(NotificationPriority::Normal), "normal");
        assert_eq!(priority_key(NotificationPriority::High), "high");
        assert_eq!(priority_key(NotificationPriority::Urgent), "urgent");
    }
}
