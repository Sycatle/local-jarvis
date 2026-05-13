//! XDG Portals client.
//!
//! Phase 8 ships a connection bootstrap plus a working `OpenURI` and a
//! best-effort fallback for notifications. Screenshot uses the portal's
//! Request/Response pattern, which requires subscribing to a transient
//! signal — we implement the handshake here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::stream::StreamExt;
use zbus::zvariant::{OwnedValue, Value as ZValue};
use zbus::{Connection, MatchRule, MessageStream};

use crate::backend::{DesktopError, NotificationOptions, NotificationPriority};

const SCREENSHOT_TIMEOUT: Duration = Duration::from_secs(30);

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

    /// Take a screenshot via the XDG Screenshot portal.
    ///
    /// The portal writes the image to a sandbox-friendly URI and returns it
    /// through the Request/Response signal. We:
    /// 1. Pick a `handle_token`, derive the expected Request object path.
    /// 2. Subscribe to its `Response` signal **before** the call (to avoid
    ///    racing the portal's emit).
    /// 3. Issue `Screenshot(parent, { handle_token, interactive: false, modal: false })`.
    /// 4. Await the Response (timeout: 30 s), parse the returned `uri`.
    /// 5. If `path` is given, copy the file there; otherwise return the
    ///    portal's local path as-is.
    pub async fn screenshot(&self, path: Option<&Path>) -> Result<PathBuf, DesktopError> {
        let unique = self
            .conn
            .unique_name()
            .ok_or_else(|| DesktopError::DBus("no unique name on session bus".into()))?
            .as_str()
            .to_string();
        let sender = munge_sender(&unique);
        let token = format!(
            "jarvis_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let request_path = format!("/org/freedesktop/portal/desktop/request/{sender}/{token}");

        let rule = MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .interface("org.freedesktop.portal.Request")
            .map_err(|e| DesktopError::DBus(e.to_string()))?
            .path(request_path.as_str())
            .map_err(|e| DesktopError::DBus(e.to_string()))?
            .member("Response")
            .map_err(|e| DesktopError::DBus(e.to_string()))?
            .build();
        let mut stream = MessageStream::for_match_rule(rule, &self.conn, Some(1))
            .await
            .map_err(|e| DesktopError::DBus(e.to_string()))?;

        let proxy = zbus::Proxy::new(
            &self.conn,
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.Screenshot",
        )
        .await
        .map_err(|e| DesktopError::DBus(e.to_string()))?;

        let parent_window = "";
        let options: HashMap<&str, ZValue<'_>> = [
            ("handle_token", ZValue::from(token.as_str())),
            ("interactive", ZValue::from(false)),
            ("modal", ZValue::from(false)),
        ]
        .into_iter()
        .collect();
        proxy
            .call_method("Screenshot", &(parent_window, options))
            .await
            .map_err(|e| DesktopError::Portal(e.to_string()))?;

        let response_msg = tokio::time::timeout(SCREENSHOT_TIMEOUT, stream.next())
            .await
            .map_err(|_| DesktopError::Portal("screenshot portal response timed out".into()))?
            .ok_or_else(|| DesktopError::Portal("screenshot signal stream closed".into()))?
            .map_err(|e| DesktopError::DBus(e.to_string()))?;

        let body = response_msg.body();
        let (response, results): (u32, HashMap<String, OwnedValue>) = body
            .deserialize()
            .map_err(|e| DesktopError::Portal(format!("response decode: {e}")))?;
        if response != 0 {
            return Err(DesktopError::Portal(format!(
                "screenshot portal reported response code {response}"
            )));
        }

        let uri_value = results
            .get("uri")
            .ok_or_else(|| DesktopError::Portal("portal response missing `uri`".into()))?;
        let uri: String = <&str>::try_from(uri_value)
            .map(str::to_owned)
            .map_err(|e| DesktopError::Portal(format!("portal `uri` not a string: {e}")))?;

        let src = uri_to_local_path(&uri)
            .ok_or_else(|| DesktopError::Portal(format!("unsupported portal uri scheme: {uri}")))?;

        match path {
            Some(dest) => {
                tokio::fs::copy(&src, dest)
                    .await
                    .map_err(|e| DesktopError::Subprocess(format!("copy {src:?} → {dest:?}: {e}")))?;
                Ok(dest.to_path_buf())
            }
            None => Ok(src),
        }
    }
}

/// Convert a session-unique name (e.g. `:1.42`) to the form the portal embeds
/// in the Request object path: leading colon stripped, dots replaced by
/// underscores (see XDG-Desktop-Portal spec, `org.freedesktop.portal.Request`).
fn munge_sender(unique: &str) -> String {
    unique.trim_start_matches(':').replace('.', "_")
}

/// Best-effort `file://` URI → local path. Returns `None` for any other
/// scheme (the portal sometimes hands back a non-file URI in flatpak setups;
/// we don't try to fetch it here).
fn uri_to_local_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    // Drop the authority component if present (`file://host/path` → `/path`).
    let path_start = rest.find('/')?;
    Some(PathBuf::from(&rest[path_start..]))
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

    #[test]
    fn munge_sender_strips_colon_and_escapes_dots() {
        assert_eq!(munge_sender(":1.42"), "1_42");
        assert_eq!(munge_sender(":1.1234"), "1_1234");
    }

    #[test]
    fn uri_to_local_path_handles_file_scheme() {
        assert_eq!(
            uri_to_local_path("file:///tmp/portal-shot.png"),
            Some(PathBuf::from("/tmp/portal-shot.png"))
        );
        assert_eq!(
            uri_to_local_path("file://host/var/lib/foo.png"),
            Some(PathBuf::from("/var/lib/foo.png"))
        );
    }

    #[test]
    fn uri_to_local_path_rejects_other_schemes() {
        assert_eq!(uri_to_local_path("http://example.com/x.png"), None);
        assert_eq!(uri_to_local_path("/already/a/path.png"), None);
    }
}
