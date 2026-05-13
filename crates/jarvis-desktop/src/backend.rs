use std::path::{Path, PathBuf};

use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DesktopError {
    #[error("D-Bus error: {0}")]
    DBus(String),
    #[error("portal call failed: {0}")]
    Portal(String),
    #[error("subprocess error: {0}")]
    Subprocess(String),
    #[error("unsupported on this session")]
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct NotificationOptions {
    pub title: String,
    pub body: String,
    pub priority: NotificationPriority,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NotificationPriority {
    Low,
    #[default]
    Normal,
    High,
    Urgent,
}

#[async_trait]
pub trait Desktop: Send + Sync {
    async fn notify(&self, opts: NotificationOptions) -> Result<(), DesktopError>;
    async fn screenshot(&self, path: Option<&Path>) -> Result<PathBuf, DesktopError>;
    async fn open_uri(&self, uri: &str) -> Result<(), DesktopError>;
    async fn focus_window(&self, pattern: &str) -> Result<(), DesktopError>;
}
