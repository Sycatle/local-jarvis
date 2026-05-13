//! Capability manifest declared by each skill.
//!
//! Inspired by the security postmortems on OpenClaw's plugin marketplace: every
//! built-in skill states up front which binaries it executes, which paths it
//! touches, and whether it needs network or D-Bus access. The registry logs
//! each invocation with the declared caps so users can audit what the
//! assistant actually did. Future dynamic skill loading will enforce these as
//! hard sandbox boundaries (capability model + subprocess), but for now they
//! serve as documentation + structured audit trail.

use std::path::PathBuf;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Capabilities {
    /// Filesystem paths the skill may read from.
    pub fs_read: Vec<PathBuf>,
    /// Filesystem paths the skill may write to.
    pub fs_write: Vec<PathBuf>,
    /// External binaries the skill may spawn (basename match).
    pub exec: Vec<String>,
    /// Whether the skill performs network I/O.
    pub net: bool,
    /// D-Bus interface names the skill talks to.
    pub dbus: Vec<String>,
}

impl Capabilities {
    pub fn none() -> Self {
        Self::default()
    }

    /// Convenience builder for the common "I spawn these binaries" case.
    pub fn exec(bins: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            exec: bins.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Convenience builder for skills that go through D-Bus / portals.
    pub fn dbus(interfaces: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            dbus: interfaces.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    pub fn with_exec(mut self, bin: impl Into<String>) -> Self {
        self.exec.push(bin.into());
        self
    }

    pub fn with_dbus(mut self, iface: impl Into<String>) -> Self {
        self.dbus.push(iface.into());
        self
    }

    pub fn with_net(mut self) -> Self {
        self.net = true;
        self
    }
}
