//! D-Bus session service `org.jarvis.Assistant`.
//!
//! Exposes:
//! - `Speak(s text) -> ()` — raw TTS, says the text verbatim
//! - `Ask(s prompt) -> (s)` — runs prompt through the LLM tool-loop and returns the reply
//! - `Listen() -> (s)` — returns the latest transcript
//! - `Status() -> (s)` — current state
//! - `Cancel() -> ()` — interrupt TTS
//!
//! Emits `StateChanged(s)`, `Transcribed(s)`, `Spoken(s)`.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, watch};
use zbus::{interface, object_server::SignalContext, Connection};

pub const BUS_NAME: &str = "org.jarvis.Assistant";
pub const OBJECT_PATH: &str = "/org/jarvis/Assistant";

/// Commands that flow from the bus to the orchestrator.
#[derive(Debug)]
pub enum BusCommand {
    Speak(String),
    Ask(String, tokio::sync::oneshot::Sender<String>),
    Listen(tokio::sync::oneshot::Sender<String>),
    Cancel,
}

#[async_trait]
pub trait Orchestrator: Send + Sync {
    async fn handle(&self, cmd: BusCommand);
}

pub struct Service {
    cmd_tx: mpsc::Sender<BusCommand>,
    state_rx: watch::Receiver<String>,
}

#[interface(name = "org.jarvis.Assistant")]
impl Service {
    async fn speak(&self, text: String) {
        let _ = self.cmd_tx.send(BusCommand::Speak(text)).await;
    }

    async fn ask(&self, prompt: String) -> String {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.cmd_tx.send(BusCommand::Ask(prompt, tx)).await.is_err() {
            return String::new();
        }
        rx.await.unwrap_or_default()
    }

    async fn listen(&self) -> String {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.cmd_tx.send(BusCommand::Listen(tx)).await.is_err() {
            return String::new();
        }
        rx.await.unwrap_or_default()
    }

    async fn status(&self) -> String {
        self.state_rx.borrow().clone()
    }

    async fn cancel(&self) {
        let _ = self.cmd_tx.send(BusCommand::Cancel).await;
    }

    #[zbus(signal)]
    async fn state_changed(emitter: &SignalContext<'_>, state: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn transcribed(emitter: &SignalContext<'_>, text: &str) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn spoken(emitter: &SignalContext<'_>, text: &str) -> zbus::Result<()>;

    /// Emitted once per ReAct iteration. `action` is empty when the model
    /// produced a final reply rather than a tool call; `observation` is empty
    /// until the tool result is available (the signal fires twice per
    /// tool-using step — once at dispatch, once after the result).
    #[zbus(signal)]
    async fn step_taken(
        emitter: &SignalContext<'_>,
        iteration: u32,
        thought: &str,
        action: &str,
        observation: &str,
    ) -> zbus::Result<()>;
}

pub struct ServiceHandle {
    pub conn: Connection,
}

impl ServiceHandle {
    pub async fn start(
        cmd_tx: mpsc::Sender<BusCommand>,
        state_rx: watch::Receiver<String>,
    ) -> anyhow::Result<Arc<Self>> {
        let service = Service { cmd_tx, state_rx };
        let conn = zbus::connection::Builder::session()?
            .name(BUS_NAME)?
            .serve_at(OBJECT_PATH, service)?
            .build()
            .await?;
        tracing::info!("D-Bus service available at {BUS_NAME} {OBJECT_PATH}");
        Ok(Arc::new(Self { conn }))
    }

    pub async fn emit_state_changed(&self, state: &str) -> anyhow::Result<()> {
        let iface_ref = self
            .conn
            .object_server()
            .interface::<_, Service>(OBJECT_PATH)
            .await?;
        Service::state_changed(iface_ref.signal_context(), state).await?;
        Ok(())
    }

    pub async fn emit_transcribed(&self, text: &str) -> anyhow::Result<()> {
        let iface_ref = self
            .conn
            .object_server()
            .interface::<_, Service>(OBJECT_PATH)
            .await?;
        Service::transcribed(iface_ref.signal_context(), text).await?;
        Ok(())
    }

    pub async fn emit_spoken(&self, text: &str) -> anyhow::Result<()> {
        let iface_ref = self
            .conn
            .object_server()
            .interface::<_, Service>(OBJECT_PATH)
            .await?;
        Service::spoken(iface_ref.signal_context(), text).await?;
        Ok(())
    }

    pub async fn emit_step_taken(
        &self,
        iteration: u32,
        thought: &str,
        action: &str,
        observation: &str,
    ) -> anyhow::Result<()> {
        let iface_ref = self
            .conn
            .object_server()
            .interface::<_, Service>(OBJECT_PATH)
            .await?;
        Service::step_taken(iface_ref.signal_context(), iteration, thought, action, observation)
            .await?;
        Ok(())
    }
}
