use anyhow::Result;
use clap::{Parser, Subcommand};

mod memory_cli;
mod orchestrator;
mod runner;
mod skill_cli;
mod skills_state;
mod streaming;
mod tts_preview;
mod tui;

#[derive(Parser)]
#[command(name = "jarvis", version, about = "Local voice assistant", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum MemoryCmd {
    /// Print the last N recorded interactions (oldest first). Default 10.
    Show {
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Drop every recorded interaction from the SQLite store.
    Forget,
}

#[derive(Subcommand)]
enum Command {
    /// Run the foreground service (state machine + D-Bus).
    Run,
    /// Interactive terminal UI (Ratatui) connected to the running daemon.
    Tui,
    /// Speak text via the running daemon.
    Say { text: String },
    /// Trigger a listen cycle on the running daemon, print the transcript.
    Listen,
    /// Query the running daemon's state.
    Status,
    /// Cancel any in-progress speech.
    Cancel,
    /// Print the resolved config file path.
    ConfigPath,
    /// Open the config file in $EDITOR (falls back to $VISUAL, then `vi`) and
    /// validate the result via Figment before returning. Bad edits are kept
    /// on disk but flagged loudly so the user can fix them.
    ConfigEdit,
    /// Directly invoke a registered skill by name (debug aid, bypasses LLM).
    Skill {
        /// Skill name, e.g. `volume`, `notify`, `media_play_pause`.
        name: String,
        /// JSON arguments object. Defaults to `{}`.
        #[arg(long, default_value = "{}")]
        args: String,
    },
    /// List every registered skill name.
    Skills,
    /// Show or forget remembered interactions from the persistent SQLite
    /// memory store (read-only access; daemon does not need to be running).
    Memory {
        #[command(subcommand)]
        cmd: MemoryCmd,
    },
    /// Enable a previously-disabled skill across daemon restarts.
    SkillEnable { name: String },
    /// Disable a skill across daemon restarts (it is dropped at boot).
    SkillDisable { name: String },
    /// Synthesise a phrase with Kokoro locally and play it. Useful to A/B-test
    /// voices without restarting the daemon.
    TtsPreview {
        /// Voice name (e.g. `im_nicola`, `bm_george`, `ff_siwis`). Defaults to
        /// the configured voice.
        #[arg(long)]
        voice: Option<String>,
        /// Phrase to synthesise.
        #[arg(long, default_value = "Bonjour Sir, ravi de vous servir.")]
        text: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();
    match cli.command {
        Command::Run => runner::run().await,
        Command::Tui => tui::run().await,
        Command::Say { text } => client::call_speak(&text).await,
        Command::Listen => {
            let t = client::call_listen().await?;
            println!("{t}");
            Ok(())
        }
        Command::Status => {
            let s = client::call_status().await?;
            println!("{s}");
            Ok(())
        }
        Command::Cancel => client::call_cancel().await,
        Command::ConfigPath => {
            println!("{}", jarvis_core::dirs::config_file().display());
            Ok(())
        }
        Command::ConfigEdit => config_edit().await,
        Command::Skill { name, args } => skill_cli::invoke(&name, &args).await,
        Command::Skills => skill_cli::list().await,
        Command::TtsPreview { voice, text } => tts_preview::run(voice, &text).await,
        Command::Memory { cmd } => match cmd {
            MemoryCmd::Show { limit } => memory_cli::show(limit),
            MemoryCmd::Forget => memory_cli::forget(),
        },
        Command::SkillEnable { name } => skills_state::enable(&name),
        Command::SkillDisable { name } => skills_state::disable(&name),
    }
}

async fn config_edit() -> Result<()> {
    use std::process::Command as Proc;
    let path = jarvis_core::dirs::config_file();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !path.exists() {
        let cfg = jarvis_core::config::Config::default();
        std::fs::write(&path, toml::to_string_pretty(&cfg)?)?;
    }
    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "vi".to_string());
    let status = Proc::new(&editor).arg(&path).status()?;
    if !status.success() {
        anyhow::bail!("editor `{editor}` exited with status {status}");
    }
    match jarvis_core::config::Config::load_from(&path) {
        Ok(_) => {
            println!("ok: {}", path.display());
            Ok(())
        }
        Err(e) => {
            eprintln!(
                "warning: config at {} fails to parse:\n  {e}",
                path.display()
            );
            eprintln!("the daemon will refuse to start until this is fixed.");
            Err(e)
        }
    }
}

fn init_logging() {
    // Send to journald when running under systemd, else stderr.
    let under_systemd =
        std::env::var_os("INVOCATION_ID").is_some() || std::env::var_os("JOURNAL_STREAM").is_some();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if under_systemd {
        if let Ok(layer) = tracing_journald::layer() {
            use tracing_subscriber::layer::SubscriberExt;
            use tracing_subscriber::util::SubscriberInitExt;
            tracing_subscriber::registry()
                .with(filter)
                .with(layer)
                .init();
            return;
        }
    }
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

mod client {
    use anyhow::Result;
    use jarvis_service::{BUS_NAME, OBJECT_PATH};
    use zbus::Connection;

    async fn proxy() -> Result<zbus::Proxy<'static>> {
        let conn = Connection::session().await?;
        let p = zbus::Proxy::new(&conn, BUS_NAME, OBJECT_PATH, "org.jarvis.Assistant").await?;
        // The connection must outlive the proxy; leak it on purpose for the
        // brief CLI lifetime.
        Box::leak(Box::new(conn));
        Ok(p)
    }

    pub async fn call_speak(text: &str) -> Result<()> {
        let p = proxy().await?;
        p.call_method("Speak", &(text,)).await?;
        Ok(())
    }

    pub async fn call_listen() -> Result<String> {
        let p = proxy().await?;
        let m = p.call_method("Listen", &()).await?;
        let body: String = m.body().deserialize()?;
        Ok(body)
    }

    pub async fn call_status() -> Result<String> {
        let p = proxy().await?;
        let m = p.call_method("Status", &()).await?;
        let body: String = m.body().deserialize()?;
        Ok(body)
    }

    pub async fn call_cancel() -> Result<()> {
        let p = proxy().await?;
        p.call_method("Cancel", &()).await?;
        Ok(())
    }
}
