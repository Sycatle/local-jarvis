//! Plays the wake chime. Useful smoke test for the audio output path.
//!
//! Run: `cargo run --example ding -p jarvis-audio`

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    tracing::info!("playing ding...");
    jarvis_audio::play_ding()?;
    tracing::info!("done");
    Ok(())
}
