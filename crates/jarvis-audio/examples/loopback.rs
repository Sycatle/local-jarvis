//! Captures mic audio at 16 kHz and prints peak amplitude per frame.
//! Ctrl-C to stop.
//!
//! Run: `cargo run --example loopback -p jarvis-audio`

use jarvis_audio::Capture;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let capture = Capture::start(16_000)?;
    let mut rx = capture.subscribe();
    tracing::info!("capture started @ {} Hz — speak into the mic", capture.sample_rate());
    let mut frames = 0u64;
    while let Ok(frame) = rx.recv().await {
        let peak = frame
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, |a, b| a.max(b));
        if frames.is_multiple_of(10) {
            tracing::info!("frame {frames}: peak={:.3}, len={}", peak, frame.len());
        }
        frames += 1;
    }
    Ok(())
}
