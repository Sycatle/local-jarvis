use std::sync::Arc;
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use thiserror::Error;
use tokio::sync::broadcast;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("no default input device")]
    NoDevice,
    #[error("device error: {0}")]
    Device(String),
    #[error("stream error: {0}")]
    Stream(String),
}

/// One frame of captured audio: mono f32 in [-1.0, 1.0] at the configured rate.
pub type CaptureFrame = Arc<Vec<f32>>;

/// Microphone capture publishing frames into a tokio broadcast channel.
///
/// Frame size = the cpal default buffer for the device. Downstream consumers
/// (clap detector, wake-word, VAD) re-buffer at their preferred window length.
pub struct Capture {
    sample_rate: u32,
    tx: broadcast::Sender<CaptureFrame>,
    _thread: thread::JoinHandle<()>,
}

impl Capture {
    pub fn start(sample_rate: u32) -> Result<Self, CaptureError> {
        let (tx, _rx) = broadcast::channel::<CaptureFrame>(64);

        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), CaptureError>>();
        let tx_thread = tx.clone();

        let handle = thread::spawn(move || {
            let host = cpal::default_host();
            let device = match host.default_input_device() {
                Some(d) => d,
                None => {
                    let _ = ready_tx.send(Err(CaptureError::NoDevice));
                    return;
                }
            };

            let supported = match device.default_input_config() {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(CaptureError::Device(e.to_string())));
                    return;
                }
            };

            let channels = supported.channels() as usize;
            let config = StreamConfig {
                channels: supported.channels(),
                sample_rate,
                buffer_size: cpal::BufferSize::Default,
            };

            let publish = move |data: &[f32]| {
                let mono: Vec<f32> = if channels == 1 {
                    data.to_vec()
                } else {
                    data.chunks(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect()
                };
                let _ = tx_thread.send(Arc::new(mono));
            };

            let err_fn = |e| tracing::error!("input stream error: {e}");

            let stream_res = match supported.sample_format() {
                SampleFormat::F32 => device.build_input_stream(
                    &config,
                    move |data: &[f32], _| publish(data),
                    err_fn,
                    None,
                ),
                SampleFormat::I16 => device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        let f: Vec<f32> =
                            data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                        publish(&f);
                    },
                    err_fn,
                    None,
                ),
                SampleFormat::U16 => device.build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        let f: Vec<f32> = data
                            .iter()
                            .map(|s| (*s as f32 - 32_768.0) / 32_768.0)
                            .collect();
                        publish(&f);
                    },
                    err_fn,
                    None,
                ),
                other => {
                    let _ = ready_tx.send(Err(CaptureError::Device(format!(
                        "unsupported sample format: {:?}",
                        other
                    ))));
                    return;
                }
            };

            let stream = match stream_res {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(CaptureError::Stream(e.to_string())));
                    return;
                }
            };
            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(CaptureError::Stream(e.to_string())));
                return;
            }
            let _ = ready_tx.send(Ok(()));

            loop {
                thread::park_timeout(std::time::Duration::from_secs(60));
            }
        });

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sample_rate,
                tx,
                _thread: handle,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(CaptureError::Stream("capture thread died".into())),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn subscribe(&self) -> broadcast::Receiver<CaptureFrame> {
        self.tx.subscribe()
    }
}
