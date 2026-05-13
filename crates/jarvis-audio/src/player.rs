use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlaybackError {
    #[error("no default output device")]
    NoDevice,
    #[error("device error: {0}")]
    Device(String),
    #[error("stream error: {0}")]
    Stream(String),
}

/// Blocking player — opens a stream, plays the buffer, returns.
///
/// Used for short, self-contained sounds (e.g. the wake chime).
pub struct Player;

impl Player {
    pub fn play_blocking(samples: &[i16], sample_rate: u32) -> Result<(), PlaybackError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(PlaybackError::NoDevice)?;

        let supported = device
            .default_output_config()
            .map_err(|e| PlaybackError::Device(e.to_string()))?;

        let channels = supported.channels() as usize;
        let config = StreamConfig {
            channels: supported.channels(),
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let buffer: Vec<f32> = samples
            .iter()
            .map(|s| *s as f32 / i16::MAX as f32)
            .collect();
        let buffer = Arc::new(buffer);
        let pos = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));

        let buffer_cb = Arc::clone(&buffer);
        let pos_cb = Arc::clone(&pos);
        let done_cb = Arc::clone(&done);

        let err_fn = |e| tracing::error!("output stream error: {e}");

        let stream = match supported.sample_format() {
            SampleFormat::F32 => device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    fill(&buffer_cb, &pos_cb, &done_cb, channels, data);
                },
                err_fn,
                None,
            ),
            // For other formats, cpal converts internally; we still write f32 via build.
            _ => device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    fill(&buffer_cb, &pos_cb, &done_cb, channels, data);
                },
                err_fn,
                None,
            ),
        }
        .map_err(|e| PlaybackError::Stream(e.to_string()))?;

        stream
            .play()
            .map_err(|e| PlaybackError::Stream(e.to_string()))?;

        // Busy-wait with sleep until the buffer is drained.
        while !done.load(Ordering::Acquire) {
            thread::sleep(std::time::Duration::from_millis(5));
        }

        Ok(())
    }
}

fn fill(
    buffer: &Arc<Vec<f32>>,
    pos: &Arc<std::sync::atomic::AtomicUsize>,
    done: &Arc<AtomicBool>,
    channels: usize,
    out: &mut [f32],
) {
    let total = buffer.len();
    let mut i = pos.load(Ordering::Relaxed);
    for frame in out.chunks_mut(channels) {
        let sample = if i < total { buffer[i] } else { 0.0 };
        for s in frame.iter_mut() {
            *s = sample;
        }
        if i < total {
            i += 1;
        }
    }
    pos.store(i, Ordering::Relaxed);
    if i >= total {
        done.store(true, Ordering::Release);
    }
}

/// Streaming PCM player.
///
/// Spawned on a dedicated thread owning the cpal stream. Samples are pushed via
/// [`PcmPlayer::write`] (i16 mono at construction-time sample rate). [`stop`]
/// drains the queue, cuts playback immediately, and lets the caller barge in.
pub struct PcmPlayer {
    sample_rate: u32,
    queue: Arc<Mutex<VecDeque<i16>>>,
    stopped: Arc<AtomicBool>,
    _thread: thread::JoinHandle<()>,
}

impl PcmPlayer {
    pub fn new(sample_rate: u32) -> Result<Self, PlaybackError> {
        let queue: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::new()));
        let stopped = Arc::new(AtomicBool::new(false));

        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), PlaybackError>>();

        let queue_t = Arc::clone(&queue);
        let stopped_t = Arc::clone(&stopped);

        let handle = thread::spawn(move || {
            let host = cpal::default_host();
            let device = match host.default_output_device() {
                Some(d) => d,
                None => {
                    let _ = ready_tx.send(Err(PlaybackError::NoDevice));
                    return;
                }
            };
            let supported = match device.default_output_config() {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(PlaybackError::Device(e.to_string())));
                    return;
                }
            };
            let channels = supported.channels() as usize;
            let config = StreamConfig {
                channels: supported.channels(),
                sample_rate: SampleRate(sample_rate),
                buffer_size: cpal::BufferSize::Default,
            };

            let q_cb = Arc::clone(&queue_t);
            let stopped_cb = Arc::clone(&stopped_t);

            let stream_res = device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    let mut q = q_cb.lock().expect("pcm queue poisoned");
                    let stopped = stopped_cb.load(Ordering::Acquire);
                    for frame in data.chunks_mut(channels) {
                        let sample = if stopped {
                            0.0
                        } else {
                            q.pop_front()
                                .map(|s| s as f32 / i16::MAX as f32)
                                .unwrap_or(0.0)
                        };
                        for s in frame.iter_mut() {
                            *s = sample;
                        }
                    }
                    if stopped {
                        q.clear();
                    }
                },
                |e| tracing::error!("pcm player stream error: {e}"),
                None,
            );

            let stream = match stream_res {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(PlaybackError::Stream(e.to_string())));
                    return;
                }
            };
            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(PlaybackError::Stream(e.to_string())));
                return;
            }
            let _ = ready_tx.send(Ok(()));

            // Park the thread; the stream stays alive until we drop.
            // We watch `stopped` with a flag flip to "drop" — but we want
            // the player to survive `stop()` (which only mutes), so we loop
            // forever until the parent drops us.
            loop {
                thread::park_timeout(std::time::Duration::from_secs(60));
            }
        });

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sample_rate,
                queue,
                stopped,
                _thread: handle,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(PlaybackError::Stream("audio thread died".into())),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Append PCM mono i16 samples to the playback queue.
    /// If the player was stopped, calling `write` re-arms it.
    pub fn write(&self, samples: &[i16]) {
        self.stopped.store(false, Ordering::Release);
        let mut q = self.queue.lock().expect("pcm queue poisoned");
        q.extend(samples.iter().copied());
    }

    /// Drain the queue and silence output immediately. Used for TTS barge-in.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        let mut q = self.queue.lock().expect("pcm queue poisoned");
        q.clear();
    }

    pub fn queued_samples(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }
}
