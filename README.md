# Jarvis

[![CI](https://github.com/sycatle/local-jarvis/actions/workflows/ci.yml/badge.svg)](https://github.com/sycatle/local-jarvis/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV: 1.90](https://img.shields.io/badge/MSRV-1.90-orange.svg)](rust-toolchain.toml)

A 100% local voice assistant for Linux, built in Rust as a proper desktop
daemon (systemd, D-Bus, XDG Portals, PipeWire). No cloud. Wake on a clap or
the word "Jarvis", transcribe, reason with an embedded LLM, speak back, and
drive the desktop through skills exposed as tool-calls.

> Status: **alpha**. Tested on Pop!\_OS 24.04 (GNOME 46, X11) with a Dell G15
> (i7-10870H, RTX 3060 Mobile 6 GB). Other distros should work with minor
> tweaks to `scripts/setup.sh`.

## Highlights

- **Rust end-to-end.** One binary, cold start in milliseconds, no Python
  runtime to manage.
- **Stack swappable per layer.** Wake / STT / LLM / TTS are traits with a
  default implementation and a stub. The binary runs even without any model
  downloaded.
- **D-Bus first.** External clients (keyboard shortcuts, scripts, the CLI)
  talk to the daemon over `org.jarvis.Assistant`. Internally, tasks
  communicate through `tokio` channels — no IPC overhead.
- **Skills as tool-calls.** A registry of typed actions (volume, windows,
  media, screenshot, …) is exposed to the LLM via grammar-constrained
  decoding (GBNF). External MCP servers plug in too.

## Architecture

```
IDLE ──(clap || "Jarvis")──▶ LISTENING ──(VAD silence)──▶ THINKING
                                                              │
                                                              ▼
IDLE ◀───────────────────────── SPEAKING ◀── (tool calls?) ── LLM
   ▲                                │
   └────────── barge-in (new wake cuts TTS)
```

Workspace crates:

| Crate | Purpose |
|---|---|
| `jarvis-core` | Event bus, state machine, config (figment/toml), XDG dirs |
| `jarvis-audio` | `cpal` capture and interruptible PCM player |
| `jarvis-wake` | Clap detector + openWakeWord |
| `jarvis-stt` | Energy VAD + `whisper-rs` |
| `jarvis-llm` | ChatML prompt builder + GBNF tool-loop (`llama-cpp-2`) |
| `jarvis-tts` | eSpeak NG phonemiser + Kokoro (default) / Piper backends |
| `jarvis-skills` | Skill registry + system + media skills |
| `jarvis-skills-macros` | `#[skill]` proc-macro (reserved for v1.1) |
| `jarvis-mcp` | MCP client bridging external tool servers |
| `jarvis-desktop` | XDG portals + X11 backend |
| `jarvis-service` | `org.jarvis.Assistant` D-Bus interface |
| `jarvis-memory` | SQLite-backed conversation store |
| `jarvis` | CLI binary + orchestrator + Ratatui TUI |

A more thorough architecture deep-dive (in French) lives in
[`docs/PROJECT.md`](docs/PROJECT.md), and detailed performance notes in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Getting started

### Prerequisites

- Linux with PipeWire and a working microphone (Linux-only by design — the
  daemon talks systemd, D-Bus, and XDG Portals)
- Rust stable, MSRV 1.90 (tracked by `rust-toolchain.toml`)
- System dependencies installed by `scripts/setup.sh`:
  `build-essential pkg-config cmake clang libdbus-1-dev libssl-dev`
  `libespeak-ng-dev libasound2-dev libpipewire-0.3-dev`

### One-shot install (Pop!\_OS / Ubuntu)

```bash
./scripts/setup.sh
```

This installs apt dependencies, the Rust toolchain (if missing), downloads
the default models (Whisper, Qwen2.5-3B, openWakeWord, Silero VAD,
Kokoro-82M), writes a default config to `~/.config/jarvis/config.toml`,
installs the systemd user unit and the `.desktop` entry, and binds
<kbd>Super</kbd>+<kbd>J</kbd> to `org.jarvis.Assistant.Listen`.

Kokoro is the default TTS and the only one auto-installed. Piper remains
supported as a fallback engine — point `[tts.piper].model` at a downloaded
voice and switch `[tts].engine = "piper"` to use it. eSpeak NG is wired
in via `apt` for phonemisation and as a last-resort speech backend.

### Manual build

```bash
cargo build --release
./target/release/jarvis --help
```

Useful subcommands:

```bash
jarvis run           # foreground daemon (logs to stdout)
jarvis tui           # ratatui TUI (state, transcripts, logs)
jarvis say "hello"   # synthesize text through the configured TTS
jarvis status        # ask the running daemon over D-Bus
jarvis config-path   # print resolved config path
```

To run as a service:

```bash
systemctl --user enable --now jarvis
journalctl --user -u jarvis -f
```

### Build features

The default profile builds with stubs so the workspace compiles on a vanilla
machine. Enable the real backends with cargo features on the `jarvis` crate:

| Feature | Effect |
|---|---|
| `llama` | Wire `llama-cpp-2` for the LLM |
| `whisper` | Wire `whisper-rs` for STT |
| `embed` | Wire embeddings into `jarvis-memory` |
| `cuda` | Enable CUDA for `llama`, `whisper`, `tts` (implies `llama` and `whisper`) |

Example: `cargo build --release -p jarvis --features cuda`.

## Configuration

`~/.config/jarvis/config.toml` is the source of truth. `scripts/setup.sh`
seeds a working default. Override with environment variables prefixed
`JARVIS_` (via `figment`).

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the workflow, commit conventions (Conventional Commits), and the local
checks expected before opening a PR (`cargo fmt`, `cargo clippy`,
`cargo test`).

Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before participating.
Security reports go to [`SECURITY.md`](SECURITY.md).

## License

Dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
