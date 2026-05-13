# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-13

Initial public release.

### Added

- Workspace of 13 crates (`jarvis-core`, `jarvis-audio`, `jarvis-wake`,
  `jarvis-stt`, `jarvis-llm`, `jarvis-tts`, `jarvis-skills`,
  `jarvis-skills-macros`, `jarvis-mcp`, `jarvis-desktop`, `jarvis-service`,
  `jarvis-memory`, `jarvis`).
- State machine wiring wake → STT → LLM → tool-calls → TTS with barge-in.
- Clap-based wake detector and openWakeWord ONNX path.
- Energy VAD + `whisper-rs` STT backend.
- ChatML prompt builder + GBNF tool-loop on `llama-cpp-2`.
- eSpeak NG phonemiser with Kokoro and Piper backends.
- Skill registry with system (volume, brightness, notify) and media skills.
- D-Bus interface `org.jarvis.Assistant` for external clients.
- SQLite-backed conversation memory.
- Ratatui TUI (`jarvis tui`) for live state inspection.
- systemd user unit, `.desktop` entry, and `gsettings` keybinding for
  <kbd>Super</kbd>+<kbd>J</kbd>.
- `scripts/setup.sh` one-shot bootstrap for Pop!\_OS / Ubuntu.

[Unreleased]: https://github.com/sycatle/local-jarvis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sycatle/local-jarvis/releases/tag/v0.1.0
