# Jarvis — assistant vocal local (Rust, Pop!_OS)

> Document de référence. Mis à jour au fil des décisions structurantes.

---

## 1. Pourquoi ce projet

Construire un assistant vocal **100 % local** sur Pop!_OS, intégré comme un
vrai démon desktop Linux (systemd, D-Bus, XDG Portals, PipeWire), capable de :
- détecter un wake (clap ou « Jarvis ») sans cloud,
- transcrire la parole, raisonner via un LLM embarqué, parler en français,
- piloter le PC (volume, fenêtres, capture, média) via des skills exposés
  comme tool-calls au LLM.

Choix de design clés :
- **Rust de bout en bout** pour un binaire unique, démarrage à froid rapide,
  pas de venv, alignement avec la trajectoire COSMIC.
- **Workspace multi-crates** : fronts clairs, swap d'implémentation propre
  (`llama-cpp-2`, `whisper-rs`, `piper-ort` derrière des traits).
- **Stack swappable** : chaque couche (wake/STT/LLM/TTS) est un trait avec
  une implémentation par défaut + un stub qui rend le binaire utilisable
  même sans modèles téléchargés.

---

## 2. Environnement cible (mesuré)

| | |
|---|---|
| Machine | Dell G15 |
| CPU | i7-10870H, 8C/16T |
| RAM | 16 Go |
| GPU | NVIDIA RTX 3060 Mobile 6 Go (Optimus + Intel UHD) |
| OS | Pop!_OS 24.04 LTS, kernel 6.18 |
| Session | X11, GNOME |
| Audio | PipeWire 1.5.85 (compat pulse) |
| Rust | 1.90 stable (pinné par `rust-toolchain.toml`) |
| Outils déjà là | `playerctl`, `wmctrl`, `xdotool`, `dbus-send` |
| Installé par setup.sh | `brightnessctl`, `libespeak-ng-dev`, `cmake`, `clang`, `libdbus-1-dev`, `libpipewire-0.3-dev`, etc. |

> ⚠️ `nvidia-smi` casse actuellement (driver/lib mismatch 580.159) ; un reboot
> recolle. Tant que c'est pas fait, llama.cpp/whisper.cpp tournent CPU
> (dégradé propre, configurable via `[llm].n_gpu_layers`).

---

## 3. Architecture

### 3.1 Vue d'ensemble

Un seul binaire `jarvis run` à base de tokio multi-thread. State machine
centrale, tâches async pour wake / STT / LLM / TTS / D-Bus. Communication par
`mpsc`/`broadcast`/`watch` et `CancellationToken` pour le barge-in.

```
IDLE ──(clap || "Jarvis")──▶ LISTENING ──(VAD silence ≥ 600 ms)──▶ THINKING
                                                                       │
                                                                       ▼
IDLE ◀────────────────────────────── SPEAKING ◀──── (tool calls?) ─── LLM
   ▲                                     │
   └──── barge-in (nouveau wake coupe TTS)
```

D-Bus est l'**interface externe** (CLI, raccourcis clavier, scripts) pas le
bus interne — single user, single GPU, single device, donc pas d'IPC inutile.

### 3.2 Workspace Cargo

```
local-jarvis/
├── Cargo.toml                  # workspace + workspace.dependencies
├── rust-toolchain.toml         # pin stable
├── crates/
│   ├── jarvis-core/            # events, state, config (figment/toml), dirs (XDG)
│   ├── jarvis-audio/           # cpal capture + PcmPlayer interruptible + ding
│   ├── jarvis-wake/            # clap detector + openWakeWord (stub) + manager
│   ├── jarvis-stt/             # VAD énergétique + whisper (stub)
│   ├── jarvis-llm/             # ChatHistory + ChatML Qwen + GBNF + tool-loop
│   ├── jarvis-tts/             # phonemiser + Piper (stub)
│   ├── jarvis-skills/          # registry + system + media (12 skills)
│   ├── jarvis-skills-macros/   # proc-macro #[skill] (réservée v1.1)
│   ├── jarvis-desktop/         # zbus portals + X11 backend
│   ├── jarvis-service/         # service D-Bus org.jarvis.Assistant
│   └── jarvis/                 # binaire CLI + orchestrator + runner + TUI Ratatui (`jarvis tui`)
├── packaging/
│   ├── jarvis.service          # systemd --user
│   ├── jarvis.desktop          # entry GNOME
│   └── org.jarvis.Assistant.xml# introspection D-Bus
├── scripts/setup.sh            # apt + rustup + modèles + cargo install
└── docs/PROJECT.md             # ce fichier
```

### 3.3 Stack par couche

| Couche | Crate Rust | Notes |
|---|---|---|
| Audio I/O | `cpal` | PortAudio-style, talks PipeWire via ALSA |
| Resampling | `rubato` | 44.1 kHz (clap) ↔ 16 kHz (STT/wake) |
| FFT | `realfft` | pour signature spectrale future |
| Wake clap | `jarvis-wake/clap.rs` (biquad bandpass + EMA baseline) | tests de régression internes |
| Wake vocal | `ort` + openWakeWord ONNX *(stubbé)* | activé via setup.sh |
| VAD | énergie + hystérésis *(Silero ONNX optionnel plus tard)* | suffisant CPU |
| STT | `whisper-rs` *(feature-gated)* | modèle ggml small FR |
| LLM | `llama-cpp-2` *(feature-gated)* + GBNF tool-call | Qwen2.5-3B Q4_K_M par défaut |
| TTS | `ort` + **Kokoro-82M** ONNX (défaut) + phonèmes IPA via subprocess `espeak-ng --ipa -q` | voix `am_onyx` (masculin ♂ grave, US sur phonèmes FR — timbre archétypal "Jarvis") ; Piper conservé en option |
| Async | `tokio` multi-thread | un seul process |
| D-Bus | `zbus` 4.x | `#[interface]`, signaux émis via `SignalContext` |
| Logs | `tracing` + `tracing-journald` | bascule auto sous systemd (`INVOCATION_ID`) |
| Config | `figment` + TOML + env `JARVIS_*` | layered defaults → file → env |
| XDG dirs | `directories` | `~/.config/jarvis/`, `~/.local/share/jarvis/`, etc. |

### 3.4 État actuel des couches ML

Pour livrer un binaire qui **boot, accepte les commandes D-Bus et joue le ding
end-to-end** avant le téléchargement des modèles, les couches lourdes (Whisper,
LLM, Piper) sont des **stubs** qui :
- détectent l'absence du modèle (`is_file()` checks),
- exposent la même API `async`,
- renvoient une erreur typée claire (`NotImplemented` / `ModelMissing`).

Activation : `scripts/setup.sh` télécharge les modèles dans
`~/.local/share/jarvis/models/` et `~/.local/share/jarvis/piper/`, puis
l'orchestrateur passe en chemin réel sans modification de code une fois les
crates ML branchés derrière feature gates (jalon v1.1).

---

## 4. Skills v1

12 skills enregistrés dans `SkillRegistry` (implémente `jarvis_llm::ToolRegistry`).

### Système (`system.rs`)
| Skill | Effet | Implémentation |
|---|---|---|
| `volume` | up/down/set/mute | `wpctl` |
| `brightness` | up/down/set | `brightnessctl` |
| `launch_app` | lance par nom | `gtk-launch` → fallback `xdg-open` |
| `focus_window` | focus fenêtre | `wmctrl` (X11) |
| `notify` | toast | XDG Portal Notification |
| `screenshot` | capture | XDG Portal Screenshot |

### Média (`media.rs` — MPRIS via `playerctl`)
| Skill | Effet |
|---|---|
| `media_play_pause` | play/pause |
| `media_next` | suivante |
| `media_previous` | précédente |
| `media_stop` | stop |
| `media_now_playing` | "artist — title" |

Hors v1 : timers, calendrier, mails, recherche web, exec shell, OCR écran,
domotique, dictée — voir §10.

---

## 5. Intégration Linux native

### 5.1 XDG Base Directory
Via crate `directories` :
- Config : `~/.config/jarvis/config.toml`
- Data : `~/.local/share/jarvis/{models,piper}/`
- Cache : `~/.cache/jarvis/`
- State : `~/.local/state/jarvis/`

### 5.2 systemd --user (`packaging/jarvis.service`)
```ini
[Unit]
Description=Jarvis local voice assistant
After=pipewire.service default.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.cargo/bin/jarvis run
Restart=on-failure
RestartSec=2s
Nice=-5

[Install]
WantedBy=default.target
```
Activation : `systemctl --user enable --now jarvis`
Logs : `journalctl --user -u jarvis -f` (journald via `tracing-journald`).

### 5.3 D-Bus session — `org.jarvis.Assistant`
Sur `/org/jarvis/Assistant`, interface `org.jarvis.Assistant` :

| Méthode | Signature | Effet |
|---|---|---|
| `Speak` | `(s text) → ()` | parle |
| `Listen` | `() → (s)` | écoute, renvoie transcript |
| `Status` | `() → (s)` | état courant |
| `Cancel` | `() → ()` | coupe TTS |

Signaux : `StateChanged(s)`, `Transcribed(s)`, `Spoken(s)`.

Usage :
```bash
busctl --user call org.jarvis.Assistant /org/jarvis/Assistant \
       org.jarvis.Assistant Speak s "Bonjour Sir"
dbus-monitor --session "interface='org.jarvis.Assistant'"

# Via la CLI fournie :
jarvis say "Bonjour Sir"
jarvis listen
jarvis status
jarvis cancel
jarvis config-path
jarvis skills                                       # liste des skills enregistrés
jarvis skill volume --args '{"action":"down"}'      # invoque un skill (debug, bypasse le LLM)
jarvis tts-preview --voice am_onyx --text "Test"  # A/B-test d'une voix Kokoro
```

Implémentation `zbus` 4.x (`#[interface]`, signaux émis via `SignalContext`
récupéré depuis `InterfaceRef::signal_context()`).

### 5.4 XDG Portals
Préférés aux DBus GNOME directs pour `Screenshot`, `Notification`, `OpenURI`,
`Inhibit`. Portabilité X11/Wayland/Flatpak/COSMIC.

### 5.5 PipeWire (optionnel pour v1)
- Activer `module-echo-cancel` (config WirePlumber dans `setup.sh` futur) →
  barge-in transparent.
- Streams Jarvis taggés `media.role = "voice-assistant"`.

### 5.6 Abstraction desktop X11 ↔ Wayland
- `jarvis-desktop/x11.rs` : `wmctrl`/`xdotool` (actif).
- `jarvis-desktop/portals.rs` : zbus portals (actif, marche partout).
- Squelette Wayland prévu via `ydotool` à terme.

### 5.7 `.desktop` file
`packaging/jarvis.desktop` → apparaît dans Activities GNOME.

---

## 6. Configuration (`~/.config/jarvis/config.toml`)

Toutes les sections sont optionnelles, defaults dans `jarvis_core::config`.
Override possible via env `JARVIS_<SECTION>__<KEY>` (figment split sur `__`).

```toml
[audio]
sample_rate_capture = 16000
sample_rate_clap = 44100

[wake.clap]
enabled = true

[wake.wakeword]
enabled = true
model = "/home/sir/.local/share/jarvis/models/hey_jarvis_v0.1.onnx"
threshold = 0.6
cooldown_s = 1.5

[stt]
model = "/home/sir/.local/share/jarvis/models/ggml-small.bin"
language = "fr"
silence_ms = 600
min_speech_ms = 250
max_utterance_s = 20

[llm]
model = "/home/sir/.local/share/jarvis/models/qwen2.5-3b-instruct-q4_k_m.gguf"
n_ctx = 4096
n_threads = -1
n_gpu_layers = 0
temperature = 0.4
max_tool_iterations = 3
system_prompt = """
Tu es Jarvis, l'assistant personnel de Sir. Tu réponds en français, court,
sobre, factuel, en vouvoiement. Tu disposes d'outils pour contrôler le PC et
la musique. Utilise-les sans demander confirmation pour les actions
réversibles. Si la requête est une simple question, réponds directement en
texte sans appeler d'outil.
"""

[tts]
engine = "kokoro"                                                    # kokoro | piper | espeak
# Piper fallback paths (utilisés si Kokoro indisponible) :
voice_model = "/home/sir/.local/share/jarvis/piper/fr_FR-siwis-medium.onnx"
voice_config = "/home/sir/.local/share/jarvis/piper/fr_FR-siwis-medium.onnx.json"

[tts.kokoro]
model      = "/home/sir/.local/share/jarvis/kokoro/kokoro-v1.0.onnx"
tokenizer  = "/home/sir/.local/share/jarvis/kokoro/tokenizer.json"
voices_dir = "/home/sir/.local/share/jarvis/kokoro/voices"
voice      = "am_onyx"                # alternatives : ff_siwis, bm_george, am_michael, em_alex
speed      = 1.0
language   = "fr"
```

---

## 7. Setup & opération

```bash
# Bootstrap complet (Pop!_OS / Ubuntu Debian-likes)
./scripts/setup.sh

# Lancer (foreground, debug)
cargo run --release -p jarvis -- run
# ou si déjà cargo-installed :
jarvis run

# Service permanent
systemctl --user enable --now jarvis
journalctl --user -u jarvis -f

# Pilotage CLI (parle au démon via D-Bus)
jarvis say "Bonjour Sir"
jarvis listen
jarvis status
jarvis cancel
```

---

## 8. Choix Rust assumés

### Pourquoi pas Python
- POC Python initial supprimé en table rase.
- Distribution : un binaire static-ish vs un venv 200+ paquets pip.
- Démarrage : pas d'import time, pas de GIL, type checks compilés.
- Audio temps-réel : callbacks cpal en Rust pur, sans bridge FFI Python↔C.
- COSMIC à venir : alignement écosystème.

### Pourquoi pas Zig/Go
- Écosystème ML Rust mature (`ort`, `whisper-rs`, `llama-cpp-2`).
- D-Bus async natif (`zbus`) sans GLib.
- Type-state pattern pour la state machine = bugs rattrapés au compile time.

### Plan B si un module bloque
Chaque crate est isolé : si `llama-cpp-2` régresse, swap par binding HTTP vers
Ollama derrière le même trait `LlmEngine`. Si `espeakng-sys` ne builde pas,
fallback subprocess `espeak-ng --ipa -q`. Décidé case-par-case, pas un *grand
plan B* monolithique.

---

## 9. Vérification — critères v1 utilisable

### Critères Rust
- [ ] `cargo build --release --workspace` produit `target/release/jarvis`
- [x] `cargo clippy --workspace --all-targets -- -D warnings` passe
- [x] `cargo test --workspace` vert
- [ ] `cargo audit` : aucune CVE critique

### Critères fonctionnels (à exécuter après setup.sh + activation features ML)
1. **Wake silencieux** : double clap → ding < 300 ms, état → LISTENING
2. **Wake vocal** : "Jarvis" prononcé → mêmes effets. 30 min de bruit ambiant
   (clavier, conversation) : 0 faux positif
3. **Q&A libre** : "Jarvis, quelle heure est-il ?" → réponse vocale FR <
   2 s sur GPU OK, < 4 s CPU only
4. **Skill système** : "Jarvis, baisse le volume de moitié" → `wpctl`
   exécuté + confirmation vocale
5. **Skill système** : "Jarvis, prends une capture d'écran" → fichier dans
   `~/Images/Captures d'écran/`
6. **Skill média** : Spotify lancé, "Jarvis, pause" → MPRIS pause +
   confirmation. "Suivante" → next
7. **Barge-in** : pendant que Jarvis parle, redire "Jarvis" → TTS coupé,
   retour LISTENING
8. **Offline** : déconnecter le réseau, refaire 1-6 → tout passe
9. **Régression clap** : `cargo test -p jarvis-wake` → tests synthétiques verts
10. **Latence wake→premier audio TTS** mesurée via timestamps `tracing`
11. **D-Bus** : `busctl --user call ... Speak s "test"` → TTS s'exécute.
    `Status` renvoie l'état. `StateChanged` reçu via `dbus-monitor`
12. **systemd** : `systemctl --user start jarvis` → actif. Kill du process
    → restart auto. `journalctl --user -u jarvis -n 20` → logs structurés
13. **XDG dirs** : config lue depuis `~/.config/jarvis/config.toml`
14. **Portails** : `screenshot` skill via portail → fichier généré

---

## 10. Hors scope v1 (notés pour v1.1+)

**v1.1 — Activer les ML stubs**
- [x] **TTS Kokoro-82M** (feature `kokoro`, défaut) — voix `am_onyx` masculine, phonèmes IPA via `espeak-ng --ipa -q`, sortie 24 kHz, fallback automatique Piper → eSpeak si modèle absent.
- Feature gates `whisper`, `llm`, `wakeword` qui montent les vrais
  bindings (`whisper-rs`, `llama-cpp-2`, `ort` + openWakeWord).
- Téléchargement automatique des modèles via setup.sh (Kokoro déjà inclus).
- Silero VAD ONNX via `ort` (remplace l'energy VAD).

**v2 — Sécurité/identité**
- FaceID embeddings ArcFace/InsightFace + anti-spoofing
- Diarisation (ignorer voix non autorisées)
- Verrou FaceID avant actions sensibles

**v2 — Intelligence**
- RAG sur docs perso (embeddings locaux)
- Mémoire long-terme (SQLite + embeddings)
- Vision LLM (Moondream, Qwen-VL) + OCR
- Mode agentic multi-étapes

**v2 — Skills additionnels**
- Timers/rappels persistants
- Calendrier, mails
- Recherche web local (SearxNG)
- Contrôle navigateur (Playwright)
- Home Assistant / MQTT

**v2 — UX**
- AppIndicator / tray icon
- Overlay GTK pendant écoute
- TUI debug live (spectrogramme)
- Packaging `.deb` / Flatpak
- GSettings schema, GNOME Search Provider
- i18n
- Backend Wayland complet

---

## 11. Décisions verrouillées (ne pas re-débattre)

- **Langage** : Rust pour 100 % du code v1.
- **Workspace** multi-crates, pas binaire mono-crate.
- **LLM** : `llama-cpp-2` embarqué (pas Ollama).
- **TTS** : Kokoro-82M ONNX direct via `ort` (Piper conservé en option). Phonemisation par subprocess `espeak-ng --ipa -q` (le wrapper safe `espeakng` 0.2 n'expose pas les bits IPA du `phonememode` C ; fork+exec ~5 ms, négligeable).
- **STT** : `whisper-rs` (pas faster-whisper côté Python).
- **Async** : tokio multi-thread.
- **D-Bus** : zbus (pas dbus-rs legacy).
- **Portals** > DBus GNOME direct.
- **Stubs ML** intentionnels en v1 : permettent boot + tests + D-Bus end-to-end
  avant téléchargement modèles. Pas de coexistence Python.

---

## 12. État d'avancement

- [x] Phase 0 — Table rase Python + bootstrap workspace Cargo (11 crates)
- [x] Phase 1 — `jarvis-core` (events/state/config figment/dirs XDG) + 3 tests
- [x] Phase 2 — `jarvis-audio` (cpal capture/player/ding + 2 examples)
- [x] Phase 3 — `jarvis-wake` (clap biquads/EMA + wakeword stub + manager) + 2 tests
- [x] Phase 4 — `jarvis-stt` (energy VAD + collector + whisper stub) + 2 tests
- [x] Phase 5 — `jarvis-llm` (ChatML Qwen + GBNF + tool-loop + stub engine) + 3 tests
- [x] Phase 6 — `jarvis-tts` (TtsBackend trait + Piper stub + Phonemiser)
- [x] Phase 7 — `jarvis-skills` (registry + 6 system + 5 media skills = 11)
- [x] Phase 8 — `jarvis-desktop` (zbus portals + X11 + facade)
- [x] Phase 9 — `jarvis-service` (org.jarvis.Assistant via zbus #[interface])
- [x] Phase 10 — `jarvis` binaire (CLI clap + orchestrator state machine + runner)
- [x] Phase 11 — packaging (jarvis.service / jarvis.desktop / dbus xml / setup.sh)
- [x] Phase 12 — réécriture `docs/PROJECT.md` (ce fichier)
- [ ] Phase 13 — smoke test end-to-end + activation features ML (whisper, llama, wakeword)
- [x] Phase 14 — TTS Kokoro-82M (voix `am_onyx` ♂ par défaut, fallback auto vers Piper/eSpeak)
- [x] Phase 15 — UI client : TUI Ratatui intégrée au binaire `jarvis` (`jarvis tui`). Remplace l'ancien scaffolding Tauri 2 + Next.js (`apps/jarvis-ui/`, supprimé) — un seul binaire, zéro webview, fonctionne en SSH/tmux.
