# Performance — latence & énergie

Cible matérielle de référence : laptop Intel CometLake-H + NVIDIA RTX 3060 Mobile (6 Go VRAM), Pop!_OS 22.04+, PipeWire.

## Build CUDA

Tous les modèles ML peuvent être déportés sur la 3060. Active la feature `cuda` du binaire ; elle propage les features `cuda` à `jarvis-llm`, `jarvis-stt`, `jarvis-tts`.

```
cargo build --release --features cuda -p jarvis
```

Pré-requis runtime :
- CUDA toolkit installé (`nvcc` pour le build, `libcudart.so.12` + `libcublas.so.12` pour l'exécution).
- ONNX Runtime CUDA EP : `libonnxruntime_providers_cuda.so` doit être trouvable. `ort` télécharge automatiquement les binaires CPU mais **pas** la lib CUDA ; installe `onnxruntime-gpu` du système ou pointe `ORT_DYLIB_PATH` vers une lib CUDA-enabled.

## Configuration recommandée

Les défauts de `~/.config/jarvis/config.toml` sont déjà alignés sur cette cible :

```toml
[stt]
model = ".../models/ggml-medium.bin"   # voir « STT alternatifs » plus bas
n_threads = 4
n_gpu_layers = 99

[llm]
n_ctx = 4096
n_threads = 4
n_gpu_layers = 99
temperature = 0.4

[tts.kokoro]
voice = "am_onyx"
speed = 1.0
gpu = true
```

Sur un build sans la feature `cuda`, `n_gpu_layers` est ignoré et `n_threads` reprend la main : tout retombe sur le CPU sans erreur.

## STT alternatifs

`ggml-medium.bin` (1.5 Go) est solide en FR mais lourd. Deux options pour gagner :
- `ggml-small.bin` (~470 Mo) — déjà téléchargé par `scripts/setup.sh`. 2× plus rapide, WER FR ~5–10 % plus élevé.
- `ggml-distil-large-v3-fr-q5_0.bin` (~750 Mo, à télécharger depuis Hugging Face `distil-whisper/distil-large-v3-fr-ggml`). 3–4× plus rapide que medium sur GPU, qualité FR comparable.

Pour changer : ajuster `stt.model` dans la config.

## NVIDIA — comportement énergie

Modèles résidents en VRAM, GPU en idle quand inactif. La 3060 Mobile descend en P8 (~8–12 W) après ~1 s sans calcul. Pour aider :

```bash
sudo nvidia-smi -pm 1          # persistence mode : pas de réinit driver
sudo nvidia-smi --auto-boost-default=0
# Plafond TGP optionnel (perte < 10 %, gros gain thermique en laptop) :
sudo nvidia-smi -pl 60
```

À ajouter dans un service systemd `nvidia-persistenced.service` (souvent déjà installé) ou un drop-in unique au démarrage.

## CPU governor

Avec l'inférence sur GPU, le CPU n'est plus le facteur de latence. Garde `schedutil` (défaut Pop!_OS) ou `powersave` :

```bash
cpupower frequency-set -g schedutil
```

## Wake-word — coût continu

`crates/jarvis-wake/src/wakeword.rs:231` fixe ORT à `intra_threads=1, inter_threads=1`. La détection openWakeWord tourne à 80 ms par chunk → < 1 % CPU sur un seul cœur, et pas de réveil thermique du GPU. Ne pas baisser le `cooldown_s` sous 1 s.

## Streaming LLM → TTS

L'orchestrateur exécute déjà :
1. `LlamaEngine::generate_stream` (token-par-token côté llama.cpp).
2. `jarvis_llm::sentence_stream` : découpe sur `. ! ? ; \n` ou flush à 160 caractères.
3. `crate::streaming::speak_stream` : synthétise et joue les phrases en pipeline (jusqu'à 2 chunks en vol).

Time-to-first-audio observé : ~400 ms après la fin du STT pour une réponse multi-phrases.

### Limite actuelle

`run_tool_loop` (`crates/jarvis-llm/src/tools.rs`) bufferise la sortie LLM pour parser les `<tool_call>` JSON avant de parler. Pour les réponses sans tool call (cas le plus fréquent), on pourrait court-circuiter et brancher `generate_stream` directement sur `sentence_stream → speak_stream`. Cela nécessite un détecteur de `<tool_call>` qui peut basculer en mode buffered dès qu'un `<tool_` apparaît dans le flux. À tenter si les mesures montrent que le TTFA est dominé par le tool loop.

## KV-cache persistant (follow-up)

`LlamaEngine::generate*` recrée un `LlamaContext` à chaque tour, donc prefill complet du system prompt + historique à chaque fois. Sur GPU, cela coûte ~50 ms ; sur CPU, plusieurs centaines. Optimisation possible : conserver un contexte par engine (lock-protégé) et utiliser `llama_kv_cache_seq_rm` pour ne re-préfilluer que le dernier tour. À évaluer après mesures : si le prefill GPU reste sous 100 ms, ne pas s'y lancer.

## Décodage spéculatif (follow-up)

Ajouter Qwen2.5-0.5B-Instruct Q4 comme draft model (~400 Mo VRAM). Gain attendu ×1.5–2 sur le throughput LLM, qualité identique. À activer via un champ optionnel `llm.draft_model` une fois `llama-cpp-2` exposera l'API speculative.

## Mesurer

Pendant une session, dans un terminal séparé :

```bash
# GPU power & VRAM (toutes les 200 ms)
nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv -lms 200

# CPU usage par cœur
mpstat -P ALL 1

# Latence end-to-end : observer les timestamps des signaux D-Bus
dbus-monitor --session "interface='org.jarvis.Assistant'"
```

Cibles avec GPU :
- TTFA (fin VAD → premier sample audio) ≤ 600 ms.
- Throughput LLM ≥ 60 tok/s sur Qwen2.5-3B Q4_K_M.
- GPU draw au repos < 15 W, pic inférence ~60–80 W (~3 s).
- CPU pic ≤ 30 % sur 2–4 cœurs.
