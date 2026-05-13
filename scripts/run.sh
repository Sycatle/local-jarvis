#!/usr/bin/env bash
# Launcher Jarvis avec env runtime CUDA correctement résolu.
#
# Trois bibliothèques doivent être trouvables au runtime quand le binaire est
# compilé avec --features cuda :
#   1. libcudnn.so.9 — pas packagé par Pop!_OS ; on prend celui bundlé par
#      `nvidia-cudnn-cu12` (pip) si présent.
#   2. libonnxruntime_providers_cuda.so — téléchargé par `ort` dans
#      ~/.cache/ort.pyke.io. ORT le dlopen au runtime ; pas dans le RPATH.
#   3. libllama.so.0 — produit par le build llama-cpp-sys-2 dans target/.
#      Cargo positionne un RPATH relatif mais on l'override souvent via
#      LD_LIBRARY_PATH ailleurs ; on l'ajoute explicitement.
#
# Sans ces chemins le binaire démarre mais Kokoro retombe silencieusement sur
# CPU et le démon ne charge pas llama.
#
# Usage : `scripts/run.sh <sous-commande jarvis>`
# Exemples :
#   scripts/run.sh run
#   scripts/run.sh tts-preview --text "Bonjour"
#   scripts/run.sh say --text "Coucou"
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
bin="$root/target/release/jarvis"
if [ ! -x "$bin" ]; then
    echo "error: $bin introuvable. Build d'abord : scripts/build.sh build --release -p jarvis --features cuda" >&2
    exit 1
fi

paths=()

# 1. cudnn (cherche d'abord pip user, puis system).
for d in \
    "$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib" \
    "$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib" \
    /usr/lib/x86_64-linux-gnu \
    ; do
    if [ -e "$d/libcudnn.so.9" ]; then
        paths+=("$d")
        break
    fi
done

# 2. ORT bundled providers.
ort_lib=$(find "$HOME/.cache/ort.pyke.io" -name "libonnxruntime_providers_cuda.so" 2>/dev/null | head -1)
if [ -n "$ort_lib" ]; then
    paths+=("$(dirname "$ort_lib")")
fi

# 3. libllama produced by cargo build.
llama=$(find "$root/target/release/build" -name "libllama.so" 2>/dev/null | head -1)
if [ -n "$llama" ]; then
    paths+=("$(dirname "$llama")")
fi

if [ ${#paths[@]} -gt 0 ]; then
    joined=$(IFS=:; echo "${paths[*]}")
    export LD_LIBRARY_PATH="$joined:${LD_LIBRARY_PATH:-}"
fi

exec "$bin" "$@"
