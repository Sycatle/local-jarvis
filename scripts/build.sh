#!/usr/bin/env bash
# Build wrapper for Jarvis.
#
# Contournement d'un bug de `llama-cpp-sys-2` 0.1.146 (build.rs:1110) :
# entre deux invocations de cargo avec des features ML différentes,
# `target/debug/` (et `target/release/`) accumulent des symlinks dangling
# `libggml-*.so` / `libllama.so` qui font paniquer le prochain build :
#   thread 'main' panicked at .../llama-cpp-sys-2-0.1.146/build.rs:1110:56:
#     called Result::unwrap() on Err(Os { kind: AlreadyExists, ... })
#
# On les nettoie systématiquement avant chaque build.
#
# Usage : `scripts/build.sh [args cargo]`
# Exemples :
#   scripts/build.sh check -p jarvis --features cuda
#   scripts/build.sh build --release --features cuda -p jarvis
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

# Purge des symlinks orphelins laissés par les builds précédents.
if [ -d target ]; then
    find target -maxdepth 4 -xtype l -delete 2>/dev/null || true
fi

exec cargo "$@"
