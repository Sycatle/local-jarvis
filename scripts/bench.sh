#!/usr/bin/env bash
# Probe Jarvis runtime cost during a live session.
#
# Lancer pendant que le démon `jarvis` tourne (avec --features cuda en
# release). Le script écrit dans ./bench-<timestamp>/ :
#   - gpu.csv           : power.draw, memory.used, utilization.gpu (200 ms)
#   - cpu.csv           : utilisation totale et par cœur (1 s)
#   - dbus.log          : trace des signaux StateChanged/Transcribed/Spoken
#   - summary.txt       : résumé (max GPU W, mean CPU %, durée).
#
# Ctrl-C arrête la capture et calcule le résumé.
set -euo pipefail

ts=$(date -u +%Y%m%dT%H%M%SZ)
out="bench-$ts"
mkdir -p "$out"

cleanup() {
    trap - INT TERM EXIT
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    summarise
}
trap cleanup INT TERM EXIT

pids=()

if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu \
        --format=csv,nounits -lms 200 > "$out/gpu.csv" &
    pids+=($!)
else
    echo "nvidia-smi indisponible — GPU non mesuré" >&2
fi

if command -v mpstat >/dev/null; then
    mpstat -P ALL 1 > "$out/cpu.csv" &
    pids+=($!)
else
    # Fallback minimal : top en mode batch
    top -b -d 1 -n 100000 > "$out/cpu.csv" &
    pids+=($!)
fi

dbus-monitor --session \
    "type='signal',interface='org.jarvis.Assistant'" \
    > "$out/dbus.log" 2>&1 &
pids+=($!)

echo "Capture en cours dans $out/ — Ctrl-C pour arrêter."

summarise() {
    {
        echo "== Résumé $ts =="
        if [ -s "$out/gpu.csv" ]; then
            awk -F, 'NR>1{p=$2+0; if(p>max)max=p; sum+=p; n++} END{
                if(n) printf("GPU power : max %.1f W, moyenne %.1f W sur %d échantillons\n", max, sum/n, n)
            }' "$out/gpu.csv"
            awk -F, 'NR>1{m=$3+0; if(m>max)max=m} END{
                if(max) printf("VRAM peak : %.0f MiB\n", max)
            }' "$out/gpu.csv"
        fi
        if [ -s "$out/cpu.csv" ]; then
            # mpstat colonne "all" → %idle en dernière colonne ; CPU % = 100 - %idle
            awk '/^Average:.*all/{print "CPU mean : "100-$NF" %"; exit}' "$out/cpu.csv" || true
        fi
        echo "Signaux D-Bus capturés : $(grep -c "^signal " "$out/dbus.log" || echo 0)"
    } | tee "$out/summary.txt"
}

# Attendre les processus enfants jusqu'à signal.
wait
