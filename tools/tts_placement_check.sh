#!/usr/bin/env bash
# The TTS placement check (docs/speech-service.md, docs/campaigns/
# qwen3tts-card-real-time.md): --tts-gen-device cpu must actually build the
# audio-generation mmproj's own clip graphs on the CPU. backend_llama.cpp
# captures libmtmd's clip_init log ("CLIP using <backend> backend", clip.cpp)
# while the mmproj loads and refuses the boot if any clip context landed on
# the card. This script runs the two cases on the card host.
#
# Green: a cpu-requested boot comes up and its log shows CPU clip contexts only.
# Red:   ARCINT_TTS_GEN_FORCE_GPU=1 negates the request; the boot must abort
#        naming the card backend. Run the red case first: a check that was
#        green before the change measures nothing (CLAUDE.md, measurement
#        discipline).
#
# The green leg binds one port and stops by pid (never pkill: the units run
# the same name). The models and the profiles directory are operator-local,
# outside the repository.
#
# usage: tools/tts_placement_check.sh <arcint binary> <backbone.gguf> <mmproj.gguf> <profiles dir> [port]
set -euo pipefail
BIN=${1:?arcint binary}; GGUF=${2:?backbone gguf}; MMPROJ=${3:?mmproj gguf}
PROFILES=${4:?profiles directory}; PORT=${5:-18084}
BASE=(--engine llama --gguf "$GGUF" --mmproj "$MMPROJ" --tts --tts-profiles "$PROFILES"
      --tts-gen-device cpu --device GPU.0 --n-ctx 4096 --host 127.0.0.1 --port "$PORT" -v)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# ---------------------------------------------------------------- red first
set +e
ARCINT_TTS_GEN_FORCE_GPU=1 "$BIN" "${BASE[@]}" > "$tmp/red.log" 2>&1
rc=$?
set -e
if [ "$rc" = 0 ]; then
    echo "RED FAIL: the forced-gpu build booted instead of refusing"
    exit 1
fi
if ! grep -q "built an mmproj clip context on the card" "$tmp/red.log"; then
    echo "RED FAIL: the boot refused for another reason --"
    tail -20 "$tmp/red.log"
    exit 1
fi
echo "red: the forced-gpu arm refuses, naming the card backend"

# ------------------------------------------------------------------- green
"$BIN" "${BASE[@]}" > "$tmp/green.log" 2>&1 &
pid=$!
up=0
for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.5
done
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
if [ "$up" != 1 ]; then
    echo "GREEN FAIL: the cpu-requested boot did not come up --"
    tail -20 "$tmp/green.log"
    exit 1
fi
if grep -q "CLIP using OpenCL" "$tmp/green.log"; then
    echo "GREEN FAIL: a clip context landed on the card:"
    grep "CLIP using " "$tmp/green.log"
    exit 1
fi
if ! grep -q "CLIP using CPU backend" "$tmp/green.log"; then
    echo "GREEN FAIL: no 'CLIP using CPU backend' in the log: the placement was not observed"
    exit 1
fi
echo "green: the mmproj clip graphs on the CPU"
