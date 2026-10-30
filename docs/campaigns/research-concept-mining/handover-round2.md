# concept-mining 14, round 2 — handover

State: item 14's save path was built as Strata's wrong rule. Round 1 implemented the CUDA `#477`
rule (`src/core/expert_cache.cpp`, `rank_learned_profile`) and cited the SYCL/xe save
(`sycl/src/program/generate.cpp`, commit `3d6f1718`). Round 2 builds the xe path as the default and
keeps cu as a named option. Evidence: `code` for both reference paths and the build; device-free
cells and the recipe are `measured-here`; no card run was made here, so the served A/B is owed.

## What changed

- Patch `0068-expert-cache-save` is now `0070-expert-cache-save`, after the trace's `0069`; its
  hunks were regenerated against 0001-0067 + 0069 (applies cleanly, `git apply` on the pin). The
  prefill slot loan keeps `0068` when it integrates.
- The save's default rule is `xe`: Strata's SYCL/xe `save_profile`, the cumulative counts mixed
  with the start's ranking as pseudo-counts, lambda 2, `STRATA_PROFILE_PRIOR_ENTRIES` default
  800,000 routed entries, no resident head. `--llama-expert-cache-save-rule cu` selects
  `rank_learned_profile` (residents first). The teardown log names the rule it used.
- The save ranks from a cumulative per-pair `heat` (`heat += usage` before every adapt decay,
  allocated only when the save is set), not the decayed usage snapshot.
- Pending swaps at save: the incoming expert is marked resident in both phases; the outgoing is
  kept resident only in phase 1 and cleared in phase 2.
- `--llama-expert-cache-save` equal to `--llama-expert-profile` is refused at config; an unknown
  rule name is refused.
- The profile header says the trailing rank table is written (the `.cpp` already wrote it).
- Tie-break: prior rank then index, as Strata. Replay `read_profile` refuses a short header and a
  rank count above the geometry. `profile_slots` models the engine's global walk (frames, not
  bytes; the table caveat is in the audit).
- CTest runs the C++ cell through pytest (`expert-profile-save`, LABELS unit).
- `tools/fn_sweep.py --prompts FILE` takes a JSON prompt list; `tools/fn_sweep_prompts.json` is
  the 36-prompt training capture (English and German, public-domain prose summaries, code
  questions, explanatory prose), nothing private.
- Authors and committers of every commit not on `qfndev` are the repository's usual identity.

## The card window commands (Claude runs these in the announced window)

One lane, `fn_sweep.py`'s unit flags. The training capture uses the training prompts only; the
served gate uses the six standard prompts (the unseen side).

```bash
cd ~/src/ligence-foresight
set -o pipefail
ARCINT=/path/to/the/recipe/arcint
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets
TRAIN=/flash-model/wp-tmp/item14-train

# 1. the training trace, from the training prompt set only
rm -rf "$TRAIN"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" item14-train \
  --prompts tools/fn_sweep_prompts.json \
  --set --llama-expert-route-trace "$TRAIN" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=64 \
  2>&1 | tee /tmp/fn-sweep-item14-train.out
find "$TRAIN" -maxdepth 1 -name '*.ert1' | wc -l      # >= 1
grep -c "WARN: expert route trace" /tmp/fn-sweep-item14-train.log   # expect 0

# 2. the profile from that trace only (Strata's make_profile.rank_profile)
python3 tools/expert_profile_from_trace.py "$TRAIN" --out /tmp/item14-trace.bin
python3 - <<'PY'
import struct
b = open('/tmp/item14-trace.bin', 'rb').read()
print('STRP', b[:4], struct.unpack_from('<5I', b, 4))
PY
```

The served A/B, adapt on (the default), alternated twice, then the adapt-off pair. Each arm is one
fresh process; the standard prompts are the unseen gate.

```bash
UNIT="$ASSETS/expert-profile.bin"
TRACE=/tmp/item14-trace.bin
for tag in g1-unit g1-trace g2-trace g2-unit; do
  case "$tag" in
    *unit)  SET=(--set --llama-expert-profile "$UNIT") ;;
    *trace) SET=(--set --llama-expert-profile "$TRACE") ;;
  esac
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" "$tag" "${SET[@]}" \
    2>&1 | tee "/tmp/fn-sweep-$tag.out"
done
# adapt off: the same two placements, LLAMA_EXPERT_CACHE_ADAPT_EVERY=0
for tag in off-unit off-trace; do
  case "$tag" in
    *unit)  SET=(--set --llama-expert-profile "$UNIT") ;;
    *trace) SET=(--set --llama-expert-profile "$TRACE") ;;
  esac
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" "$tag" "${SET[@]}" \
    --env LLAMA_EXPERT_CACHE_ADAPT_EVERY=0 2>&1 | tee "/tmp/fn-sweep-$tag.out"
done
grep -h '^SWEEP ' /tmp/fn-sweep-g1-unit.out /tmp/fn-sweep-g1-trace.out \
  /tmp/fn-sweep-g2-trace.out /tmp/fn-sweep-g2-unit.out \
  /tmp/fn-sweep-off-unit.out /tmp/fn-sweep-off-trace.out
```

Report per arm: decode-mean, hits-mean, accept-mean, and the answers' capital check. The gate is
the answer-level bar (the facts stay right; the six standard prompts were never trained on); the
speed comparison is decode-mean within its spread, alternated twice.

The save path's warm-start arm (also owed), xe default and cu option, distinct paths:

```bash
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" save-xe \
  --set --llama-expert-profile "$UNIT" \
  --set --llama-expert-cache-save /tmp/item14-saved-xe.bin \
  --env STRATA_PROFILE_PRIOR_ENTRIES=800000 2>&1 | tee /tmp/fn-sweep-save-xe.out
grep "saved the expert profile" /tmp/fn-sweep-save-xe.log   # names the xe rule
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" restart-xe \
  --set --llama-expert-profile /tmp/item14-saved-xe.bin 2>&1 | tee /tmp/fn-sweep-restart-xe.out
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" save-cu \
  --set --llama-expert-profile "$UNIT" \
  --set --llama-expert-cache-save /tmp/item14-saved-cu.bin \
  --set --llama-expert-cache-save-rule cu 2>&1 | tee /tmp/fn-sweep-save-cu.out
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" restart-cu \
  --set --llama-expert-profile /tmp/item14-saved-cu.bin 2>&1 | tee /tmp/fn-sweep-restart-cu.out
```

The warm-start gate: a restart on the saved file is at or above the arm it was saved from, within
the window's spread, and the answers stay right. `xe` is the default; `cu` is named only in the
`save-cu` arm, so the record can say which rule each arm used.

## Owed

- the training capture, the trace-built profile and the four alternated arms plus the adapt-off
  pair;
- the warm-start arms above.

The recipe line is in the audit: `RECIPE a809775: 100% tests passed, 0 tests failed out of 7
EXIT 0 deb arcint_1.2.6-1_amd64.deb` (`measured-here`), patch 0070 in sequence on the pinned
tarball.
