# Patches carried against the pinned OpenVINO

Per `DESIGN.md` §1 in the arcint repository, rung 2 of "smallest sufficient
divergence": a numbered patch set **applied at build time here**, not a
divergent checkout. Each patch stays PR-shaped so it can be re-offered
upstream, and this file names what each one is for.

The pin is `2026.4.0-22849-71640275d29` — upstream commit `71640275`. A patch
that does not apply cleanly to that commit is a bug in this directory, not a
reason to move the pin.

## Applied

### 0003-moe-batched-gemv-expert-mask-subbuffer-churn.patch

`prepare_internal_buffers` in the GPU plugin's MoE implementation rebuilds its
per-expert mask subbuffers on **every** inference when `token_num > 1`:
`create_subbuffer` twice per expert, 512 calls per layer, **20,480 per
two-token forward**. Those masks are read only by the per-expert prefill
fallback; the batched-GEMV path that a small-token forward actually takes never
looks at them. At `token_num == 1` the whole block is skipped, which is why
plain decoding never showed the cost.

The patch skips the mask creation below the batched-GEMV threshold and caches
it against `(token_num, buffer)` for the prefill path proper.

Measured on an Arc Pro B60, u8 KV, 300 tokens, temperature 0, output
**byte-identical** to the unpatched plugin in both arms:

| | unpatched | patched |
|---|---|---|
| verify forward wall | 27.3 ms | 18.1 ms |
| MoE host execute per verify | 8.91 ms | 0.74 ms |
| `--mtp on`, prose | 44–46 t/s | 60.5–61.2 t/s |
| `--mtp off` | 62.3 t/s | 61.7–62.3 t/s |

Full derivation: `docs/moe-m2-path.md` in the arcint repository.

Upstream: not yet filed. This is the patch that motivated giving this recipe a
patch path at all; it should be offered upstream, and this line should then
name the PR.

## Deliberately NOT applied

These live in the arcint repository's `patches/` as records of measurements.
They are listed here so that nobody re-derives the decision by trying them.

- **0001-null-implementation-control.patch** — an instrument, not a fix: it
  forces a null implementation so a node's cost can be measured by removal.
  Shipping it would disable real work.
- **0002-fc-horizontal-fusion-bound.patch** — raises the horizontal FC fusion
  bound. Measured and rejected: fusing the MLP quartet produces wrong output
  (its fourth member is the width-1 `shared_expert_gate`), and with the bound
  restricted to the GDN sets the gain is 66.6 against 66.4 t/s — inside the
  noise. DESIGN records it as "not carried".

## Not carried either: the measurement instrument

The arcint session's working tree also carries per-stage timing accumulators
(`network.cpp`, `primitive_inst.cpp`, `stage_acc.hpp`). Those are how the
20,480 calls were found. They are **not** part of any patch here, and the
build script resets to the pinned commit and applies only this directory, so a
dirty measurement tree cannot leak into a package.
