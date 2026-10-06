# strata-sycl-b60 — Strata's own engine (its SYCL port) on the B60, as the measured reference for Flash-Next

**Open** (2026-10-06, operator).

## Charter

Build Strata's SYCL port (upstream `sycl/`, `docs/INTEL.md`) for the B60,
run Qwen3.8-Flash-Next on it, and measure it on our card, our link and our
host. The number is the reference that arcint's Flash-Next path
(`flash-next-llama-engine.md`) is judged against: what this card reaches
with the expert engine's mechanisms all in place. This campaign changes
nothing in arcint; the decision it informs (keep porting Strata's
mechanisms into libllama, or serve Flash-Next on Strata's engine) is the
operator's.

Why: Strata's published B70 numbers are about five times arcint's served
Flash-Next decode on the B60 (below), and the decomposition of that gap
(speculative decoding, expert bytes, kernel alignment, launch overhead) is
an estimate until the reference runs here.

## Reference to follow

Strata upstream `6f32ec0` (engine 0.1.39, `code`), github.com/Niko1221/Strata:
- `sycl/`: the CUDA engine migrated with SYCLomatic plus
  `sycl/tools/fixups.py`; AOT device code (`STRATA_SYCL_AOT`, upstream
  builds `bmg-g31` for the B70; this campaign builds `bmg-g21`).
- `docs/INTEL.md`, Arc Pro B70 32 GB at PCIe 3.0 x8 (`paper`):
  - Flash-Next IQ2_XS (ISTA-DASLab GSQ-RCO, 35.5 GB of experts), 18,329 of
    24,576 experts in VRAM, 8.4 GiB in a pinned host mirror read by the
    expert kernels over PCIe: decode **58.5-64.2 t/s** (MTP draft layer,
    19- and 2,184-token prompts, 256 greedy tokens), prompt 549 t/s at
    2,184 tokens.
  - the Coder IQ1_M, every expert in VRAM: decode 78.2 t/s, prompt 790 t/s
    at 2,184 tokens, 1,201 at 40K.
- What it says moved decode on Xe2 (`paper`): the MTP draft layer (2.9-3.4
  tokens a round); aligned 16-byte loads for Q6_K / IQ4_XS / IQ4_NL / Q8_0
  (2.3-4.7x per kernel; Coder 45 -> 78 t/s with the rest of "decode
  round 2"); window graphs captured at load; the commit overlapped with the
  drafter.

arcint's side for scale (`measured-here`, B60, 2026-10-06, patch 0021,
UD-Q3_K_XL from NVMe, no speculative decoding): prefill 235 t/s on the
20,045-token needle, decode 12.5-13.0 t/s.

## Gate

The reference counts when, on the B60 with the IQ2_XS GGUF:
- it answers: the capital check, and the 20,045-token needle that the
  flash-next-llama-engine arms use;
- its prefill and decode are measured at that needle and on the 500-token
  long answer, greedy, through its server's API or its own `--tokens`
  run, with the configuration, the experts in VRAM and in the mirror, the
  draft acceptance and the tokens per round stated;
- the build is identified: upstream sha, AOT target, oneAPI version, and the
  B60 by PCI id.

A quality comparison of IQ2_XS against UD-Q3_K_XL (the answer-level bar's
KL, against the CPU reference) is owed before any serving decision; it is
not part of this measurement.

## Current state

- Strata `6f32ec0` staged on the dev host from `git archive`; ggml from
  llama.cpp `3cf03257` (Strata's pin); oneAPI 2026.1.1 native (no container).
  Build for `bmg-g21` started.
- IQ2_XS shards downloaded from the pinned revision
  (`ed59f920`), sha256 against the Hub's LFS ids.

## Where it lives

Outside the tree: the build and the measurement scripts live on the dev
host (`*.local.md`). Results land here and in DESIGN §7.0.2 when closed.
