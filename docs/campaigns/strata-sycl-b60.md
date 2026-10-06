# strata-sycl-b60 — Strata's own engine (its SYCL port) on the B60, as the measured reference for Flash-Next

**Open** (2026-10-06, operator). The reference ran on the B60: 620.5 t/s
prefill at 20k, 37.2-37.8 t/s decode, answers right (`measured-here`).

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

Strata upstream `7ba023e` (the port's 0.1.39 sync, `code`), github.com/Niko1221/Strata.
Later `main` (`6f32ec0`) does not build for SYCL: 86 upstream commits changed
shared headers (`ThreadAffinity`, `NativeDense::load`) that the `sycl/` copies
have not taken yet.
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

Built and measured 2026-10-06 (`measured-here`). The B60 (PCI 8086:E211,
PCIe 4.0 x8). Strata `7ba023e`, AOT `bmg-g21`, oneAPI 2026.1.1 native, the
Level Zero loader 1.34.0: the system's 1.20.6 makes every SYCL program
segfault in the UR adapter's device enumeration. The ISTA-DASLab IQ2_XS
(revision `ed59f920`, sha256 against the Hub). Shard 2, the PLE table, on
NVMe; shard 1 on ZFS (read once at load). The configuration is
`sycl/setup_intel.py`'s for 32K: `--expert-cache auto --stream-experts
--vram-reserve-mib 1024 --prefill auto --spec 4 --spec-min-p 0.5 --mtp
--kv int8`, `STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1`, served
through its `serve/server.py` with the same requests as
`flash-next-llama-engine.md`'s gate, greedy, thinking off.

| | Strata SYCL, IQ2_XS | arcint libllama, UD-Q3_K_XL (patch 0021) |
|---|---|---|
| prefill, the 20,045-token needle | **620.5 t/s** (32.3 s) | 235 t/s |
| decode, the 500-token long answer | **37.2 / 37.8 t/s** (two runs) | 12.5-13.0 t/s |
| answers | Paris; ORANGE-FALCON-77; the long answer | the same |
| experts on the card | 12,332 slots, 16.53 GiB (50 %), placed by the profile, no eviction | 31 % of the bytes, adapted every 6 tokens |
| experts over the link | 12,244 in a pinned host mirror, 16.49 GiB, read by the GPU's expert kernels | the rest in a USM bank, gathered into VRAM |
| speculative decoding | MTP draft layer: 288 of 433 drafts accepted, ~2.36 tokens a round | none |
| VRAM free with everything loaded | 1,559 MiB | |

arcint on the same IQ2_XS file (patch 0022, 12,500 MiB of slots,
`flash-next-llama-engine.md` stage 3, `measured-here`): prefill 337.0 t/s,
decode 16.0 t/s without MTP, 13.8 with it; answers right.

What decode spends: 212 verify rounds for 500 tokens in 13.4 s, 63 ms a
round, each verifying ~3 tokens (2.04 drafts offered a round). arcint
spends 77 ms on one token. So the gap is about 2.4x from the drafts, times
about 1.2x per forward pass, where Strata's model is smaller (IQ2_XS, 35.5
GB of experts against 52 GiB) and half of it is on the card. A
no-speculation arm was not measured: the serve mode refuses to start
without `--mtp`, and a native pack refuses `--spec` below 2.

Load: 495 s, almost all of it the slot fill and the mirror read from ZFS
(0.11 GB/s), which is a load cost only.

Owed: the quality of IQ2_XS against UD-Q3_K_XL at the answer-level bar
(the existing KL reference is the UD-Q3_K_XL GGUF on the CPU, so it cannot
judge a different quant); the task battery on Strata's answers.

## Where it lives

Outside the tree: the build and the measurement scripts live on the dev
host (`*.local.md`). Results land here and in DESIGN §7.0.2 when closed.
