# sub4bit-vram-kernel — VRAM-resident sub-4-bit expert weights: the re-scoped M10, paid only by a new GPU kernel

## The defect, as measured

Not a defect: a milestone re-scope, recorded 2026-09-05 (DESIGN §7.0.2ah,
`docs/milestone-0.3.0.md` M10 row). M10's original gate had two halves —
more context per card, and a mixed sub-4-bit artifact to buy it with. The
first half is already discharged, by a different lever: `--paged-kv u8:i4`
auto-fits +28.3% more context on the 24 GB card (155,376 → 199,424 tokens,
dense agent, MTP on, prefix cache 8 GiB) and +28.4% on the 16 GiB card
(133,456 → 171,312 tokens, coder, no offload, prefix cache 2 GiB), both
Prüfstand 10/10, greedy text byte-identical u8 vs u8:i4 at 16 tokens on
both cards (§7.0.2y). What is left of M10 is the second half alone:
VRAM-resident expert weights below 4 bits, gate now priced in that
currency — "≥ 15% more max context" means KV headroom freed by smaller
resident expert weights, not a KV-precision flag.

Neither route the row named reaches that gate without a new kernel. NNCF
3.3.0's `INT3_SYM`/`INT2_SYM` are implemented (not stubs), producing a
plain dequantize subgraph (packed `u3`, an `i8` zero-point, `f16` group
scales) — representable, but the pinned runtime's MoE fusion matcher
(`keep_moe_3gemm_const_precision.cpp`) requires `u4` on all twelve weight
and zero-point constants of the fused op, a `u3` artifact never reaches
it, and the kernel type table is `{u4, i4, u8, i8}` — oneDNN has no 3-bit
type. GGUF K-quant import (Q3_K 110 B/3.4375 bpw, IQ3_XXS 98 B/3.0625 bpw,
Q2_K 84 B/2.625 bpw per 256-weight block) fares no better on the gated
axis if it dequantizes to int4 at load, as the row specified: the
resident representation stays int4, byte-identical to the baseline on
VRAM — the format's real wins (disk, host-pool bytes, native host compute)
are `kquant-host-storage`'s territory, not this one's. Both routes
converge on the same new kernel; what remains open is the resident
*format* once it exists — `u3` group-quant or K-quant blocks — decided by
measurement on one expert layer, same f16 source and calibration, against
the int4 baseline.

## Known against hypothesised

Known: the matcher's `u4`-only requirement and the oneDNN type table
(§7.0.2ah); the K-quant byte counts above; that NNCF 3.3.0 has
`INT3_SYM`/`INT2_SYM` in its mode enum but not its public documentation,
undocumented until a compression call succeeds (§7.0.2y); that per-expert
granularity is inexpressible in the batched IR the served artifacts use
and constructible, unverified, in the unrolled form (`tools/
export_mtp.py`'s `--moe-lowering unrolled`); the magnitude bound — the
coder's experts are ≈ 85% of its parameters (≈ 23.2 B of 27 B, ≈ 10.9 GiB
of the 12.8 GiB int4 weights), 4 → 3.44 bpw on all of them freeing ≈ 1.5
GiB against a ≈ 1.4 GiB KV budget on the 16 GiB card (§7.0.2y's cost
model) — an estimate, explicitly not the obstacle; the kernel is.
Hypothesised: the kernel's size, "800–1,500 lines" (HYPOTHESIS, §7.0.2y),
and its decode-regression sign — unmeasured for a sub-4-bit expert kernel,
with only a same-shaped precedent on record: symmetric `u4` KV quarters
memory and costs 6% at depth 32k because the in-kernel dequant path costs
more than the bandwidth it saves (`PagedAttentionExtension` 58.6 ms f16 →
95.4 ms u4, +63%, coder, 24 GB card, §7.0.3) — a different kernel and
tensor class, cited for the shape of the cost, not its number. Also
whether "rarely-routed" experts form a real threshold at all — the
acceptance prompt left most of 7,360 experts at 0–2 routings (§7.0.2ah).

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and
Arc applicability: `research-sub4bit-weights.md` (same directory). Its "what transfers"
section is the recon's starting point, not a substitute for it.

## Gate

The M10 row's own gate, unchanged by the re-scope: a mixed artifact serves
with ≥ 15% more max context than pure int4 on the same card at Prüfstand
10/10; per-expert bpw map recorded in the artifact; quality delta vs int4
measured, not asserted (`docs/milestone-0.3.0.md` backlog row 1; §7.0.2ah).
Quality delta vs int4: not on the record.

## Entry criteria

Partially met. (1) The recon is on the record (§7.0.2ah) — met. (2) An
`INT3_SYM` `compress_weights` smoke test against a real IR, NNCF 3.3.0
already in the tooling venv, clearing §7.0.2y's "undocumented" caveat —
unrun. (3) A routing histogram (patch 0013, `MOE_OTD_ROUTING_HIST`) over a
corpus long enough that "rarely-routed" has a distribution to threshold
on — unrun, and shared prework with the `partition-seeding` campaign's own
entry criterion (2) (DESIGN §7.0.2ah names the same corpus gap for both).
(4) The bpw map's artifact format (sidecar, XML attribute, or
per-expert constant naming — the unrolled form makes the last natural,
the batched form does not) — undecided. (5) The resident format (`u3`
group-quant vs K-quant blocks), picked by measurement on one expert layer
against the int4 baseline — unrun, also the criterion `kquant-host-
storage` waits on.

## Scope — in / out

In: the two pre-work measurements; the bpw artifact-format decision; the
resident-format measurement; the kernel work — a fusion matcher for the
new element type, an oneDNN bypass, a GEMV/GEMM with in-kernel dequant —
judged by a fusion-impact profile (ground rule 2); the decode-regression
measurement against the u4-KV precedent; the VRAM-headroom and Prüfstand
measurements the gate names.

Out: the context claim via KV precision (discharged by M8, §7.0.2y); the
host-pool byte count and disk-format win (`kquant-host-storage`, fed by
but not measured here); the static partition's prefill/cold-start costs
(separate backlog items); any change to which experts are resident under
the M9/M14 offload path.

## Where it lives

DESIGN §7.0.2ah (re-scope and recon), §7.0.2y (discharged context claim,
NNCF/K-quant/runtime findings), §7.0.3 (u4-KV precedent); `docs/milestone-
0.3.0.md` M10 row and backlog row 1; the pinned runtime's MoE fusion
matcher (`keep_moe_3gemm_const_precision.cpp`, not in this tree);
`contrib/packaging/marfrit-openvino/patches/0013-moe-otd-routing-
histogram.patch` (`MOE_OTD_ROUTING_HIST`) and its `patches/README.md`
header; `tools/export_mtp.py` (`moe_block_unrolled`, `--moe-lowering`),
the tooling the bpw map's unrolled form would use; the tooling venv's
NNCF 3.3.0 (`compress_weights`); `docs/model_requirements.md` §2/§6.

## Pipeline for this campaign

Recon (done, §7.0.2ah) → pre-work: the `INT3_SYM` smoke test → pre-work:
the routing histogram over a longer corpus → design note: the per-expert
bpw artifact format → design note plus measurement: the resident format,
decided on one expert layer against the int4 baseline → **the kernel
work**, by a wide margin the largest item on this backlog (the fusion
matcher, the oneDNN bypass, the GEMV/GEMM with in-kernel dequant — new
plugin code with no analog in the patch series, none of it started) →
fusion-impact profile and decode-regression measurement → one card
window: Prüfstand 10/10, VRAM headroom (≥ 15% more max context), quality
delta vs int4 → review before commit → DESIGN §7.0.2x record, CHANGELOG
line, the milestone row's closing line. Everything before the kernel work
is a measurable step with a defined output; the kernel work cannot be
scoped further until the resident-format decision lands — the open-ended
remainder, where the 800–1,500-line HYPOTHESIS lives.

## Invariants

DESIGN §3.4 (history-independent greedy output): the new kernel must not
introduce residency- or arithmetic-dependent history sensitivity — the
mistake the M14 host tier's LRU residency made, fixed by patch 0018's F2
static partition (§7.0.2ae/§7.0.2ai), is the standing warning. Ground
rule 2 (a fusion-impact profile, not a kernel micro-benchmark) applies to
the matcher and the GEMV/GEMM alike. Ground rule 3's survey obligation
transfers from M10's close (§7.0.2ah), mostly on the record already
(§7.0.2y — no published Intel/OpenVINO IR below int4 for a Qwen3-MoE-class
model, nothing below W4A16 on vLLM's roadmap, ik_llama.cpp/QuantMoE-Bench
as comparators), inherited rather than repeated.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog; nothing started.
