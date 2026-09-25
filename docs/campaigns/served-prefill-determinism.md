# served-prefill-determinism — the served Flash-Next path is not run-to-run
# deterministic at long context, so no KLD gate is readable on it

## STATUS — 2026-09-20 (read this before the static sections below)

The sections below are the campaign's **original framing**, kept for the
record; the dated status log further down supersedes them. Current standing,
all measured:

- **The gate is no longer blocked.** On the **A770** (same bytes, same request,
same harness) the depth-48 served floor is **0** — r0↔r1 bit-identical,
0/1367 rows moved, argmax 1.0000 — so clause (d) reads **READABLE on the A770
as the measurement card**. The B60's UNREADABLE reading stands and is a
**per-card** defect.
- **The chunk attribution is REFUTED.** Unchunked prefill (`--prefill-chunk 0`)
is **worse**, not better: 0.095497 / 0.202143 against chunk-512's
0.072601 / 0.073234 (D4, 2026-09-20). §3.2's chunk non-exactness is not the
mechanism, and neither a single-chunk mode nor a bit-exact carry is the fix.
- **Warm-up and the GPU/host residency mix are refuted** (0.072601 vs
0.073234; force-the-tier 0.081553).
- **The mechanism is OPEN, narrowed to a WITHIN-KERNEL nondeterminism in the
GDN arithmetic on Xe2** (execution level, below the kernel-choice level): the
JIT is byte-identical, serialization changes nothing (233 enqueues each
finished), the minimal gemm is clean, launch geometry is static, the reduction
is a fixed tree at both widths, and `xe2` **requires** subgroup size 16 so the
width pin is dead. Location `layer0/mixer_out`, input-bit-identity proven;
fingerprint: `dim0 = row 0`, heads
[3,5,6,7,10,13,17,22,31,39,41,42,43,47], one f16 ulp, flip count 2423..3924.
- **Upstream:** the sibling report is posted —
openvinotoolkit/openvino#38099 `issuecomment-5751935449` (2026-09-20).

Where a static sentence below disagrees with the dated status log, the status
log governs.

**Evidence classes in the status log:** an entry is `[measured-here]` unless it
names a different class (`[code]`, `[paper]`); the arm or dump it names is the
evidence. Static sections above are framing, not dispositions.

## The defect, as measured

Two independent forwards of the **same** 2,735-token window, same process,
same artifact, differ by:

| window | KL(A‖B) mean | max | argmax agreement | max abs logit diff |
|---|---|---|---|---|
| 0 | **0.1361047** | 4.4250 | 0.8500 | 13.220 |
| 1 | **0.1512423** | 9.6483 | 0.9020 | 17.807 |

`RUN` of 2026-09-19 (B60, 24 GB Arc; depth 48; native artifact; ratio 99 +
host tier; u8 KV; prefill chunk 512), four replays of both windows, dump
held in the operator-local `*.local.md` record. 2 of 2 window pairs moved.

The argmax column is the user-visible half: **~10–15 % of scored positions
pick a different top token from two identical forwards**, so at greedy
decoding the served answer is not reproducible run to run — a breach of the
invariants below, not merely a measurement artefact.

Against it, every bar in force is unreadable:
`bar_0.5.1 = 3.0905e-03` (below 2051) sits **~44x under the floor**; even
llama.cpp's own median error against the f32 reference (0.0649 w0 /
0.0283 w1) is ~2x under it on window 0 and ~5x under it on window 1. Clause
(d)'s own rule therefore fires: the row reads **UNREADABLE, not PASS** on the
B60.

## Known against hypothesised

- **KNOWN, documented in-tree:** DESIGN.md §3.2 — chunk boundaries are not
  bit-exact on this backend; the served loader warns so at every boot.
  2,735 tokens at chunk 512 is **6 chunks**, and the defect is exactly
  chunk-to-chunk state carry. [CORRECTED 2026-09-20: **REFUTED** — D4's
  unchunked arm is *worse* (0.095497 / 0.202143 against 0.072601 / 0.073234),
  so the defect is **not** chunk-to-chunk carry. See STATUS and the dated log.]
- **KNOWN upstream, OPEN:** openvinotoolkit/openvino **#38099** — "[GPU]
  Deterministic wrong values: batched unrolled forward-substitution, slice 0
  correct and every other chunk-slice wrong". Reported by this project
  2026-09-12; Intel acknowledged and requested a repro 2026-09-16; the
  self-contained reproducer was posted the same day; **no fix or PR since**.
  It is a deterministic miscompile in the same chunked-GatedDeltaNet unroll
  region (chunk-2 row 0 correct, row 1 wrong; ~1e-1 error on GPU where CPU
  is at the f64 floor), on both the A770 and the B60, on two builds.
- **HYPOTHESISED:** the served variance is that family plus run-order
  effects, amplified by 6 chunks. Not yet isolated — a single-chunk control
  and a chunk-carry instrument are what decide it.
  [CORRECTED 2026-09-20: **REFUTED as stated.** The single-chunk control was
  run (D4) and is *worse*, so the variance is not chunk amplification; the
  mechanism is a within-kernel GDN nondeterminism on Xe2. See STATUS.]
- **KNOWN, CLOSED-NOT-FIXED, adjacent:** openvinotoolkit/openvino **#37607**
  — GPU plugin model-cache import breaks fused MoE (`expert weight provider
  not initialized`). Closed 2026-09-18 as COMPLETED with **no upstream fix**;
  the operator's workaround (`CACHE_MODE=OPTIMIZE_SPEED`, which stops weights
  being embedded in the blob) is the only known route. Relevant to any cached
  warm start.

## Gate

`F_served ≤ bar_0.5.1` on both capture windows, measured by the existing
floor instrument (KL(A‖B) over ≥ 2 replays), with the decided bound beside
it. Today `F_served` is 44x over that bound, so the gate is RED by the
UNREADABLE rule. [CORRECTED 2026-09-20: true **on the B60**; on the **A770**
`F_served = 0` (bit-identical r0↔r1), so clause (d) reads **READABLE** there
and the gate closes on that card. See STATUS.] **The gate is on determinism, not on model fidelity** —
the artifact's KL-vs-reference means cannot be read to better than the floor.

## Entry criteria

- The acceptance commit -001 lands first: rows EMPTY, the gate stated, no
  code. (This document is not that commit.)
- The single-chunk control already on disk (one forward of the same 2,735
  ids) is available as the reference arm.
  [CORRECTED 2026-09-20: the control was run and it **refuted** the chunk
  attribution (D4, unchunked is worse). It is no longer the reference arm.]
- An instrument exists: `tools/kld_served.py --replay --repeat N` +
  `--compare` (floor pair); `tools/boot_serving_shape.py --repeat/--cut` for
  the per-node bisect. The `--timeout` on the replay must exceed the per-window
  cost (~3,600 s) or the leg is cut before a second replay exists.

## Scope — in / out

**In:** a red-first cell that asserts determinism; the per-node localiser
for the within-kernel GDN nondeterminism on Xe2.
[CORRECTED 2026-09-20: the original "In" list (the chunked-prefill path and
its cross-chunk carry; a single-chunk mode for the gate measurement; a
bit-exact carry option) is **superseded** — D4 refuted the chunk attribution.
The live item is the within-kernel nondeterminism in the GDN arithmetic, and
the gate item is the A770 measurement card. See STATUS.]
**Out:** the quality term itself (now measured to be inside the floor);
the f16-state/attention hypotheses (FALSIFIED by the artifact IR); the
model-cache path (#37607, closed-not-fixed, separate).

## Where it lives

The prefill chunking and the stateful carry in the executor backend; the
DESIGN.md §3.2 note; the floor/compare instrument in `tools/`; the card
window ritual and sampler in the operator-local record.

## Pipeline for this campaign

recon (read §3.2 and the chunk-carry code, and re-read #38099's repro) →
design note (single-chunk mode vs bit-exact carry, stated as an A/B) →
red-first implementation (a cell that fails when two forwards differ) → one
card window at the end → review before commit → DESIGN §7.0.2x record and a
CHANGELOG line when it closes.
[CORRECTED 2026-09-20: the A/B framed here (**single-chunk vs bit-exact
carry**) was run and **refuted**; the pipeline now runs on the within-kernel
GDN nondeterminism (STATUS, dated log, and the posted upstream sibling
report).]

## Invariants

DESIGN §3.4 (history-independent greedy output); LISBON-001's restart
determinism ("two cold boots, byte-identical answers"); the §5 ladder; the
measurement discipline of CLAUDE.md. A change that would trade determinism
for a number does not close.

## Status

- 2026-09-19: opened from the `F_served` measurement at depth 48 (numbers
  above); clause (d)'s row filled and marked UNREADABLE in
  `docs/window-051.md`; upstream #38099 confirmed OPEN and acknowledged,
  #37607 confirmed CLOSED-NOT-FIXED. Not committed: CLAUDE.md requires
  Fable review before every commit.
- 2026-09-19 (late): **the boot-driver route for D2/D4 at depth 48 is
  memory-infeasible in a 44 GiB container.** It binds the 26.82 GiB PLE
  table *and* allocates fresh state tensors per repeat (72 state tables
  zeroed each repeat), so the cgroup peaked at 49.2 GiB against 44 GB RAM
  and `MemAvailable` fell under 4 GiB — at which point the **sampler's own
  watchdog** (the fdinfo sampler, `MemAvailable < 4 GiB
  -> SIGKILL`) killed the leg during repeat #2 (rc=137). Forward #1 alone
  was clean: `INFER OK 728.4 s`, out (1,512,248320), finite, absmax 38.5.
  **Consequence for the discriminator: it runs through the SERVED path**,
  which reuses state and survived four replays. D2 (warm-up, then the
  counted replays) launched as `--windows 0 --warmup 1 --repeat 2`; the
  boot driver stays the tool for the `--cut` per-node bisect, but at repeat
  2 it needs the watchdog disabled or a smaller working set — recorded
  here so the next leg does not rediscover it.
- 2026-09-20 (01:10Z, **D2 read**): **the warm-up halves the floor but does not
  close it — the served path is intrinsically non-deterministic at steady
  state.** Window 0, all on the tool's own 1367-row subset (no warm-up arm from
  the earlier leg, for comparison):

  | pair | mean KL(A‖B) | rows moved | argmax | max \|diff\| |
  |---|---|---|---|---|
  | first vs second, **no** warm-up | **0.136105** | 205/1367 | 0.8500 | 13.22 |
  | warmup vs replay 0 | 0.072601 | 98/1367 | 0.9283 | 13.18 |
  | warmup vs replay 1 | 0.085348 | 123/1367 | 0.9100 | 12.51 |
  | **replay 0 vs replay 1** (warmed) | **0.073234** | 114/1367 (8.3 %) | 0.9166 | 12.30 |

  So **the warm-up contributes nothing measurable** (0.072601 vs
  0.073234 — same floor with and without it), and **the residual
  steady-state floor is ~24x the decided bound** (3.0905e-03) while still
  flipping the argmax at ~8 % of scored positions, `bit_identical False`.
  Clause (d) therefore stays **UNREADABLE**, and the fix must be a determinism
  mechanism (pin the path / serialise it / single-chunk), not a warm-up.
- 2026-09-20 — **CORRECTION to the bullet above (same day, after re-reading
  the replay ORDER): the 0.1361 → 0.0732 gap is NOT a warm-up effect.** The
  tool's `--repeat` is rep-major, so the earlier leg's two window-0 forwards
  (0.136105) **straddled a window-1 forward**, while D2's pairs are
  consecutive same-content forwards. D2 tests the warm-up head-on and it
  adds nothing: **warmup vs replay 0 = 0.072601 ~= replay 0 vs replay 1 =
  0.073234**. What moves the number is **what ran in between**: a different
  window raises it (0.136105 across w1; 0.085348 across a same-content
  replay), not the temperature of the start. So **H1 (cold->warm residency
  ramp) is NOT supported**; the divergence tracks **content/residency churn
  between forwards**, laid on top of an intrinsic ~0.073 floor for two
  identical consecutive forwards.
- 2026-09-20 — **instrument caveat recorded**: with only window 0 replayed,
  `kld_served --compare` takes the LAST `n_chunk` replays to be the capture's
  windows, so replay 1 was silently compared against capture **window 1**
  (argmax 0.026, mean 14.5) — the driver's own report is contaminated, and
  window 1 of the capture was never replayed. The floor above was computed
  directly from the dump's own window-0 replays (same rows, `floor_pair`).
  A single-window arm must be read that way, or the arm must replay all
  capture windows in order.
- 2026-09-20 (**force-the-tier read**): with every expert forced onto the host
  kernel (`ARCINT_MOE_DEVICE_POOL_BYTES=16777216`, too small for one 419 MB
  expert), the floor **does not collapse** — `warmup vs replay 0` = mean
  **0.081553**, max 4.765, **137/1367 (10.0 %) rows moved**, argmax 0.8998,
  max |diff| 13.058, `bitid False`. Slightly *higher* than D2's 0.072601.
  **So the GPU/host residency MIX is refuted as the cause**, and the
  divergence lives in a part of the graph the mix does not touch.
- 2026-09-20 — **the project's own localisation, which this campaign had not
  cited** (`docs/window-051.md`, the cut bisect table). That record says the
  variable is **the card**: on the **B60 (GPU.0, Xe2)** this graph's first
  layer carries a **per-forward nondeterminism of one f16 ulp in a few rows
  at or after row 32** (`layer0/out`: 1-3 of 1,024 rows, 2.44e-4 against 2.54),
  which 12 layers amplify to an 11 % argmax agreement at the logits; on the
  **A770 (GPU.1, Alchemist) the same bytes are bit-identical** (x8, x12). At
  depth 4 on the A770 the floor pair is **bit-identical x3** (KL 1.0e-18,
  |diff| 0). The kernel itself is **not localised** — the cut names the block
  (short-conv / GDN core / hyper-connection / MoE / PLE), and "the kernel-level
  cut needs names the emitter does not set yet".
- 2026-09-20 — **two attributions now sit on the record, and they are not the
  same mechanism**: (i) window-051's filled clause (d) attributes the depth-48
  served floor to **the chunk-boundary non-exactness of §3.2** (2,735 tokens
  over 6 chunks at chunk 512) and names the fix as **"a single-chunk prefill,
  or a bit-exact chunk-carry"**; (ii) the cut table above attributes it to
  **the B60's layer-0 per-forward step**, with the A770 as the clean
  measurement card. Our force-the-tier arm held chunk 512 constant throughout,
  so it separates neither. **D4 (`--prefill-chunk 0`, unchunked) is the
  discriminator**: collapse -> (i), the chunked path, and the fix is a serving
  config; persist -> (ii), the card's layer-0 kernel, and the fix needs the
  kernel named (emitter cut names) or the A770 as the measurement card.
- 2026-09-20 — the older dumps are **not** floors: `kld-d48g-r99.bin` and
  `kld-d48f-r99.bin` each hold one replay of BOTH capture windows, so their
  "pair" is w0 against w1 (mean 13.926929 / 3.637777, argmax 0.0227 / 0.0000)
  — different content, not repeats. `ctrl3-d48g-p18.bin` and
  `kld-d48n-r99.bin` are **0 bytes**. So there is no free GPU-only
  (`d48g`) floor in the store.
- 2026-09-20 — **handoff written** for the seconds-scale reproducer:
  `docs/handoff-served-prefill-determinism.md`. It carries the established
  findings, the depth ladder (**4 -> 2 -> 1**, because the defect is already
  present at depth 4), the chunk axis, what must be held fixed, the zeros
  trap (all-zero Constants get constant-folded away), the card A/B as the
  sharpest artefact, the P(fixed) table, the operational packet, and the
  seven traps already paid for. Read it before touching a card.
- 2026-09-20 (**D4 read — the discriminator ran, and the chunk attribution is
  REFUTED**). Unchunked prefill (`--prefill-chunk 0`; served path, ratio 99 +
  tier, B60 GPU.0, launched 09:47Z, all three forwards complete 13:37:30Z,
  dump `d48n-d4-unchunked.bin`) gives a floor **higher** than the chunked
  path, not lower: `warmup vs replay 0` mean **0.095497** (max 6.648,
  143/1367 moved, argmax 0.8954); **`replay 0 vs replay 1` mean 0.202143**
  (max 6.488, **284/1367 rows moved**, argmax **0.7922**). Beside chunk-512:
  D2 0.072601 / 0.073234 and force-the-tier 0.081553. So **removing the chunk
  boundaries does not remove the divergence — it increases it**, and
  window-051's 2026-09-19 clause-(d) amendment ("the indeterminism is the
  chunk-boundary non-exactness") is **corrected**: the depth-48 served floor
  is the **B60/Xe2 per-forward step** of window-051's cut table (layer 0, one
  f16 ulp at or after row 32; the A770 bit-identical), not §3.2's chunk
  non-exactness. The fix is therefore **NOT a single-chunk prefill**; it is
  (a) the **A770 as the measurement card** (bit-identical, a decision, zero
  code), or (b) the layer-0 **kernel named and pinned** (emitter cut names,
  then the seconds-scale reproducer in the handoff).
- 2026-09-20 (the **peer session**, **one level deeper — the mechanism is
  named**): the B60-vs-A770 difference is the **GDN subgroup width**. The
  plugin JITs ONE source and specializes per arch:
  `ocl_v2/paged_gated_delta_net.cpp::get_subgroup_size()` returns 8 for
  gen9/gen11/xe_lp/xe_hp/xe_hpg and **16 for xe2/xe3/default**, so
  `K_LANE_ELEMS = K_HEAD_DIM / SUBGROUP_SIZE` is 16 (A770) vs 8 (B60). The
  width-sensitive sites are the core's `sub_group_reduce_add` tree and the
  state write `BLOCK_WRITEN(recurrent_state_table, ...)`; the same bucket
  carries `paged_causal_conv1d_ref`. Evidence: B60 `src_002.cl` vs A770
  `src_013.cl`, 1615 vs 1607 lines, 17 diff lines, **one substantive** (the
  width define).
  - **Launch geometry is ruled out from code.**
    `get_dispatch_data_func` gives `wgs.global = {sequences, head_nums,
    v_blocks * subgroup_size}` and `wgs.local = {1, 1, subgroup_size}` -- every
    term a static shape plus the arch-derived width; no
    `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`, no `max_work_group_size` or
    occupancy query, no divisor-of-token-count local size, and
    `!params.is_dynamic()` is asserted. At a fixed width the lane->data mapping
    cannot vary per forward.
  - **The C is ruled out as the mechanism.** Neither `.cl` has `atomic`,
    `barrier`, `__local` or `volatile`; the only reductions are
    `sub_group_reduce_add` (33/38/179/180/221/242) and the state write is a
    plain store (265/270; conv1d 109). A fixed tree and a plain store are both
    deterministic per launch, so the tiebreak must be in the **compiled**
    width-16 code -- the `ocloc disasm` target: the reduction's lowering
    (SLM+barrier vs shuffle) or a read-modify-write in the state path.
  - **The fix candidate is in OUR tree.** Pinning `SUBGROUP_SIZE 8` for `xe2`
    is a one-line change to `get_subgroup_size()`. The third arm is: rebuild
    with 8 for xe2, re-run the B60 `layer0/mixer_out` r0<->r1 pair. The
    interpretation is a **determinism** test, not parity -- width 8 changes
    the VALUES (the B60 would then match the A770), so a collapse means the
    width-16 path WAS the nondeterminism, while "still varies" **exonerates
    the width** and moves the search to the state table's producers/consumers
    around this kernel. Only the second branch would overturn the
    localisation.
  - **Controls.** `zero_state(rq, label)` runs before every forward (a
    `copy_from` of a zero host tensor into each state table, exactly
    `zero_paged_rows`), so a state-carry **ramp** cannot explain "every forward
    differs" -- and the A770 control (bit-identical x4 on the same harness) is
    what proves it, since a ramp would differ there too. The peer added
    `--digest-ports` to `tools/boot_serving_shape.py` (sha256 of the request's
    input ports before each forward), which makes the **input-bit-identity
    proof** available: inputs bit-identical across repeats WHILE `mixer_out`
    differs == divergence provably internal to the kernel. That is stronger
    than eight differing outputs.
- 2026-09-20 (**A770 depth-48 leg — the card branch is decided, and it is
  clean**). `RUN@b6dbca5`, A770 (GPU.1), d48n, `--offload-ratio 99
  --moe-cpu-tier`, chunk 512, served path, `WARMUP=0 REPEAT=2`, dump
  `d48n-a770-d48.bin`, launched 14:22:12Z, replay 0 3585.2 s / replay 1
  3585.4 s, `KLD-D48G-DONE` 16:59:00Z. Read with `floor_pair` over the tool's
  own 1367-row subset:

  | pair | mean KL(A‖B) | rows moved | argmax | max \|diff\| |
  |---|---|---|---|---|
  | A770 r0 vs r1 | **-0.000000** | **0/1367** | **1.0000** | **0.000** |

  `bit-identical True`. So **`F_served(A770, d48) = 0`**, the decided bar
  (3.0905e-03) sits ABOVE the floor, and **BERLIN-001 clause (d) reads
  READABLE on the A770 as the measurement card** — no longer UNREADABLE. The
  B60's floors (0.1361/0.1512 headline, D2 0.072601/0.073234, force-the-tier
  0.081553, D4 unchunked 0.095497/0.202143) are therefore a **per-card
  defect**, and the width localisation above says which card and why.
  Caveats kept: **×2 only** [DISCHARGED 2026-09-21 — see the entry below]
  (the **depth-4** evidence is ×8/×12 — both A770
  rows in window-051's cut table are depth 4, no A770 depth-12 leg exists;
  §4.11's "A770 steps once at an unpredictable forward" is not refuted by two
  forwards — a repeat-8 A770 arm is the follow-on), and the A770 is the
  **measurement** card, not the deployment card. [CORRECTED 2026-09-20: the
  width pin is **DEAD** (`xe2` requires 16), so the B60 is not made readable
  that way — it is a per-card caveat until the within-kernel mechanism is
  found or upstream fixes it.] Peers' arms are open:
  width-only `xe2` disasm, `--digest-ports` repeat-8, `FORCE_IMPLEMENTATIONS`,
  width-pin rebuild + B60 r0↔r1.
- 2026-09-20 (**CORRECTION — the width is not the mechanism, and the pin is
  DEAD**). Two ISA/compiler facts, both measured by the peer session:
  1. **`xe2` REQUIRES subgroup size 16.** A minimal
     `intel_reqd_sub_group_size(8)` + `sub_group_reduce_add` kernel fails to
     compile on **every** Xe2 target (`bmg-g21`, `bmg-g31`, `lnl-m`, `ptl-h`)
     with *"Kernel compiled with required subgroup size 8, which is
     unsupported on this platform"*; it compiles for the A770, `ACM-G10`. So
     `get_subgroup_size`->16 is **forced by the platform** — the one-line pin
     is not a choice we can flip, and the middle row of the handoff's
     P(fixed) table needs a different mechanism.
  2. **The reduction is a fixed deterministic tree at BOTH widths.** Disasm:
     width 16 / `bmg-g21` -> `add(8)+add(4)+add(1)+add(1)` register-halving
     tree; width 8 / `ACM-G10` (A770) -> `add(4)+add(1)+add(1)`. No SLM, no
     `barrier`, no send-to-SLM, on either arch. So the width explains the
     **card-to-card VALUE difference** (harmonisation) and **not the run-to-run
     variance**; the width is a **correlate** of the card, not the mechanism.
     Also refuted with it: the co-resident RMSNorm's launch geometry is static
     (`rms_kernel_base.cpp`: `GetOptimalLocalWorkGroupSizes(gws, engineInfo)`
     with a static `engineInfo`; `RmsSchedulingPolicy` constants).

  **The mechanism is OPEN**, bounded to two shapes: the **state write/read
  path** (`BLOCK_WRITEN(recurrent_state_table, ...)`) or a plugin/driver-level
  **ordering race between co-resident kernels** — neither visible in the
  kernel source. What remains pinned is the **location only**:
  `layer0/mixer_out`, input-bit-identity across 8 repeats (all nine ports
  bit-identical, both state tables exactly the all-zero hash) while the output
  differs every forward; the A770 bit-identical.

  Next narrowing (GPU.0, free): stub the GDN's input (the
  attn_hyper_connection output) and cut `mixer_out` to split GDN vs
  hyper-connection; digest the **OUTPUT** state tables per repeat (inputs
  identical + post-forward states differing => the variance is in the kernel's
  state write/read; post-forward states identical while `mixer_out` differs =>
  a co-resident kernel or the read of `mixer_out`'s producers); audit the
  plugin's enqueue dependencies for a buffer shared without an event.

  **Consequence for the fix:** with the pin dead, the B60's determinism has no
  one-line fix in view. The options are (i) find and fix the real mechanism in
  the plugin, (ii) `OV_GPU_FORCE_IMPLEMENTATIONS` opt->ref **if** `ref` is
  deterministic and its rate tolerable, or (iii) the **A770 as the measurement
  card**. The A770 result (floor 0 -> clause (d) readable) never depended on
  the width being the cause and is untouched.
- 2026-09-20 (**state-digest split + fresh-process test — the mechanism is a
  stochastic write, and the FIRST FORWARD IS SPECIAL**). `--digest-ports`
  extended to the post-forward state tables (`layer0/mixer_out`, native d48n,
  GPU.0, T=1024, ×8): `conv_state_table.0` post = `7000565fc4eb` **every
  repeat in every process** — one distinct value, the clean control (its data
  input being a model port) — while `gated_delta_state_table.0` post is
  **stochastic**: 5 distinct hashes in one process (`ffdd…`, `82e7…`,
  `d526…`, `e8ff…`×4, `bb20…`) and 1–4 distinct among repeats in cold
  processes. **The ping-pong / reset-coverage branch is REFUTED** (never
  exactly 2); the shape is a **stochastic state write/read**.
  - **THE FIRST FORWARD IS SPECIAL AND REPRODUCIBLE**: `#1 =
    ffdd9c82bf32` in **all three cold processes**, while only later forwards
    vary (p2 settles to `e8ff`×7; p1/p3 spread over 4 values). **Leading
    mechanism hypothesis: SERIALIZATION.** The first forward pays the
    feedback-driven JIT, so nothing overlaps it — compilation drains the
    queue and the kernel runs alone; later forwards run the **cached** kernels
    with the graph's normal pipelining, so co-resident kernels overlap. The
    defect would then be a **missing event/dependency between co-resident
    kernels**, not floating-point arithmetic — consistent with every
    exoneration (no atomics/SLM, static geometry, fixed reduction tree, conv
    clean).
  - **Test that is also the fix candidate**: force serialization (a
    `stream.finish()`/event wait after each kernel of the layer, or a single
    in-order queue with explicit dependencies) and re-run the repeats. If the
    later forwards become bit-identical, the fix is an **added dependency in
    the plugin's graph scheduling**.
  - **FLOOR-PAIR RULE (new, and it propagates backwards): every floor must
    name its transition.** With #1 reproducible and #2+ stochastic, D2
    `warmup↔r0` 0.072601 is a **#1↔#2** transition while D2 `r0↔r1` 0.073234
    is **#2↔#3**; D4's 0.095497 is **#1↔#2** and 0.202143 is **#2↔#3**; and
    the fserv 0.136105 pair (`w0#1` vs `w0#2`) **straddled** the special #1,
    making it the most inflated of the set. The A770 arm was `WARMUP=0
    REPEAT=2`, so its pair is **#1↔#2** and bit-identical — the A770 is clean
    **including** the cold→warm transition, which is the transition the B60
    fails.
- 2026-09-20 (**serialization test refutes the overlap hypothesis**). An
  `LD_PRELOAD` interceptor for `clEnqueueNDRangeKernel` calling
  `clFinish(queue)` after **every** one of 233 kernel enqueues
  (`CLDUMP_SERIALIZE=1`) leaves the result unchanged: all 7 repeats still
  **DIFFER**, and `gated_delta_state_table.0` still spreads (`ffdd…`,
  `e8ff…`×4, `fc9a…`, `daa0…` — 3 distinct among repeats). So it is **NOT
  inter-kernel overlap and NOT a missing event/dependency**, and the
  "#1 is JIT-serialized" reading is **refuted by its own test**.
  `GPU_QUEUE_THROTTLE=HIGH` also fails to collapse it (6 distinct) —
  throttling is not serialization, so that proxy is **void**. **The mechanism
  narrows to a WITHIN-KERNEL nondeterminism**, bounded to two shapes:
  (i) a lane/address race in the `recurrent_state_table` write (two
  work-items writing the same element, or an uninitialized read) — testable by
  diffing the post-state **row by row** across repeats, where a tile-boundary
  cluster names the racing mapping; or (ii) a nondeterministic upstream
  `jit:gemm:any__f16` feeding q/k/v — testable by a minimal same-shape gemm
  run, with the **conv as the standing control** (input = a port AND an
  internal write, deterministic). Note the distinct-value **sets** differ
  between the serialized and non-serialized runs (`{ffdd, e8ff, fc9a, daa0}`
  vs `{ffdd, 82e7, d526, e8ff, bb20}`), so **quote the sets, not the counts**,
  when comparing arms. If both come back clean, the residual is a
  **compiler/hardware-level effect** we cannot patch and must route around
  (A770 card, or `ref`).
- 2026-09-20 (**REVIEW — the commit gate**). `CLAUDE.md` requires an external
  review before every commit; this harness has no Fable agent and
  `.claude/agents/fix-implementer.md` substitutes the external reviewer, so
  two reviewer passes were run on the uncommitted set. **Addressed:** (i)
  operator-local host names and paths were removed from the tracked documents
  and moved to `docs/handoff-served-prefill-determinism.local.md` (git-ignored
  by `.gitignore`'s `*.local.md`; the repository is public); (ii) the static
  sections that still named the chunk/single-chunk/carry fix, and the
  window-051 / window-052 / README / design-note sentences that still carried
  the refuted ping-pong, ordering-race or width-pin shapes, are corrected in
  place with a date and a STATUS block at the top; (iii) the 23.6 t/s figure
  attributed to `d48g` in window-052 is corrected (it is the HF-exported 35B
  control); (iv) the upstream posting is recorded below; (v) `tools/kld_served.py`
  docstring updated for the landed bar; (vi) the `--stub` comment's
  'kernel-neutral' claim is corrected (the stub changes the compiled kernel
  set, 1.37 -> 5.72 GiB, so the confound applies); (vii) the debug-caps build
  script now refuses to run without an explicit `OV_BUILD_PREFIX`, since its
  defaults match `build-openvino.sh` and would clobber the measurement plugin.
  **Verified:** `tests/python/test_kld_bar.py` + `test_kld_served.py` = **15
  passed** (2026-09-20). **Acknowledged minors, not fixed:** `cldump.c`
  silently captures nothing if `CLDUMP_DIR` does not exist; its header claims
  `clCompileProgram` is captured while the implementation only logs; IL and
  binary buckets share one counter; and `boot_serving_shape.py` assumes the
  artifact's hidden size equals the config, that `outputs[0]` is the logits
  Result under `--cut`, and that `--fresh` needs the KV ports fed.
- 2026-09-20 — **the seconds-scale reproducer is BUILT and its A770 control is
  read** [measured-here]. Method: `tools/boot_serving_shape.py --artifact
  <served bytes> --cut <block> --cut-prune --repeat N` — read the artifact's
  own IR, run `SDPAToPagedAttention`, cut after the suspect node, prune the
  parameters it no longer reaches (which strips the 26.82 GiB `ngram_table.*`
  ports for a layer-0 cut), compile on the card, run N sequential forwards on
  one request (state zeroed per forward as `zero_paged_rows` does), compare the
  f32 bytes. A layer-0 cut loads one layer's constants, not the artifact.
  **The tool needed one fix** [code]: on a cut leg the driver fed the served
  feed list unconditionally, so ports pruned by `--cut-prune` threw at
  `set_tensor` and ended the leg before a forward. It now feeds only the
  declared ports (9 kept on a `layer0/out` cut) and reports the skips. The
  change is in `tools/boot_serving_shape.py`, uncommitted (CLAUDE.md: Fable
  review before every commit).
- 2026-09-20 — **why the ladder uses the depth-4 artifact and not a depth-1
  build** [measured-here]: a depth-1 (or any <4) graph holds no full-attention
  layer, so `SDPAToPagedAttention` refuses it verbatim —
  `sdpa_to_paged_attention.cpp:81: No ScaledDotProductAttention operation
  observed in the graph`. The served shape needs >= 4 layers, so the cut
  ladder reads the existing `qwen38-flash-next-d4-ov` and cuts at
  `layer0/out`; the surviving graph is layer 0's block alone.
- 2026-09-20 — **the native artifact only compiles with the served plugin and
  its props** [measured-here]: `d48n` (patch 0043 native expert format) needs
  the built plugin's `PYTHONPATH` *and* the served properties
  `OFFLOAD_RATIO=99`, `MOE_CPU_TIER=YES`, `WEIGHTS_PATH=<art>/openvino_language_model.bin`.
  Without the built plugin the compile dies
  `clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`;
  with the plugin but a device that did not take the props it reports
  `Option not found: MOE_CPU_TIER`. The prior cut legs (`cut48n-b6dbca5`) used
  exactly this pairing.
- 2026-09-20 — **A770 (GPU.1) controls**, window 0 first 1024 ids, `--repeat 4`,
  same request, state zeroed each forward [measured-here]:

  | artifact | cut | ops | compile | forward #1 | repeats |
  |---|---|---|---|---|---|
  | `d48n` native | `layer0/mixer_out` | 194 | 5.18 s | 0.051 s | **bit-identical x4** (`7dc926cb9943`) |
  | `d48n` native | `layer0/moe/mix` | 332 | 9.64 s | 46.6 s | **bit-identical x4** (`1470739d9751`) |
  | `d48n` native | `layer0/out` | 367 | 4.55 s | 53.6 s | **bit-identical x4** (`3546c8ee1ebc`) |
  | `d4-ov` u4 | `layer0/out` | 334 | 1.66 s | 1.14 s | **bit-identical x4** (`f86020b05f7d`) |

  The A770 column agrees with `docs/window-051.md`'s cut table (A770
  bit-identical at this block); the reproducer is seconds-scale (54 s
  load+compile + ~50 s/forward native, 1.2 s/forward u4), which is the point.
  Note `layer0/moe` is not a node; the MoE output is `layer0/moe/mix`.
- 2026-09-20 — **the B60 ladder is queued behind the D4 arm** [measured-here]:
  the D4 arm still holds GPU.0, so the peer's ladder script waits for it to
  clear, then runs cells `d48n` {`layer0/out`, `layer0/mixer_out`,
  `layer0/moe/mix`}, `d4n` `layer0/out`, `d12` `layer0/out`, and a T sweep
  (1024/512/256/64) at `layer0/out`, all at `--repeat 8`, logs under
  the ladder's log. Results pending.
- 2026-09-20 — **the D4 arm is unchunked as intended** [measured-here, source +
  `ps`]: its command line carries `--prefill-chunk 512 ... --prefill-chunk 0`;
  `src/config.cpp:616` assigns in order, so the trailing 0 wins and the arm
  runs unchunked. The chunk-vs-card discriminator the campaign named is
  therefore actually in flight.
- 2026-09-20 — **the known-good-substitution harness is built** [measured-here]:
  `tools/boot_serving_shape.py --stub NAME=FILE.npy` replaces the named node's
  output with a fixed tensor BEFORE `--cut`, so a sub-block downstream is fed
  IDENTICAL bytes on every repeat. Trap paid for in the build: the replacement
  must be a **Parameter**, not a Constant — a Constant makes the whole
  downstream a pure function of it and the compiler constant-folds every
  MoE/gather op away (measured: 133 ops remain, **0 parameters**, compile fails
  in `program_builder.cpp:168`). A Parameter keeps the ops alive and the driver
  feeds it. Validated on A770 (u4 `d4-ov`, GPU.1): stub `layer0/mixer_out` with
  its own dumped output, `--cut layer0/out`, `--repeat 4` -> compiles, runs,
  **bit-identical x4** (`195453d754e5`). Note the stub changes the compiled
  kernel set (device_resident 1.37 -> 5.72 GiB), which is the substitution
  confound; for run-to-run DETERMINISM the test is still valid (fixed bytes in,
  is the sub-block stable?), it is only the semantic-error reading that the
  kernel swap would confound.
- 2026-09-20 — **`--kernel-names` names every node's kernel** [measured-here]:
  the built plugin has no `ENABLE_DEBUG_CAPS` (so `OV_GPU_DUMP_SOURCES_PATH` /
  `..._TENSORS_PATH` are inert), but `PERF_COUNT=YES` gives
  `InferRequest.get_profiling_info()`. A770, u4 `d4-ov`, `layer0/out` cut,
  T=1024 [measured-here]: `ocl::paged_gated_delta_net::opt___f16` (GDN core),
  `ocl::paged_causal_conv1d::ref___f16`, `rms_gpu_bfyx_opt__f16`,
  `jit:gemm:any__f16` (all dense projections, incl. the GDN half's),
  `jit:gemm:any__i8` (FullyConnectedCompressed, the expert GEMMs),
  `dynamic_quantize_gpu_opt__f16`, `reduce_ref__f16`, `softmax_gpu_bf__f16`.
  Library only: `layer0/moe/mix` is a node in the NATIVE `d48n` artifact but
  NOT in the u4 `d4-ov` artifact (there the MoE output is unnamed).
- 2026-09-20 — **the OpenCL kernels are capturable without debug caps**
  [code]: an `LD_PRELOAD` shim intercepts
  `clCreateProgramWithSource` / `...WithIL` / `...WithBinary` and
  `clBuildProgram`, writing each bucket. Two traps paid for: (a) resolving the
  real symbol with `dlsym(RTLD_DEFAULT)` when `RTLD_NEXT` is NULL recurses into
  the shim and SIGSEGVs — resolve through an explicit
  `dlopen("libOpenCL.so.1")`; (b) the plugin creates programs from SOURCE (not
  IL), so the build-time `CL_PROGRAM_SOURCE` query is a driver quirk — the real
  source is at `clCreateProgramWithSource`. Captured 14 buckets for the u4
  `layer0/out` cut; persisted in the A770 capture dir.
- 2026-09-20 — **the GDN kernel source, read** [code]:
  the A770-generated `src_013.cl` (`paged_gated_delta_net_opt`, 107 KB)
  is a **sequential per-token recurrence** — `for (; token<chunk_end; ...)`,
  `state *= b_g; h = state·k; update=(v-h)*beta; state=fma(k,update,state);
  out=state·q` — with `tokens_to_next_boundary`/`interval` advancing the state
  table at chunk boundaries. Its only reductions are `sub_group_reduce_add`
  (fixed tree). **No floating-point atomics, no unordered reduction** — so the
  GDN core is not an obvious run-to-run variance source, and #38099's
  deterministic-wrong-values family is a different failure mode than ours.
- 2026-09-20 — **ISA disassembly works, and the kernel source is
  DEVICE-SPECIFIC** [measured-here]: `ocloc compile -file src_013.cl -device
  xe-hpg -options "-cl-std=CL3.0 -cl-mad-enable"` builds for the A770 (`ACM-G10`) and
  `ocloc disasm` yields `.text.paged_gated_delta_net_opt...asm` (1143 lines) +
  the SPIR-V section; persisted in the A770 disasm dir.
  The SAME bucket fails to compile for `-device xe2` (`bmg-g21`, error -11):
  the plugin JITs the source per device, so the A770 bucket is not the B60
  kernel. **The B60-generated bucket must be captured on the B60** — the
  disassembly of the divergent kernel requires a B60 run, which the queued
  ladder provides.
- 2026-09-20 — **leading kernel candidate after the map** [measured-here +
  code]: the resident variance survives when all experts go to CPU
  (force-the-tier, 0.0816), so the expert GEMMs (`jit:gemm:any__i8`) are
  exonerated. What remains on the device is the dense `jit:gemm:any__f16`
  (oneDNN GEMM in the GDN/attention projections) and the GDN core. oneDNN's
  `jit:gemm:any` can select a split-K kernel that accumulates with FP atomics
  (unordered) — a classic run-to-run variance source, and a plausible reason it
  is card-specific (kernel selection differs by device). The GDN core, by
  source, is not. Testable next: force the FullyConnected impl/kernel via
  `OV_GPU_FORCE_IMPLEMENTATIONS` (available in this build) and watch the B60
  rate; and disassemble the B60-generated `jit:gemm` bucket.
- 2026-09-20 — **A770 T-bisect at `layer0/out` (u4 `d4-ov`, device-resident)**
  [measured-here]: GPU.1, window 0, `--repeat 4`, T in {1, 2, 4, 8, 16, 24, 32,
  33, 48, 64, 96, 128, 192, 256, 512, 1024}. **Every T is bit-identical x4**;
  the forward succeeds down to T=1 (output shape (1,T,10240), finite); compile
  1.5-2.7 s, forward #1 0.35-1.73 s, repeats 0.006-1.14 s. **There is no T
  threshold on the A770** — the block is deterministic at every tested size.
  The row>=32 precondition needs T>=33; that is the smallest shape the B60 leg
  needs to carry the defect. This is the clean-card half of the minimal-shape
  search and it ran entirely on GPU.1, before the B60 window.
- 2026-09-20 — **the native `d48n` T-bisect is deferred, and why** [measured-here]:
  the native form asserts `!native || (_cpu_tier && ...)`
  (`moe_3gemm_swiglu_op`), so it needs `MOE_CPU_TIER=YES`; the CPU tier runs on
  the same 8-core host the D4 arm is saturating (arcint 613% CPU, load 7.8 over
  8 cores). A tiered A770 leg now would perturb the in-flight D4 measurement,
  so the native sweep waits for D4. The native T=1024 control is already read:
  bit-identical x4 on the A770 (`3546c8ee1ebc`). The u4 result above carries the
  structural threshold; native differs only in constant placement, not ops.
- 2026-09-20 — **the B60 block cut localises to the GDN half** [measured-here].
  The seconds-scale reproducer (`tools/boot_serving_shape.py --artifact
  the native artifact `--cut <node> --cut-prune --repeat 8`,
  GPU.0 = B60, window 0 first 1024 ids, native, tier) read:

  | cut (native d48n, B60) | ops | first row | rows differ | max |diff| | argmax moved | fwd | A770 x4 |
  |---|---|---|---|---|---|---|---|
  | `layer0/mixer_out` (GDN/hyper-connection only) | 194 | 137 | **887/1024** | 1.22e-4 | 0 | 0.025 s | bit-identical |
  | `layer0/moe/mix` (+ MoE) | 332 | 137 | 887/1024 | 4.88e-4 | 0 | 27 s | bit-identical |
  | `layer0/out` | 367 | 137 | 887/1024 | 4.88e-4 | 0 | 27 s | bit-identical |

  So the variance is **generated inside `layer0/mixer_out`** (1.22e-4 = one f16
  ulp at ~0.125) and the MoE merely **amplifies** it (-> 4.88e-4). The expert
  GEMMs, the MoE, and chunking are exonerated; the reviewer's independent arms
  agree (D4 unchunked `--prefill-chunk 0`: 0.0955 / 0.2021, worse than chunk
  512; force-the-tier: 0.0816). This is attribution **(ii)**, and inside it the
  **GDN/hyper-connection half** — `ocl::paged_gated_delta_net::opt` + the short
  conv + the dense `jit:gemm:any__f16`/RMS hyper-connection projections. It
  matches openvino **#38099** as a **sibling** (deterministic-wrong values vs
  run-to-run variance), not the same bug. Forward is 0.025 s at the GDN-only
  cut: the reproducer made the localisation a seconds-scale experiment.
- 2026-09-20 — **follow-on capture queued** [measured-here]: the A770-generated
  OpenCL bucket does NOT compile for `-device xe2` (B60) — the plugin JITs the
  source per device — so the B60 `layer0/mixer_out` kernels are captured on
  the B60 (`repro-b60-cldump.sh`, `LD_PRELOAD` shim, waits for the ladder),
  then disassembled for `xe2` with `ocloc` and tested by
  `OV_GPU_FORCE_IMPLEMENTATIONS`. The reviewer session runs the A770 depth-48
  gate leg after this clears (~14:40Z).
- 2026-09-20 — **the card-specific kernel difference is the SUBGROUP WIDTH**
  [measured-here + code]. The B60-generated bucket for `layer0/mixer_out`
  (the B60-generated `src_002.cl`, 107,570 B, captured
  on GPU.0 with the `LD_PRELOAD` shim) was diffed against the A770-generated
  bucket (`src_013.cl`, 107,569 B). Extracting the
  `paged_gated_delta_net_opt` kernel body from each gives 17 diff lines, **one
  substantive**:

      A770:  #define SUBGROUP_SIZE 8
      B60 :  #define SUBGROUP_SIZE 16

  with `K_LANE_ELEMS = K_HEAD_DIM / SUBGROUP_SIZE` therefore 16 (A770) vs 8
  (B60). The plugin JITs ONE GDN source and specializes the subgroup width by
  device; the Xe2/B60 specialization is the 16-wide one, and 16-wide is the
  nondeterministic one (8-wide A770 bit-identical x4). The two places the width
  changes the lane mapping are the core's `sub_group_reduce_add` tree (8 vs 16
  lanes) and the state write `BLOCK_WRITEN(recurrent_state_table, ...)`. The
  short-conv `paged_causal_conv1d_ref` rides in the same bucket and gets the
  same width change, so it is a second candidate on the same axis.
  Next (deferred while the reviewer's A770 gate leg holds the 8 cores):
  `ocloc compile -device xe2` + `ocloc disasm` on `src_002.cl`, then
  `OV_GPU_FORCE_IMPLEMENTATIONS` to swap `ocl::paged_gated_delta_net::opt` for
  `ref`/another kernel and watch the B60 rate.
- 2026-09-20 — **trap paid for: `pgrep -f` self-match** [measured-here]. The
  wait guards `while pgrep -f "bash ./repro-l0-b60.sh"; do sleep; done` matched
  the log-watcher's OWN `sh -c` command line (which carries the pattern
  literally), so they never cleared; the ladder finished 13:57:24Z and the
  capture sat until killed at 14:19Z. Fix: guard on a marker file
  (`grep -q "REPRO DONE" run.log`) or anchor the pattern
  (`pgrep -f "^bash \./repro-l0-b60\.sh$"`). The same bug hit the tracked
  child's wait command.
- 2026-09-20 — **the B60 ladder, full table** [measured-here] (all 8 cells
  compiled and ran, none refused; the ladder's log, `--repeat 8`,
  window 0, same request, state zeroed per forward). Every cell's #2..#8 differ
  from #1 except the two marked bit-identical:

  | cell | artifact | cut | T | compile | fwd #1 | #2..#8 | first row | rows | max |diff| | argmax |
  |---|---|---|---|---|---|---|---|---|---|---|
  | d48n-l0mixer | d48n native | layer0/mixer_out | 1024 | 0.82 s | 0.087 s | DIFFERS | 137 | 887/1024 | 1.2207e-04 | 0 |
  | d48n-l0moe   | d48n native | layer0/moe/mix | 1024 | 0.99 s | 26.80 s | DIFFERS | 137 | 887/1024 | 4.8828e-04 | 0 |
  | d48n-l0out   | d48n native | layer0/out | 1024 | 4.37 s | 27.60 s | DIFFERS | 137 | 887/1024 | 4.8828e-04 | 0 |
  | d48n-l0out-t512 | d48n native | layer0/out | 512 | 0.32 s | 13.56 s | DIFFERS | 139 | 1-3/512 | 1.5259e-05 | 0 |
  | **d48n-l0out-t256** | d48n native | layer0/out | 256 | 0.29 s | 6.78 s | **BIT-IDENTICAL x8** | - | 0 | 0 | - |
  | **d48n-l0out-t64** | d48n native | layer0/out | 64 | 0.28 s | 1.73 s | **BIT-IDENTICAL x8** | - | 0 | 0 | - |
  | d4n-l0out    | d4n native | layer0/out | 1024 | 1.52 s | 38.73 s | DIFFERS | 137/248 | 582-887/1024 | 4.8828e-04 | 0 |
  | d12-l0out    | d12 u4 | layer0/out | 1024 | 9.37 s | 2.64 s | DIFFERS (bimodal) | 32 then 14 | 1-3/1024 @2.44e-04, 248-249/1024 @5.86e-03 | 0 then 1 |

  Two new facts. **(a) There is a T threshold on the B60** that does not exist
  on the A770: T=64 and T=256 are bit-identical x8, T=512 carries 1-3 rows at
  1.5e-5, T=1024 carries 887 rows at 1.2e-4 (mixer_out) / 4.9e-4 (out). The
  defect therefore needs T in (256, 512]; the minimal reproducing shape is not
  T=33 but ~512. The A770 showed no threshold at any T in 1..1024. **(b) the d12
  u4 cell is bimodal**: half the repeats carry the cut-table's documented mode
  (1-3 rows / 2.44e-4 / argmax 0), half a larger mode (248-249 rows / 5.86e-3 /
  argmax 1) not in the cut table — so the d12 block has two regimes, and the
  native d48n block's 887-row mode is the broad one.
- 2026-09-20 — **the harness's controls, stated [measured-here + code]**: (a)
  the state tables (`conv_state_table.*`, `gated_delta_state_table.*`) are
  ZEROED before every forward (`zero_state` -> `copy_from` a zero host tensor,
  the log line `state: N state table(s) zeroed before the forward`), so
  "every forward differs" is not a state-carry ramp — and the A770 running the
  same harness is bit-identical x4, which a ramp would not be. Keep the A770
  control attached to every repeat count quoted. (b) `--digest-ports` (new,
  `tools/boot_serving_shape.py`, uncommitted) prints a sha256 of the request's
  own input ports (`inputs_embeds`, `conv_mask`, `subsequence_begins`, `la.*`,
  both state tables) before each forward, giving the INPUT-BIT-IDENTITY proof:
  if those are identical across repeats while the cut output differs, the
  divergence is provably internal to the kernel. [CORRECTED 2026-09-21: the
  primary read path is `t.data` on the request's own tensor; `copy_to` is the
  fallback for a remote one. Same bytes either way.] To run on the deferred B60
  arms behind the A770 gate leg.
- 2026-09-20 — **two refinements to the controls [code + measured-here]**: (a)
  `--digest-ports` hashes the STORED bytes of each port (a host tensor of the
  port's own element type via `copy_to`, then the contiguous raw buffer; dtype
  printed, "no re-render") — so a state-table digest is over f16 bytes, not a
  float view. (b) the disasm arm compiles the SAME B60-generated source
  (`src_002.cl`) for BOTH `xe2` and `xe_hpg`/`dg2` and diffs the two `.asm`
  files, turning the source's one substantive define into the instruction-level
  consequence (SLM+barrier vs shuffle for `sub_group_reduce_add`; the
  `BLOCK_WRITEN` addressing at `K_LANE_ELEMS` 8 vs 16). All-zero state-table
  digests on every repeat independently confirm `zero_state()` is effective.
- 2026-09-20 — **the disasm arm is width-only, arch fixed** [method, agreed]:
  compile the B60-generated `src_002.cl` twice for `-device xe2`, once as-is
  (`SUBGROUP_SIZE 16`) and once with the define forced to 8 (copy + sed the
  `#define SUBGROUP_SIZE 16`; `-D` cannot cleanly override a define the source
  already sets), then diff the two `.asm`. Same arch removes every
  target-feature confound (dot-product instructions, FP16, work-group limits),
  so the diff is purely the width's consequence. The cross-arch
  (`xe2` vs `xe_hpg`) diff is kept as a secondary. Datum: the A770-generated
  `src_013.cl` FAILS to compile for `xe2` (`-11`) — a bucket JITed for one arch
  does not necessarily recompile for another, consistent with the plugin
  specializing per device and with the width as the specialization seam.
- 2026-09-20 — **INPUT-BIT-IDENTITY PROOF on the B60** [measured-here]. The
  `layer0/mixer_out` cut, native `d48n`, GPU.0, T=1024, `--repeat 8`,
  `--digest-ports --kernel-names`. Every repeat fed BIT-IDENTICAL inputs, and
  the state tables' digests are exactly the all-zero sha256 (the zeroing is
  effective, computed independently):

  | port | digest (every repeat) | shape | dtype |
  |---|---|---|---|
  | `inputs_embeds` | `479969949951` | 1024x2560 | f32 |
  | `conv_mask` | `e9bac255f4ad` | 1x1024 | f32 |
  | `subsequence_begins` | `ba4c2184969f` | 2 | i32 |
  | `la.block_indices` | `af5570f5a181` | 2 | i32 |
  | `la.block_indices_begins` | `2fcd151b8295` | 2 | i32 |
  | `la.past_lens` / `la.cache_interval` | `df3f619804a9` | 1 | i32 |
  | `conv_state_table.0` | `9eb8fa54d87a` = zero hash of 245,760 B | 3x10240x4 | f16 |
  | `gated_delta_state_table.0` | `7971f869259a` = zero hash of 4,718,592 B | 3x48x128x128 | f16 |

  Yet `layer0/mixer_out` differs on every repeat (first row 137 or 248; 749-887
  of 1024 rows; max |diff| 1.2207e-04; argmax 0). **Inputs identical, output
  differs -> the divergence is provably INTERNAL to the kernel**, and no
  upstream-producer or state-ramp explanation survives. The all-zero state
  digests are the independent confirmation that `zero_state()` works.
- 2026-09-20 — **the width-only `ocloc` arm is CONFOUNDED** [measured-here]:
  forcing only `#define SUBGROUP_SIZE 8` in the B60-generated `src_002.cl` and
  compiling for `-device xe2` FAILS for `bmg-g21` (`-11`) while the as-is 16
  builds for bmg-g21/bmg-g31/lnl-m. The JIT output has other width-derived
  constants baked for 16 (`K_VEC_SIZE` from `get_vec_size`, `K_LANE_ELEMS`,
  `K_CHUNKS`), so the sed'd source is an inconsistent kernel, not a valid
  width-8 variant. **The only valid width-8 test is a plugin rebuild** (so
  every derived constant is recomputed), which is the arm in flight.
- 2026-09-20 — **the width pin is NOT implementable** [measured-here]: a minimal
  `__attribute__((intel_reqd_sub_group_size(8)))` kernel with
  `cl_intel_required_subgroup_size`/`cl_khr_subgroups` pragmas FAILS to compile
  for every Xe2 target with an explicit message — `bmg-g21`, `bmg-g31`,
  `lnl-m`, `ptl-h`: **"Kernel compiled with required subgroup size 8, which is
  unsupported on this platform"** (backend `-11`). The same kernel compiles for
  the A770, `ACM-G10`. So `get_subgroup_size` returning 16 for `xe2` is FORCED by
  the platform, not discretionary; the one-line pin is dead, and the middle row
  of the handoff's fix table ("B60 deterministic in code we own") does not have
  the width pin as its mechanism. A fix must make the 16-wide path
  deterministic (or an upstream/IGC change), or move the state path.
- 2026-09-20 — **suspect (a) refuted: the reduce lowering is a fixed tree at
  both widths** [measured-here]: `sub_group_reduce_add` in a minimal kernel,
  disassembled with `ocloc` — width 16 / `bmg-g21` lowers to a register-halving
  add tree (`add (8)`, `add (4)`, two `add (1)`), width 8 / `ACM-G10` (A770) to the
  same shape (`add (4)`, `add (1)`, `add (1)`). No SLM, no `barrier`, no
  `send`-to-SLM for the reduction, on either arch. So the reduction is a fixed,
  deterministic tree per launch; it cannot be the run-to-run variance.
- 2026-09-20 — **launch geometry also refuted for the co-resident RMSNorm**
  [code]: `rms_kernel_base.cpp` sets `lws = GetOptimalLocalWorkGroupSizes(gws,
  engineInfo)` — `engineInfo` is static per device, not per launch — and
  `rms_kernel_bfyx_opt.cpp` derives `LWS`/`ONE_SUBGROUP_ROW` from
  `RmsSchedulingPolicy` constants, not from a per-launch driver query. So the
  RMS geometry is static too. Both hypotheses the review raised
  (reduce-lowering; launch geometry) are now refuted by code/ISA, and the
  location remains `layer0/mixer_out` (input-bit-identity proven) with the
  mechanism OPEN — candidates left: the state write/read path
  (`BLOCK_WRITEN(recurrent_state_table, ...)`) or a plugin/driver-level
  ordering race, neither visible in the kernel source.
- 2026-09-20 — **post-forward state digests split the remaining suspects**
  [measured-here]. Same arm (`layer0/mixer_out`, native `d48n`, GPU.0, T=1024,
  `--repeat 8`, `--digest-ports` now extended to digest the state tables AFTER
  each forward):

  | port | pre-forward | post-forward |
  |---|---|---|
  | `conv_state_table.0` | all-zero `9eb8fa54d87a` every repeat | `7000565fc4eb` **every repeat** (deterministic) |
  | `gated_delta_state_table.0` | all-zero `7971f869259a` every repeat | `e8ff0176dc46` / `bb20806d026f` **alternating** (varies) |

  So the variance lands in the **GDN state write/read**, while the conv state
  write is clean — consistent with the conv's data input being a model port
  (bit-identical) and the GDN's being an internal tensor. CAVEAT, stated rather
  than smoothed: the GDN's data inputs q/k/v are the outputs of the dense
  `jit:gemm:any__f16` projections INSIDE the cut, which the port digest does not
  cover. So this does not yet prove the GDN core writes the variance; the q/k/v
  producers (the dense gemms) remain equally live. The conv being clean is the
  control that shows a kernel whose input IS a port is deterministic. Next:
  freeze/observe q/k/v (a cut or stub at the linear_attn's projection output),
  or a minimal same-shape `jit:gemm` determinism run on the B60.
- 2026-09-20 — **CORRECTION to the entry above: it is NOT a 2-value alternation**
  [measured-here]. The full post-forward `gated_delta_state_table.0` sequence
  over the 8 repeats is: `ffdd9c82bf32` (#1), `82e760e4a098` (#2),
  `d5269bbd7cfc` (#3), `e8ff0176dc46` (#4, #5, #6, #8), `bb20806d026f` (#7) —
  **five distinct hashes**, with `e8ff0176dc46` dominant. The reviewer's own
  discriminator reads: exactly 2 distinct = structural ping-pong/period-2;
  3+ = ordering race / stochastic. Five distinct therefore **refutes the
  ping-pong hypotheses** (double-buffered reset mismatch, ping-pong page) and
  supports a **stochastic state write/read**. `conv_state_table.0` stays at
  **one** distinct value (`7000565fc4eb`) across all 8, the deterministic
  control. The first forward (#1) differs from every repeat, so the state is
  not converging to a fixed point either.
- 2026-09-20 — **fresh-process test: the FIRST forward is deterministic, later
  ones are not** [measured-here]. Three cold processes (p1/p2/p3), same arm
  (`layer0/mixer_out`, native d48n, GPU.0, T=1024, x8). Post-forward
  `gated_delta_state_table.0` in order (repeats after #1):

  | proc | #1 | #2..#8 | distinct (#2..#8) |
  |---|---|---|---|
  | p1 | `ffdd9c82bf32` | `82e760e4a098`, `d5269bbd7cfc`, `e8ff0176dc46` x4, `bb20806d026f` | 4 |
  | p2 | `ffdd9c82bf32` | `e8ff0176dc46` x7 | 1 |
  | p3 | `ffdd9c82bf32` | `e8ff0176dc46`, `bb20806d026f`, `e7f0386ec258`, `fc9a1c607cf4`, `bb20806d026f`, `e8ff0176dc46` x2 | 4 |

  In ALL three processes **#1 is `ffdd9c82bf32`** (deterministic, reproducible
  across cold processes), and `conv_state_table.0` is `7000565fc4eb` on every
  forward of every process (fully deterministic). So the defect is
  **first-launch-deterministic / later-launch-stochastic**: the cold first
  forward is reproducible; subsequent forwards vary and sometimes settle (p2)
  or spread (p1/p3). This is NOT a strict 2-value ping-pong (the distinct count
  is 1-4, not 2), so the reviewer's ping-pong branch is refuted; but the fact
  that #1 is special across processes points at a **first-launch effect**
  (the plugin's feedback-driven kernel compilation on the first request) plus a
  stochastic cached-kernel behaviour, or a hidden buffer initialised by #1.
  Consequence for the gate: `floor_pair` must state WHICH pair was read; a
  (warmup, r0) pair and an (r0, r1) pair are not the same observation.
- 2026-09-20 — **serialization does NOT fix it: the overlap/missing-event
  hypothesis is REFUTED** [measured-here]. Added an `LD_PRELOAD`
  `clEnqueueNDRangeKernel` interceptor to the CL shim that calls
  `clFinish(queue)` after EVERY kernel enqueue (`CLDUMP_SERIALIZE=1`; 233
  enqueues in the run). Under full serialization, `layer0/mixer_out` native d48n
  on GPU.0, T=1024, x8: all 7 repeats still DIFFER, and the post-forward
  `gated_delta_state_table.0` still spreads (`ffdd9c82bf32`, `e8ff0176dc46` x4,
  `fc9a1c607cf4`, `daa0e8511036` — 3 distinct among repeats). So the defect is
  NOT inter-kernel overlap / a missing event between co-resident kernels.
  `GPU_QUEUE_THROTTLE=HIGH` likewise fails to collapse it (6 distinct) —
  throttling is not serialization. Mechanism therefore narrows to a
  WITHIN-kernel nondeterminism: a lane/address race inside one kernel, or a
  compiler/hardware-level effect, given identical inputs and no overlap.
- 2026-09-20 — **per-row state diff: stable 14 heads, row 0 only, 1 ulp**
  [measured-here]. Post-forward `gated_delta_state_table.0` (shape 3x48x128x128)
  dumped per repeat and diffed element-by-element against #1:

  - differing elements are confined to **dim0 = row 0** (of the 3 bound rows);
  - they touch **exactly the same 14 of 48 heads in every repeat**:
    `[3, 5, 6, 7, 10, 13, 17, 22, 31, 39, 41, 42, 43, 47]`;
  - within those heads the diffs are scattered across the whole 128x128 plane
    (no contiguous tile/lane block), and `max |diff| = 9.7656e-4` = **one f16
    ulp**;
  - only the NUMBER of flipped elements varies (2423..3924), never the head set.

  The STABLE head set makes the effect **data-dependent**, not
  scheduling-dependent, and the scatter rules out a contiguous lane-address
  overlap signature. Combined with the clean minimal gemm (same-shape f16
  MatMul, 1024x10240x2560, **bit-identical x8** on GPU.0), the leading
  remaining cause is a **compiler/hardware-level nondeterminism on Xe2** in the
  GDN arithmetic — not a plugin-code race. That is the boundary the review named:
  fixable inside a kernel, or not fixable in our tree (route around: A770 card,
  `ref` impl, or upstream IGC/OpenVINO).
- 2026-09-20 — **the JIT is deterministic; the variance is at EXECUTION level**
  [measured-here]. Compiling the B60 `src_002.cl` twice with `ocloc` for
  `bmg-g21` gives **byte-identical** binaries
  (`be20f259e064e763c47090615ee013da61d00daa66e2f65bbd894e1fb944dc78`, 64,712 B
  each). So the compiler's own output does not vary run to run: the
  nondeterminism is at **execution** level (IGC/hardware), not at the compiler.
  With serialization refuted (`clFinish` after every enqueue) and the minimal
  same-shape gemm clean, the mechanism is a compiler/hardware-level execution
  nondeterminism in the GDN arithmetic on Xe2.
- 2026-09-20 — **the in-tree substitution routes are blocked** [measured-here]:
  `OV_GPU_FORCE_IMPLEMENTATIONS` and `QUEUE_TYPE` are
  `OV_CONFIG_RELEASE_INTERNAL_OPTION`s — rejected via `core.compile_model`
  (`Option not found`) in both `--plugin-prop KEY=VALUE` and the documented
  `OV_GPU_FORCE_IMPLEMENTATIONS=...` env form, because this plugin build has no
  `ENABLE_DEBUG_CAPS`. So the `ref` substitution cannot be run either, and no
  incremental plugin build dir exists under the workspace to rebuild one.
  **Boundary, stated flatly:** the GDN arithmetic on Xe2 is nondeterministic
  below the kernel-choice level; the in-tree options are (i) the **A770
  measurement card** (clause (d) already readable there) or (ii) an upstream
  IGC/OpenVINO fix (the #38099 family). The `--digest-ports`/per-row/per-process
  harness and the serialization shim remain the instruments for any upstream
  report.
- 2026-09-20 — **the defect's fingerprint** [measured-here]: post-forward
  `gated_delta_state_table.0` differs from #1 in **dim0 = row 0 only**, in
  **exactly the 14 heads `[3,5,6,7,10,13,17,22,31,39,41,42,43,47]`**, scattered
  across the 128x128 plane, at **one f16 ulp** (9.7656e-4); only the flip COUNT
  varies (2423..3924), never the head set. Any future arm that does not
  reproduce this head set is measuring something else.
- 2026-09-20 — **the upstream instrument set is preserved as a unit** [code]:
  `tools/cldump.c` (the OpenCL capture + `CLDUMP_SERIALIZE` shim, now in the
  repo, uncommitted) with `--digest-ports`/`--dump-state`/`--kernel-names`/
  `--stub`/`--cut --cut-prune` in `tools/boot_serving_shape.py`; the per-row
  diff, the fresh-process test, the two-arch `ocloc` disasm, and the
  fingerprint. Together they are the **sibling reproducer** for #38099 (same
  GDN-unroll family, different failure mode: run-to-run at execution level vs
  deterministic-wrong at chunk >= 2). [2026-09-20: operator approval granted
  and the comment **POSTED** — openvinotoolkit/openvino#38099
  `issuecomment-5751935449`, full text kept at
  `docs/upstream-38099-comment.md`.] The in-tree fix route is
  closed (JIT byte-identical, execution-level variance, internal-option
  substitution blocked, no build dir), so the gate closes on the A770
  measurement card with a repeat-8 arm owed for rigour.
- 2026-09-20 — **why the debug-cap rebuild is required (gate confirmed in source)**
  [code]. `OV_GPU_DUMP_SOURCES_PATH`, `OV_GPU_DUMP_TENSORS_PATH`,
  `QUEUE_TYPE` and `FORCE_IMPLEMENTATIONS` are all `RELEASE_INTERNAL`/debug
  options, and `PluginConfig::set_user_property`
  (`src/inference/src/dev/plugin_config.cpp:81-88`) rejects any option whose
  visibility is not covered by the caller's `allowed_visibility`, throwing
  `Couldn't set unknown property: GPU_FORCE_IMPLEMENTATIONS`. The env-prefix
  initializer that would make `OV_GPU_...` work lives under
  `#ifdef ENABLE_DEBUG_CAPS` (`src/inference/dev_api/openvino/runtime/plugin_config.hpp:45`).
  So no property spelling (`GPU_FORCE_IMPLEMENTATIONS` at Core or compile time,
  or the `OV_`-prefixed env form) can reach the option in this build; only a
  rebuild with `ENABLE_DEBUG_CAPS=ON` can. The canonical recipe is
  `contrib/packaging/marfrit-openvino/build-openvino.sh` (pin `71640275`, 41
  patches through `0043`, applied by `git apply` at build time), with the two
  `-DENABLE_*DEBUG_CAPS` flags flipped ON. The built plugin tree is not a git
  tree (`git` exits 128), so it cannot reproduce the series by hand-cmake.
- 2026-09-21 (**the A770 repeat-8 arm lands — clause (d) is settled at the
  depth-4 evidence's own repeat count**). [measured-here] `RUN@b6dbca5`, A770
  (GPU.1), d48n, ratio 99 + tier, KV u8, chunk 512, served path, `WARMUP=0
  REPEAT=8`, dump `d48n-a770-rep8.bin`, eight forwards at 3,563-3,593 s each
  (replay done 2026-09-21T05:41:50Z): **all seven consecutive pairs read
  KL(A‖B) mean -0.000000, 0/1367 moved, argmax 1.0000, max |diff| 0.000,
  `bit-identical True`**. So the **×2 caveat in the 2026-09-20 entry is
  discharged**, `F_served(A770, d48) = 0` at the same repeat count as the
  depth-4 ×8/×12 evidence, and BERLIN-001 clause (d) is settled **on the A770
  as the measurement card**. No A770 depth-12 leg exists; the ×8 here is
  depth 48. Recorded in window-051's clause (d), DESIGN §7.0.2cb and the
  handoffs. The next city (0.5.2 VENICE) is open in its own campaign,
  `docs/campaigns/expert-hot-set-lru.md`.

- 2026-09-25 — **dated correction: the measurement card is ACM-G10, not
  `acm-g12`** [code]. Every row in this document that named the A770 as
  `acm-g12` is corrected in place to **ACM-G10** (DG2-512, PCI `0x56A0`).
  `acm-g12` is a DIFFERENT DG2 die — DG2-256, 16 Xe-cores, shipped in Arc Pro
  A60 / A570M / A530M. Both dies are **Xe-HPG**, which is the distinction every
  argument here actually rests on (Xe-HPG vs `xe2`), so **no conclusion moves**:
  the subgroup-width pin stays dead, the reduction stays a fixed tree at both
  widths, and the ×8 settlement on the A770 is untouched. The `acm-g12` strings
  in the `ocloc` logs are compiler-target labels the toolchain printed; their
  provenance is unverified and is now recorded as such rather than read as
  evidence about the silicon. The same label error is present in the comment
  already posted upstream (`docs/upstream-38099-comment.md`) and needs a
  follow-up there.
