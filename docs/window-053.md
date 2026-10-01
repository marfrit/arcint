# window-053 — 0.5.3 LISBON acceptance (LISBON-001: every measured row EMPTY)

Recorded 2026-09-23, before the LISBON card window exists and before the
expert store can be filled correctly at all. This file is the acceptance
commit of 0.5.3 in the form the roadmap's law demands
(`ROADMAP-0.5.x.local.md`:2–6): *the first commit is the acceptance criteria,
with the measured rows EMPTY; prediction commits precede measurement; gates
red-first; counts generated; comparisons to FreeToken pinned ONLY by our own
measured runs.* The commit that fills a row is a measurement commit and pastes
the raw output.

[FILLED 2026-09-24: the three measured rows below are filled from the LISBON
gate window — both B60 arms in one window (cold TTFT, `wait4`/`ru_maxrss`) and
the A770 restart-determinism confirmation. The criteria as written 2026-09-23
are unchanged; only the EMPTY cells moved. Every measured value carries
`measured-here`; the raw command and output are pasted in each row and the full
packet is in the git-ignored `docs/handoff-nvme-direct-expert-tier.local.md`. ]

The markers are `docs/window-050.md`'s (`RUN@<sha>`, `RUN@wt+<sha>`,
`RUN@unrecorded`, `DRY`, `UNTESTED`); a row with no marker is EMPTY, and EMPTY
is the honest state of every measured column in this commit. Every disposition
below carries an evidence class (`paper` / `code` / `measured-here`) where it is
a disposition of fact; the acceptance-cell bullets are configuration, not
dispositions. The pinned threshold is arithmetic (`code`), **not**
`measured-here`.

## Feature (roadmap 0.5.3)

NVMe miss tier — cold start with **NOTHING prebound**; bytes stream
NVMe→host→card under the same residency policy. The milestone's second half is
the PLE pin removal: the Flash-Next n-gram table is now a bounded per-forward
staging buffer instead of 26.82 GiB of USM host for the life of the process
(`docs/campaigns/ple-disk-backend.md`, gate PASSED depth-4 on 2026-09-23).

## What LISBON-001 owns

Three acceptance rows, all EMPTY:

1. **cold time-to-first-token, nothing prebound** — the arcwell path
   (NVMe→VRAM direct) against the host-fed path (NVMe→host→VRAM), same card,
   same artifact, same residency policy, **both arms in one window**, arcwell
   at or below host-fed, and an absolute threshold `X` pinned here;
2. **RSS bounded through boot** — the boot child's peak RSS under a stated
   bound, read with the `wait4` child-rusage discipline;
3. **restart determinism** — two cold boots, byte-identical greedy answers in
   digest form.

The three are the campaign gate of `docs/campaigns/nvme-direct-expert-tier.md`
(Gate section) and §7.4 of `docs/design-nvme-direct-expert-tier.md`. This
document does not fill them.

## Scope depth — pinned at depth 4, with full depth as an operator decision

**Pinned: depth 4.** The mechanism under test is the byte path
(NVMe → {host | VRAM} → VRAM) under the static-partition residency policy. That
path is **depth-independent**: one expert slice is 2,457,600 B
(`code`: `src/exec/flash_next_offload.h:45`) at every layer, the partition's
membership is a pure function of configuration fixed at `bind()`
(`code`: patch 0018; patch 0046), and the plugin's slot layout is per layer, not
per depth. The accepted precedent for exactly this shape is the PLE gate, which
ran its staged-vs-pinned served window at **depth 4** because "the n-gram table
is a property of the model, not of the depth"
(`docs/campaigns/ple-disk-backend.md`, 2026-09-23 close). Depth 4 keeps both
arms in ONE window on ONE card, which is the campaign's own requirement.

**Flagged, not assumed: a full-depth (48-layer) variant is an OPERATOR
DECISION.** The document does **not** assume full depth. Two facts make that a
real decision rather than a formality: (a) the depth-4 export's pinned expert
set is a truncated subset, so the *byte volume* the fill moves at depth 4 is
small and the bandwidth claim is not exercised at milestone scale; (b) the
native full-depth artifact is where the 8.38 GB (ratio 86) / 15.10 GB
(ratio 75) pinned fills and the ~35 s cold-forward reads live, but it also
carries the B60 determinism caveat below. The operator decides whether the
gate closes at depth 4 (mechanism) or is repeated at depth 48 (scale); the
arithmetic for the full-depth variant is printed separately in §1 and is
**not** the gate.

## The acceptance cell (fixed before measuring)

- **Card: B60** (`GPU.0`, PCI `8086:e211`). The gate's measurement card is the
  B60 because arcwell's own README excludes the A770; the A770 (`GPU.1`, PCI
  `8086:56a0`) is the determinism-confirmation card only. DRM numbering is
  inverted vs OpenVINO numbering — cards are identified by PCI id, never by
  number (`docs/sop-card-window.md` §2).
- **Artifact: the depth-4 staging-PLE twin** (`qwen38-flash-next-d4s-ov`), the
  correction-matched artifact whose `config.json` is identical to the pinned
  twin and whose only difference is the n-gram port partition
  (`docs/campaigns/ple-disk-backend.md`, 2026-09-23). Using it makes LISBON-001
  one window for both halves and lets row 2 read the freed 26.82 GiB.
- **Residency policy: the static partition, `--offload-ratio 86`** → the
  plugin's integer-division pool is `512*(100−86)/100 =` **71 slots/layer**
  (`code`: patches 0041/0047; `docs/window-052.md` dated correction). The
  design note's prefetch schedule is one batch per layer at this budget
  (`docs/design-nvme-direct-expert-tier.md` §3).
- **Cold state: page cache dropped between arms and boots; nothing prebound**
  (the pinned slots start empty on the host and device).
- **Prompt: the 5-token France prompt** ("The capital of France is", greedy,
  temperature 0), the reference cell's own minimal request.
- **Instrument hygiene: one fresh process per arm**, sampler on the physical
  host started before the leg (`docs/sop-card-window.md` §1), 4 GiB watchdog,
  plugin `ov-0047` (`f021de51b5812ee2`, patches 0003–0047), KV u8, the served
  binary named by sha in the measurement commit.

## 1. Cold TTFT, nothing prebound — MEASURED 2026-09-24

**The gate.** Both arms measured in one window on the cell above; the arcwell
arm required to land **at or below** the host-fed arm; and the measured cold
TTFT required to land **at or below `X`**. The prefetch depth that achieves it
is recorded (a win at a depth the serving loop cannot reach is not a win); the
`AW_IOC_STATS` delta proving `via_host_bounce = 0` is pasted.

**The pinned threshold.**

    X = T_boot + T_fill + T_prefill

| term | value | basis | evidence class |
|---|---|---|---|
| `T_boot` | **136 s** | B60 depth-4 served boot to `/props → 200` = 2 min 16 s (`docs/window-050.md` §4.9, 2026-09-13) | `measured-here` (input) |
| `T_fill`, host-fed | **0.387 s** | pinned bytes (697,958,400 B) ÷ the in-container NVMe read rate (~1.68 GiB/s) | `code` arithmetic |
| `T_fill`, arcwell | **0.240 s** | same bytes ÷ arcwell's own 2.91 GB/s — **arcwell's number, labelled, never ours** | `code` over arcwell's `measured-here` |
| `T_prefill` | **3.08 s** | B60 depth-4 cold prefill, 5 tokens (`docs/window-050.md` §4.9) | `measured-here` (input) |

The fill arithmetic, shown explicitly:

    slice            = 2,457,600 B              (code: src/exec/flash_next_offload.h:45)
    slots/layer      = 71                        (code: 512*(100-86)/100, plugin integer division)
    moe layers       = 4 in scope                (the depth-4 artifact)
    pinned bytes     = 71 x 4 x 2,457,600
                     = 697,958,400 B             (code)
    host-fed rate    ~= 1.68 GiB/s ~= 1,803,882,782 B/s
                       (measured-here: the in-container NVMe miss-feed rate,
                        docs/design-qwen-flash-next.md WP6b) = 0.387 s
    arcwell rate     = 2.91 GB/s  (arcwell's own measurement, labelled) = 0.240 s

    X = 136 + 0.387 + 3.08 = 139.467 s   ->   X = 139.5 s

**Evidence class of `X`: `code`** (arithmetic over `code` byte counts and
`measured-here` inputs). `X` is a predictively pinned threshold and is
**never** labelled `measured-here`. arcwell's 2.91 GB/s and 1.125 ms-expert
figures are arcwell's own measurements on arcwell's hardware
(`docs/campaigns/nvme-direct-expert-tier.md`, Known section); they are used
only as a labelled projection and are never quoted as arcint measurements.

**Conservative note, stated not smoothed.** The `T_boot` input is the
*non-staging* depth-4 B60 boot, which paid a 44.1 s full-table bind; the
staging twin does not copy the 26.82 GiB table, so the LISBON cell's boot is
expected **below** 136 s and `X` is an upper bound. The measured value reads
the true term. If a component is wrong, `X` is corrected in place with the
measurement's date — the gate value is not moved silently after the fact.

**Full-depth variant — arithmetic only, NOT the gate, OPERATOR DECISION.**
For the native depth-48 artifact (`qwen38-flash-next-d48n-ov`) at ratio 86 on
the B60: pinned bytes `= 71 × 48 × 2,457,600 = 8,375,500,800 B`; `T_boot` =
34.7 s (model-ready, B60 probe 2026-09-23, `measured-here`); host-fed fill
`4.64 s`; arcwell-labelled fill `2.88 s`; `T_prefill` = 11.11 s (B60 ratio-86
5-token prefill). `X_full(host-fed) = 34.7 + 4.64 + 11.11 = 50.45 s`;
`X_full(arcwell-labelled) = 34.7 + 2.88 + 11.11 = 48.69 s`. This is printed so
the operator can open the scale gate with a pinned number if they choose; it is
**not** the acceptance threshold of this commit and is **not** assumed.

| quantity | predicted (`code`) | measured (2026-09-24) |
|---|---|---|
| arcwell arm cold TTFT, depth 4 | ≤ host-fed arm; ≤ 139.5 s | **92.492 s** (`measured-here`) |
| host-fed arm cold TTFT, depth 4 | ≤ 139.5 s | **99.679 s** (`measured-here`) |
| prefetch depth that achieves it | recorded | **4 batches in flight** — one batch per layer over the 4 depth-4 layers (`code`: patch `0049`, `lgc::nvme_fill::Scheduler(transport, /*depth=*/4)`); the per-arm counter is `batches +4`, and `max_inflight` is module-global reading **220**, a high-water inherited from the warm-up leg (`measured-here`) |
| `AW_IOC_STATS` delta (`via_host_bounce`, `max_inflight`, `batches`, `batch_reads`, `segments`) | `via_host_bounce = 0`, `max_inflight > 1` | arcwell arm: `bytes +697,958,400`, `reads +852`, `segments +871`, `batches +4`, `batch_reads +852`, `via_host_bounce 0→0`, `max_inflight 220` (`measured-here`); host-fed arm: `bytes +0`, no arcwell traffic |
| decode t/s at the reference cell, both arms | must not regress vs the measured host-tier baseline (`docs/window-052.md`: 0.5–0.8 t/s B60; 0.526 t/s A770 same-day host comparand) | arcwell **4.1 t/s** (32 tok in 7.76 s) vs the same-window host-fed **3.4 t/s** (32 tok in 9.28 s) — no regression (`measured-here`; the `window-052` 0.5–0.8 t/s figure is a different depth/artifact comparand, not this cell's control) |

**Raw evidence (`measured-here`, the whole row).** Card B60 (`GPU.0`, PCI
`8086:E211`); artifact `qwen38-flash-next-d4s-ov` (`openvino_language_model.xml`
sha256 `823997733f0b4b07…`, bin 9,270,599,557 B); plugin `ov-0049` (sha256
`2d83e2a6…`, patches through `0049`); served binary `a6dac5b5…`;
`--offload-ratio 86 --moe-cpu-tier --paged-kv u8 --prefill-chunk 512 --n-ctx
8192`; the 5-token France prompt, greedy, `max_tokens 32`. The container CPU
topology regression was worked around with `unshare -rm` presenting a
contiguous `online=0-7` / `possible=0-15` view (the launcher logs
`[ns] online=0-7 possible=0-15`). Both arms are fresh processes in ONE window;
a first arcwell boot warmed the compile cache, then page cache was dropped
before each measured arm, so each arm reads cold. `cold TTFT = launch→/props
200 + request→first streamed token`.

```
# Arm B, host-fed (MOE_OTD_PINNED_NVME_FILL unset), B60
RESULT {"arm": "hostfed", "device": "GPU.0", "ratio": "86", "tag": "armB-hostfed",
  "ready": true, "t_boot_s": 97.19, "ttft_s": 2.491, "cold_ttft_s": 99.679,
  "digest": "9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f",
  "exit_code": 0, "wait4_ru_maxrss_gib": 3.697, "vmhwm_prefix_gib": 3.697,
  "vmhwm_le_wait4": true, "server_has_arcwell_refusal": false}
# Arm A, arcwell (MOE_OTD_PINNED_NVME_FILL=1), B60
RESULT {"arm": "arcwell", "device": "GPU.0", "ratio": "86", "tag": "armA-arcwell",
  "ready": true, "t_boot_s": 90.25, "ttft_s": 2.247, "cold_ttft_s": 92.492,
  "digest": "9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f",
  "exit_code": 0, "wait4_ru_maxrss_gib": 3.697, "vmhwm_prefix_gib": 3.697,
  "vmhwm_le_wait4": true, "server_has_arcwell_refusal": false}
# AW_IOC_STATS delta, arcwell arm (separate client fd; module-global counters)
pre : reads=4299 bytes=3521740800 segments=4394 batches=32 batch_reads=4299
      max_inflight=220 via_host_bounce=0
post: reads=5151 bytes=4219699200 segments=5265 batches=36 batch_reads=5151
      max_inflight=220 via_host_bounce=0
# host-fed arm, same instrument: bytes 3521740800 -> 3521740800 (delta 0)
```

**Verdict — L1 holds.** Arm A (92.492 s) ≤ arm B (99.679 s), a 7.19 s margin,
and the mechanism is visible in the same run's own counters: the host-fed arm
read the expert bytes with `total_disk_io_ms 12,497` (`avg_disk_io_us 7,386`,
`tensor_loads 1,692`) against the arcwell arm's `1,928` (`1,131`).
**L2 holds** — both arms land below `X = 139.5 s` (`code`), with the measured
`T_boot` terms (90.25 / 97.19 s) below the pinned 136 s input.
**L3 holds** — decode does not regress at the reference cell.

**Clause L1** — if the arcwell arm's cold TTFT exceeds the host-fed arm's, the
fill does not pay and the record says so with the prefetch depth that was
reached. **L2** — if the measured arm exceeds `X`, the pin was wrong; the row
names which term missed. **L3** — if the decode t/s regresses at the reference
cell in trade for the cold-boot number, the change does not close.

## 2. RSS bounded through boot — MEASURED 2026-09-24

**The bound: the boot child's peak RSS must stay at or below 32 GiB.** That is
the host class the milestone's own charge names: removing the 26.82 GiB PLE pin
"frees ~26.82 GiB of host RAM (making a 32/44 GiB host viable — the class the
external 16 GB/32 GB Flash-Next runs live in)"
(`ROADMAP-0.5.x.local.md`, 0.5.3). A boot that needs more than 32 GiB fails
this row, because it has not delivered the class the PLE half exists to make
reachable. This is the row where the **freed term is read**.

**What reads it.** The `wait4` child-rusage discipline, CF-KEYSTONERSS
(`DESIGN.md` §7.0.2 area; `docs/design-qwen-flash-next.md`): the parent reaps
the boot child with `os.wait4` and reads `ru_maxrss` **for that child**; the
child's own mid-run `getrusage(RUSAGE_SELF)` is kept beside it and asserted
**not to exceed** the `wait4` value. The hole this closes is demonstrated, not
argued: a child that allocates 2 GiB after its self-read reports 0.52 GiB where
the kernel accounts 2.52 — 2.00 GiB invisible to the self-read. In parallel,
the SOP physical-host sampler records `MemAvailable` / `Shmem` / ZFS ARC every
2 s with a 4 GiB watchdog (`docs/sop-card-window.md` §1), because driver /
USM-host memory is charged to the physical host, not the container.

**Arithmetic for the freed term** (`measured-here`, PLE gate, depth 4): the
pinned twin's container `VmRSS` peak was **19.79 GiB** (20,753,068 KB); the
staged arm's was **4.93 GiB** (5,173,232 KB); Δ **14.86 GiB**. The staged
n-gram resident is **2.884 MiB** against the pinned **26.82 GiB USM host**, and
the physical `MemAvailable` minimum rose 9.89 → 32.91 GiB
(`docs/campaigns/ple-disk-backend.md`, 2026-09-23). The depth-4 arithmetic RSS
for the LISBON cell is therefore ≈ 5–7 GiB; the 32 GiB bound is the class
ceiling, not the expected value.

[CORRECTED 2026-09-24: the measured boot-child `ru_maxrss` is **3.697 GiB**.
The ≈ 5–7 GiB was an upper shape from the container-`VmRSS` instrument, not a
prediction the gate was read against; the row is read from the kernel's
per-child `wait4` accounting, a different instrument, and it lands below the
estimate.]

| quantity | bound | measured (2026-09-24) |
|---|---|---|
| boot child `ru_maxrss`, read by `os.wait4` | ≤ 32 GiB | arcwell arm **3.697 GiB**; host-fed arm **3.697 GiB** (`measured-here`, B60) |
| child `RUSAGE_SELF` vs `wait4` | self ≤ wait4 (CF-KEYSTONERSS) | the mid-run `/proc/<pid>/status:VmHWM` prefix **3.697 GiB** ≤ `wait4` **3.697 GiB** (both arms). The child is an exec'd C++ binary with no self-report, so the mid-run witness reads the same `mm->hiwater_rss` quantity `getrusage(RUSAGE_SELF).ru_maxrss` is derived from; the field is a prefix of the whole-run `wait4` value |
| physical-host `MemAvailable` minimum under the sampler | > 4 GiB (watchdog) | B60 window **48,092,424 kB = 45.86 GiB**; **0 watchdog trips**. A770 window 45,709,112 kB = 43.59 GiB, 0 trips |
| freed PLE term on the ledger | the 26.82 GiB USM-host pin is gone; staging is `T x H x 90 B` | `lgc load: ngram table STAGED: 1 port(s) of 33600 rows x 90 B = 2.884 MiB` — the pinned twin's 26.82 GiB is gone (`measured-here`) |

**Raw evidence (`measured-here`).** The two B60 arms in row 1 are the boot
children; each was launched as a direct child of the measurement interpreter,
reaped once with `os.wait4`, and the kernel's `ru_maxrss` for that child is
above. A 4 GiB watchdog armed before the leg never fired; the physical sampler
minima are from `lisbon-gate-b60-phys-sampler.log` and
`lisbon-gate-a770-phys-sampler.log`. `ngram table STAGED` is read from the
arcwell arm's own `server.err`.

**Clause L4** — if `ru_maxrss` exceeds 32 GiB, the boot has not reached the
32 GiB host class and the row is RED. **L5** — if the `wait4` value is below
the child's self-read, the instrument is void (the accounting was taken before
teardown); the row is re-run, not passed.

## 3. Restart determinism — MEASURED 2026-09-24 (A770 host-fed arm)

**The gate.** Two cold boots of the same served configuration — process
restarted, page cache dropped between them, nothing prebound — the same greedy
requests, **byte-identical answers in digest form** (the digest form of the old
restore witness, DESIGN §3.4: greedy output is a pure function of the request,
never of which fetches happened to land first).

**Scope, decided up front and stated exactly.** The acceptance cell's
measurement card is the B60, but the B60 cannot carry a byte-identity claim
(the determinism caveat below). The row is therefore read on the A770
(`GPU.1`, PCI `8086:56A0`), the bit-readable card, **on the only arm that can
run there: the host-fed arm** (fill disabled). `~/src/arcwell` excludes the
A770, so **arcwell is B60-only** (`docs/campaigns/nvme-direct-expert-tier.md`,
Entry criteria 2); the arcwell arm's restart determinism is therefore **OWED**
and is governed by design rule D3's load barrier (§4 of the design note), not
by cross-boot byte-identity. The row's table below is the host-fed arm on the
A770, `--offload-ratio 86`, same d4s artifact, same flags, two cold boots with
page cache dropped between.

| probe | boot 1 digest | boot 2 digest | identical? |
|---|---|---|---|
| "The capital of France is", greedy | `9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f` | `9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f` | **YES, byte-identical** (`measured-here`, A770 host-fed) |
| the reference cell's prompt, greedy | (the acceptance cell's own minimal request = the same prompt) | (same) | **YES** |

**Raw evidence (`measured-here`).** A770 (`GPU.1`, PCI `8086:56A0`), host-fed
arm (fill disabled), `qwen38-flash-next-d4s-ov`, ratio 86, same flags. A first
A770 boot warmed the A770 compile cache; page cache was dropped before each
measured boot.

```
# A770 cold boot 1
RESULT {"arm": "hostfed", "device": "GPU.1", "ratio": "86", "tag": "a770-boot1",
  "ready": true, "t_boot_s": 104.18, "ttft_s": 2.736, "cold_ttft_s": 106.917,
  "digest": "9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f",
  "exit_code": 0, "wait4_ru_maxrss_gib": 3.698, "vmhwm_prefix_gib": 3.698}
# A770 cold boot 2
RESULT {"arm": "hostfed", "device": "GPU.1", "ratio": "86", "tag": "a770-boot2",
  "ready": true, "t_boot_s": 103.17, "ttft_s": 2.476, "cold_ttft_s": 105.651,
  "digest": "9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f",
  "exit_code": 0, "wait4_ru_maxrss_gib": 3.697, "vmhwm_prefix_gib": 3.697}
```

**Verdict — L6 holds for the measured scope.** The two A770 host-fed cold boots
are byte-identical. The **arcwell arm's restart determinism is OWED**, with the
reason recorded: no bit-readable card can run arcwell. In this same window the
B60 arcwell and host-fed arms also returned the same digest — but that reading
is **not** admissible as a byte-identity claim (the B60 caveat below).

A digest that differs between boots is a finding; its mechanism is measured
with the instrument that exists (`tools/boot_serving_shape.py --cut layerN/out
--repeat`), not narrated. The arcwell fill's byte-transparency is separately
enforced at the load barrier by design rule D3 (`docs/design-nvme-direct-
expert-tier.md` §4): a pinned expert whose fetch has not landed is a **load
failure**, never a silent demotion — so a boot's residency set cannot depend on
I/O timing.

**Clause L6** — any digest differs → RED, localise the layer that first
diverges, do not narrate.

## The B60 determinism caveat (operator decision 2026-09-23)

On the B60, **timing and statistics are admissible**; **byte-identity claims
are admissible only where the B60 is known readable**, otherwise they are
paired with an **A770 confirmation** (`8086:56A0` = A770 = `GPU.1`;
`8086:E211` = B60 = `GPU.0`; DRM inverted, identify by PCI id). The reason is
the recorded per-card defect: the depth-48 served path is bit-identical on the
A770 (`F_served = 0`, 0/1367 rows moved) and carries a per-forward within-kernel
nondeterminism on the B60 (KL mean 0.1361/0.1512)
(`docs/campaigns/served-prefill-determinism.md`; `docs/window-051.md` clause
(d)). Row 3's B60 reading therefore pairs with an A770 confirmation of the same
served answer; the arcwell arm itself is B60-only and its byte path is judged by
the load-barrier rule, not by cross-boot byte-identity alone.

## Dependencies — THE GATE IS BLOCKED, NOT MEASUREMENT-READY

The acceptance document must not be read as a runnable measurement plan. Three
things block it today. Every one is recorded, not worked around.

1. **The ext4 expert store is synthetic arcwell test data.** `measured-here`
   (B60 probe 2026-09-23): the store is 1,700 files of 2,457,600 B, each
   resolving to exactly one plain extent — but each file is **one deterministic
   4096-byte block repeated 600×** (600/600 blocks identical, distinct files
   differ only by seed). It carries **no expert tensors and no scales/zp**.
   Nothing can be filled correctly from it today.
   [CLEARED 2026-09-23, artifact-format step: a REAL store now exists on the
   ext4 partition — 3,408 files of 2,457,600 B (`8,375,500,800 B`), every file
   exactly ONE plain extent, `aw_fiemap` byte-verifying all 3,408 against the
   raw device with its `--mutate` leg failing on content, and a 24-expert
   byte-exactness sample exact to the u4 half-step. See the campaign's
   "Artifact-format step" section. The synthetic store itself was NOT touched.]
2. **The scales/zp store-layout precondition is OWED and belongs to the
   artifact-format step.** `code`: the plugin's device slot layout and the
   weight-file layout differ for scales/zp — device `[group][oc]`, file
   `[oc][group]` — and `maybe_transpose_scale_zp` transposes on upload
   (patches 0011:75-90, 0006:291). The weights are `[oc][ic]` in both, so a
   byte-transparent full-slice DMA would land the weights correctly and the
   scales **transposed**. Resolving it (lay files in device order, and adapt
   the host tier's scale indexing, or move the small scale/zp tensors through
   the existing host path) is a **store-layout decision** owed to the
   artifact-format step, not to this gate. No correct fill is claimed.
   [RESOLVED 2026-09-23, artifact-format step: the verdict is **weights-only,
   device order** — the 2,457,600-byte slice is the three u4 weight matrices,
   which are byte-identical file↔device, so a naive full-slice DMA is
   byte-transparent; scales/zp are EXCLUDED from the DMA slice (adding them
   gives 2,553,600 B = 623.4375 pages, not page-aligned, and the plugin's
   per-tensor scale destinations are unaligned too) and stay on the existing
   host path that transposes. The device-order alternative is implemented for
   a future integration and tested. See the campaign's "Artifact-format step"
   section, §2.]
3. **The consumer does not exist.** The `AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`
   fill's timing is measured standalone (submit 20.9 ms of a 64.5 ms batch;
   2.71 GB/s at 71 experts; `via_host_bounce` delta 0), but the D2/D3
   integration that runs it inside the serving loop is **OWED**. The
   "serving step with the fill overlapping" has no number.
   [PARTLY ADDRESSED 2026-09-23, D2/D3 integration leg — dependency 3 STANDS.
   The schedule now exists and is wired plugin-side in patch `0048` (one batch
   per layer, four in flight, collect→`set_filled`, a load barrier that retries
   once then REFUSES, and a red-first guard refusing a synchronous
   `AW_IOC_READ_BLOCKS` on the decode path); it applies on the full 0003–0048
   series and compiles clean, and its 8-cell ladder plus 3 mutants is green/red
   as designed. But the fill cannot LAND: the static partition's slot buffers
   are host-mapped (the B60 probe's `device_slot_buffers=0`) and arcwell
   requires a dma-buf from an xe VRAM BO, so there is no destination a
   byte-transparent fill can land in. The transport and the per-expert
   dma-buf BO-backed slot destination are OWED, and with them the "serving
   step with the fill overlapping" number. See the campaign's "D2/D3 consumer
   integration" section.]
   [DESTINATION CLEARED 2026-09-24, byte-destination proof leg — dependency 3
   STANDS on its consumer-integration half. The "there is no destination a
   byte-transparent fill can land in" clause is **CLEARED**: the destination is
   settled and proved end-to-end at the smallest scale on the B60 by a
   non-arcint client (`tools/arcwell_bo_dma_proof.c`). arcwell provides **no**
   allocator/helper — `stub/src/arcwell.c` says "userspace creates a
   host-visible VRAM BO on xe and exports it as a dma-buf, then hands us the
   fd" — so the plugin creates the BO itself with the raw xe DRM ioctls
   (`DRM_IOCTL_XE_GEM_CREATE` VRAM + `NEEDS_VISIBLE_VRAM` + `CPU_CACHING_WC`,
   64 KiB-rounded; `DRM_IOCTL_PRIME_HANDLE_TO_FD`), registers it peer-to-peer,
   and imports the same dma-buf into OpenCL (`cl_khr_external_memory_dma_buf`,
   the path arcwell's own E2E cell proved). Proof: one real 2,457,600 B expert
   from the real store landed in the BO, host readback **byte-identical**
   (sha256 `4a4bb0f9…`), `AW_IOC_STATS` delta `via_host_bounce = 0`,
   `max_inflight > 1`. What still STANDS: the plugin-side transport wiring, the
   OpenCL import into OpenVINO's slot descriptors, and the "serving step with
   the fill overlapping" number — none exists yet, so the three rows stay OPEN.
   See the campaign's "D2/D3 byte destination" section.]
   [TRANSPORT + OPENCL IMPORT BUILT 2026-09-24, plugin transport leg —
   dependency 3 STANDS. The plugin now ships the production `Transport`
   (`moe/pinned_nvme_transport.hpp`: raw xe VRAM BO creation, dma-buf export,
   `AW_IOC_MAP_BUFFER` peer-to-peer, `AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`)
   and the OpenCL import (`bind_pinned_nvme_pool()`: `engine.import_buffer()`
   replaces the layer's host-mapped `gate_w`/`up_w`/`down_w`, so the resident
   expert is read from the BO; the six scale/zp tensors stay on the host path).
   Patch `0049` reverse-applies/re-applies on the 0048 tree and compiles clean
   (rc 0; `apply-check-0049.txt` and `build-0049.log` in the packet); the
   mechanism is proven on the B60 by
   `tools/arcwell_cl_slot_proof.c` — two real store experts DMA'd into
   per-tensor VRAM BOs, imported into OpenCL, read back **byte-identical**
   (sha256 `d463d1d5…`), `via_host_bounce` delta 0, `max_inflight` 6 — with
   five red legs. What still STANDS: the **integrated served number** (the fill
   running inside the serving loop, cold TTFT both arms, byte-identity across
   two cold boots, decode non-regression) and the depth-4-artifact↔store key
   match; none is measured here. The three rows stay OPEN. See the campaign's
   "D2/D3 plugin transport + OpenCL slot import" section.]
   [CLEARED 2026-09-24, D4 integrated served leg — dependency 3's
   consumer-integration clause is CLEARED. A depth-4 store was built with the
   tracked writer (`tools/q4e/expert_store.py`, unchanged; 4 layers × 71 experts
   = 284 experts / 697,958,400 B, one plain extent each, `aw_fiemap` verifies
   all 284 against the raw device, manifest exact) and its four layer keys are
   the served artifact's `weight_0` bin offsets
   (`284632533 / 2033390357 / 3650420373 / 5369067397`) — the plugin's own
   `MOE_OTD_ROUTING_HIST` dump reproduces exactly those keys, and 852/852
   pinned expert-role slices are byte-identical to the artifact's `weight_u4`
   constants. The integrated B60 run with `MOE_OTD_PINNED_NVME_FILL=1` did not
   refuse: boot to `/props` 97 s, `T_prefill` 1.43 s (arcwell-arm cold TTFT
   **98.4 s** = `T_boot` 97 s + `T_prefill` 1.43 s, `code` arithmetic over
   `measured-here` terms), `AW_IOC_STATS` delta `bytes +697,958,400` exactly,
   `via_host_bounce 0→0`, `max_inflight 220`, 12 BOs live and released. The three
   gate rows stay **OPEN** — they are the next leg: both arms in one window
   (cold TTFT, arcwell ≤ host-fed, ≤ `X`), the `os.wait4`/`ru_maxrss` row, and
   the two-cold-boot determinism row with its A770 confirmation. See the
   campaign's "D4 integrated served leg" section.]

**Consequence:** the three rows stay **OPEN**. [UPDATED 2026-09-23,
artifact-format step: dependencies 1 and 2 are CLEARED, so the gate is no
longer blocked on "nothing can be filled correctly". It is now blocked ONLY on
dependency 3, the D2/D3 consumer integration, with its three rows still OPEN.]
[UPDATED 2026-09-23, D2/D3 integration leg: dependency 3 STANDS — the schedule
is built, wired and compile-verified (patch `0048`), but there is no dma-buf
VRAM destination for the fill, so nothing lands and the overlapping-step number
still does not exist. The three rows stay OPEN.]
[UPDATED 2026-09-24, byte-destination proof leg: dependency 3's **destination**
clause is CLEARED — the fill's destination is settled and proved on the B60
(a caller-created xe VRAM BO whose dma-buf meets arcwell's mapping contract),
so the gate is no longer blocked on "there is no destination". Dependency 3
STANDS only on its consumer-integration half: the plugin-side `Transport`, the
OpenCL import into the slot descriptors, and the "serving step with the fill
overlapping" number. The three rows stay OPEN.]
[UPDATED 2026-09-24, D4 integrated served leg — **dependency 3 is CLEARED**: its
last clause, the consumer integration with an integrated served number, now
exists. The plugin transport + OpenCL slot import run inside the serving loop on
the depth-4 artifact whose store holds the same layer keys; the arcwell arm's
integrated cold TTFT is `T_boot` 97 s (served boot to `/props → 200`) +
`T_prefill` 1.43 s = **98.4 s** (`code` arithmetic over `measured-here` terms;
the run-1 streamed-TTFT client call failed, so 98.4 s is the composed figure,
not an instrument reading), the `AW_IOC_STATS` delta is `bytes +697,958,400` (the
exact pinned payload) with `via_host_bounce 0→0` and `max_inflight 220`. The gate
is no longer blocked on a missing consumer. The three gate rows stay **OPEN** —
the NEXT leg is the two-arm window, the `os.wait4` RSS row and the two-cold-boot
determinism row.]
[CORRECTED 2026-09-24, D4 integrated served leg: the retained sentence this
note replaces ("The gate's own bytes cannot be laid down.") is no longer true —
a byte-transparent store exists and the fill **landed** on the B60
(`AW_IOC_STATS bytes +697,958,400`, `via_host_bounce 0`). The three acceptance
rows are still EMPTY; this document remains the criteria, not a measurement of
the gate.]

## Harness and tools the measurement will use (named now)

- **served binary** `arcint` + plugin `ov-0047` (`f021de51b5812ee2`, patches
  0003–0047), KV u8, `--moe-cpu-tier`, `--offload-ratio 86`, the pinned
  `--prefill-chunk` and `--n-ctx`; artifact sha pasted at measurement.
  [UPDATED 2026-09-24, D4 integrated served leg: the LISBON gate window needs the
  plugin that carries the pinned NVMe fill — patches through `0049`
  (sha256 `2d83e2a6…`), not `ov-0047`; `ov-0047` cannot run the arcwell arm.
  The D4 integrated leg used the 0049 plugin and the served binary
  `a6dac5b5…`. The two-arm gate must use that plugin.]
- **device-free probe** `tools/boot_serving_shape.py` (`--stage compile` /
  `--stage forward`, `--cut layerN/out`, `--repeat`) for the determinism
  localisation.
- **OTD counters** `MOE_OTD_PERF_LOG` plateau probe (device-byte plateau,
  evictions) and the `[OTD_PERF]` lines (`gpu_hits`, `gpu_misses`,
  `evictions`, `device_slot_buffers`, `host_slot_buffers`).
- **arcwell client** derived from `~/src/arcwell`'s `stub/test/aw_async_test.c`
  (the `aw_fill_budget` shape): `AW_IOC_MAP_BUFFER` with the
  `AW_MAP_F_REQUIRE_P2P` assertion, `AW_IOC_SUBMIT_BATCH` / `AW_IOC_BATCH_WAIT`
  at DEPTH 4, `AW_IOC_STATS` read as a **delta** before/after the leg's work.
- **RSS instrument** `os.wait4` + `ru_maxrss` (CF-KEYSTONERSS), plus the SOP
  physical-host sampler (`docs/sop-card-window.md` §1) with the 4 GiB watchdog.
- **admission ledger** `--fit-ledger-dir` for the fit arithmetic, so the
  load-time probes are skipped on the second matching arm.

[USED 2026-09-24, the LISBON gate window: plugin `ov-0049` as required (sha256
`2d83e2a6…`); served binary `a6dac5b5…`; the `unshare -rm` contiguous-CPU-view
workaround (operator-local, no tracked change; the launcher logs
`[ns] online=0-7 possible=0-15`); `--fit-ledger-dir` was passed but the ledger
directory did not exist (`fit ledger write failed` in the run log), so each arm
ran its own load-time probes. Raw command and output are pasted in rows 1–3.]

Nothing is run in this commit.

## Falsifiable clauses, listed

| # | clause | dies if |
|---|---|---|
| L1 | the arcwell arm's cold TTFT is at or below the host-fed arm | it exceeds the host-fed arm — the fill does not pay |
| L2 | both arms' cold TTFT ≤ `X` = 139.5 s | either exceeds `X` — the pin's arithmetic is corrected in place, dated |
| L3 | decode t/s at the reference cell does not regress | it falls below the measured host-tier baseline in trade for the cold-boot number |
| L4 | boot child `ru_maxrss` ≤ 32 GiB | it exceeds 32 GiB — the 32 GiB host class is not reached |
| L5 | `wait4` value ≥ child `RUSAGE_SELF` | the self-read exceeds the reaped value — the instrument is void |
| L6 | two cold boots are byte-identical in digest form | any digest differs — localise, do not narrate |

## Explicit NOT claims

- **No measurement of any kind is made in this commit; every measured cell
  reads EMPTY (pending measurement).**
- **arcwell's numbers are arcwell's.** 2.91 GB/s, 1.18 vs 2.91 GB/s, 16.3×
  less CPU/GiB, 1.125 ms/expert, 2.18 GB/s serial, submit comparands — none is
  an arcint measurement. The only use here is a labelled projection.
- **The store cannot be filled today** (a repeated 4096-byte block, no expert
  tensors, no scales/zp), and the scales/zp layout is an OWED precondition.
- **The consumer integration does not exist**; the standalone fill budget is
  not the integrated gate.
- **Full depth is NOT assumed.** The pinned scope is depth 4; the full-depth
  arithmetic is printed for an operator decision only.
- **No FreeToken comparison here** — that is GENEVA (0.5.8), pinned only by our
  own runs.
- **No byte-identity claim on the B60** outside the A770-paired rule above.
- `X` is arithmetic (`code`), not `measured-here`, and is not moved after the
  fact.

## Where it lives

`docs/window-053.md` (this file). Campaign:
`docs/campaigns/nvme-direct-expert-tier.md`; design note:
`docs/design-nvme-direct-expert-tier.md`; PLE half:
`docs/campaigns/ple-disk-backend.md`, `docs/design-ple-disk-backend.md`.
Precedents: `docs/window-051.md` (BERLIN-AS-LISBON), `docs/window-052.md`
(VENICE). External convention: `~/src/arcwell` (tracked, established);
`~/src/FreeToken-ref` (tracked, established). Operator-local facts (host,
paths, lock, raw output) live only in the git-ignored packet
`docs/handoff-nvme-direct-expert-tier.local.md`.

## Status

- 2026-09-23 — **LISBON-001 drafted as the 0.5.3 acceptance commit; every
  measured row EMPTY.** Criteria and the threshold `X = 139.5 s` pinned before
  any LISBON card window. Scope pinned at depth 4 (mechanism, PLE precedent);
  full depth flagged as an operator decision. The gate is recorded as
  **BLOCKED** — the synthetic ext4 store and the OWED scales/zp store-layout
  precondition mean nothing can be filled correctly today. No measurement, no
  card leg, no wake lock, no host/store/module mutation. Review before commit
  (reviewer subagent); corrections in place with dates.
- 2026-09-23, later — **artifact-format step: the store precondition is
  CLEARED.** A REAL expert store was built on the ext4 partition (3,408 files
  of 2,457,600 B, `8,375,500,800 B`), every file exactly ONE plain extent,
  `aw_fiemap` byte-verifying all 3,408 and its red leg failing on content; the
  scales/zp layout verdict is **weights-only, device order**, with scales/zp
  excluded from the DMA slice and left on the transposing host path. Dependencies
  1 and 2 are cleared; dependency 3 (the D2/D3 consumer integration) remains,
  so the three gate rows stay **OPEN** and the gate is blocked only on that
  integration. No card leg, no module load, no wake lock; the arcwell module was
  found loaded/carved (inherited) and left as found; the synthetic store was
  not touched.
- 2026-09-23, later — **D2/D3 integration leg: the schedule is built, wired and
  compile-verified; dependency 3 STANDS.** Patch `0048` adds the load-time
  pinned-fill schedule plugin-side (one batch per layer, four in flight,
  collect→`set_filled`, a load barrier that retries once then REFUSES, and a
  red-first guard refusing a synchronous `AW_IOC_READ_BLOCKS` on the decode
  path), tracked device-free as `src/exec/pinned_nvme_fill.h` and tested by
  `tests/test_pinned_nvme_fill.cpp` (8 cells green; 3 mutants each fail their
  named cell). It applies on the full 0003–0048 series and compiles clean with
  the production target. But the fill cannot LAND: the static partition's slot
  buffers are host-mapped (B60 probe `device_slot_buffers=0`) and arcwell needs
  a dma-buf from an xe VRAM BO, so the per-expert dma-buf BO destination and
  the arcwell transport are OWED, and an enabled fill refuses the load. The
  three gate rows stay **OPEN**; the gate is now blocked on the BO-backed slot
  destination, not on a missing schedule. No card leg, no module load, no wake
  lock, no store/host mutation; the arcwell module was found loaded/carved and
  left as found.
- 2026-09-24 — **byte-destination proof leg: the destination is SETTLED AND
  PROVEN; dependency 3's destination clause is CLEARED and its
  consumer-integration clause stays standing.** One card leg on the B60 alone
  (`8086:E211`), non-arcint; the A770 was untouched. New tracked client
  `tools/arcwell_bo_dma_proof.c` creates a caller-owned xe VRAM BO
  (`DRM_IOCTL_XE_GEM_CREATE` with VRAM + `NEEDS_VISIBLE_VRAM` + `CPU_CACHING_WC`,
  size rounded to 64 KiB), exports it via `DRM_IOCTL_PRIME_HANDLE_TO_FD`,
  registers it peer-to-peer (`AW_MAP_F_REQUIRE_P2P` asserted), transfers one
  real 2,457,600 B expert from the real store, and verifies the BO by host
  readback: **byte-identical**, sha256 `4a4bb0f9…` both sides; `AW_IOC_STATS`
  delta `via_host_bounce = 0`, `max_inflight 261 (> 1)`, `batches`/`batch_reads`/
  `segments` moved by the leg's own work, `bytes +2,457,600`. Five mutation
  legs; four go red as required (partition offset dropped → mismatch; unaligned
  `in_dest_offset` refused/`out_err=-22`; system-memory BO refused `-ERANGE`;
  readback corrupted → mismatch). Three dated corrections: on the B60 the 64 KiB
  BO gate is **not** enforced (a 37.5 × 64 KiB BO was accepted end-to-end,
  unlike the A770/DG2 measurement in `~/src/arcwell/KERNEL_FACTS.md`); a
  submission-time geometry error is reported in `out_submitted`/`out_err`, not
  the ioctl return; and the system-memory refusal is `-ERANGE` at the carve
  range check, before the `via_host_bounce` sites. What stays OWED: the
  plugin-side transport, the OpenCL import into the slot descriptors, and the
  overlapping-step number. Three gate rows stay **OPEN**. No `arcint` leg, no
  module load/unload, no store write; the arcwell module was found loaded and
  carved and left as found; wake lock taken and released; sampler minimum
  32.9 GiB, 0 watchdog trips.
- 2026-09-24, later — **plugin transport + OpenCL import leg: built and
  mechanism-proven; dependency 3 STANDS; the three rows stay OPEN.** Patch
  `0049` adds `moe/pinned_nvme_transport.hpp` (production `Transport`: raw xe
  VRAM BO + dma-buf export + `AW_IOC_MAP_BUFFER` peer-to-peer +
  `AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`, store ordinal `dense_layer *
  capacity + slot`) and `moe/aw_uapi.h`, wires it into the coordinator, and
  replaces each layer's host-mapped `gate_w`/`up_w`/`down_w` with BO-backed
  memories imported through `engine.import_buffer()`; the six scale/zp tensors
  stay on the transposing host path and are uploaded at load. The patch
  reverse-applies/re-applies on the 0048 tree and compiles clean (rc 0). The
  mechanism was proven on the B60 end-to-end by `tools/arcwell_cl_slot_proof.c`
  (three per-tensor VRAM BOs, two real store experts as six requests, OpenCL
  import, byte-identical readback sha256 `d463d1d5…`, `via_host_bounce` delta
  0, `max_inflight` 6) with five red legs. What stays OWED: the integrated
  served number (cold TTFT both arms, byte-identity across two cold boots,
  decode non-regression) and the depth-4-artifact↔store key match. One card
  leg on the B60 alone (`8086:E211`), A770 untouched; the arcwell module was
  found unloaded (host sleep) and loaded for the leg, left loaded; `/dev/arcwell`
  and the store were passed into the container to run the proof; sampler
  minimum `MemAvailable` 30.66 GiB (32,153,844 kB), 0 watchdog trips. The wake
  lock was found held by the coordinator and left untouched.
- 2026-09-24, later — **D4 integrated served leg: the pinned NVMe fill runs
  inside the serving loop; dependency 3 is CLEARED and the integrated number
  exists.** A depth-4 store was built with the tracked writer (unchanged): 284
  files of 2,457,600 B, every file one plain extent, `aw_fiemap` byte-verifies
  all 284 against the raw device with its `--mutate` leg failing on content,
  manifest mapping and sha256s exact; its four layer keys are the served
  artifact's `weight_0` bin offsets, and the plugin's own `MOE_OTD_ROUTING_HIST`
  dump reproduces exactly them (`key_collisions=0`), with 852/852 pinned
  expert-role slices byte-identical to the artifact's `weight_u4` constants. The
  integrated B60 run with `MOE_OTD_PINNED_NVME_FILL=1` did not refuse: boot to
  `/props` 97 s, `T_prefill` 1.43 s (arcwell-arm cold TTFT **98.4 s** = boot +
  prefill, `code` arithmetic), and the
  `AW_IOC_STATS` delta is `bytes +697,958,400` exactly, `via_host_bounce 0→0`,
  `max_inflight 220`, 12 BOs live and released. `docs/window-053.md` dependency
  3 is cleared in place; the three gate rows stay **OPEN** (the next leg). No
  new tracked code; the A770 was untouched; the arcwell module was left loaded
  and carved; the coordinator's wake lock was left held and untouched.
- 2026-09-24, later — **LISBON-001 gate window: the three measured rows are
  FILLED.** Rows 1–3 above now carry `measured-here` values. One B60 two-arm
  window (one card, one artifact, ratio 86, plugin `ov-0049`, `unshare -rm`
  CPU-view workaround): arcwell arm cold TTFT **92.492 s**, host-fed arm
  **99.679 s** — **L1 holds** (arcwell ≤ host-fed, 7.19 s margin) and **L2
  holds** (both ≤ `X = 139.5 s`); prefetch depth **4 batches in flight** (one
  per layer); `AW_IOC_STATS` arcwell delta `bytes +697,958,400`, `reads +852`,
  `segments +871`, `batches +4`, `via_host_bounce 0→0`, `max_inflight 220`;
  host-fed delta `bytes +0`. RSS (`os.wait4` `ru_maxrss`) **3.697 GiB** both
  arms, mid-run VmHWM prefix 3.697 ≤ wait4 (CF-KEYSTONERSS), physical sampler
  minimum 45.86 GiB, 0 watchdog trips, PLE term staged at 2.884 MiB. Row 3 on
  the **A770 host-fed arm**: two cold boots byte-identical (`9a7e2e77…9f`),
  **PASS**; the **arcwell arm's restart determinism is OWED** (arcwell is
  B60-only, no bit-readable card can run it) and is governed by design rule
  D3's load barrier. No tracked code; no module load/unload; the arcwell module
  was left loaded and carved; the coordinator's wake lock was left held and
  untouched; both cards left as found.

## Acceptance commit — LISBON, operator-approved re-gating (2026-10-01)

Appended 2026-10-01. The operator re-gated 0.5.3 to two readable rows — the
**host bank's served gate** and **NVMe-staged serving** — with **cold start
with nothing prebound** marked owed. Nothing here is a new measurement.

| row | value / status | marker | evidence class | on record |
|---|---|---|---|---|
| L-host-bank | the host bank's served gate | **PASS.** B60, one fresh process per arm, 257 greedy tokens each: mapping arm **48.55 s**, bank 46 GiB **38.18 s** (the operator's "38.2 s"); greedy text **byte-identical across arms**; gate met (−18 to −21 %). At 46 GiB: ~16,000 experts filled, 2,161–2,506 demand reads, 0 evictions; the fill read 59.8 GiB sequentially and added ~21 s to the load | `measured-here` | `measured-here` | `docs/campaigns/host-expert-bank.md` 2026-09-28 window table (campaign record); `docs/campaigns/README.md` host-expert-bank row |
| L-nvme | NVMe-staged serving | **PASS for rows 1–3 of this file.** One B60 two-arm window, ratio 86, plugin `ov-0049`, one artifact: arcwell arm cold TTFT **92.492 s** ≤ host-fed **99.679 s** ≤ `X = 139.5 s` (L1, L2 hold); prefetch depth **4 batches in flight**; `AW_IOC_STATS` arcwell delta `bytes +697,958,400`, `reads +852`, `segments +871`, `batches +4`, `via_host_bounce 0→0`, `max_inflight 220`; host-fed `bytes +0`; boot-child `wait4` `ru_maxrss` **3.697 GiB** both arms ≤ 32 GiB (L4), mid-run VmHWM prefix 3.697 ≤ wait4 (L5), physical sampler minimum 45.86 GiB, 0 watchdog trips; restart determinism on the **A770 host-fed arm**: two cold boots byte-identical `9a7e2e77…9f` (L6). The **arcwell arm's restart determinism stays OWED** (arcwell is B60-only; no bit-readable card can run it), governed by D3's load barrier | `measured-here` | `measured-here` | this file §§1–3 (rows filled 2026-09-24) |
| L-cold | cold start with nothing prebound | **owed.** The LISBON served cell is the **load-time pinned fill** at ratio 86: the 71 slots/layer are prebound at load, and the row's "cold" drops only the page cache. There is no miss-tier path — the routing warning horizon is **zero layers** (a layer's top-k ids are host-visible only at that layer's own MoE hook), so no fetch can be hidden ahead of the 1.125 ms/expert arcwell cost, and LISBON keeps the host hop as a miss tier (`docs/campaigns/nvme-direct-expert-tier.md`, verdict 2026-09-23; `docs/design-nvme-direct-expert-tier.md`). A true nothing-prebound cold start therefore has no served number. Standing beside it: the served artifact↔store key match exists only at depth 4, and the arcwell arm's byte-identity is judged by the load barrier, not cross-boot | `EMPTY` | `code` / `measured-here` | this file §Dependencies and §3; nvme-direct-expert-tier.md verdict and status log |

**Verdict.** LISBON reads **PASS on L-host-bank and L-nvme**, with the
**arcwell-arm determinism sub-row** and **cold start with nothing prebound**
(L-cold) owed with the reasons above.
