# benchmark-served-services — packaged 0.5.0 vs the qfndev tip, two deployed services

Recorded 2026-09-24 from one card window, then corrected the same day when the
operator's direct answers to the arm-design questions arrived. Protocol, exact
flags, raw rows, deltas and caveats for eight served sessions: the two deployed
services, each run as a direct invocation of a single `arcint` process (the
deployment units stayed disabled and stopped throughout), under the packaged
binary and under the qfndev tip, plus the configuration-delta arms the
operator ordered — and two earlier delta arms retained because they were
already measured (see §4b).

Every measured value below is `measured-here`. Dispositions of fact carry
`paper` / `code` / `measured-here`; the flag-applicability dispositions are
`code` (the engine's own guard) and `measured-here` (the exact stderr).

## 1. What was measured

Two services, named here by port and model because unit names stay in the
operator-local packet:

| service | port | model | class | card |
|---|---|---|---|---|
| coder | 8080 | `qwen3.6-27b-a3b-coder` | MoE (`qwen3_5_moe`, 184 experts pruned from 256, 40 layers, 30 GDN + 10 attn) | A770, PCI `8086:56a0`, OpenVINO `GPU.1` |
| dense agent | 8087 | `qwen3.8-27b` | dense (`qwen3_5`, 64 layers, + MTP head) | B60, PCI `8086:e211`, OpenVINO `GPU.0` |

DRM numbering is inverted against OpenVINO numbering; the cards are identified
by PCI id, never by number (`docs/sop-card-window.md` §2). Classes are `code`
(`src/core/model_registry.cpp`: `moe = true` / `moe = false`).

## 2. Protocol

- **One session per arm**: one fresh server process, sweeps `1 → 4096 →
  16384` through the prefix cache with three sequential `/v1/completions`
  requests.
- **Prompt**: the same fixed text corpus for every arm and both models,
  tokenised with the shared Qwen tokenizer (`tokenizer_hash
  87a7830d63fcf43b` on both registry entries) to **23,680 token ids**; the
  request body carries the `prompt` as an int array, `ids[:1]`, `ids[:4096]`,
  `ids[:16384]`. Id-file sha256 `31b351d6f95c5f08fc877e1750c3126610f16443ae568b930374deaad4c5e91e`.
- **Greedy settings**: `max_tokens=32`, `temperature=0`, `ignore_eos=true`,
  streamed.
- **Recorded per point**: `T_boot` (launch → `/props` 200, model-ready,
  separate from TTFT); TTFT (request → first streamed token); the server's own
  `slot 0: prefill`/`decode` lines; the prefix-cache hit the prefill line
  reports; `sha256(text)` of the 32 greedily served tokens.
- **Extension-vs-cold labelling**: the server's prefill line counts the whole
  prompt, including the cached part. The extension figures here are
  `prompt_tokens − cache_hit_tokens` over the printed prefill wall time.
- **Instrument hygiene** (`docs/sop-card-window.md`): a physical-host sampler
  started before the leg (2 s interval: `MemAvailable`, `Shmem`, ZFS ARC,
  leg `VmRSS`, `drm-resident-vram0`, `drm-resident-gtt`) with a `MemAvailable
  < 4 GiB` watchdog; page cache dropped between arms; a fresh process per arm;
  the container's sparse `/sys/devices/system/cpu/online` parse fed a
  contiguous CPU view through `unshare -rm` (the known workaround — it was
  used for every arm; no arm tripped `free(): invalid next size`).

## 3. The arms and the exact flags

**Binaries and plugins** (`measured-here`, by sha256):

| role | binary | plugin prefix | GPU plugin sha256 |
|---|---|---|---|
| P (packaged) | installed `arcint` 0.5.0+git20260916.75d28d4, `6b607acc84e4e344…f901af3c` | installed `+p17` | `4f881ffff6c514de…1661fc` |
| Q0 and deltas | qfndev tree at `459121a`, `5f7d2155f2a9806d…00484322d` | `ov-0049` (patches 0003–0049), build string `…-marfrit-p19` | `2d83e2a6d2aa7894…e4b855` |

The P→Q0 comparison therefore mixes **binary and plugin provenance**: it is
the packaged 0.5.0 binary on the installed `+p17` plugin against the qfndev
tip on `ov-0049`. No arm isolates one cause, and no disposition below
attributes the delta to either.

**Service C (coder, port 8080, A770 / `GPU.1`)**, the unit's flags exactly
after the `--cache-dir` path (operator-local, omitted):

```
--model /models/ov/qwen36-coder-b5-ov
--model-id qwen3.6-27b-a3b-coder --served-model-name qwen3.6-coder
--device GPU.1 --host 0.0.0.0 --port 8080
--n-ctx 98304 --prefix-cache-mib 2048
--queue-timeout 30 --paged-kv u8
```

- **C-P** = the above, packaged binary, installed plugin.
- **C-Q0** = the above, qfndev binary, `ov-0049` (via `LD_LIBRARY_PATH`). This
  is the operator-requested isolated binary/plugin arm.
- **C-D-ratio-99** = C-Q0 + `--offload-ratio 99 --moe-cpu-tier` (the
  operator's direct answer).

`code`: `--moe-cpu-tier` is refused without `--offload-ratio` — the engine's
own guard (`src/config.cpp`) reads *"--moe-cpu-tier needs --offload-ratio > 0:
with every expert resident there is nothing for the host tier to compute"*.
`measured-here`: the bare flag on the coder flags exits `2` with exactly that
message (captured in the packet), so the ratio the operator names is the only
way it becomes applicable. **R = 99** is the percentage OFF the card, so the resident share is
`ceil(184·(100−99)/100) = 2` experts per layer (`code`: `src/exec/fit.h`).

**Service D (dense agent, port 8087, B60 / `GPU.0`)**, the unit's flags exactly
after the `--cache-dir` path:

```
--model /models/ov/qwen38-b7c1-ov
--model-id qwen3.8-27b --served-model-name qwen3.8-agent
--device GPU.0 --host 0.0.0.0 --port 8087
--n-ctx 122880 --prefix-cache-mib 8192 --cache-host-mib 4096
--queue-timeout 30 --prefill-chunk 512
--paged-kv i8:u8 --mtp on --gate-pad 16 --repetition-penalty 1.0
```

- **D-P** = the above, packaged binary, installed plugin.
- **D-Q0** = the above, qfndev binary, `ov-0049`. This is the
  operator-requested isolated binary/plugin arm. The operator decided
  **keep the unit's `i8:u8`** — no `--paged-kv u8` is added.
- **D-mtp** = D-Q0 with `--mtp off` (DESIGN §7.0.2ag measures MTP as a loss on
  the dense agent: a 390 ms cycle against a 130 ms break-even at depth).

### 3a. Provenance of the arm design (two operator inputs)

The arm set was named in two steps, and the record keeps both because they
disagree on one arm:

- A reviewer-relayed ruling named coder **R = 75** and a dense
  **`--paged-kv u8` override**; those two arms (C-D-ratio-75, D-kv) were run.
  The relayed ruling also named **D-mtp** (`--mtp off`); it stays in the
  operator's set because the direct answers did not contradict it.
- The operator's direct answers to the arm-design questions then arrived:
  coder **ratio 99**, dense **keep the unit's `i8:u8`**, isolated arms **yes,
  both models**. These supersede the relayed ruling.

The record therefore executes R = 99 and keep_unit, and retains the two
superseded arms in §4b rather than dropping measured data.

## 4. Results — service C, coder, A770 (digests admissible)

All raw `slot 0` lines are in §7. Derived extension rate =
`(prompt − hit) / prefill_s`.

| arm | T_boot s | depth | prompt tok | cache-hit tok | ext tok | prefill s | ext prefill t/s | TTFT s | decode t/s | served digest |
|---|---|---|---|---|---|---|---|---|---|---|
| C-P | 79.334 | 1 | 1 | 0 | 1 | 0.30 | 3.3 | 0.570 | 36.0 | `0edc8dd7e703` |
| C-P | | 4096 | 4096 | 0 | 4096 | 2.96 | 1383.8 | 2.966 | 40.4 | `709556a93505` |
| C-P | | 16384 | 16384 | 3072 | 13312 | 10.99 | 1211.3 | 10.991 | 43.0 | `d8d45653ff32` |
| C-Q0 | 77.767 | 1 | 1 | 0 | 1 | 0.48 | 2.1 | 0.763 | 34.4 | `0edc8dd7e703` |
| C-Q0 | | 4096 | 4096 | 0 | 4096 | 2.97 | 1379.1 | 2.970 | 43.9 | `709556a93505` |
| C-Q0 | | 16384 | 16384 | 3072 | 13312 | 11.01 | 1209.1 | 11.010 | 42.4 | `d8d45653ff32` |
| C-D-ratio-99 | 580.378 | 1 | 1 | 0 | 1 | 0.60 | 1.7 | 0.809 | 1.1 | `f732f7fce8be` |
| C-D-ratio-99 | | 4096 | 4096 | 0 | 4096 | 374.59 | 10.9 | 374.591 | 14.8 | `2c2c31439df7` |
| C-D-ratio-99 | | 16384 | 16384 | 2048 | 14336 | 1187.47 | 12.1 | 1187.476 | 11.9 | `e458e88df4d8` |

**P→Q0 (fork + plugin), coder.** Extension prefill: 1383.8 → 1379.1 t/s
(−0.3 %) at 4096 and 1211.3 → 1209.1 t/s (−0.2 %) at 16384. Decode:
36.0 → 34.4 (−4.4 %) at 1, 40.4 → 43.9 (+8.7 %) at 4096, 43.0 → 42.4
(−1.4 %) at 16384 — the depth-1 figure is a first-token warm-up and within run
noise. `T_boot` 79.3 → 77.8 s. The served digests are equal at all three
depths.

**Q0→C-D-ratio-99 (configuration), coder.** Extension prefill:
1379.1 → 10.9 t/s (−99.2 %) at 4096 and 1209.1 → 12.1 t/s (−99.0 %) at
16384. Decode: 34.4 → 1.1 t/s (−96.8 %) at 1, 43.9 → 14.8 (−66.3 %) at 4096,
42.4 → 11.9 (−71.9 %) at 16384. `T_boot` 77.8 → 580.4 s (×7.5).
**The served digest differs from C-P/C-Q0 at every depth**
(`f732f7fce8be`, `2c2c31439df7`, `e458e88df4d8`): at R = 99 the CPU tier
carries essentially every routed expert, and the route is not
answer-equivalent to the resident route even at the 1-token start. This is
the static-partition/CPU-tier answer-dependence class already on this record
(`docs/campaigns/expert-hot-set-lru.md`, `docs/campaigns/sub4bit-vram-kernel.md`),
now shown at R = 99 to be present from the first token.

The 580.4 s `T_boot` needs its own sentence: the load line says *"language
model ready in 22.1 s (paged)"*, but `/props` became ready only at 580.4 s.
The gap is the distinct-token plateau probe, which runs forwards through the
CPU tier at R = 99; **`T_boot` here is probe-dominated, not
compile-dominated** (`measured-here`: the probe line follows the ready line in
`server.err`; `code`: the whole load sequence completes before
`http: listening`).

## 4b. The two superseded arms (retained, not part of the operator's set)

Both were run under the earlier reviewer-relayed ruling; the operator's direct
answers superseded them. They are kept because they were measured.

| arm | T_boot s | depth | hit | ext | prefill s | ext prefill t/s | decode t/s | served digest |
|---|---|---|---|---|---|---|---|---|
| C-D-ratio-75 | 563.347 | 1 / 4096 / 16384 | 0 / 0 / 2048 | 1 / 4096 / 14336 | 0.61 / 305.70 / 915.94 | 1.6 / 13.4 / 15.7 | 1.4 / 7.9 / 7.2 | `0edc8dd7e703` / `709556a93505` / `460dee9d6d7c` |
| D-kv (`--paged-kv u8`) | 102.243 | 1 / 4096 / 16384 | 0 / 0 / 3584 | 1 / 4096 / 12800 | 0.20 / 3.60 / 15.02 | 5.0 / 1137.8 / 852.2 | 21.9 / 23.9 / 20.2 | `851f3dfa4518` / `29344a729932` / `f580ae279d58` |

The contrast between the two coder ratios is the sharpest single result here:
at **R = 75** (46/184 resident) the digest matches C-P/C-Q0 at depths 1 and
4096 and diverges only at 16384; at **R = 99** (2/184 resident) it diverges at
every depth. So the CPU-tier route's answer-dependence is a function of the
resident share, not a single fixed property.

## 5. Results — service D, dense agent, B60 (digests NOT admissible)

| arm | T_boot s | depth | prompt tok | cache-hit tok | ext tok | prefill s | ext prefill t/s | TTFT s | decode t/s | served digest |
|---|---|---|---|---|---|---|---|---|---|---|
| D-P | 110.419 | 1 | 1 | 0 | 1 | 0.20 | 5.0 | 0.381 | 23.6 | `851f3dfa4518` |
| D-P | | 4096 | 4096 | 0 | 4096 | 3.59 | 1140.9 | 3.591 | 24.2 | `29344a729932` |
| D-P | | 16384 | 16384 | 3584 | 12800 | 14.98 | 854.5 | 14.979 | 20.3 | `f580ae279d58` |
| D-Q0 | 106.471 | 1 | 1 | 0 | 1 | 0.20 | 5.0 | 0.427 | 20.5 | `851f3dfa4518` |
| D-Q0 | | 4096 | 4096 | 0 | 4096 | 3.59 | 1140.9 | 3.595 | 24.6 | `29344a729932` |
| D-Q0 | | 16384 | 16384 | 3584 | 12800 | 15.03 | 851.6 | 15.035 | 20.2 | `f580ae279d58` |
| D-mtp | 100.248 | 1 | 1 | 0 | 1 | 0.22 | 4.5 | 0.541 | 19.5 | `851f3dfa4518` |
| D-mtp | | 4096 | 4096 | 0 | 4096 | 3.27 | 1252.6 | 3.273 | 21.4 | `29344a729932` |
| D-mtp | | 16384 | 16384 | 3584 | 12800 | 13.43 | 953.1 | 13.434 | 21.3 | `f580ae279d58` |

**P→Q0 (fork + plugin), dense.** Extension prefill: 1140.9 → 1140.9 t/s
(0 %) at 4096 and 854.5 → 851.6 t/s (−0.3 %) at 16384. Decode:
23.6 → 20.5 (−13.1 %) at 1, 24.2 → 24.6 (+1.7 %) at 4096, 20.3 → 20.2
(−0.5 %) at 16384. `T_boot` 110.4 → 106.5 s (−3.6 %).

**Q0→D-mtp (configuration), dense.** Extension prefill:
1140.9 → 1252.6 t/s (+9.8 %) at 4096 and 851.6 → 953.1 t/s (+11.9 %) at
16384. Decode: 20.5 → 19.5 (−4.9 %) at 1, 24.6 → 21.4 (−13.0 %) at 4096,
20.2 → 21.3 (+5.4 %) at 16384. `T_boot` 106.5 → 100.2 s. The served digests
are equal at all three depths. (The decode move is mixed and single-run; the
prefill move is the larger and consistent one.)

## 6. Digests, and what they are admissible for

- **A770 (coder arms): admissible.** C-P and C-Q0 are byte-identical at all
  three depths (and so is the superseded C-D-ratio-75 at depths 1 and 4096).
  **C-D-ratio-99 diverges at every depth** — the configuration delta moves the
  answer, so it is reported as a configuration result, never as a fork result.
- **B60 (dense arms): NOT admissible as byte-identity.** This card carries a
  known per-card determinism defect (`docs/campaigns/served-prefill-determinism.md`,
  DESIGN §7.0.2cb), so the dense arms' matching digests are recorded as
  *observed equal*, not as byte-identity evidence. No claim of byte-identity
  is made on the B60.

## 7. Raw server output (accepted numbers)

Command shape for every arm (one fresh process; `LD_LIBRARY_PATH` set to the
`ov-0049` runtime + TBB lib directories for the Q0/delta arms, unset for P;
run inside `unshare -rm` with the contiguous CPU view; flags as §3):

```
<binary> <unit flags> [<delta flags>]
```

Verbatim `slot 0` lines from each arm's `server.err` (stderr is where arcint
logs; the stdout file is empty). The C-D-ratio-99 block also reproduces its
three load lines, verbatim. The two superseded arms' lines follow the
operator-set arms.

**C-P**
```
lgc  slot 0: prefill     1 tok in  0.30 s (  3.3 t/s) | graph 0.30 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  0.89 s ( 36.0 t/s) | graph 0.63 s, embed 0.01 s, sample 0.00 s, emit 0.25 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in  2.96 s (1381.6 t/s) | cache snapshot 0.04 s | graph 2.90 s, embed 0.03 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  0.79 s ( 40.4 t/s) | graph 0.69 s, embed 0.01 s, sample 0.00 s, emit 0.09 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 10.99 s (1490.9 t/s) | cache hit 3072 tok (18.8%) | cache snapshot 0.03 s | graph 10.84 s, embed 0.07 s, pages 0.00 s, restore 0.04 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  0.74 s ( 43.0 t/s) | graph 0.72 s, embed 0.01 s, sample 0.00 s, emit 0.01 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
```

**C-Q0**
```
lgc  slot 0: prefill     1 tok in  0.48 s (  2.1 t/s) | graph 0.48 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  0.93 s ( 34.4 t/s) | graph 0.63 s, embed 0.01 s, sample 0.00 s, emit 0.29 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in  2.97 s (1379.5 t/s) | cache snapshot 0.03 s | graph 2.91 s, embed 0.03 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  0.73 s ( 43.9 t/s) | graph 0.69 s, embed 0.01 s, sample 0.00 s, emit 0.02 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 11.01 s (1488.4 t/s) | cache hit 3072 tok (18.8%) | cache snapshot 0.04 s | graph 10.84 s, embed 0.07 s, pages 0.00 s, restore 0.06 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  0.75 s ( 42.4 t/s) | graph 0.72 s, embed 0.01 s, sample 0.00 s, emit 0.01 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
```

**C-D-ratio-99**
```
lgc  load: MoE host compute tier enabled (threads=auto)
lgc  load: host tier: plugin reports a static residency partition; the prefix cache is allowed
lgc  load: expert slots host-side: 0.12 GiB (GTT, source: config)
lgc  slot 0: prefill     1 tok in  0.60 s (  1.7 t/s) | graph 0.60 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in 27.86 s (  1.1 t/s) | graph 27.58 s, embed 0.02 s, sample 0.01 s, emit 0.26 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in 374.59 s ( 10.9 t/s) | cache snapshot 0.04 s | graph 374.52 s, embed 0.03 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  2.17 s ( 14.8 t/s) | graph 2.14 s, embed 0.01 s, sample 0.01 s, emit 0.01 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 1187.47 s ( 13.8 t/s) | cache hit 2048 tok (12.5%) | cache snapshot 0.04 s | graph 1187.32 s, embed 0.08 s, pages 0.00 s, restore 0.04 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  2.68 s ( 11.9 t/s) | graph 2.16 s, embed 0.01 s, sample 0.01 s, emit 0.50 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
```

**D-P**
```
lgc  slot 0: prefill     1 tok in  0.20 s (  5.1 t/s) | graph 0.19 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.36 s ( 23.6 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.22 s, wait 0.00 s, other 0.04 s | draft accept 65.0% (13/20), propose 0.13 s, verify 0.97 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in  3.59 s (1141.0 t/s) | cache snapshot 0.06 s | graph 3.23 s, embed 0.02 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.28 s
lgc  slot 0: decode     32 tok in  1.32 s ( 24.2 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.05 s, wait 0.00 s, other 0.05 s | draft accept 65.0% (13/20), propose 0.15 s, verify 1.06 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 14.98 s (1093.9 t/s) | cache hit 3584 tok (21.9%) | cache snapshot 0.20 s | graph 13.42 s, embed 0.05 s, pages 0.00 s, restore 0.05 s, wait 0.00 s, other 1.26 s
lgc  slot 0: decode     32 tok in  1.58 s ( 20.3 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.01 s, wait 0.00 s, other 0.14 s | draft accept 88.2% (15/17), propose 0.21 s, verify 1.22 s, re-forward 0.00 s, rollback 0.00 s
```

**D-Q0**
```
lgc  slot 0: prefill     1 tok in  0.20 s (  4.9 t/s) | graph 0.20 s, embed 0.00 s, pages 0.00 s, restore 0.01 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.56 s ( 20.5 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.41 s, wait 0.00 s, other 0.04 s | draft accept 65.0% (13/20), propose 0.14 s, verify 0.98 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in  3.59 s (1139.6 t/s) | cache snapshot 0.05 s | graph 3.23 s, embed 0.02 s, pages 0.00 s, restore 0.01 s, wait 0.00 s, other 0.28 s
lgc  slot 0: decode     32 tok in  1.30 s ( 24.6 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.03 s, wait 0.00 s, other 0.05 s | draft accept 65.0% (13/20), propose 0.15 s, verify 1.07 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 15.03 s (1089.9 t/s) | cache hit 3584 tok (21.9%) | cache snapshot 0.21 s | graph 13.43 s, embed 0.05 s, pages 0.00 s, restore 0.08 s, wait 0.00 s, other 1.28 s
lgc  slot 0: decode     32 tok in  1.58 s ( 20.2 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.01 s, wait 0.00 s, other 0.14 s | draft accept 88.2% (15/17), propose 0.21 s, verify 1.23 s, re-forward 0.00 s, rollback 0.00 s
```

**D-mtp**
```
lgc  slot 0: prefill     1 tok in  0.22 s (  4.6 t/s) | graph 0.21 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.64 s ( 19.5 t/s) | graph 1.31 s, embed 0.01 s, sample 0.00 s, emit 0.33 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in  3.27 s (1251.9 t/s) | cache snapshot 0.02 s | graph 3.22 s, embed 0.02 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.50 s ( 21.4 t/s) | graph 1.39 s, embed 0.01 s, sample 0.00 s, emit 0.09 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 13.43 s (1219.7 t/s) | cache hit 3584 tok (21.9%) | cache snapshot 0.02 s | graph 13.33 s, embed 0.05 s, pages 0.00 s, restore 0.03 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.50 s ( 21.3 t/s) | graph 1.49 s, embed 0.00 s, sample 0.00 s, emit 0.01 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
```

**C-D-ratio-75 (superseded, §4b)**
```
lgc  slot 0: prefill     1 tok in  0.61 s (  1.6 t/s) | graph 0.61 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in 23.54 s (  1.4 t/s) | graph 23.28 s, embed 0.01 s, sample 0.01 s, emit 0.24 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in 305.70 s ( 13.4 t/s) | cache snapshot 0.04 s | graph 305.63 s, embed 0.03 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  4.04 s (  7.9 t/s) | graph 4.00 s, embed 0.01 s, sample 0.01 s, emit 0.03 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 915.94 s ( 17.9 t/s) | cache hit 2048 tok (12.5%) | cache snapshot 0.04 s | graph 915.79 s, embed 0.08 s, pages 0.00 s, restore 0.04 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  4.43 s (  7.2 t/s) | graph 4.00 s, embed 0.01 s, sample 0.01 s, emit 0.42 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
```

**D-kv (superseded, §4b)**
```
lgc  slot 0: prefill     1 tok in  0.20 s (  4.9 t/s) | graph 0.20 s, embed 0.00 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.46 s ( 21.9 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.32 s, wait 0.00 s, other 0.04 s | draft accept 65.0% (13/20), propose 0.13 s, verify 0.97 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill  4096 tok in  3.60 s (1137.2 t/s) | cache snapshot 0.06 s | graph 3.23 s, embed 0.02 s, pages 0.00 s, restore 0.01 s, wait 0.00 s, other 0.29 s
lgc  slot 0: decode     32 tok in  1.34 s ( 23.9 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.06 s, wait 0.00 s, other 0.05 s | draft accept 65.0% (13/20), propose 0.15 s, verify 1.07 s, re-forward 0.00 s, rollback 0.00 s
lgc  slot 0: prefill 16384 tok in 15.02 s (1091.1 t/s) | cache hit 3584 tok (21.9%) | cache snapshot 0.20 s | graph 13.41 s, embed 0.05 s, pages 0.00 s, restore 0.08 s, wait 0.00 s, other 1.28 s
lgc  slot 0: decode     32 tok in  1.58 s ( 20.2 t/s) | graph 0.00 s, embed 0.00 s, sample 0.00 s, emit 0.01 s, wait 0.00 s, other 0.14 s | draft accept 88.2% (15/17), propose 0.21 s, verify 1.23 s, re-forward 0.00 s, rollback 0.00 s
```

## 8. Caveats

1. **Plugin provenance delta.** P→Q0 mixes binary and plugin: packaged
   0.5.0 on installed `+p17` against the qfndev tip on `ov-0049`
   (`…-marfrit-p19`). No arm isolates one cause, so no disposition attributes
   the delta to either binary or plugin. `code` + `measured-here`.
2. **Extension-vs-cold labelling.** At depth 4096 the cache hit is **0 on
   every arm**: the 1-token start is below the 32-token KV block and below the
   snapshot grid, so the depth-4096 row is a **cold prefill**, not an
   extension. At 16384 the hit is 3584 on all four dense arms and 3072 on
   C-P/C-Q0; the CPU-tier arms (C-D-ratio-75/99) are the exception at 2048 —
   their auto-fit chunk grid is 2048, not the resident route's 1024, so the
   extension base differs (§8.4). `measured-here`.
3. **B60 determinism.** Dense-arm digests are **not** byte-identity evidence;
   the B60 carries a per-card determinism defect. The observed equality is
   recorded, no claim follows from it. `measured-here` + the campaign record.
4. **Coder chunk grid differs between Q0 and the CPU-tier arms.** The coder
   unit does not set `--prefill-chunk`; the auto-fit chose chunk 1024 for
   C-P/C-Q0 (snapshot grid 1024) but 2048 for C-D-ratio-75/99 (snapshot grid
   2048), so the 16384 extension base is 3072 vs 2048. A configuration
   artifact of the delta, not a fork difference. `measured-here`.
5. **Single run per arm.** Each cell is one session; depth-1 decode is a
   first-token warm-up and carries more variance than the deeper rows.
   `measured-here`.
6. **The paged graph's blob cache is off.** The engine sets
   `ov::cache_dir("")` on the paged path, so every boot pays the compile; no
   arm benefited from a warm compile cache. `code`.
7. **CPU-view workaround.** The container's sparse `/sys/devices/system/cpu/online`
   trips the OpenVINO CPU-plugin parse; every arm ran under `unshare -rm`
   with a contiguous view. No arm hit `free(): invalid next size`.
   `measured-here`.

## 9. What failed, and what was missing

Nothing failed and no context point is missing: all eight arms returned `rc=0`,
reached `/props`, completed all three depths, and left no `arcint` process.
The physical samplers recorded **0 watchdog trips** (minimum
`MemAvailable` ≈ 28.8 GiB on the main window, ≈ 43.0 GiB on the R = 99 window),
far above the 4 GiB threshold. No arm was refused. The only point that needs a
label rather than a number is the depth-4096 row (cold, not an extension;
§8.2).

## 10. The packet

The operator-local packet (host, unit, paths, raw JSON, sampler logs, the
harness) is in the git-ignored `docs/benchmark-served-services.local.md` and on
the card host under a persistent `bench-out` directory, not `/tmp`.
