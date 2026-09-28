# ple-disk-backend — stage the n-gram table per chunk from disk instead of pinning 26.82 GiB

## The defect, as measured

`bind_ngram_ports` (`src/exec/backend_ov.cpp`:8076–8165) binds the whole
Flash-Next n-gram table into **USM_HOST_BUFFER** tensors: one `ov::RemoteTensor`
per `ngram_table.K` port, `std::memcpy` of the entire payload, and the tensors
are retained in `ngram_table_tensors_` for the life of the process. The table is
28,800,138,240 B (26.82 GiB) over seven chunks of 47,718,400 rows x 90 B
(`docs/window-050.md`:1331).

Measured on the served path, both cards, hash ordinal 0 (`docs/window-050.md`
§4.8 run 2, R3, 2026-09-13):

    R3 | A770 / B60 | table bound in 33.03 s / 29.49 s, 26.82 GiB USM host,
         `usm_device` unchanged; peak host 27.67 GiB; no OOM | yes

Run 1's copy times were 3.6/3.6/3.5/4.3/3.7/3.5/2.6 s = 28.33 s, with the shard
pages evicting under the 28.78 GiB host peak. The user-stated figure for the
served loader is 17.2 s and 26.82 GiB; the window record's own reading is
29.49–33.03 s. Both agree on the resident bytes. **EVIDENCE CLASS:
`measured-here`.**

The pin is a CHOICE, not a constraint. The reference ships a disk backend as its
default:

- `~/src/FreeToken-ref/python/freetoken/engine/config.py`:32 —
  `ple_backend: str = "disk"`. The `"pinned"` backend (preload into page-locked
  host RAM) is the alternative, not the default. **EVIDENCE CLASS: `code`.**
- `~/src/FreeToken-ref/python/freetoken/models/qwen4_exp/ple_disk.py`
  (`DiskRowTable`) allocates **bounded** pinned staging up front —
  `alloc_pinned_tensor(max_graph_rows * token_bytes)` for decode
  (`max_graph_rows=256`) and `alloc_pinned_tensor(max_extend_tokens *
  token_bytes)` for prefill (`max_extend_tokens=8192`) — and `fill()`/`lookup()`
  stage only the rows a forward names, per fill. `source_from_safetensors`
  maps the checkpoint shards **in place, no copy** (`PleRowSource`). The
  io_uring switch (`FREETOKEN_PLE_IO_URING`, default `"1"`) changes the I/O
  backend, not the residency model. **EVIDENCE CLASS: `code`.**

For the served geometry (`num_ngram_heads = (ngram_size-1)*heads_per_ngram =
16`, 90 B/row), the reference's staging is `T x 16 x 90` B a forward — 737 KiB
at T=512, or 11.8 MB at its own `max_extend_tokens=8192` prefill bound — against
26.82 GiB pinned. The pin consumes ~26.82 of the 48 GiB host's ~44 usable GiB
(`docs/design-qwen-flash-next.md` WP6b), the same RAM `FIX E`'s host-resident
expert pool wants.

**Note on the figure `T x 7 x 90`.** The operator's working figure used H=7;
the served artifact's own id ports are `[1, T, 16]`
(`docs/window-050.md` §4.8; `src/exec/ngram_row_ids.h`:59
`(ngram_size - 1) * heads_per_ngram`). The formula is `T x H x row_bytes`; at
T=512 with the served H=16 it is **720 KiB**. The design note states both.

## Known against hypothesised

**Known (`code`, read from this tree):**

- The port contract already generalises to a chunked table. `ngram_ports.h`'s
  `PortPlan` reads the partition off the compiled model's port shapes; the ids
  are computed **host-side** and carried as ports (`ngram_chunk_ids` = which
  chunk, `ngram_local_ids` = the row within it). Nothing in the graph does
  integer arithmetic on an id (`ngram_ports.h`:11–17; `serving_shape.py`:
  `ngram_chunked_gather`).
- `PLETableBackend.lookup` is implemented (`src/exec/ngram_table.h`,
  `NGramLookup`), with the reference's row-id hash (`ngram_row_ids.h`).
- A **host-side** gather-with-dequant exists and is byte-exact tested against
  the scalar reference (`src/exec/ngram_gather.h`,
  `tests/test_ngram_gather.cpp`: 9/9 cases, AVX2 + scalar, 2026-09-10).
- Disk admission with a host-RAM fit refusal exists
  (`src/core/artifact.cpp::admit_ngram_table_from_disk`,
  `src/exec/fit.h::host_ram_fit_must_refuse`).
- The mmap path (`NGramLookup::mmap_table`) is genuinely lazy: `MADV_RANDOM`,
  rows paged in on demand. It is the `NGramLookup` path; `bind_ngram_ports` is
  the one that does the eager full copy.

**Hypothesised (unmeasured):**

- Per-forward `pread` of `T x H` rows (up to 720 KiB at T=512) is cheap enough
  on the decode/prefill path. The table's rows are random-access; the mmap
  path already reads exactly these rows. Whether plain synchronous `pread`
  (arcint scope) is fast enough without the reference's io_uring is unmeasured.
- Bounded staging does not perturb numerics: same bytes, same gather order,
  same IQ4_NL decode, no accumulation. To be proven by the gate below, not
  assumed.

## Gate

**Gate (copied from the operator's charge, 2026-09-22):** one served window,
same prompt and greedy settings, through the staged path vs the pinned path,
**byte-identical answer**, plus:

- the **freed host RAM measured** — the 26.82 GiB PLE term leaves the ledger;
- the load-time `memcpy` / 17.2 s gone.

**A numeric difference is a FINDING and blocks the change.** This touches
numerics and cannot be a silent optimisation. **EVIDENCE CLASS: `measured-here`
once run; the gate itself is a measurement that can fail.**

The mmap `NGramLookup` path is the byte-exact oracle for the device-free half;
`tests/test_ngram_gather.cpp` stays unchanged and unwidened.

## Entry criteria

1. The served IR declares a **staging** port: one `ngram_table.0` of
   `[staging_rows, row_bytes]` rows, `staging_rows >= T_max x H`, instead of
   seven full-chunk ports. (Same `ngram_table_ports(staging_rows, row_bytes)`
   helper; the caller passes the staging bound, not the table's row count.)
2. A real Flash-Next GGUF carrying `per_layer_token_embd.weight` (IQ4_NL) and
   the artifact's n-gram config, as `docs/window-050.md` §4.9 already admits.
3. A card window coordinated with the session holding the A770/GPU.1, per
   `docs/sop-card-window.md` (do not run two `arcint` legs at once; identify
   cards by PCI id).

## Scope — in / out

**In:** the host-side staging mechanism (row source, forward staging plan,
`pread` of named rows, capacity and out-of-range refusal); the fit arithmetic
term for staging bytes; the emitter port-count change; the served-window
acceptance; a DESIGN `§7.0.2` record and a CHANGELOG line on closure.

**Out (named so they are not silently implied):**

- **io_uring** (`FREETOKEN_PLE_IO_URING`). arcint scope is plain `pread`;
  the reference's io_uring is an I/O-backend optimization, not part of the
  residency model, and can be a follow-up campaign if the plain path is too
  slow.
- **Dedup / an LRU of staged rows.** The first cut stages in token order, no
  dedup (the reference's own `fill` order). A repeated row is read twice; that
  is correct and measurable, and dedup is a later lever.
- **Multiple PLE layers.** The served IR carries one PLE layer with ordinal 0;
  `bind_ngram_ports` already refuses a config with more (`REVIEW F3`).
- **The `NGramLookup` mmap path.** It is already lazy; this campaign does not
  change it, and it stays the oracle.

## Where it lives

| file | role |
|---|---|
| `src/exec/ngram_staging.h/.cpp` | the staging mechanism: row source, forward staging, `pread` |
| `src/exec/ngram_ports.h` | the port contract; a note that a staging port is a small chunk |
| `src/exec/fit.h` | `ngram_staging_bytes` and the pinned-vs-staged distinction |
| `src/exec/backend_ov.cpp` | `bind_ngram_ports` staging branch (card side) |
| `tools/q4e/serving_shape.py` | emit `ngram_table.0` at the staging bound |
| `tests/test_ngram_staging.cpp` | the device-free red-first cells |
| `tests/test_ngram_gather.cpp` | the byte-exact oracle — unchanged |

## Pipeline for this campaign

recon (this document + the cited code) → design note
`docs/design-ple-disk-backend.md` → red-first implementation (device-free) →
one coordinated card window → review before commit → DESIGN `§7.0.2` record and
CHANGELOG line on closure.

## Invariants

DESIGN §3.4 (history-independent greedy output) and §3.8; the §5 ladder;
`CLAUDE.md`'s measurement discipline. A campaign that would trade one for a
number does not close; it records the trade as a finding and stops.

## Status

- **2026-09-22** — campaign opened. Recon read: `CLAUDE.md`, `AGENTS.md`,
  `docs/campaigns/README.md`, `docs/research-freetoken-code.md`,
  `docs/design-qwen-flash-next.md` FIX D, `docs/design-routing-aware-expert-
  execution.md` §5.1, DESIGN §7.0.2 records on the PLE, the arcint mechanism
  (`ngram_ports.h`, `ngram_gather.h`, `ngram_table.{h,cpp}`,
  `admit_ngram_table_from_disk`, `bind_ngram_ports`), and the reference source
  (`ple_disk.py`, `config.py`:32). Design note written; contradicting claims
  revised in place, dated, with evidence classes. Device-free staging mechanism
  and its red-first cells in progress.
- **2026-09-22 (implementation)** — `src/exec/ngram_staging.h` (header-only:
  `StagingGeometry`, `check_staging_geometry`, `plan_staging_fill`,
  `pread_staging_rows`, `stage_from_file`) and `tests/test_ngram_staging.cpp`
  (6 cells) landed; `ngram_staging_bytes` added to `src/exec/fit.h`; the
  CMakeLists test list registers the new file.

  Green (this session, device-free build, host without AVX2):

      ./build/arcint-test ngram_staging   ->   6 cases run, 0 failed, 0 skipped
      ./build/arcint-test ngram           ->  44 cases run, 0 failed, 3 skipped
      ./build/arcint-test fit             ->  50 cases run, 0 failed, 0 skipped
      ./build/arcint-test                 -> 581 cases run, 0 failed, 5 skipped

  Red-first deletion proof (three mutations, each reverted; raw output
  captured in the session): breaking the slot order (`local[i] = 0`) fails
  `ngram_staging_places_row_i_at_the_row_the_id_names` and
  `ngram_staging_output_matches_the_pinned_gather_byte_exact`.

  **Red-first, MEASURED (2026-09-23, `measured-here`):** the file contributes
  **6 cells** to the `arcint-test` ladder, and every refusal is pinned by
  mutating the mechanism and watching that cell fail --
    * capacity refusal dropped (`if (global.size() > g.staging_rows)` ->
      `if (false)`), build rc = 0 -> FAIL
      `ngram_staging_refuses_a_forward_that_overruns_the_port` (run exit 1);
    * out-of-range refusal dropped, build rc = 0 -> FAIL
      `ngram_staging_refuses_an_out_of_range_table_id` (run exit 1);
    * the row placement shifted by one (`rows[i] = id` -> `id + 1`),
      build rc = 0 -> FAIL at `tests/test_ngram_staging.cpp:148` (run exit 134,
      the harness aborts on the first failure).
  Restored, the ladder reads **581 cases run, 0 failed, 5 skipped**, exit 0, and
  the restored header is byte-identical to the one committed. The raw mutation
  output is kept on the operator-local session evidence path (not in this
  PUBLIC repository). NOTE, stated rather than
  smoothed: the harness reports the WHOLE ladder, so "the file is 6/0/0" is not
  a run it can produce -- **6** is the file's own cell count and **581/0/5** is
  the ladder's result.

  The byte-exact cell diffs the staged gather against the pinned
gather (`gather_dequant` over the whole table) with `memcmp`, so it is the
campaign's numeric gate in device-free form.

  **Still open:** the emitter port-count change (`ngram_table_ports` at the
  staging bound), the `backend_ov.cpp::bind_ngram_ports` staging branch, and
  the one coordinated card window. The mechanism is implemented and proven
  device-free; the served acceptance needs a card and must not run beside
  another `arcint` leg.

- 2026-09-23 — **the wiring landed: the emitter can declare a staging window, and
  the runtime fills it per forward.** [code; `measured-here` for the cells]
  - **Emitter** (`tools/q4e/serving_shape.py`): `build_serving_shape_ir(...,
    ngram_staging_rows=N)` passes the STAGING BOUND to `ngram_table_ports`
    instead of the table's row count, so the IR declares ONE `ngram_table.0`
    port of `[N, 90]` — deliberately SMALLER than the source tensor, which is
    exactly how `bind_ngram_ports` recognises staging. The report carries
    `ngram_staging_rows` beside the unchanged `ngram_table_rows`. Two cells in
    `tests/python/test_serving_shape.py`: the staging shape, and the regression
    half (no bound -> the ports still cover the whole table). **RED-FIRST
    MEASURED:** against the unpatched emitter both cells FAIL (`2 failed`); with
    it they PASS (`2 passed`); the full suite is **36 passed, 1 skipped**.
  - **Runtime** (`src/exec/backend_ov.cpp`): `bind_ngram_ports` recognises a
    single port whose row count is below the source's, validates it with
    `check_staging_geometry`, opens the GGUF path for `pread`, allocates ONE
    `[S, row_bytes]` USM-host staging tensor, and SKIPS the full-table copy;
    `feed_ngram_ports` then stages exactly the rows this forward names
    (`ngram::stage_from_file`, slot `i` = the `i`-th named row) and feeds the
    slot ids with chunk id 0. **Compile-verified with the production flags:**
    `-fsyntax-only` against the OpenVINO toolchain returns **rc = 0 with no
    diagnostics** under `-Wall -Wextra -Wpedantic`.
  - **Still open, unchanged: the served card window.** Byte-identical answer
    staged vs pinned, the freed host RAM measured (the 26.82 GiB term off the
    ledger), and the load-time copy gone. It needs a card and must not run
    beside another `arcint` leg. The acceptance is a numeric gate: a difference
    BLOCKS the change. Until that window runs, the pin is still what the served
    path does — the staging path is implemented and proven device-free, not yet
    served.

  **[DATED IN PLACE 2026-09-23: the served window RAN and the gate PASSED —
  see the entry below. The paragraph above stands as the pre-window state.]**

- **2026-09-23 (served acceptance — GATE PASSES, `measured-here`).** Scope
  approved as depth-4 (the n-gram mechanism is depth-independent). One card,
  one binary, one fresh process per arm.

  **The confound found first, and why a twin was required.** The obvious pinned
  arm is the stale pinned artifact `qwen38-flash-next-d4-ov` (window-050 §4.9, tree `wt+6743ffb`,
  `lm_xml_sha 2910a860bf9dc6bb`). Diffing its `config.json` against the staged
  artifact's showed two extra keys in the staged one:
  `"output_gate_type": "sigmoid"` and `"gdn_key_head_map": "tiled"` — the
  corrected fill (DESIGN §7.0.2bz), absent from the stale 2026-09-13 artifact.
  A staged-vs-stale comparison would have measured the backbone, not the PLE.
  So a **pinned twin** was exported from the SAME tree `0d9bb3d`, with the SAME
  corrected fill, differing ONLY in the n-gram ports:

      artifact alias `qwen38-flash-next-d4p-ov`  --ngram-staging-rows omitted
        ports ['ngram_table.0'..'ngram_table.6'] (47718400 rows each; last 33691136)
        lm_xml_sha 9a65886dc653e0ad, lm_bin_bytes 9270599581
        export: build 595.4 s + save 64.4 s + hash 15.0 s, peak host 20.11 GiB
      artifact alias `qwen38-flash-next-d4s-ov`  --ngram-staging-rows 33600
        ports [['ngram_table.0', 33600, 3024000]]
        lm_xml_sha 823997733f0b4b07, lm_bin_bytes 9270599557

  Their `config.json` files are `CONFIG_IDENTICAL`; the two differ only in the
  n-gram port partition. (The stale d4-ov pinned arm was also run as a
  cross-check and returned `c983da7e…`; that is the pre-correction answer and is
  **discarded for gate purposes**, kept only to show the fill is what moved.)

  **Setup.** The serving binary built from the scratch qfndev tree (`wt-ple`) sha256
  `a6dac5b57cc5fa40` (carries the staging branch; both arms on this ONE binary).
  The measurement plugin prefix sha256 `b2754b8fe8a9b89b`.
  Card GPU.1 = PCI `8086:56a0` (Arc A770; DRM `card0`), identified by PCI id.
  Flags identical for both arms: `--device GPU.1 --ngram-gguf <the checkpoint's
  shard 2> --prefill-chunk
  512 --n-ctx 8192 --parallel 1 --offload-ratio 99 --moe-cpu-tier --paged-kv u8
  --no-logits-slice`. Prompt: the pinned capture (first 256 ids),
  `max_tokens=32`, `temperature=0`, `ignore_eos=true`.

  **Pinned-twin arm** (window log directory `ple-staging/pinned_twin/`), raw:

      lgc  load: ngram table bound: 7 port(s), 320001536 rows x 90 B = 26.82 GiB
           of USM host memory from per_layer_token_embd.weight in 37.5 s;
           id ports declared, conv_mask declared; hash ordinal 0
      SHA256_TEXT=d7f998cd8bff32d71a1b1b9153ad8874f5bff61d684c6fad268c3b1325ea5b8f
      peak rss_kb=20753068 peak vmhwm_kb=20753160 min memavail_kb=31780445
      physical-host window: min MemAvailable 10,368,488 KB = 9.89 GiB;
           ZFS ARC fell 38.05 -> 10.15 GiB

  **Staged arm** (window log directory `ple-staging/staged2/`), raw:

      lgc  load: ngram table STAGED: 1 port(s) of 33600 rows x 90 B = 2.884 MiB
           of USM host staging from per_layer_token_embd.weight (the 320001536-row
           table stays on disk, read per forward); id ports declared, conv_mask
           declared; hash ordinal 0
      SHA256_TEXT=d7f998cd8bff32d71a1b1b9153ad8874f5bff61d684c6fad268c3b1325ea5b8f
      peak rss_kb=5173232 peak vmhwm_kb=5178752 min memavail_kb=44706027
      physical-host window: min MemAvailable 34,503,944 KB = 32.91 GiB;
           ZFS ARC 25.28 -> 21.87 GiB

  **The gate, row by row.**

  | row | pinned twin | staged | verdict |
  |---|---|---|---|
  | answer digest | `d7f998cd…2ea5b8f` | `d7f998cd…2ea5b8f` | **PASS, byte-identical** |
  | n-gram resident | 26.82 GiB USM host, 37.5 s copy | 2.884 MiB staging, no copy | **the 26.82 GiB term is off the ledger** |
  | container VmRSS peak | 19.79 GiB (20,753,068 KB) | 4.93 GiB (5,173,232 KB) | Δ **14.86 GiB** |
  | physical MemAvailable min | 9.89 GiB | 32.91 GiB | Δ **23.02 GiB** |

  **[DATED IN PLACE 2026-09-23: the staged VmRSS peak above was first written
  4.85 GiB. The sampler's own `5,173,232 KB` is 4.93 GiB (÷1024²); the Δ
  14.86 GiB was already the correct 19.79 − 4.93, so only the absolute staged
  figure was mistyped. Corrected when the raw `sampler.log` was re-read during
  review.]**

  The numeric difference is **zero** on the gate; the staging change is invisible
  under DESIGN §3.4 on this window. A difference would have been the finding; it
  is not there.

  **The one confound, stated rather than smoothed.** The FIRST staged run (before
  the twin existed) returned `d7f998cd…` against the stale pinned artifact's
  `c983da7e…`, i.e. a numeric difference. Reading `config.json` showed the cause
  was the corrected backbone fill in the newly-exported artifact, not staging:
  the corrected twin reproduces `d7f998cd…` exactly, so the difference belonged
  to `output_gate_type`/`gdn_key_head_map`, not to the PLE. The stale arm is
  retained in the window log directory for the record.

  **Perf observation, not a defect:** the first staged arm's plateau probe/load
  was cold-table-bound (props 370 s, request 56.5 s); re-run with the table pages
  warm it was props 45 s, request 25.6 s, against the pinned twin's 155 s / 33.2 s.
  The staged load is now FASTER than pinned. Dedup and io_uring remain out of
  scope.

  **Local, uncommitted, scratch-only:** the scratch tree's
  `src/core/model_registry.cpp`
  gained two entries (`qwen38-flash-next-d4s-ov`, `qwen38-flash-next-d4p-ov`) so
  the served binary admits the two new basenames; the pinned `qwen38-flash-next-d4-ov`
  entry and artifact are untouched. **Not committed** (operator-local window
  scaffolding).

  **Remaining open:** LISBON-001's RSS-bounded-through-boot cell is where the
  freed 26.82 GiB is read; `ngram_table_rows` in the export manifest still
  reports the source table's row count beside the staging port (by design).
  io_uring, dedup, and multi-PLE-layer IRs stay out of scope.

- 2026-09-23 — **the export entry point exposes the staging bound.**
  [code] `tools/export_serving_artifact.py` gained `--ngram-staging-rows N`, passed
  into `build_serving_shape_ir` at the export call site, so an artifact can be
  written with the staging window declared. One avenue is CLOSED and worth
  recording: the existing full-depth artifact **cannot** be turned into a staging
  IR by editing its XML, because the artifact declares **three** chunked
  `ngram_table.K` ports (the whole table under the per-object cap) while staging
  needs **one** — the port COUNT changes, so the graph changes, so an export is
  required. The cheap acceptance path is therefore a TRUNCATED export
  (`--layers 4`), which is legitimate for this gate: the n-gram table is a
  property of the model, not of the depth, so a depth-4 staging artifact exercises
  the same mechanism and the same freed 26.82 GiB term.

- 2026-09-27 — **full depth: 2.2-2.8x faster, the gate FAILED (cause
  open), and a precision defect found in the row decode shared by both
  paths** (DESIGN §7.0.2cz; `measured-here` unless a class is named).
  - Setup: `d48s` (tree 0e0ef26, `--ngram-staging-rows 33600`) and `d48n`,
    each on the A770 at ratio 75 + tier + dispatch, one binary, the 0068
    prefix, one run each.
  - Speed, `d48s` against `d48n`:
    - d1 decode 16.33 s against 36.72 s;
    - d512 prefill 30.65 s against 75.72 s, decode 9.56 s against 26.66 s.
  - Quality: the greedy digests differ. On window 0 of the f32 capture, the
    mean KL below 2051 is 0.3386 staged against 0.2840 pinned.
  - The graphs differ only in the gather, and the weight files only in 24
    bytes. One 4-token forward already differs on rows 0–3. Re-binding the
    staging tensor every forward changes nothing.
  - **The divergence's cause is open.** The next measurement is the
    gathered bytes of both arms, compared byte for byte.
  - Separately, a card probe of the emitter's own decode
    (`ngram_dequant_iq4nl`, stock OpenVINO via Python) found 1.41 %
    relative error on the A770 at the served f16 execution:
    - the same with 1 port and with 3, so it does not by itself separate
      the arms;
    - exact at f32.
    The scale's bit pattern `lo + 256*hi` exceeds f16's exact range.
  - Fixed in the emitter: the fields come from the two bytes apart. The
    card then returns the exact decode rounded to f16, bit for bit.
  - A new cell asserts every intermediate is f16-exact; it is red on the old
    decode.
  - Open: both twins re-exported with the new decode (`d48p2`, `d48s2`),
    their digest gate, their KLD and the gathered-bytes comparison.
  - [Correction, 2026-09-27] The 2026-09-23 entry above says the full-depth
    artifact declares three chunked ports. `d48n`'s `serving-shape.json`
    lists seven (`ngram_table.0`–`6`).

- 2026-09-27 (afternoon) — **the gate's failure is of the order of a neutral
  change; its cause is not found** (DESIGN §7.0.2cz, "Later the same day";
  `measured-here` unless marked).
  - The new-decode twins `d48p2` / `d48s2` agree at d1 (`82df3c78…`) and
    differ at d512. On window 0 they read 0.2757 / 0.4657 / 0.8310 against
    0.3273 / 0.5273 / 0.7864.
  - `d48s2` decodes d512 in 9.90 s against 23.13 s. The page cache was not
    dropped.
  - The gathered rows are right in both paths:
    - staged: in the served process's 512-token prefill;
    - pinned: its gather subgraph, cut out and run on stock OpenVINO with
      the real 26.82 GiB table.
  - Staging into a fresh plain tensor reads the same KL as the shared USM
    buffer.
  - The runtime graphs match in type, primitive, precision and layout
    outside the gather. One FullyConnected trades places with two
    independent neighbours.
  - On the pinned path, changing only the prefill chunk (512 to 256) moves
    the KL by +0.043 below 2051, +0.019 above and -0.028 in argmax. The
    staged gap is +0.052 / +0.062 / -0.045.
  - Reading (not measured): at depth 48 any f16-rounding change is expected
    to change greedy text, so the byte-identical gate cannot separate one
    from a defect.
  - Operator's decisions:
    - replacing the gate, for example with a KL criterion sized by several
      neutral perturbations;
    - serving the staged artifact.
  - Owed: the decoded embedding of both served processes compared byte for
    byte, a wider spread, and a cold-cache speed.

- 2026-09-27 (evening) — **the digest gate passes on the old-decode twins
  with dispatch off (the 2026-09-23 gate's route); with dispatch on the
  twins part at layer 1's MoE** (DESIGN §7.0.2cz, "Evening").
  - [measured-here] With `--moe-per-expert-dispatch` off (ratio 75 + CPU
    tier, u8 KV), the twins' 4-token logits are bit-identical, and the
    digests agree: d1 `8ec20481a399…`, d512 `eb89a675a8d2…`. One run each.
  - [measured-here] With dispatch on:
    - in the unperturbed program the twins agree at `layer0/out` and differ
      from `layer2/out` on;
    - in a probed program whose logits the probes moved, they agree through
      the PLE block and layer 1's mixer, and part at `layer1/out`, after its
      MoE.
    - The divergence is fixed per artifact and program, not run-to-run
      noise.
  - [code + measured-here on plugin 0047, before 0056, not re-measured]
    Under dispatch, residency decides whether an expert runs on the GPU
    kernel or the host tier, which were not bit-identical (§7.0.2ce/cf).
    Whether the twins' resident sets differ is not measured. Residency is
    one candidate, a per-program kernel choice another.
  - Reading (not measured): on the old-decode twins, on the forwards
    measured, the staging change is byte-exact on the tier-only route.
  - Owed:
    - the new-decode twins on the tier-only route, before the full-depth
      gate is called passed;
    - the staged tier-only speed from the same volume;
    - whether, and why, the dispatch route's layer-1 MoE differs.

- 2026-09-27 (late evening) — **the new-decode twins agree with dispatch off
  (ratio 75); on the dispatch route their divergence is measured, on a 4-token
  forward, to come from the static partition's `.bin`-offset key** (DESIGN
  §7.0.2cz, "Late evening").
  - [measured-here; A770 at 2000 MHz, `--offload-ratio 75 --moe-cpu-tier`,
    u8 KV; the gate was defined at ratio 99] `d48p2` and `d48s2` with
    dispatch off, both from ext4 (`d48p2` first, page cache not dropped),
    one run each:
    - digests: d1 `82df3c7881cf…`, d512 `c1d2077f1a84…` in both;
    - staged speed: d1 decode 16.16 s against 25.58 s, prefill 512 37.89 s
      against 66.10 s, d512 decode 10.77 s against 29.12 s.
  - [code] The static partition keys each layer on its `weight_0` constant's
    `.bin` offset (patches 0013/0018).
  - [measured-here] `cmp`: `d48p2` holds 24 extra bytes at offset
    1,515,618,509, and everything after is shifted by 24. Recomputed from
    each xml, the partition reproduces all 48 logged checksums per twin: 47
    keys differ by 24, layer 0's is equal.
  - [measured-here] With census seeds (0046) giving both twins `d48p2`'s own
    resident sets (the logged checksum sets confirm it), the dispatch-on
    4-token logits of the twins are bit-identical.
  - [measured-here] On that forward, with equal resident sets, the staged
    path gives the pinned path's bits.
  - Owed:
    - the seeded dispatch-on d1/d512 digests and KLD window;
    - a layout-independent layer key in the plugin (it moves every baseline
      on a residency-dependent route, and the census seed key space);
    - the new-decode twins' KLD on the tier-only route.

- 2026-09-28. **Paper trail closed; the gate verdict stands.** DESIGN
  §7.0.2cg records the depth-4 gate (PASSED, byte-identical, 26.82 GiB off
  the ledger), and §7.0.2cz records the full-depth window (staged 2.2–2.8x
  faster, **gate FAILED**, cause localised to the static partition's
  `.bin`-offset key on the dispatch route). CHANGELOG carries the staged-disk
  subsection. The owed measurement legs listed just above (seeded dispatch-on
  digests and KLD, the layout-independent layer key, the tier-only twins' KLD)
  stay open and are not closed by this line.
