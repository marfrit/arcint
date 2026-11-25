# 1a: fa::Kern's structure in OpenCL C — a design plan, no code

Revised after the second-model review (14 findings,
2026-11-24). Design only; the changes that review forced are listed first, then
the plan as it now stands. Every claim about our kernels is `code` with a
file:line; every claim about the reference is `code` with a line; measurements
are `measured-here` and attributed; reasoning from the two is `inferred`.

Line numbers in `flash_attn_dpas2.cl` here refer to the reviewed revision
(the earlier draft's PV line numbers were ~34 stale; PV is `K2:400-406`).

## Changes in the second session (2026-11-24)

The plan below is the first session's. What the code does now differs in five
places, all recorded in README.md, "1a, second session": group classes
(1 linear / 2 empty / 0 mixed) replace the single flag; the keys a tile visits are
bounded by a test a 32-key block (fa::Kern's `[lo, hi)`), not by a host hint; the
online softmax updates `m` a block, with the sink's term in lane 0 from the start;
the 256-GRF build holds Q in registers and reads it from the op's f32 rows; and
the port's graph is a prompt batch's graph only, its sources on the op before the
backend is asked.

## Changes after review

1. **The pad bin is not zero, and the sink is unfolded.** `INK:386-389` (our pin,
   `src/models/inkling2.cpp`) subtracts the sink from **every** table bin, so the
   pad bin holds `-sink[q,h]` and a `need_bias` skip would drop the sink for far
   keys. Adopted from the reference: the table carries the bias only, and the
   sink is `src[4]` f32 `[n_q, n_head]` seeded into `(m, l)` (`REF:549-550`,
   `SGL:405,415`). The graph's `ggml_sub` pass is deleted (patch 0194). Off by
   one too: the pad bin is `extent`, not `E-1` — bins are `clamp(dist,0,extent)`,
   count `extent+1` (`KV:2206`); with `extent = 1024` the table has 1,025 bins,
   bin 1,024 is the zero-weight pad row, bins 0-1,023 are real bias. `E` is 1,025
   and `need_bias` skips only when a whole group is at bin `>= E-1` *and* the sink
   is already unfolded, so skipping cannot lose it.
2. **The sink counts once.** `l` is a per-lane partial sum (`K2:396,446`); seeding
   `(0,1)` in all 16 lanes would count the sink 16 times. The sink enters lane 0's
   `l` only (`l[0] += exp2(sink - m_final)`), or the kernel keeps `K2:418-446`'s
   per-lane `l` and reduces once at the end — both plans; the second is the
   default because it is what our fast kernels already do.
3. **No `bix` in the kernel; positions and linearity flags instead.** `dist =
   qpos - kpos` (`REF:609`), pad/future/window are compares (`REF:611-616`), and
   linearity belongs to two axes only: the key axis (consecutive kv rows) and the
   row axis (consecutive query rows). The host already walks every pair
   (`KV:2194-2209`) and emits, in patch 0194: `kpos` I32 `[n_kv]`, `qpos` I32
   `[n_q]`, and **one flag a group**: a 16-key group's flag says its keys are
   consecutive and belong to one sequence with no empty cell; an 8-row group's
   flag says its query rows are consecutive. A first/last-bin check is no proof.
   Only groups whose both flags are set take the linear path; all others take the
   general path (`K2:308-317`), which is a per-key mask/bias read and a clamped
   index.
4. **The wrap cost is named, not absent.** One break in the 192 key groups of
   q2048×kv3072 costs ~50 µs (`measured-here` arithmetic: 8,894 µs / 1,024 keys ×
   16 keys × 2). A wrapped ring breaks at the ring seam and at the chunk seam, so
   the SWA layers take 1-2 slow groups per tile-row band; the frequency is
   **unmeasured** and is counted in `set_input_relative_pos` (CPU only) as
   non-consecutive groups per call. The reference materialises the ring itself
   (`SGL:410-413`) and has no slow path, so its 319/627/843 µs bias numbers are
   the linear path's.
5. **The table read form was wrong.** The bias is wanted per (row, key), and DPAS
   lanes are keys (`K2:79,299`), so a lane-local `vload16` would load 16 bins to
   keep one. The plan's first form (a block read with lane = key on a
   bin-reversed table) was not the one built: instead the table stays **as
   `ggml_mul_mat` emits it** (no producer reversal), the run of 16 distances
   `d0-15 .. d0` is one sub-group block read a row at `(d0-15)` with lane =
   position in the run, and the lane whose key has distance `d0-j` takes lane
   `15-j`'s value by `intel_sub_group_shuffle` — one shuffle a row a group, and
   the run is inside the row by construction (`15 <= d0 <= n_bins-1`), no padding
   and no overhang. Outside that (recent keys below 15, keys past the extent) the
   lane reads its clamped bin scalar. Both ends stay bounds-checked
   (`GGML_OPENCL_BOUNDS=1`). The f32 table is what the kernel reads (`ggml_mul_mat`
   is f32-only, `ggml.c:3402`): 4 B a bin, `K2:315`'s f16 rounding of the mask+bias
   is unchanged for the f32 add.
6. **bf16 is not free.** `ggml_mul_mat` is hard-wired F32 (`ggml.c:3402`), the
   reference casts separately (`SGL:405`) *outside* the kernel timing, and our
   backend stores BF16 as f16 (`CL:16524`), where `<<16` misreads. So the kernel
   **reads the f32 `mul_mat` output directly**: 64 B a run for 8 rows × 16 keys ×
   4 B… (16 keys × 4 B = 64 B a row), 0 casts, and `bf16` exists only as a fusion
   option where a GEMM kernel writes 16-bit. Note `mul_mat`'s output is already
   bin-fastest (`INK:386`: `table[q, h, bin]` with the bin axis fastest); the
   current `ggml_permute` + `ggml_cont` (`INK:390`) undoes that only to feed the
   dense gather, and patch 0194 keeps the bin-fastest order.
7. **Rounding.** Today's bias is `half(mask + bv)` on visible cells (`K2:315`),
   i.e. `half(bv)` where the mask adds nothing. An f32 table read keeps the f32
   value and adds it in f32 before the same f16 rounding of `p`/`m`; a bf16 table
   would keep 8 mantissa bits. Either way a KL arm against today's kernel is owed.
8. **No GRF mode is requested, and 256 is the measured default.** The gate
   (ocloc 26.27.1, IGC 2.38.2, bmg-g21) measured both kernels: our port spills
   8,192 B at 128 GRF and is 0 at 256 (3/3 variants, no calls, no retry); the
   reference, compiled from its own SPIR-V, spills 2,112 B at IGC's default 128
   and is 0 with `-doubleGRF`. So the 256 GRF default is evidence, not taste: the
   8-row tile with O and the DPAS operands does not fit 128, and the reference's
   measured times are a spilling build. The cost is occupancy (256 halves the
   threads a core). Originally: `REF:671` submits with no properties and
   `build_inkling2.sh:7-10` passes no GRF flag, while `exl3_ops.sycl:121,634`
   request `grf_size<256>` for their kernels; our card build also passed none, so
   the measured 2,500-3,500 µs are the **default** mode, `inferred` 128 GRF
   (confirm in the ISA dump). **Target: 128 GRF**; a 256-GRF variant only with a
   measurement that shows the 128 build spilling (finding 9).
9. **GRF arithmetic, corrected.** Xe2 GRF = 64 B (`PROGRESS.md:145`), not 32 B.
   Per lane, 8 rows: `O` = 8×128 f32 = 32 GRF; `Q` = 32 GRF if resident, 0 if
   re-read per tile (`K2:274`); `S` = 8×32 f32 = 16 GRF; `P` = 8 GRF. Resident:
   ~120 GRF; re-read: ~90-110 — borderline at 128. Two arms carried: **128 +
   Q re-read** (default) and **256 + Q resident** (a measured arm; 256 halves the
   threads a core, `exl3_ops.sycl:176`, `CL:5341`, 640 against 1,280 on the B60;
   q512 has 1,536 tiles). An offline spill check (the ocloc script) precedes any
   timing.
10. **Tile classes, not a bool.** `MCLS_NONE` in `K2:105-107` means *nothing
    visible*, not "no mask work"; the labels were wrong. Classes now: `ZERO`
    (nothing visible), `LINEAR` (dist compares, no mask read), `LINEAR_BOUNDARY`
    (one edge: last keys or last rows, dist compares + bounds), `GENERAL` (mask or
    table read per key, `K2:308-317`). `need_bias` is a class test too: `LINEAR*`
    groups at bin `< E-1` read the table, others do not — after finding 1, never
    dropping a sink.
11. **ZERO tiles.** With window 1,024 = `extent`, every visible key has
    `dist < extent` (`REF:603-604`): in the measured shapes and in the SWA layers
    a ZERO tile has no bias run at all; only global layers past `extent` do, and
    then the pad bin is read (zero bias), which is cheap and safe.
12. **Geometry.** `K2:51,163` puts one head's 256 rows in a work-group, while the
    reference shares keys across `G × QTW` (`REF:529`) and runs `qtw = 4` in
    production (`SGL:39`). The port keeps the reference's sharing (a
    work-group = `G × QTW` sub-groups, `G = 3`), and **times the `qtw = 1` and
    `qtw = 4` arms**; the grid is sized by the B60's one-work-group-a-core rule
    (`0045`).
13. **Softmax reductions.** `l` stays per lane and is reduced once at the end
    (`K2:446`); `sub_group_reduce_max` runs only when `anyg` is set
    (`K2:366-383`), as does the `shuffle_xor` tree.
14. **Line numbers** in `flash_attn_dpas2.cl` are the reviewed ones throughout
    (`DPAS:57`, QREAD/QREAD2/KREAD/VREAD `70-87`, prefetch `243-244`, mask
    `315-319`, PV `400-406`, reductions `446`, `366-383`).

## Why this plan, and the numbers it must reach

`measured-here` (the operator, B60, 2026-11-24; the reference kernel inside its
own SYCL stack, our kernel through our OpenCL path, the same shapes, one call a
shape, Hq 24, Hk 8, D 128, window 1,024, E 1,025, sinks, table). These are
kernel-level times. Any *served* prefill comparison runs on the operator's
standard arm — `GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1`, ub
1024 main with 512/2048 as extra columns (reference pp4096 722.2 / 765.7 / 723.4
t/s at ub 512 / 1024 / 2048, an earlier window, B60), decode named on the same arm and
unchanged:

| shape | theirs | theirs, table removed | ours (today) |
|---|---|---|---|
| q2048 × kv3072 | 2,500-3,500 µs | 1,630 µs | 12,583 µs |
| q2048 × kv2048 | 1,828 µs | 1,201 µs | 9,938 µs |
| q512 × kv1536 | 856 µs | 537 µs | 2,293 µs |

Bias lookup alone: **843 / 627 / 319 µs** theirs against **8,894 / 6,844 /
1,275 µs** ours. Their base loop without the table is **~2.3-2.6×** faster than
ours with the bias removed. Two levers, taken together.

## fa::Kern's structure, as the reference writes it

`fa_kern.sycl` (verbatim, `xpu/csrc/inkling2_ops.sycl:509-654`): `QR = 8` rows a
tile, `KT = 16` keys a DPAS step, `KBT = 2` → **32 keys a key-block**, `NC = 8`
D chunks; work-group = `G × QTW` work-items sharing keys (L1); one work-item
owns 8 rows of Q, its `O` (8×128 f32), `m`, `l`. Per block: S by DPAS
`8×8×16` fp16→f32, `need_mask`/`need_bias` tests, `mb = hmax`, `exp2` base 2
(`LOG2E` in `scale_log2`), `l = l·alpha + rowsum(p)`, `O = O·alpha + p·V` by
DPAS; sink as the initial `(m, l)` (`REF:549-550`); bias read per row with a
clamped index and only for `dist < E` (`REF:617-619`); K by a transposed 2D block
load (`REF:96-99`), V by a transformed one (`REF:145-148`).

## The OpenCL C translation

Geometry: **one sub-group (SG16) a (head, 8-row) tile**, `QTW` such sub-groups a
work-group, `G = Hq/Hk = 3` heads sharing keys. `K2`'s `intel_reqd_sub_group_size`
(108, 134), `DPAS` (57), the read macros (70-87), prefetch (243-244), the mask
select (315-319), PV (400-406).

| ESIMD construct (`fa_kern.sycl`) | OpenCL form | note |
|---|---|---|
| `xmx::dpas<8,8,float,float,fp16,fp16>` for S (`REF:96-107`) | `intel_sub_group_f16_f16_matrix_mad_k16(a,b,c)` (`K2:57`); `short8` A (lane = row, 16 halves), `int8` B (lane = key), `float8` C (lane = key, components = rows) | N = 16 keys an instruction; a 32-key block = 16 DPAS for S |
| DPAS for `O += p·V` (`REF:145-152`) | same; A = `as_short8(half8)`, B = a VREAD fragment, C = `float8 o[c]` (`K2:400-406`) | 16 DPAS a 32-key block |
| Q rows `block_load<fp16,D>` (`REF:62-69`) | `__builtin_IB_subgroup_block_read_flat_u16_m8k16v1` / `…v2` (`QREAD/QREAD2`, `K2:70-77`) | 8 rows × 128 = 8 or 4 reads a tile; re-read per tile in the 128-GRF arm |
| K `lsc_load_2d<uint32_t,8,KT,transpose>` (`REF:96-99`) | `…_transpose_u32_k8` (`KREAD`, `K2:79`) | 16 keys × 16 dims |
| V `lsc_load_2d<fp16,16,KT,transform>` (`REF:145-148`) | `…_transform_u16_k16` (`VREAD`, `K2:87`) | same |
| prefetch (none in `fa::Kern`) | `intel_sub_group_2d_block_prefetch_16b_8r16x2c` (`K2:243-244`) | keeps the next block in flight |
| per-row max over the block (`REF:130-132`) | components of the `float8` + `sub_group_reduce_max` only when `anyg` (`K2:366-383`) | the block's 32 keys are the lane's component pairs; sub-group max only on rescale |
| row sums (`REF:138`) | per-lane `l` summed a row, reduced once at the end (`K2:446`) | no per-block reduction |
| `exp2` base 2 | `native_exp2` (or `exp2`); the `… > 0.5·M_INIT ? … : 0` shape (`K2:361`) | ours uses `native_exp` with natural-log scores |
| `sv.merge(sv + bias*LOG2E, near)` / `merge(NEG, bad)` (`REF:118-129`) | per-component select (`K2:315-319`), mask sentinel by `as_ushort(mh) != 0xFC00` (`flash_attn_dpas.cl:338`) | no vector `merge` |
| bias read `gather<uint16_t>` (`REF:122`) | `intel_sub_group_block_read_us` with lane = key, one a row a 16-key group (`cl_intel_subgroups_short`; us8 form `mul_mm_few_i8.cl:56`), on a **bin-reversed** f32 table | 8 calls a (tile, key group); 64 B a row |
| `bit_cast_view<fp16>` of uint vectors (`REF:101`) | `as_short8` / `as_int8` (`K2` macros) | same |
| `block_store<float,D>` to `o` (`REF:182-186`) | lane = dim after PV, 8 rows: 8 `intel_sub_group_block_write` a row a D chunk (`mul_mm_id_i8_sg8.cl:97`), or `intel_sub_group_2d_block_write` if IGC has it (`inferred`) | 64 coalesced writes worst case a tile |
| `simd<fp16,…>` 128/256/1024-element vectors | private arrays of DPAS fragments (`K2:205-207,356`) | arbitrary widths are not expressible |
| vertical `select`/`replicate`/`merge` | scalar/8-component array code (`K2:341-352,361-367`) | no equivalent |

## What has no OpenCL equivalent

1. **Sub-group size 1 semantics.** A DPAS needs 16 lanes; the 8-row tile becomes a
   sub-group's collaboration. Register budget is per lane either way; the
   geometry and `reqd_sub_group_size` are what change.
2. **`gather`**: replaced by the block read on a bin-reversed table (§read form);
   an arbitrary-bin table would fall back to scalar gathers = today's 8,894 µs.
3. **Arbitrary vector widths, vertical select/replicate/merge** → DPAS fragments
   and array code; the 128-GRF arm therefore re-reads Q.
4. **Exact lsc 2D widths**: four builtin shapes cover K/V 1:1; Q is 8 or 4 calls.

## Register budget (`inferred`, 64 B GRF)

Per lane, 8 rows: `O` 32 GRF; `Q` 32 GRF resident / 0 re-read; `S` 16; `P` 8;
`m,l` 1; K/V fragments 2-4 → **~90-110 GRF re-read, ~120 resident**. Default arm
**128 GRF + Q re-read**; the 256-GRF arm only with a spill measurement, and it
halves the threads a core (640 against 1,280 on the B60).

## The table: f32, bin-reversed, block read

- Producer: `mul_mat`'s output is already bin-fastest `[rows, n_head, bins]`
  (`INK:386`); patch 0194 keeps it (deletes `INK:390`'s permute+cont for the FA
  path) and reverses `attn_rel_bias`'s columns at load so the kernel's ascending
  lane index is `extent - dist`. Type **f32 in place** (0 casts, 64 B a row a
  16-key group); a bf16/f16 table is a *fusion* option only, where a GEMM writes
  16-bit, and the f16 variant matches today's `half(bv)` rounding (`K2:315`).
- Bins: the pads make 1,025 bins (`extent + 1`); the bin-reversed table's last
  row is padded by 15 bins so the final 16-key run cannot overhang; the first
  row starts at 0. Both ends are bounds-checked (`GGML_OPENCL_BOUNDS=1`).
- Sink: `src[4]` f32 `[n_q, n_head]` as the initial `(m, l)`; the table carries
  the bias alone (finding 1).

## Positions, linearity, and the tile classes

- `kpos` I32 `[n_kv]`, `qpos` I32 `[n_q]` (new op sources, patch 0193), one flag a
  16-key group and one a 8-row group (patch 0194), built in the host's pair walk
  (`KV:2194-2209`). A flag is set only for consecutive rows of one sequence with
  no empty cell.
- `dist = qpos - kpos`; validity is `dist >= 0`, `dist < window`, `key < n_kv`,
  `row < n_q` — compares, no mask read (finding 3, `REF:609-616`).
- Classes: `ZERO` (nothing visible), `LINEAR` (both flags set and interior),
  `LINEAR_BOUNDARY` (one edge), `GENERAL` (a flag clear or a mask/table read a
  key, `K2:308-317`). `need_bias` is per class: `LINEAR*` groups with bin
  `< E-1` read the table; `GENERAL` may read any bin with a clamp.
- The pack bin (`bin = extent`) is the zero-bias pad row and is read safely; the
  sink never depends on it (finding 1).
- Wrap: a ring seam or a chunk seam clears the key group's flag → that group takes
  `GENERAL`; cost ~50 µs a break in 192 groups at q2048×kv3072 (`inferred`),
  frequency counted in `set_input_relative_pos`.

## Binding plan per layer call (Hq 24, Hk 8, D 128, `G = 3`)

1. Q f16 `[Lq, Hq, D]` (already the graph's order, `binding-notes.md` §2);
2. K/V f16 `[Lk, Hk, D]` contiguous, or the paged gather (6.29 MB a call at
   Lk 1536, 134.2 MB at 32k, 536.9 MB at 128k for the 4 global layers);
3. table f32 `[rows, Hq, bins]` bin-reversed, 1,025 bins, 15-bin end pad: 50.4 MB
   a 512-token layer call in place, **0 extra** for the cast/permute;
4. `kpos`, `qpos`, group flags: 4 B a kv row, 4 B a query row, a few bytes a group;
5. sink f32 `[n_q, n_head]` (`fill(0)` for the model → `(0,1)`);
6. out f32 `[Lq, Hq, D]`.

## Order of work (code starts now)

1. `flash_attn_port.cl`: tile, DPAS S/softmax/DPAS PV, LINEAR classes, no bias —
   the 1,630 / 1,201 / 537 µs baselines are what this step must land near; red
   case: 1b's oracle (`fa_ref_host` unchanged).
2. the bias: block read a (row, 16-key group) on the bin-reversed f32 table,
   `need_bias` per class, both-ends guard.
3. the classes and `GENERAL`, with the campaign design note's red cases plus a pad
   bin and a zero sink and a wrapped ring.
4. variants keyed `(dk, dv, kvq, bias)`, `GGML_OPENCL_FA_PORT=1`, a `DOWNGRADE
   reason=` line for every shape or layout the port does not cover; unset = the
   current dispatch exactly.
5. 128-GRF default arm, Q re-read; the 256-GRF arm only after the spill check.

Open operator decisions: the bin-reversed table (a graph-side change to
`attn_rel_bias`'s load; it does not change the op's shape contract), and the
`supports_op` gate.
