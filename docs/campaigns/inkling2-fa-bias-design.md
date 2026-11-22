# Inkling 2 Mini: relative-position bias + sinks inside the OpenCL flash-attention kernels

Design note. No code, no builds, no GPU. Every claim about our code is
`code` with `file:line`; the reference and the open upstream PR are `code`
too; anything reasoned from sizes is marked `inferred`.

Trees read:
- `llama-ink-port` — llama.cpp pin `3e4dfa8` with the arcint patch
  series applied (commit `bf8909e`). Paths below relative to
  `ggml/src/ggml-opencl/` unless said otherwise.
- `llama-inkling-ref` — mainline with the model
  (`src/models/inkling2.cpp`).
- `pr32740-fa-compact-bias.diff` + `pr32740-description.md` — open upstream
  PR (cited as `PR:<diff line>`).
- `godgmn-inkling-rdna3.md` — an AMD fork's notes (`godgmn`).

There is no `docs/campaigns/inkling2-mini-b60.md` in this repository
(`code`), so the model facts below come from the reference tree.

---

## 1. What the model asks for (reference)

`llama-inkling-ref/src/models/inkling2.cpp`:

- `attn_rel`  `{n_embd, n_rel*n_head}`, `attn_rel_bias` `{n_rel, n_extent}`,
  `attn_rel_clip` `{n_rel, n_head}`, `attn_sink_proj` `{n_embd, n_head}`
  — lines 122-126. `n_extent = is_swa ? hparams.n_swa :
  hparams.n_relative_extent` (line 110).
- Per token: `Rcur = w_rel·input` reshaped `[n_rel, n_head, n_tokens]` and
  clipped (line 337), then scaled by `tau` on global layers (341-344).
- The bias is **low-rank and per-token**, not a static table:
  `table = mul_mat(pad(attn_rel_bias, 0,1,0,0), Rcur)` (358) →
  `[n_extent+1, n_head, n_tokens]`; the pad adds a zero weight row, i.e. bin
  `n_extent` = "no relative term".
  `table -= sink_logit(head, token)` (360), permuted to
  `[n_head, n_bins, n_tokens]` (361).
- The dense `kq_b = get_rows(table, rel_pos)` (364-366) is what we must not
  build.
- The sink logit is folded out of the table and a **zero sink** is passed
  separately: `sinks = fill(0, n_head)` (369), `build_attn(..., kq_b, sinks, ...)`
  (371-374). The comment at 359-360 gives the identity:
  `exp(s-t)/(1 + Σ exp(s-t))`. So the sink term is exactly `exp(0-m)` in the
  denominator — **not** a no-op, and `-t` rides inside the bias table.
- Bin = clamped one-sided distance:
  `data[i*n_kv+j] = (distance >= 0 && distance < extent) ? distance : extent`
  — `src/llama-kv-cache.cpp:2011-2022`. Bins are `0..extent`, count
  `extent+1`. Negative/out-of-window/empty cells all land in bin `extent`.
  There is no symmetric `±` window in the index.
- Index tensor: `self_rel_pos` I32 `[n_kv, n_tokens]`
  (`src/llama-graph.cpp:4467-4473`).

Sizes for Inkling 2 Mini (`inferred` from the shapes above, with
`n_head=24`, `n_kv_head=8`, `d_head=128`, `n_rel=16`, `extent=1024` →
`n_bins=1025`):

| thing | per | bytes |
|---|---|---|
| compact `table` f32 | one query token | 1025·24·4 = **98.4 KB** |
| compact table, 512-token prefill | one layer call | **50.4 MB** |
| dense `kq_b` f32 (`get_rows`) | layer call, 1025 KV rows | 25.2 MB |
| dense `kq_b` f32, 100k KV rows | layer call, 512 queries | **4.92 GB** |

The table does **not** fit local memory (98.4 KB/token against the ~32-64 KB
of local memory our DPAS kernels already spend: `kernels/flash_attn_dpas.cl:187`,
`flash_attn_gqa_dpas.cl:216-238`). Any note that says "a few KB, fits local
memory" is wrong for this model's table; only the *slice a tile touches*
(8 rows × 32 keys = 1 KB of values) is small.

---

## 2. Q1 — our kernels today, and what each accepts

Kernel inventory (`code`, `kernels/`):

| kernel | file:line | role | mask | ALiBi | softcap | sinks |
|---|---|---|---|---|---|---|
| `flash_attn_f32_f16` | `flash_attn_f32_f16.cl:132-133` | prefill+decode, f32 Q / f16 KV | yes | yes (`slope·mask`) | yes | **yes** 569-641 |
| `flash_attn_f32_f16_q1_split_intel` | `flash_attn_f32_f16.cl:2916` | decode/MTP-verify 1-8 rows, f16 KV, split | yes 2862-2865 | yes 2949 | yes 2947 | in merge |
| `flash_attn_q8q8/q8q4/q4q4_q1_split_intel` | `flash_attn_f32_f16.cl:3028,3030,3032` | same, quantized K/V | yes | yes | yes | in merge |
| `flash_attn_f32_merge` | `flash_attn_f32_f16.cl:2703` | combine splits | — | — | — | **yes** (sink joins `m`/`l`) |
| `flash_attn_f32_f16_dpas` | `flash_attn_dpas.cl:160` (SG 16) / 482 (SG 8) | **Intel XMX prefill** | yes 331-341 | no | no | no — file header line 21: "No ALiBi, softcap or sinks" |
| `flash_attn_f32_f16_dpas2` | `flash_attn_dpas2.cl:136` | Xe2 prefill, + `flash_attn_q_to_f16` 90, `flash_attn_mask_cls` 109, `flash_attn_split_combine` 410 | yes 293-302 | no | no | no |
| `flash_attn_gqa_dpas` | `flash_attn_gqa_dpas.cl:188` | **decode / verify, GQA, f16 + q8q8/q8q4/q4q4, page table** | yes 559-564, 700-713 | args at 199 unused | arg at 200 unused | no (merge does) |
| `flash_attn_kv_pad_f16` / `mask_pad_f16` / `blk_f16` | `flash_attn_pre_f16.cl:3,53,93` | prepass for the generic kernels | — | — | — | — |
| `flash_attn_f32`, `flash_attn_f16`, `flash_attn_f32_q8_0`, `flash_attn_f32_q4_0` | upstream | f32/f16 KV legacy | yes | yes | yes | `flash_attn_f32.cl:227-238,324,399-400`; `flash_attn_f16.cl:115,316` |

Host choice (`code`, `ggml-opencl.cpp`):

- `supports_op`, `GGML_OP_FLASH_ATTN_EXT`: 13248. Head dims accepted include
  `{128,128}` (13354-13360). Type combos f32/f16, f32/q8_0, f32/q4_0, mixed
  (13375-13410). **sinks / ALiBi / softcap are not declined here** — the
  call is accepted and the dispatcher routes it.
- Prompt (n_q > 8) to XMX: `fa_dpas_ok` at **25789-25792** requires
  `sinks == nullptr && max_bias == 0.0f && logit_softcap == 0.0f`, contiguous
  Q/K/V, f16 mask wide enough. Head-128 is `fa_dki == 1` (25781). If the gate
  fails, a `DOWNGRADE` line is written (25794) and the call falls to the
  generic `flash_attn_f32_f16.cl` prefill — i.e. **today a sink (GPT-OSS) or
  ALiBi costs the whole prompt path**.
- Decode / verify (n_q ≤ 8): `intel_q_decode` 26203-26206 (quantized K/V,
  split+merge), `use_fd_intel` 26455-26461 (f16 K/V, split+merge), and the
  GQA kernel from 4 rows (`GGML_OPENCL_FA_GQA_MIN_ROWS`, 26476-26478) under
  26539: `max_bias == 0.0f && logit_softcap == 0.0f` and
  `G·n_q ≤ fa_gqa_tiles·8` — **sinks allowed**, applied by the merge kernel
  (`ggml-opencl.cpp:26971-26990`).
- Sliding window: **no kernel implements a window.** A window is the f16
  mask (`-inf` outside), and only whole-tile skipping is available
  (`flash_attn_dpas.cl:194-200` scans `kv_end` back from the mask;
  `flash_attn_dpas2.cl:109` and `flash_attn_pre_f16.cl:93` classify
  tiles). Our pin has no K/V row-list feature (no
  `ggml_flash_attn_ext_rows` anywhere in the tree, `code`).
- Paged K/V (the streaming path) is head-256 only and **declines sinks,
  ALiBi and softcap**: `ggml-opencl-fa-paged.h:85-90` (`k_ne0 == 256`),
  96-98 (mask), **101** (`!sinks_present && max_bias == 0.0f && logit_softcap == 0.0f`).
  Head-128 Inkling therefore stays on the dense-cache path; paged is out of
  scope for this campaign (`code`).

---

## 3. Q5 — the op interface: what PR 32740 changes, and our collisions

At our pin (`code`, `ggml/src/ggml.c`): `ggml_flash_attn_ext` 5543-5583
uses `src[0..3]`; `op_params` float 0 = scale, 1 = max_bias, 2 = softcap;
i32 3 = prec (5588), 4 = n_kv_max (5604); sinks at `src[4]` (5616-5631).
Paged attention has taken everything else (`ggml/src/ggml-impl.h:186-204`):

- `op_params`: **5** page size, 6-7 ready, 8-9 consumed, 10 page-table row,
  11-12 page runs, 13-14 LSE bounds.
- `src`: **5** pages, **6** ring_k, **7** ring_v, 8 tag, 9 host_k, 10 host_v,
  11 plan; `GGML_MAX_SRC = 12` (`ggml/include/ggml.h:225-230`, patch
  `contrib/llama.cpp/patches/0126-ggml-max-src-paged-attention.patch`).

PR 32740 (`code`, `PR:831-860`) adds on top of the same op — no new op:

- `ggml_flash_attn_ext_add_indexed_bias(a, table, indices)`:
  **table → `src[6]`, indices → `src[7]`, flag → `op_params` i32 5**
  (`ggml_flash_attn_ext_has_indexed_bias` reads it).
- Shapes it asserts: `table` F32 contiguous `[n_head, n_bins,
  n_q1·n_q3]`, `n_bins = table->ne[1] > 0`; `indices` I32 contiguous matrix
  `[n_kv_cols, n_q1·n_q3]`, values in `[0, n_bins)`; a mask is required with
  `ne[0] == n_kv`, `ne[1] == n_q`, `ne[2] == 1`, `ne[3] == n_batch`.
- Semantics, verbatim from its header comment (`PR:12-13`): *"Add
  `table[head, indices[key, query], query]` to the mask before its F16
  rounding. Queries from all streams are flattened in stream order."* Its own
  test reproduces that: the expected weight is
  `exp(qk + f16(mask + bias))` (`PR:1330`).
- Backends that decline it: meta `PR:29`, **CPU `PR:59`**, CANN, **OpenCL
  `PR:479-481`**, openvino, rpc, sycl, virtgpu, webgpu, zdnn.
- Graph side: `build_attn_mha` gains `bias_table`/`bias_indices`; if the
  layer device does not support the op, or `offload_kqv` is off, it clears the
  flag and materializes `dense_bias_mask()` = `get_rows` + add + cast f16
  (`PR:905-918`, `946-958`). The rel-pos indices follow the mask's row list
  (`PR:1113-1149`).

**Collision for us:** the PR's `src[6]`/`src[7]` are our paged `ring_k`/
`ring_v`, and its `op_params[5]` is our `GGML_FA_PARAM_PAGE_SIZE`. Copying the
PR's slot assignment onto our pin is undefined-behaviour-adjacent: the two
features read the same slots. Port decision (`inferred`, size argument above):

- keep the PR's *semantics and shapes* (so an upstream model file and our
  graph agree), but assign our own slots:
  `GGML_FA_SRC_BIAS_TABLE = 12`, `GGML_FA_SRC_BIAS_IDX = 13`,
  `GGML_MAX_SRC = 14`, and the flag at `op_params` i32 **15** — all with the
  existing `static_assert` pattern (`ggml-impl.h:200-204`) extended to the
  new constants. One new patch beside 0126, same idiom.
- the port must also carry the model (`src/models/inkling2.cpp`) and the
  `build_attn_mha` change; our pin's graph still asserts
  `use_flash_attn = cparams.flash_attn && kq_b == nullptr`
  (`src/llama-graph.cpp:3083-3085`) and adds a dense `kq_b` to `kq`
  (3157-3159) — with a dense bias, FA is off for that layer entirely.

**CPU reference.** Because the PR declines indexed bias on CPU, its
correctness comes from a standalone test with hand-computed expectations
(`PR:1239-1380`). `test-backend-ops` compares against CPU, so for our own
red/green cases CPU must implement the same arithmetic, or every case is
skipped. Minimal CPU change: in
`ggml/src/ggml-cpu/ops.cpp:8963` (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`):
at the score loop — `s += mv; // apply mask` at **9104** — insert before that
line `mv = f16(f32(mv) + table[(query*n_bins + bin)*n_head + head])` with
`bin = indices[query*n_kv + ic]`, and leave the sinks block at 9160-9162
untouched (it applies the sink on the first kv-chunk only). The chunked
callers are at 9627 and 9689 — there is no second score loop to fix. Mark it
`GGML_OP`-identical to the kernels: the f16 rounding of `mask+bias` is part
of the contract, not an optimization.

---

## 4. Q2 — where the bias goes, per kernel

Common to all three: `bin = indices[query·n_kv_cols + key]`;
`bias = table[(query·n_bins + bin)·n_head + head]`; add it to the f16 mask
value in f32, round to f16 once, then `score += mask_value` (our kernels use
`slope == 1` whenever `max_bias == 0`, which is what Inkling wants — keep
`max_bias == 0` so the ALiBi slope never multiplies a relative bias).
Guard `bin < n_bins` as the PR does. A `-inf` mask entry must stay `-inf`:
test the sentinel on the mask, not on the sum (`flash_attn_dpas.cl:338`
uses `as_ushort(mh) != 0xFC00`).

Layout warning (`inferred` from the PR assert at `PR:855`): with the PR's
`[n_head, n_bins, n_tokens]` (head fastest), one lane-run of 16 consecutive
bins reads 16 locations `n_head·4 = 96 B` apart — 16 separate cache lines,
where the mask read of the same shape is one line. A `[n_bins, n_head,
n_tokens]` (bin-fastest) table would coalesce. **Operator decision**: stay
upstream-compatible and eat the gather, or deviate and relax the assert.
Measure both on one leg; do not pre-decide by dropping the mechanism.

### 4.1 Intel XMX prefill — `flash_attn_dpas.cl`

Work-group = one (head, batch); a sub-group owns 8 query rows
(`qr = q_sg + r`, 334) and one key column per lane
(`key = t0 + g·16 + lane`, 332). The mask epilogue is
**331-341**; the bias goes inside it, in the `vis` branch before `sv[r] =
sv[r] + mk` (339). Normalisation is 421-432. The same loop exists in the
SG-8 build (626-629) and in `flash_attn_dpas2.cl:293-302`; all three need
the edit, plus `FA_CARRY` mode (172-177, 413-418) which carries `m,l,O`
across key blocks — the bias needs no carry state.

Read volume per tile: 8 rows × `BK` keys = 256 table reads per sub-group —
exactly the mask's own traffic, and indices are monotone in `key` (distance
shrinks as the key position grows, `llama-kv-cache.cpp:2011-2022`), so
staging one `int4`/`float8` run per lane per row into local memory is
possible if the raw gather shows up.

### 4.2 Decode / verify — `flash_attn_gqa_dpas.cl`

Rows are `(head, token)` pairs: `rh[r] = hk·G + m%G`, `rt[r] = m/G`
(255-264); keys `key = k0 + g·16 + lane` (700). The mask is read per row as
`mrow[r][key]` (559-564, 707) and folded at 707-713. The bias goes there:
`query = rt[r]`, `head = rh[r]`, `key`. Note the kernel has **no position
argument** — but it does not need one: the graph already hands it the index
tensor keyed by (query row, key), which is exactly what the PR passes.
Sinks: this kernel has none; the merge (`flash_attn_f32_f16.cl:2703`, host
args `ggml-opencl.cpp:26971-26990`) already joins the sink across splits, so
decode sinks need no kernel change.

### 4.3 Generic split decode — `flash_attn_*_q1_split_intel`

The score loop is `flash_attn_f32_f16.cl:2916+` (body shared through
`FDI_PARAMS`/`fdi_split_body`, 3028-3032): softcap 2947, `slope·mask` 2949,
`fmax` clamp 2950. Bias goes between softcap and the mask add, with
`query = q_idx`, `head = head_idx`, `key = k0+j`. Merge unchanged.

### 4.4 Sinks in the online softmax (Q3)

A sink is a per-head logit with no V row: it enters `m` and `l` but never
`O`. Two forms already in the tree:
- single kernel: seed `m_i = sink[head]` and add `exp(sink-m_final)` to
  `l_final` (`flash_attn_f32.cl:324, 399-400`), or rescale at the end
  (`flash_attn_f32_f16.cl:569-575, 596-605, 630-641`).
- split + merge: the split writes `(m,l,O)` and the merge takes
  `m_final = max(max_c m_c, sink)`, `l = Σ l_c·e^{m_c-m_final} +
  e^{sink-m_final}` (`flash_attn_f32_f16.cl:2703+`; the merge launch passes
  `sinks_buffer` — `ggml-opencl.cpp:26971-26990`).
For the two Intel kernels that have no sinks (`flash_attn_dpas.cl`,
`flash_attn_dpas2.cl`), add the first form: `l[r] += exp(sink[head] - m[r])`
once after the key loop, before the division at 421-432 — one value per row,
no per-element cost. Inkling's sink is the constant 0
(`inkling2.cpp:369`), so `exp(-m)`; correctness at the answer requires it
(the denominator changes by `e^{-m}`), and `m` can be `-3e38` for an
all-masked row, so keep the `lv[r] > 0` guard at 430.
The PR's CUDA tile kernel has to move the sink out of the first parallel
block when a bias is present (`PR:364`, `PR:375-384`) — our merge-based split
does not have that problem (`inferred`: the sink is applied once, in merge).

---

## 5. Q4 — specialisation and the variant to add

How we specialise today (`code`): each kernel family is built per (dk,dv)
with `-D` options and cached in per-variant maps —
`load_cl_kernels` 5279-5410 (`flash_attn_dpas.cl`, `-DSG=16/8`, `FA_CARRY`,
`KT`/`VT` for K/V type), `flash_attn_dpas2.cl` 5413-5430, `flash_attn_gqa_dpas.cl`
5506-5761 (`GQA_I8`, `GQA_ST`, `GQA_K_STAGED`, `GQA_V2`, `GQA_PRE`,
`GQA_SIDES`, `BK`, `QB`, `NDH`), variant enum `FA_VARIANT_*` with the source
pick at 7604, ensure-on-demand `ggml_opencl_ensure_fa_variant(...)` (13301),
kernel slots `kernel_fa_gqa_dk[dk][kvq]`, `kernel_fa_gqa_st_dk`,
`kernel_fa_dpas_dk[pi]`, `kernel_fa_dpas_q_dk[pi][kvq]`,
`fa.f32_f16_q1_split_intel[{dk,dv}]`, `fa.q_q1_split_intel[{dk,dv}][slot]`,
`fa.f32_merge[{dk,dv}]` — all keyed on (dk,dv) plus a type slot, plus env
switches (`GGML_OPENCL_FA_*`).

The PR's evidence about the decode cost, read from its code rather than the
brief's summary: CUDA passes the bias as a **runtime** pointer struct and
branches `if (bias.table)` inside the tile kernel (`PR:286-291`), with the
tile kernel templated `if constexpr (ncols2 == 1 && (DKQ==64||DKQ==128))`;
the vec/mma kernels take the struct and `GGML_UNUSED` it (`PR:173`, `PR:416`),
and the kernel picker forces the tile kernel for indexed bias at DK 128
(`PR:419-424`) and turns off the GQA fast path when a bias is present
(`PR:1277`). Vulkan keeps `n_bins` in a push-constant word
(`n_bins = (Flags & 0x7fffffff) >> 8`) with a runtime test (`PR:741-756`),
disables its GQA path for bias (`if (!bias && N <= 8 ...)`), and adds a
compile-time variant only for the row-list form (`PR:794-800`). So: upstream
uses **runtime** branches in the hot loop, and its own compile-time
specialisation only for the row-list axis.

What we should do, given our kernels are already per-(dk,dv,type) programs
and our DPAS inner loops are register-tight (`flash_attn_dpas.cl:366-371`:
"select costs the q8_0 build 896 bytes of spill and 15 % of pp512
(`measured-here`)) — add
**build-time** defines rather than a runtime test —

- `-DFA_BIAS_TABLE=<n_bins>` (0 = no bias): the loop body then knows the
  table exists and folds `indices`/`table` reads in; with `n_bins` a constant
  the multiply is a shift-add chain.
- `-DFA_SINKS` where the kernel applies them itself (4.4).
- Variant key becomes `(dk, dv, kvq, bias)`: extend the existing maps with a
  bias boolean instead of inventing a registry, and gate on it in
  `supports_op` so a shape that needs bias is declined exactly where a
  variant is missing (the pattern `ggml_opencl_fa512_intel_supported` uses at
  12684-12728, which writes a `DOWNGRADE ... reason=` line at 12722).
- Only the three kernels that Inkling actually reaches get the bias form at
  head 128: `flash_attn_dpas.cl` (f16 and the three quant pairs), `flash_attn_dpas2.cl`,
  `flash_attn_gqa_dpas.cl` (f16 + q8q8/q8q4/q4q4), `flash_attn_*_q1_split_intel`
  + merge. Build the bias variants lazily on the first call that needs them,
  as `ggml_opencl_ensure_fa_variant` does, so the no-Inkling startup cost is
  zero.

---

## 6. Q6 — test plan

Everything below is a `test-backend-ops` case plus a red/green proof. Our
pin's case class is `test_flash_attn_ext` (`tests/test-backend-ops.cpp:9135`;
17 fields in `vars()` through `VARS_TO_STR17` at 9158 — new fields go into
both): add `bias` (bool) and `n_bins`, and in `build_graph` create a `table`
F32 `[nh, n_bins, nb·nr23[1]]` and an `indices` I32 `[kv, nb·nr23[1]]`
through the new accessors (our slot names, §3). In `initialize_tensors`,
random bins in range and table values in `[-4,4]` — large enough to move the
argmax; the sinks tensor is already built there (9230-9236) and stays at
`(-10,10)`.

Shapes for this model: `hsk=hsv=128`, `nh=24`, `nr23={8,1}` (GQA 3:1, one
stream) and `nr23={1,8}` (8 streams, the flattened-query case the PR
asserts), `kv = 1024, 1025, 1088` (window edge and the `+1` pad bin),
`nb = 1..8` (decode, and the 4-8 row verify that must hit
`flash_attn_gqa_dpas`), `nb = 512` (prefill → XMX), `mask = true` always
(the op requires it), `sinks = true` always (Inkling's sink is zeros:
include a `sinks = fill(0)` variant, because a zeros sink is exactly the
bug that "no-op" implementations pass), `n_bins = 1025`, plus `n_bins`
smaller than some index value to exercise the `bin < n_bins` guard, and
`type_K/V = f16, q8_0/q8_0, q8_0/q4_0, q4_0/q4_0`.

Red cases that prove each new path is taken (`code` mechanisms already in
the tree):
1. **Path naming.** `GGML_OPENCL_FA_DEBUG=1` prints `form=gqa|intel_split|
   split|other` (26579); `ggml_cl_downgrade` (helper at 734) prints
   `op=FLASH_ATTN head=... path=... reason=...` — prompt downgrade at 25794,
   fa512 decline at 12722. A bias case that runs on the generic
   prefill kernel instead of `fa_dpas` must be visible as
   `path=f32_f16 reason=shape` — and after the change it must *not* appear.
   Gate: the DOWNGRADE count for an Inkling-shaped run goes to zero.
2. **Bias variant present.** A `supports_op` decline when the (dk,dv,kvq,
   bias) variant is missing — force it with an env switch mirroring
   `GGML_OPENCL_FA_DPAS=0` and check the case is declined, not silently
   rerouted.
3. **Index path.** With `bias` on, permute `indices` to a *constant* bin and
   check the output equals the same graph with `bias` off and a dense mask
   filled with that bin's value. This distinguishes "the bias was read" from
   "the bias was folded into the mask host-side".
4. **Sink join under splits.** decode `nb=1`, `n_kv` large enough to force
   `n_splits > 1` (`GGML_OPENCL_FA_INTEL_KV_PER_SPLIT`), zeros sink: the
   output must differ from the same call with `sinks = nullptr` by the
   `e^{-m}` denominator term and match the CPU reference within
   `max_nmse_err` (9163).
5. **Window.** mask = `-inf` outside a 1024 window with `kv = 4096`: the
   tile-class prepass must still fire (`flash_attn_mask_cls`,
   `flash_attn_blk_f16`), and the answer must match the CPU reference.
6. CPU-vs-OpenCL agreement on the f16 rounding of `mask + bias`: one case
   where `bias` and `mask` are both non-zero and the sum's f16 rounding is
   observable (choose values straddling a rounding boundary), because that is
   the one contract the PR states in a comment rather than in the op.

Not a test-backend-ops item, but owed before any serving claim: the
answer-level bar from `AGENTS.md` rule 9 (`CLAUDE.md` wording) on an actual
Inkling 2 Mini run — facts/needle/battery, mean KL ≤ +0.03 nats against the
non-FA (dense-mask) arm on the same card and window, argmax agreement within
1 point. And a measured price for the layout question in §4 (upstream table
layout vs bin-fastest) and for the compile-time variant vs the PR's runtime
branch — those two are the numbers the operator decides on.

---

## 7. Sequencing (proposal, one campaign, one gate)

1. op + slots patch (`src[12]`/`src[13]`, `op_params[15]`, `GGML_MAX_SRC=14`,
   static asserts) and the CPU reference. Gate: test-backend-ops CPU/CPU and
   the PR's standalone expectation test ported.
2. `flash_attn_gqa_dpas.cl` + `flash_attn_*_q1_split_intel` bias and sinks
   variant, DOWNGRADE lines. Gate: decode cases (1-8 rows, all four K/V type
   pairs) green and named `form=gqa`.
3. `flash_attn_dpas.cl` / `dpas2.cl` bias and sinks variant, and drop the
   `sinks == nullptr && max_bias == 0 && softcap == 0` clause from `fa_dpas_ok`
   (25789-25792) into a per-variant check. Gate: prefill 512 green, no
   DOWNGRADE line.
4. Model + graph port (`inkling2.cpp`, `build_attn_mha` with
   `bias_table`/`bias_indices`, dense-mask fallback kept for devices that
   decline) and the served-answer gate above.

Reference to follow: PR 32740 for the op contract and the arithmetic
(`code`, `PR:12-13`, `PR:286-291`, `PR:1239-1380`); `godgmn` for the
alternative of computing the bias from per-cell positions inside the kernel
and a dedicated `GGML_OP_INK_REL_BIAS` (`godgmn` §"Long context", §"What
changed" 2) — that route trades the 98.4 KB/token table for ~16 MACs per
(row,key) in the hot loop and doubles decode on RDNA3 (`paper` in that note:
31.9 → 64.0 t/s, `paper`, on a different card and a different FA stack). It
is not the starting point here because the PR's contract is the one upstream
model code will carry, but it is the mechanism to price if the gather
dominates.
