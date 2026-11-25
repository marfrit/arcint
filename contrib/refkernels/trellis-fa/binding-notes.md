# Feeding trellis-serve's kernel from our `GGML_OP_FLASH_ATTN_EXT` call

No code. Every item below is one of three classes: `code` (read out of a
tree, with file:line), `inferred` (reasoned from shapes and sizes, a
measurement still owed), `measured-here` (this host, 2026-11-24).

The kernel is `fa::Kern<128>` (`fa_kern.sycl`): f16 q `[Lq, Hq, D]`, f16 k/v
`[Lk, Hk, D]`, **contiguous rows**, one contiguous run of positions per launch
(`qpos0`, `kpos0`, `window` scalars), bf16 table `[Lq, Hq, E]` indexed
`table[q, h, clamp(qpos-kpos)]` for `dist < E`, f32 sinks `[Lq, Hq]` as the
initial `(m, l)`, f32 out `[Lq, Hq, D]`, D=128 only.

Ours is `ggml_flash_attn_ext`, at our pin `src[0..3]` = q, k, v, mask,
`src[4]` = sinks, `op_params[0]` scale, `[1]` max_bias, `[2]` softcap; the
indexed-bias port adds the table and the per-pair bins (`code`,
`docs/campaigns/inkling2-fa-bias-design.md` §3, on `qfndev`).

Surface: 1 integer handful of scalars per launch, 3 pointer-buffer conversions,
2 layout conversions, 1 dtype conversion, 1 shape conversion, and the mask
disappears. What blocks it outright is item 1 (positions) and item 2 (paging).

---

## 1. Positions: ours gives a mask and per-pair bins, theirs wants two scalars

`code`: the graph hands the FA op a mask f16 `[n_kv, n_q]` and, with the
indexed-bias port, bins I32 `[n_kv, n_q]` (design note §3; `build_attn_mha`,
`src/llama-graph.cpp:3083-3165` on the llama-ink-port tree). The bins are
`bin = clamp(dist, 0, E)` with `E = 1024` (`src/llama-kv-cache.cpp:2011-2022`),
so `bin == E` conflates *dist ≥ E*, *dist < 0*, *out-of-window* and *empty
cell*. **Positions cannot be recovered from bins**: everything above 1,024
collapses into one bin, and the invalid rows vanish.

Their kernel does not take bins at all; it recomputes `dist = qpos - kpos` from
`qpos0 + row` and `kpos0 + row`. Consequence: our bins tensor is **not needed**
(one tensor and one pass saved), but only if the caller can name the two run
origins and guarantee the kv rows are one contiguous run whose positions count
up by one.

- Prefill of one sequence into a dense cache: `qpos0 = pos of q row 0`,
  `kpos0 = pos of kv row 0`, **1 launch**, and the cache's pos array already
  holds the positions (`code`, `llama-kv-cache.cpp` per-cell `pos`). Cost: 0.
- A batch of B sequences: B contiguous runs, **B launches**, each with its own
  pair. Cost: B launches instead of one; the compute per launch is its own
  pairs, so the FLOPs are unchanged — the cost is launch latency only, and it
  has not been measured (`inferred`).
- A windowed (ring) cache: the window's rows are a *ring slice*, not a run
  whose position counts up by one. The reference has the same problem and
  solves it by materialising: `torch.cat([kb[rows], k[r0:r0+n]])`
  (`xpu/exl3xpu/inkling2/sgl_model.py:409-412`, `code`). Cost: a copy of the
  window's rows — see item 2.
- Padding: their kernel's only invalidation is `kpos ≤ qpos`, `dist < window`,
  `row < Lk`. A mask with arbitrary `-inf` cells (pad, empty paged cells,
  mixed sequences) **cannot be expressed**. Such a call must be split into
  contiguous valid runs, or stay on our kernel.

## 2. Layouts: identical element order, incompatible storage

The pleasant part (`code`, `src/llama-graph.cpp:3108-3110`): `build_attn_mha`
hands the op `ggml_permute(ctx0, q, 0, 2, 1, 3)` — q's `ne` becomes
`[D, Lq, Hq]`, physical order `[Lq][Hq][D]` with D fastest, which **is** their
`[Lq, Hq, D]` element order. Same for k/v: `[D, Lk, Hk]` view,
`[Lk][Hk][D]` physical order, which **is** their `[Lk, Hk, D]`. And `o` f32
`[D, Lq, Hq]` physical `[Lq][Hq][D]` is their f32 out. So **no transpose is
needed for any of the four**; what is needed is dtype and contiguity.

- **q**: the op's q is f32 (the graph's projection output) and the kernel takes
  f16. The reference does the same conversion at call time
  (`xpu/exl3xpu/inkling2/attn_triton.py:664`, `q.to(torch.float16)`, `code`).
  Cost per 512-token layer call: read 512·24·128·4 = 6.29 MB, write
  512·24·128·2 = 3.15 MB → **9.44 MB of traffic**, against 6.29 MB if a kernel
  reads q f32 itself (i.e. the pre-materialisation costs +3.15 MB per layer;
  ×24 layers = **75.6 MB per prefill call**). Cheaper route: let the q
  projection write f16 directly (we own that kernel; `flash_attn_q_to_f16`
  already exists in our tree, design note §2) → the conversion's write is then
  the projection's own output write, cost 0 extra.
- **k/v**: `[Lk][Hk][D]` contiguous f16 is what our *dense* cache already is
  (`build_attn_mha` casts F32 to F16 at `llama-graph.cpp:3122-3128`, `code`), so
  for the dense path the buffer is the cache and there is no copy at all — one
  `ggml_view`-equivalent pointer. The **paged** cache and the q8_0 forms are
  the incompatible ones:
  - paged: cells are scattered over pages, so `[Lk, Hk, D]` as one run does not
    exist. A gather is required per launch: `Lk·Hk·D·2` bytes written. For the
    512-token case with `Lk = 1536`: 1536·8·128·2 = **3.15 MB per tensor**,
    6.29 MB for k+v per layer. At 32,768 kv: 32,768·8·128·2 = **67.1 MB per
    tensor**, 134.2 MB per layer; at 131,072: 268.4 MB per tensor, 536.9 MB per
    layer. With 4 global layers per 512-token prefill call: **536.9 MB** at 32k,
    **2.15 GB** at 128k (SWA layers stay at `Lk ≤ 1535`, 3.15 MB per tensor).
  - q8_0: a block is 34 B per 32 elements (d f16 + 32 int8) = 1.0625 B/elem
    against f16's 2 B/elem. Dequanting `Lk·Hk·D` elements reads
    `nelem·34/32` B and writes `nelem·2` B: for the case per tensor 1,671,168 B
    read + 3,145,728 B write = **4.82 MB** (×2 tensors = 9.64 MB); at 32,768
    kv: 35.6 MB read + 67.1 MB write = 102.7 MB per tensor. If the quantized
    cache is what serves decode, the prefill pays the dequant *and* keeps the
    quantized copy — the design note's 0037 measured quantized-to-f16 staging to
    be the winning route in our own kernels, so this is the same cost we already
    pay, only now outside the kernel.

## 3. The bias table: ours f32 `[24 heads, 1025 bins, rows]` head-fastest,
theirs bf16 `[Lq, Hq, E]` bin-fastest

Both are per-(query token, head) tables, so the *content* transfers; the order,
dtype and one bin differ.

- Order: ours is `table[(query·n_bins + bin)·n_head + head]` — for fixed
  `(query, head)` the bins sit `n_head·4 = 96 B` apart (`inferred` from the
  design note §4 layout warning, `code` for the shape). Theirs is
  `table[(query·Hq + head)·E + bin]` — one contiguous bf16 row per
  `(query, head)`. A transpose + dtype pass costs the whole slab:
  512·24·1025 = 12,595,200 elements; read f32 **50.38 MB**, write bf16
  **25.19 MB** = **75.6 MB of traffic per 512-token layer call**; at a 4,096
  token prefill: 100.8 M elements → 403.2 MB read + 201.4 MB write =
  **604.6 MB**. ×24 layers that is 1.8 GB at 512 tokens — larger than the
  table itself. Cheaper route: build the table in their order upstream. The
  table is a low-rank projection we compute in the graph (`attn_rel_bias ·
  Rcur`, design note §1, `code`); reordering the GEMM's output columns and
  writing bf16 makes the cost **0 extra** (the write we already do), at the
  price of departing from upstream PR 32740's `[n_head, n_bins, rows]` shape —
  an operator decision, not a free one.
- Bins: ours has 1,025 (bins 0…1,024, bin 1,024 = the padded zero-weight
  row); theirs clamps `dc = min(dist, E-1)` and applies bias only for
  `dist < E`. Passing **`E = 1025`** reproduces our semantics exactly: `dist =
  1,024` reads bin 1,024 through the clamp (the zero row), `dist ≥ 1,025` gets
  no bias, which equals our zero row. Table bytes at 512 rows: 25,190,400 B
  (1,025 bins) against 25,165,824 B (1,024) — **+0.098 %**.
- `-inf` mask entries: the kernel's `sv.merge(NEG, bad)` is its only `-inf`, so
  a bias entry that is `-inf`-like is not supported; our bins never carry one.

## 4. The sinks

Ours: `src[4]`, f32 **per head** `[n_head]`, and for Inkling 2 Mini the graph
folds the sink into the bias table and passes `fill(0, n_head)` (design note §1,
`code`). Theirs: f32 `[Lq, Hq]`, one logit per **(query token, head)**, used as
the initial `m` and `l`. A zero sink is per-head-constant, so `[Lq, Hq]` filled
with zero once at model load serves every layer and every launch: cost **one
49,152 B buffer per 512-token case** (512·24·4), written once, read by the
kernel — 0 per call. If the model's true per-token sink logits are fed instead:
4 B per (row, head) = 49 KB per 512 rows per layer, 1.2 MB per 24-layer call.
No change of meaning: their `(m,l)` seed is exactly our `exp(sink - m)` in the
denominator (design note §4.4).

## 5. What disappears

- The mask: **1.57 MB** for the case (1536·512·2). Their kernel needs none; if
  the mask exists only for causal+window, dropping it also drops its per-layer
  read traffic: `n_q·n_kv·2` B per layer = 1.57 MB per layer for the case,
  **134.2 MB per layer at 128k** (4 global layers → 536.9 MB per prefill call).
- The bins tensor: 1536·512·4 = **3.15 MB** for the case, 268.4 MB per batch at
  128k kv, and the gather that builds it (`get_rows`) — no longer needed.
- ALiBi, softcap, sinks-in-mask: the kernel has none of them, and Inkling asks
  for none (`max_bias == 0`, `logit_softcap == 0` — the design note's own gates,
  `code`). Any layer that does ask stays on our kernels.

## 6. Per-launch cost, summed

Case: `Lq 512, Hq 24, Hk 8, D 128, window 1024`, 512-token prefill chunk,
`Lk = 1536` (window-sized SWA: `window-1 + chunk = 1535`, rounded to 1536 here),
global layers 4 of 24.

| item | per layer | ×24 layers | global at 32k (×4) | global at 128k (×4) |
|---|---|---|---|---|
| q f16 pass (9.44 MB) | 9.44 MB | 226.5 MB | 9.44 MB | 9.44 MB |
| k/v gather, paged f16 (6.29 MB) | 6.29 MB | 151 MB | 134.2 MB | 536.9 MB |
| k/v dequant q8_0→f16 (9.64 MB) | 9.64 MB | 231 MB | 205.4 MB | 822 MB |
| table transpose + bf16 (75.6 MB) | 75.6 MB | 1.81 GB | 75.6 MB | 75.6 MB |
| sinks (0 for zero sinks) | 0 | 0 | 0 | 0 |
| mask skipped (saves) | −1.57 MB | −37.7 MB | −134.2 MB | −536.9 MB |
| bins skipped (saves) | −3.15 MB | −75.6 MB | −268.4 MB | −1.07 GB |

Two readings, both `inferred` from the arithmetic and owed a measurement on the
card:

1. **The table reformat dominates** at 512-token chunks (75.6 MB per layer,
   larger than the 25.2 MB table it produces). Building the table in the kernel's
   order at graph level removes it; keeping upstream's shape pays it per call.
2. **The paged gather dominates at depth** (536.9 MB per 128k prefill call at
   f16, 822 MB if the cache is q8_0). That is the reference's own recipe
   (`gather_f16` from the paged fp8 pool, `sgl_model.py:425-427`, `code`), so it
   is not a port defect — but it is the reason a paged Inkling prefill would
   stand at ~1 GB of copies per 512 tokens, and it prices the alternative:
   keeping a dense f16 cache for the 4 global layers (`Lk ≤ 128k` f16 K+V =
   536.9 MB resident) instead of gathering every call.

## 7. What remains, in order

1. Operator decision, item 3: table shape `[n_head, n_bins, rows]` (upstream,
   pays 75.6 MB per call) or `[rows, n_head, bins]` bf16 (free, departs from
   PR 32740's assert).
2. The graph must be able to name `qpos0`/`kpos0` per contiguous run, and split
   a call into runs when the rows are not one. That is a host-side
   `build_attn_mha` change, not a kernel change.
3. Then the arm here runs the same shapes and the answer-level KL gate
   (`docs/campaigns/inkling2-mini-b60.md`, on `qfndev`) decides.
4. `--dry` on the card first: build + kernel names only. Then the enqueue.
