# kv-streaming-design — adaptive KV streaming on the B60: a larger quant at 262,144 tokens

**Open.** Design note only (2026-11-14); no code. Charter decided by the
operator: serve a larger quant of the dense agent on the 24 GB card at the
same context by keeping part of the q8_0:q4_0 KV in VRAM and streaming the
rest from host RAM. This document is the recon, the plan and the handover for
the implementation rounds. It follows the campaign rules of
`docs/campaigns/README.md`: one lever, one gate, an evidence class on every
disposition, no operator-local names.

The prior read is `docs/campaigns/research-achyudh.md` §2; it is the survey,
this is the design. Where this document contradicts it, this one was written
against the fork's source lines named below.

## Charter

Today the agent serves Qwen3.8-27B `qwen35` (64 layers: 48 GDN + 16
attention, MTP layer; DESIGN §2 table, `code`) on one 24 GB card at
`--n-ctx 262144`, q8_0 K / q4_0 V. The three quants' file sizes from the
model source, `paper` (a served load must re-read them from /props'
`weights_bytes`): Q4_K_M 17.11 GB (today), UD-Q5_K_M 19.87 GB, UD-Q6_K
21.98 GB. The KV is 26 KiB a token over the 16 attention layers (4 KV heads
of 256, `code`/`measured-here`, `agent-context.md:33-35`), 6,656 MiB at
262,144 tokens.

The lever: the attention cache's tensors become a **pool of resident pages**
plus a **host store** of the rest; the attended pages that are not resident
are copied into two device slots beside the compute queue and read there
through a page table. VRAM freed by the smaller pool pays for Q5_K_M's
+2.76 GB (`paper`, nominal bits; a file size must be measured). At 4.0 GiB
resident the agent holds about 634 of 1,024 pages, 162k tokens; below that
depth nothing streams.

Out of scope: the A770 (1.83 GB/s through the chipset, `measured-here`,
`layer-split-two-cards.md:76-82`). The flag **refuses** on it (§4).

## Reference to follow

`~/src/achyudh-ref`, branch `feature/adaptive-kv-stream-rocm` at `b540a4b9b`,
commits `143b62090`..`a6a685ab4` on upstream `a37cc4610`. MIT.
- core: `src/llama-kv-stream.h` (264 lines), `src/llama-kv-stream.cpp` (728);
- the paged FA kernels: `ggml/src/ggml-cuda/fattn-paged-decode-kernel.cuh`,
  `fattn-paged-decode.cu`, `fattn.cu`;
- its measured effect is `paper` for us (one user, one RDNA4 card, its own
  launcher): 28.4 t/s plain decode at 128k against 10.5 at 256k without
  streaming, 80.9 and 46.6 with MTP + n-gram; 607 pages x 16 layers streamed
  a step at 210k, "bound by the copies" at 74-81 ms.

Everything below marked `code` in the fork was read at source on 2026-11-13/14.
Nothing of the fork was built or run here.

## Gate

- **Answer-level bar** (`CLAUDE.md`, DESIGN §3.4 as amended 2026-10-01): the
  facts, the needle and the task battery stay right; mean KL against the
  reference no more than 0.03 nats worse than the unpaged arm on the same
  card and window; argmax agreement down by at most 1 point. Byte identity of
  *output* is not required. **Integrity of copied data stays exact**: a page
  written back and read again must be bit-identical to what was written (a
  unit record, §5 I2).
- **Speed, per phase**: decode at depth (the phase the lever targets) must
  not lose more than the run-to-run spread against the unpaged arm in the same
  window; prefill likewise. The first number to publish is the **served**
  agent leg at 262k with MTP 5 on the target quant, against the Q4_K_M
  unpaged arm, with the copy queue's busy share and the input wait reported.
- **A red case per increment** that can fail (§5). A green that was green
  before the change measures nothing.

## 1. The fork's mechanism, step by step (`code`)

### 1.1 Pages and the pool

- One sequence's 16 attention layers' K/V are **pages of 256 cells**
  (`llama-kv-stream.h:56`). A cell is one token; a page's rows are
  `[n_embd_head, 256, n_head_kv, 1]` in the cache's own layout, and the pool
  tensor of a layer is `[n_embd_gqa, n_slots*page_size]` (`h:78-80`). The
  page's row layout equals the ring's (`fattn-paged-decode-kernel.cuh:44-46`).
- Host store: pinned host memory, per layer K pages then V pages in page order
  (`llama-kv-stream.cpp:206`, `host()`; `host_buf`/`host_offs`, `h:220-224`).
- Bookkeeping: `page_slot[n_pages]` (pool slot or -1), `slot_page[n_slots]`,
  `page_ring[n_pages]` (ring index in the current ubatch), `dirty`, `saved`
  (`h:227-231`); a `layout_id` bumped whenever a page moves so graphs built
  around the layout are not reused (`h:143-144`, `llama-graph.cpp:607-611`).

### 1.2 Placement, eviction, write-back (`prepare`)

`prepare(cells, n_kv)` (`cpp:271-393`) runs before a ubatch:
- the pages the ubatch writes are made resident first; if no slot is free the
  **oldest page** that is neither written nor attended is evicted, and the
  pool asserts it is at least as large as a ubatch (`cpp:334-341`);
- free slots then take the newest attended pages (`cpp:343-349`);
- written pages are marked dirty (`cpp:351-353`).
- `assign` (`cpp:223-240`): a page returning to the pool is copied from the
  host store with `ggml_backend_tensor_set_async` on the **compute stream**, so
  it is ordered with the graphs that read the pool.
- `evict` (`cpp:242-270`): a dirty page is copied back to the host store with
  `ggml_backend_tensor_get_async` on the compute stream; a copy event is
  recorded (`wb_pending`) so the copy engine's reads of that page wait for the
  write-back, and the copies are ordered against the graph that wrote it.

### 1.3 Streaming a ubatch: two slots, a page table, a copier and a publisher

- Attended, non-resident pages **stream in every ubatch, decode included**
  (`h:15-19`, `cpp:355-393`). They are packed at the start of the layer's
  **device slot**, one of **two** slots, chosen by attention-layer index
  (`ikv % 2`, `h:134-135`, `cpp:249`); resident pages stay in the pool.
- The page table input is `I32 [n_kv/256, n_layer]` (`h:99-101`,
  `cpp:586-599`, built at `llama-kv-cache.cpp:1630`): row per layer, entry
  `>= 0` is the pool page, entry `< 0` is ring page `-1-entry`.
- `prepare` turns the streamed pages into `run`s of consecutive pages that are
  consecutive in source and destination (`cpp:357-374`), fills `page_ring`,
  and queues one **job** to the copier: the runs, the write-back event, the
  ubatch tag, and whether it is a decode-sized ubatch (`cpp:379-391`).
- **Copier thread** (`cpp:422-540`): takes a job, waits for its write-backs
  (`cpp:465-468`), then for every layer: waits until the slot's previous
  reader released it (`consumed >= that layer's tag`, `cpp:430-434`), waits
  for a free copy event (a ring of 64, `h:264`), queues K then V copies of
  each run on the copy engine into the slot (`cpp:441-445`), records an event,
  remembers `{layer, tag}` against it (`spans`, `cpp:447-452`). It reads the
  slot's release under the job mutex so the publisher cannot miss it.
- **Publisher thread** (`cpp:540-566`): spins briefly, then `ce_iface->done`
  for the event, and stores `ready[layer] = tag` with release ordering.
- **Kernel-side wait**: the attention waits for `ready >= tag` and writes
  `consumed = tag` when no block needs the ring any more
  (`fattn-paged-decode-kernel.cuh:71-96`: an acquire load loop on a
  system-scope atomic; `:200-207` waits once per block, only for blocks whose
  chunks reach a ring page; the store is in the combine, `:422`). Flags are
  monotonic and compared with `>=`; the tag is a **graph input** in device
  memory, so a captured graph serves every step (`cuh:73-74`).
- The graph hook (`llama-graph.cpp:3236-3328`): `ggml_flash_attn_ext_set_pages`
  (table, page size, ring K/V views) and `ggml_flash_attn_ext_set_sync` (tag,
  ready, consumed) are set on the FA node after it was expanded; the extra
  inputs are expanded into the graph by hand.

### 1.4 The kernels

- **Paged decode** (`fattn-paged-decode.cu:39-54` gate; kernel
  `fattn-paged-decode-kernel.cuh`): Q8_0 K / Q4_0 V, head 256, GQA-packed,
  6-8 rows one tile, larger ones two (`:17`); up to **96 query tokens**
  (`llama-kv-stream.h:64-70`; 2.3x the stock kernel at 17 rows, 1.2x at 65 at
  64k, slower from 128). Chunked over KV with split-K partials merged by
  log-sum-exp (`cu:205-229`); a fully masked row's combine is 0 (`801b3ddee`).
- **Paged prefill** (`9e9e46f86`): the **page runs** are attended as chunks by
  the stock attention, each writing its per-row log-sum-exp, merged afterwards
  (`cu:142-157`, `cu:170-196`; `fattn.cu:786-790`). Pages a prefill can also
  take the paged decode kernel when they fit its row tiles.
- **Prefetch** (`5de600d4b`, `cpp:490-530`): after a decode ubatch, if no job
  is queued, the copier copies the **first two layers** of the next ubatch into
  the two slots; the publisher does not publish them; the next job claims them
  only if it is the ubatch right after and streams the **same copies**
  (`cpp:471-486`), otherwise they are copied again over the prefetched data
  (`+6.5 %` at 210k, `paper`).
- A fuller batch than 96 tokens with no paged prefill attends the slot as a
  **contiguous copy of the whole layer**: resident pages by `ggml_cpy` nodes in
  the graph, streamed pages by the copy engine to their logical place
  (`llama-kv-cache.cpp:1830-1850`).

### 1.5 Two phase budgets, the failure rule, the copy engine

- `--kv-resident-mib` sizes the prefill pool; `--kv-resident-mib-decode` the
  pool plus slots between prefills (`cb62702a4`, `fe5f9b999`; flags at
  `llama-context.cpp:376-377`). The swap is ordered so that both never need
  the memory: the pool shrinks before the prefill's compute buffers come back,
  and grows after they shrink (`llama-context.cpp:2028-2042`); graphs whose
  views point into the old pool are rebuilt (`:2042-2048`).
- **A pool that does not fit fails the batch, not the process**
  (`8facec115`): `llama_decode` returns **-2** with the budget logged, nothing
  of the batch ran, and the cache keeps its pages in host memory
  (`llama-context.cpp:2054-2057`).
- The copy engine (`bd63910e1`, `ggml-copy-engine.h`, `copy-engine.cu`) is the
  SDMA engine through ROCr, created when either phase streams (`19bf02286`).
  HIP's own queue would block behind a readback that waits on kernels.
- Every wait is bounded: a stall logs the queue state and aborts
  (`cpp:395-420`); the attention's waits trap after 60 s (`h:30-34`).
- State save/restore goes through `get_rows`/`set_rows`, reading wherever the
  rows live, with the backend idle (`cpp:678-726`).

## 2. arcint's counterpart per step (OpenCL, B60)

Evidence classes: arcint machinery below is `code` in
`contrib/llama.cpp/patches/`, `measured-here` where the README names a number.

| fork step | arcint's machinery | deviation |
|---|---|---|
| host store: pinned host pages | **USM host memory** (`clHostMemAllocINTEL`), the bank's buffer type: `0000-02:715`, `:1329`; a `cl_mem` over host memory is migrated to the card by the driver (`measured-here`, `flash-next-llama-engine.md:155-160`), so the host store is a USM buffer, read/copied by pointer | memory source |
| copies beside compute: SDMA engine through ROCr | **a second in-order OpenCL command queue** (`bs.q = clCreateCommandQueue`, `0000-03:1649`), copies `clEnqueueCopyBuffer`, one event per ring slot (`0000-03:840`), the form measured at 2 GiB / 150 ms = 14.3 GB/s under a kernel on queue 1 (README 0032) | the queue shares the engine with graph input writes (§2.1) |
| placement in the graph: `set_async` on the compute stream | the compute queue is in order; page moves go on it as tensor writes (`set_tensor_async`) as in the fork, so no extra sync is needed there | — |
| kernel waits on a pinned `ready` flag | **0056's doorbell**: an uncached load loop in the kernel (`__builtin_IB_lsc_load_global_uint`, `MISS_L1UC_L3UC`), `0000-03:1262-1311`, measured on the B60 (README 0056) | sync form: the fork's HIP system-scope atomic becomes an L1/L3-uncached load; an event wait would stall host enqueue between layers, so it is not taken |
| publisher thread sets `ready` | the copy queue's events are checked by a publisher thread with `clGetEventInfo` (`0000-03:866`, the `cl_event evt` form) and host flags are set with a release store in USM host memory | — |
| page table `I32 [n_pages, n_layer]` | one USM/cl_mem `I32` tensor per layer, uploaded once per ubatch **behind the graph** (0042's rule), or as a graph input if the backend keeps it | upload cost 1024x16x4 B = 64 KiB, `inferred` negligible |
| two slots per layer, alternating | the same: K and V slot tensors `[n_embd_head, n_ring*256, n_head_kv, 1]` per layer parity, `n_ring` sized to the streamed pages (max at 262k: 390 pages, 416 KiB a page in K+V, 155 MiB a slot, `inferred` from the geometry) | slot count 2 as the fork |
| copies are whole-tensor rings (0032) | 0032's ring copies a whole bank tensor; the KV analogue is the **page run** copy: `clEnqueueCopyBuffer` per run, K then V, per layer (README 0032's order/event discipline; 0044's `min-tok` threshold does not apply, pages are multi-MiB) | batch size: a run is up to the streamed pages, not one 187.7 MiB tensor |
| prefill reads runs, chunk + LSE merge | arcint's prompt kernels already take `kv0`/a key range and 0037 carries the softmax state between blocks (`0000-04:637-700`, `:823-840`); the merge exists (`flash_attn_f32_merge`, `0000-04:2273`) | the chunk is 0037's 16,384-key block, not the fork's `pd_prefill_chunk_pages` |
| two phase budgets (`kv_resident_size*`) | **0068's slot loan** is arcint's phase-wise lend (compute buffers vs cache slots, README 0068:4013); the KV stream takes the same pattern: pool sized to the phase, graphs rebuilt on a phase switch | the loan is a separate mechanism; the design reuses its reserve/refusals, not its code |
| pool does not fit: batch fails (-2) | arcint's `llama_decode` return path and backend loop must carry **-2** the same way (a refusal, no partial run); the server reports it like §3.8's 400 family, not as an exception | — (invariant §3.8: no truncation) |

### 2.1 Deviations that rest on an invariant

- **§3.8 (no truncation, no shift):** the pool may be smaller than the
  context, so the *logical* context stays 262,144; a page that does not fit
  streams, and a pool that cannot be allocated **fails the batch**, never
  shortens `n_ctx`, shifts, or drops cells. This is why the fork's failure
  rule is adopted rather than a fallback to a shorter context.
- **§3.4 (answer-level correctness; copied data exact):** the page host store
  may hold every attended page; a write-back and a read-back must be exact,
  and the round that streams a page must pass the answer-level bar against the
  unpaged arm. No bit-identity of outputs is claimed or needed.
- **0042's placement rule** is not an invariant but a measured one
  (`measured-here`): copies enqueued *before* the next graph delay its input
  writes by 2.4-5.9 ms and lose ~1.5 % decode; placed **right after** a graph
  they gain 2.4 %. The stream's copies and prefetch follow it (README 0042).
- **Batch-size deviation:** the fork's paged decode takes up to 96 query
  tokens; arcint's verify is 6 rows (MTP 5) and its prefill ubatch is 512
  (agent: 2,048 for Flash-Next). So the decode kernel's row tiles serve 6 rows
  first; a 96-row path is not needed for this lever and is not built here.
- **Alternative form, not chosen:** the attention kernel reading a missed page
  in USM host memory **in place** (as the expert cache's `BANK_DIRECT` does,
  patch 0022, `measured-here` 26.0 -> 28.2 t/s) instead of copying to a slot.
  A decode attention reads each streamed byte once, so the link rate bounds
  either (`inferred`). Copied-into-slots is chosen because 0032's queue/event
  form and the kernel's coalesced reads are the measured pair; the in-place
  form stays a later A/B if the copy queue's wakeups prove the cost.

## 3. The FA kernels that learn a page table

All in `contrib/llama.cpp/patches/0000-04-attention-at-depth.patch` (folded
series; `FOLDED.md` maps old numbers). A770 kernels (0015's
`flash_attn_*_q1_split_intel`, `0000-04:2418-2534`, 0039-0041) are out of
scope: the flag refuses on the A770 (§4).

**The addressing change, once, for all of them.** Today a kernel reads a K row
at `k_offset + key*k_nb1 + h*k_nb2 + b*k_nb3` (e.g.
`flash_attn_gqa_dpas`, `0000-04:2705-2720`; the f16 conversion kernel,
`0000-04:1242-1247`). Paged form (the fork's rule, `cuh:44-60`): for logical
key `t`, `e = table[t/256]`; source = `e >= 0 ? pool : ring`, row =
`(e >= 0 ? e : -1-e)*256 + t%256`; the pool and the ring share `nb1/nb2`.
The kernel gains a page-table pointer, the ring pointer with its strides, and
`page_size`. **Built in 0122** as one helper with a `table == 0` runtime check
rather than the fork's `PAGED` compile switch (one build per type pair; an
unpaged call pays one uniform compare per key row; the cost is measured on the
card, not inferred here).

1. **0016, `flash_attn_gqa_dpas.cl`** (`0000-04:2705`): a work-group per (KV
   head, split), up to 8 (head, token) rows, 512-key splits of the KV head —
   the decode and MTP-verify kernel. Its three load sites are K staged
   (`GQA_K_STAGED` local `ks[]`), K direct, V staged — each resolves the row
   from the table/ring. The split loop already walks keys in a stride
   (`kv_per_split`), so a ring page inside a split is a base change per 256-key
   page; the split's partial merge needs nothing (`flash_attn_f32_merge`).
   Gate: Q8_0 K, Q4_0 V, head 256, rows 1-6, keys to 262k. A ring page that
   begins mid-split is the red-prone case.
2. **0088, the int8 Q Kᵀ inside the same kernel** (`0000-04:1037`): the
   `GQA_I8 && KT == 1` path stages the raw q8_0 rows into local `ks8` and
   builds the DPAS a-operand from them. The staging load is the same row
   resolution; the int8 product itself is untouched. It must be in the same
   increment as 0016, because the default build takes it.
3. **0037, `flash_attn_kv_to_f16` + the carry path**
   (`0000-04:1242-1267`, `:823-840`): converts blocks of 16,384 keys from the
   cache's view into two 32 MiB scratch buffers, then runs the f16 DPAS
   kernel with `kv0` and carried state. The conversion kernel takes the table
   and ring, and the block's logical range stays `[kv0, kv0+nb)`. This is also
   the natural place for the **prefill** path: blocks that are fully streamed
   read the ring, resident ones the pool. The block size is a compile constant
   (`GGML_OPENCL_FA_DPAS_F16KV_BLOCK`), unchanged.
4. **0038, `flash_attn_dpas2.cl`** (`0000-04:1951`): K by 16-byte 2D block
   reads and V by VNNI-transforming 2D reads from a surface with a pitch
   (`GGML_OPENCL_FA_DPAS2`). A page table breaks a 2D read when a run of
   contiguous key rows crosses a page boundary. Two changes, and the first is
   preferred: (a) **page runs** — the host passes the attended pages as runs of
   contiguous pages (`ggml_fa_page_runs`, the fork's form), and the kernel is
   called per run, or per chunk of a run, with its own `kv0`, merged by
   log-sum-exp as in 0037's carry (the fork's `9e9e46f86` mechanism). Then the
   2D reads stay inside one surface and the pitch is the run's. (b) a per-tile
   base/pitch selection if a tile would straddle a boundary — only if (a) does
   not fit the geometry. Do not do both.
5. **0007's split kernel, rows 1-3** (`0000-04` split form, `GGML_OPENCL_FA_GQA_MIN_ROWS`
   default 4, dispatch `0000-04:947-1000`): with streaming on it would read
   the pool/ring without a table. Two acceptable dispositions, one to be
   chosen at build: route 1-3 rows to the paged 0016 (0089's staged form
   exists, `GGML_OPENCL_FA_GQA_MIN_ROWS=1`), or refuse a streamed 1-3-row
   batch. The first is preferred; a 6-row MTP verify never needs it.

What must NOT change: the mask stays full `[n_kv, n_tokens]` f16 (0038's
mask classes, README 0038); this lever does not touch it.

## 4. Hooks in libllama and arcint's backend

### 4.1 ggml (op API, CPU, tests)

Mirror the fork's API so the backends and a CPU reference share one op:
- `ggml_flash_attn_ext_set_pages` / `_set_page_runs` / `_set_sync` and their
  src slots (`ggml.h:2528-2560`), setters in `ggml.c:5610-5660`, getters in
  `ggml-impl.h:211-215`; `GGML_FA_SRC_PAGES`/`_RING_K`/`_RING_V`/`_TAG` are
  appended after the mask src so existing kernels' indices do not move.
- **CPU**: `ggml_compute_forward_flash_attn_ext_paged`
  (`ggml-cpu/ops.cpp:9685-9811`): wait on `ready` if a tag is set, gather the
  logical rows into the work buffer, run the existing contiguous path, then
  store `consumed`. The gather is a plain copy, so the result is **bit-identical**
  to the unpaged op on the same bytes — the red/green reference (§5 I1).
- **test-backend-ops**: the fork's `test_flash_attn_ext_paged` case
  (`1f6e1fccc` test diff): pool with shuffled pages and unused gaps, a ring
  every `ring_every` pages, two table rows (row 0 garbage), NMSE 5e-4.

### 4.2 libllama (KV cache, graph inputs)

- A `llama_kv_stream` class (fork `llama-kv-stream.h/.cpp`) owned by the
  attention cache: pool tensors of a layer `[n_embd_gqa, n_slots*256]`, USM
  host store, the two slot tensors, the table input, the flags.
- `llama_kv_cache::resize_kv_stream(size, decode, n_tokens)` (fork
  `llama-kv-cache.cpp:1663-1780`), `kv_stream_pages_next`, and the pool view
  builders replacing `get_k`/`get_v`'s offsets (`0000-08:3911-3934` shows how
  arcint's KV views are built today — `sinfo.kv_off` and `sinfo.s0`; the pool
  view keeps the same row layout).
- Graph inputs: `llm_graph_input_attn_kv` gains `self_kv_pages` and
  `self_kv_tag` (fork `llama-graph.cpp:3236-3300`); `build_attn` sets the op
  params on the FA node and expands the extra inputs
  (`llama-graph.cpp:3325-3336`). `can_reuse_kv_stream` gates graph reuse on
  `layout_id` and on the batch ≤ the decode row cap (fork
  `llama-graph.cpp:607-611`) — arcint's scheduler reuse check needs the same
  line.
- `llama_context_params`: `kv_resident_size`, `kv_resident_size_decode`
  (fork `llama-context.cpp:376-377`), and the phase switch in `decode`
  (`:2033-2057`) returning **-2** when a pool does not fit. Unchanged:
  `type_k`/`type_v` are still the q8_0/q4_0 that the kernels read.
- State save/restore (`get_rows`/`set_rows`) reads wherever the rows live;
  arcint's checkpoint path (`backend_llama.cpp:2660` area) does not see pages.

### 4.3 arcint's side: flags, refusal, /props, logging

Style follows the expert cache and the slot loan: a compile define, env
handoff, a refusal at start, a line under /props' downgrades, no silent
fallback.

- Compile define `ARCINT_LLAMA_KV_STREAM=1` next to the others
  (`CMakeLists.txt:206-218`); the code compiles only where the patch series
  provides the class.
- Config, proposed names: `--llama-kv-resident MIB` (prefill pool) and
  `--llama-kv-resident-decode MIB` (decode budget; absent = one budget both
  phases), both 0 = off; carried in `Config` beside `llama_kv_k/llama_kv_v`
  (`src/config.h:126-127`) and parsed like `--llama-kv` (`config.cpp:637-644`).
- Handoff: `setenv("LLAMA_KV_RESIDENT_MIB", ...)` in the same place the
  expert cache's env is set (`backend_llama.cpp:553-561`).
- **A start that cannot honour the flag is refused** (`throw
  std::runtime_error`, before the load, the Dory refusal's pattern
  `backend_llama.cpp:661-680`). Refusals, each with its own message:
  - the build lacks the patch, or the flag is set while the card backend is
    not OpenCL;
  - flash attention resolves OFF (fork refuses exactly this:
    `llama-context.cpp:378-382`); arcint's `cp.flash_attn_type = AUTO`
    (`backend_llama.cpp:934`) must resolve to enabled or the start fails;
  - the device is the A770 (1.83 GB/s; `GPU.N` identity by PCI id, per the
    SOP) — this lever is B60-only;
  - K/V types other than q8_0/q4_0 (the paged kernels' pair);
  - the pool exceeds what remains after the weights on the card by the
    allocator's margin — read from the device's free bytes, not a guess;
  - `--llama-kv-resident` larger than `--n-ctx`'s KV (nothing would stream:
    accepted, logged, not refused);
  - with the MTP drafter: supported (6-row verify is the paged kernel's row
    tile), but the draft context's KV stays unpaged (a separate context).
- **The paged op's fallback, and where I5 stops it.** ggml's scheduler places a
  node on any backend whose `supports_op` says true. 0119 makes the OpenCL
  backend **decline** a `FLASH_ATTN_EXT` carrying a page table or a sync tag
  (its kernels do not take pages until I3), so a graph with pages would be
  placed on the **CPU reference** -- correct, and for I1's tests the intended
  arm; for serving it would be a silent collapse, one host thread gathering
  every page. I5's start refusal is what catches it:
  1. before the load, the backend probes exactly what the fork's
     `set_backend` probes -- one `FLASH_ATTN_EXT` node with a 1-page table,
     the pool views and a dummy run, through
     `ggml_backend_supports_op(graph_backend, probe)` -- and caches the answer
     (`prefill_paged` / `decode_paged` in the fork, `llama-context.cpp:156-199`);
  2. `--llama-kv-resident` is honoured only when the node's owning backend
     says true for the **paged decode shape** (head 256, q8_0/q4_0, 6 rows);
     otherwise the start is **refused** with a message naming what is missing
     (the build's patch, or the kernels of I3), and no session runs. The flag
     never degrades to the CPU reference;
  3. the same probe result is stored in `status_` and shown under /props
     (`kv_stream.deferred = "cpu-reference: no paged kernel"` when it would
     have fallen back), so a build that cannot serve the flag says so before a
     request.
  **Built in I5**: the rules are `src/exec/kv_stream_refusal.h`, driven by the
  enabled cell `tests/test_kv_stream_refusal.cpp` (ctest `kv-stream-refusal`);
  its red arm is `ARCINT_KV_STREAM_REFUSE_OFF=1` (branch `kv-stream-red5`,
  `381660e9`), which accepts an ask the card cannot honour and fails every
  refusal check.
- **/props**: a `kv_stream` object, `phase`, `pool_mib`, `slots_mib`,
  `resident_pages`, `context_pages`, `streamed_pages_last_round`, and
  `downgrades()` gains `kv-stream:<n> pages read from host` so nothing is
  silent; the accessor sits beside `slot_loan_state()`
  (`backend_llama.cpp:1535-1551`), read under `mu_`.
- **Logging**: one INFO line at the load with the phase budgets, the pages,
  the ring pages and the link arithmetic; one WARN the first time a ubatch
  streams, with the page count and MiB; a per-round DEBUG line (tag, pages,
  MiB, copy queue busy ms) behind an env switch; a stall logs the queue state
  and aborts as the fork's `wait_for` (`cpp:395-420`) does.
- **Failure rule**: a batch that cannot be placed returns -2; the server maps
  it to a 400 with the numbers (DESIGN §3.8's family), not a 500 and never a
  truncated context.

## 5. Increments, each independently testable

Every increment is a small commit on its own branch, red case first, with the
CPU test where possible; the card commands are for Claude in an announced
window (SOP `docs/sop-card-window.md`: sampler first, card identity by PCI id,
zombie sweep by pid; never run outside the window).

### I1 — ggml op plumbing + the CPU paged reference

- API and srcs (§4.1), the CPU gather path, the backend-support predicate
  (`supports_op` false unless a backend declares paged FA, so an unported
  backend refuses instead of reading garbage — no silent fallback).
- **Device-free red/green**: `tests/` case that builds one pool with shuffled
  pages + a ring, computes the same Q/K/V twice — paged via the table and
  unpaged as a contiguous tensor — and requires **bit-identical** f32 output
  (a `memcmp` on the f32 dst, not an NMSE: the gather path is a copy, so
  equality is expected). Red mode: one table entry shifted by a page, or the
  ring order reversed; the difference must show. A second case: an entry
  `-1-e` in a ring every other page.
- Card: none. This increment never touches a GPU.

### I2 — the page map, the host store, eviction/write-back (device-free + CPU backend)

- The stream class's placement logic with a **malloc host store and no
  backend** (no copies), used first by the CPU backend: pool, slots, dirty,
  saved, `prepare`'s eviction of the oldest page and the written-page rule,
  the table, the runs, a rotating debug mode (the fork's
  `LLAMA_KV_STREAM_DEBUG=rotate`, `h:47-49`) that forces every page through
  the host store.
- **Red/green**: a unit test of the map (write-back and re-assign freshness,
  an eviction of a written page must be a write-back, a page re-assigned must
  be read back; the red mode evicts without the write-back and the test
  fails). Then a **CPU-backend served leg** (the ggml CPU backend, `arcint`
  with the card backend absent) at 8k/32k with the stream on: answers right,
  and the paged output **bit-identical** to the stream off, because the CPU
  path gathers exact bytes.
- Card: none.
- **Built in 0120** (`measured-here`, CPU; recipe green `99d94fe0` 38 of 38,
  EXIT 0; red tree `5a01de69` 38 with 1 failure, EXIT 8): the class has the page map, the host
  store, write-back/read-back, the table, the runs, the rotation debug switch
  and `get_rows`/`set_rows`; its cell is `kv-stream-map` (part 1 the map, part 2
  four ubatches through 0119's paged attention, `memcmp`-equal to the contiguous
  cache; the fourth ubatch streams 2 of 4 pages). Red `kv-stream-red2`: `evict`
  without the write-back, 6 checks failing.
- **Deviations from the fork, named:** no copier/publisher threads, no copy
  engine, no ready/consumed flags, no prefetch -- page moves are synchronous
  `ggml_backend_tensor_set/get` on the owning backend (I3/I5 move them to the
  second in-order queue with events, and add the kernel-side wait); slots are
  sized by the caller, not by a backend probe or the phase's streamed pages
  (I3); no phase budgets or pool resize (I5); and **no `llama_kv_cache`
  wiring** in this increment: the pool views, the graph inputs and the prepare
  call are I3's prerequisite. The reason it is not here: arcint's llama engine
  constructs its OpenCL cards at start and throws without one
  (`backend_llama.cpp:505-520`, `code`), so a CPU-backend served leg does not
  exist in this tree; the CPU arm is the four-ubatch cell above, which is the
  path the design's I2 asks to be byte-identical. The served leg at 8k/32k
  moves to I5's window as part of its gate.

### I3 — OpenCL paged decode: 0016 and 0088

- The `PAGED` variants, the table/ring args, the wait/consumed flags, the
  dispatch gate (only q8_0/q4_0, head 256, rows 1-6 on the B60 when the flag
  is on; otherwise the existing kernel as before).
- **Device-free**: the kernel-source compile check (ocloc/IGC on the recipe
  host) and the arg-count unit; the CPU reference stays I1's.
- **Built in 0122/0123** (`code`; the recipe host compiles the OpenCL C++ only,
  the kernel text had no `ocloc`/driver there). The kernel resolves rows in its
  five K/V load sites (staging, 0089's int8 staged rows, 0088's int8 Q K^T
  direct and staged b operands, the V staging); a paged op's attended length is
  `pages->ne[0]*page_size`, it routes to this kernel from 1 row, and
  `supports_op` accepts only head 256, the four type pairs, GQA rows <= 64, a
  full mask and whole-page rings; anything else (a sync tag included) is
  declined. Paged `test-backend-ops` cases: 10 of 10 OK on the CPU
  (`measured-here`, the case plumbing; no card).
- **Card (Claude)**, in one window, B60, with 0122/0123 applied:
  ```
  cmake --build build --target test-backend-ops
  ./build/bin/test-backend-ops test -o FLASH_ATTN_EXT -p 'n_slots=' -b OpenCL0
  ./build/bin/test-backend-ops perf -o FLASH_ATTN_EXT -p 'n_slots=' -b OpenCL0
  ./build/bin/test-backend-ops test -o FLASH_ATTN_EXT -p 'kv=8192,nb=6' -b OpenCL0
  ```
  The first is the 10 paged cases (1k/8k/65k/262k keys, 1 and 6 rows, pools
  half the context or 4 pages all-ring, f16 and q8_0:q4_0); the second prices
  them; the third the unpaged twin at 8,192/6 rows for the A/B. A red arm flips
  one ring entry's sign in the case's `initialize_tensors` (a private tree):
  the paged cases must fail. Report whether the kernel builds at all, and
  paged against unpaged ms at each depth; a table load and one uniform compare
  per key row are the only kernel changes.
- Served leg: the agent at 131k, stream on against off in one window, MTP 5,
  answers + the capital + the needle; prefill and decode both.

### I4 — paged prefill: 0037 and 0038 by page runs

- `flash_attn_kv_to_f16` takes the table/ring; the prefill calls per run (or
  per block) with `kv0`, merged by the existing `flash_attn_f32_merge`; 0038
  gets the run/pitch form (§3.4(a)), not a per-tile table.
- **Card**: `test-backend-ops test -o FLASH_ATTN_EXT` with the 512-row
  prompt cases (f16 and q8_0:q4_0, 8k/65k) — green; the red per I3. Then a
  prefill leg at 8k/58k with stream on against off: the needle + KL (16 x 512
  against the unpaged arm), prefill t/s.
- CPU check: a 512-row paged prefill case on the CPU backend, bit-identical
  to the unpaged output.

### I5 — budgets, refusal, /props, prefetch; the served agent

**Built this round, and what is deferred, named:**
- `--llama-kv-resident MIB` (arcint's config → `cparams.kv_resident_size`),
  the start refusal (`src/exec/kv_stream_refusal.h`: flash attention, the 24 GB
  card, the device's paged-`supports_op` probe, `--llama-kv q8_0:q4_0`), the
  `/props` `kv_stream` object, and the table's device upload (0123z: the used
  row to a device buffer on the compute queue, released after the queued
  command). The refusal cell is enabled (`kv-stream-refusal`) with its red arm
  `ARCINT_KV_STREAM_REFUSE_OFF=1`.
- **Deferred, and why:** `--llama-kv-resident-decode` is parsed and **refused**
  at start (the phase resize/re-reserve is not built; the single budget
  serves); the copier/publisher threads and the ready/consumed flags are not
  built -- the stream's page moves are `ggml_backend_tensor_set/get` on the
  in-order compute queue, which is **correct and unhidden** (0032's second
  queue, 0042's after-graph placement and the prefetch are the overlap, and
  they need a card to validate); the -2 pool-does-not-fit path is the cache's
  construction refusal plus the prepare assert (a phase resize can be refused
  only after I5's queue). A start that asks for the decode budget or a device
  without paged FA is refused, not silently degraded.

- The per-phase budgets and the -2 path (§4.2), the prefetch (fork
  `cpp:490-530`) placed **after** the graph (0042), the refusal set (§4.3),
  the /props line and the logs.

- **Card (Claude), the I5 gate, one announced window, B60, SOP first:** two
  arms, one process an arm, the same requests. Baseline: the agent on
  UD-Q4_K_M, unpaged (`--n-ctx 262144 --llama-kv q8_0:q4_0 --llama-mtp 5`,
  no `--llama-kv-resident`). Candidate: UD-Q5_K_M with
  `--llama-kv-resident 4096` (the pool at half the context, ring every 2).
  ```
  arcint --model <...UD-Q4_K_M.gguf> --n-ctx 262144 --llama-kv q8_0:q4_0 --llama-mtp 5 ...
  arcint --model <...UD-Q5_K_M.gguf> --n-ctx 262144 --llama-kv q8_0:q4_0 --llama-mtp 5 \
         --llama-kv-resident 4096 ...
  ```
  Per arm: `/props`'s `kv_stream` (`requested`, `resident_mib`, `paged_fa`);
  the capital; a needle at ~250k; twelve 300-token answers; decode t/s at
  32k, 128k and 250k depth; prefill t/s at both depths; peak VRAM and GTT from
  the sampler (per the SOP's 2 s recording); the copy queue's busy share
  (`GGML_SCHED_TIMING`) **only once I5's queue lands** — today the moves are on
  the compute queue, so the gate's shape is correctness and the wall, not
  overlap. Red: the same candidate with `ARCINT_KV_STREAM_REFUSE_OFF=1` is
  accepted at start and then fails at the cache's construction (the shapes) --
  it must not serve.
- **Device-free red/green**: the refusal matrix as a unit (each arm's message;
  red: a flag accepted by a build that cannot honour it must fail — that is
  the red case) and the -2 mapping (unit test on the decode return path, no
  card).
- **Card, the gate leg**: the agent, Q5_K_M (19.87 GB) and Q6_K (21.98 GB)
  against Q4_K_M unpaged, all at `--n-ctx 262144`, q8_0:q4_0, MTP 5; the
  needle at ~40 % of a ~255k prompt, the capital, the task battery, KL against
  the unpaged arm; prefill and decode t/s; peak VRAM, GTT, the copy queue's
  busy share, the verify's input wait (`GGML_SCHED_TIMING`); one process an
  arm, A B A B where the phase allows. Report the pool MiB, pages, ring pages,
  streamed MiB per round and the card identity.

## 6. Expected cost, and what must be measured

- **Link**: 14.3 GB/s measured on the B60 (README 0032's 2 GiB / 150 ms probe,
  `measured-here`). **75 ms a streamed GiB a round** (`inferred`: 1.074e9 /
  14.3e9 s; 187 ms for 2.5 GiB).
- **At the agent's shape** (`inferred` from the geometry, 26 KiB a token,
  416 KiB a page over K+V, 16 layers): 1,024 pages at 262,144 tokens;
  4.0 GiB resident = 634 pages, 162k tokens. Above that depth 390 pages
  stream a round = 2.5 GiB = 161 MB a layer (the review's 111 MB/layer at
  210k with a 215-page pool is smaller because its resident set is larger).
- **Overlap**: 1.6 GiB could hide under a 115k 6-row verify (124.4 ms,
  `measured-here`, README 0091 window 61) — `inferred`, and the 262k verify
  is longer but not measured. The lever therefore has three outcomes: fully
  hidden (no decode loss, the quant fits), partly hidden (+ the difference
  a round), or not hidden (the full 187 ms, a decode collapse at depth). The
  numbers decide which, not the design.
- **Prefill**: a 512-token ubatch's compute over a 255k prefill averages
  ~1.5 s (348.6 t/s, README 0091), so a ubatch's 2.5 GiB of copies would hide
  (`inferred`); the first chunk's pages are written and resident, so streaming
  starts with the second.
- **What must be measured, in one window, before the gate leg:**
  1. the copy queue's overlap under a served 6-row verify graph at 262k:
     `GGML_SCHED_TIMING` per node, the verify's wall with and without the
     copies, and the input write/upload wait (0042's discriminator);
  2. the paged 0016's own cost at 131k/262k, 6 rows against unpaged (the
     table adds a load and a branch per key tile);
  3. the prefetch's effect (the fork's `+6.5 %` is `paper`, one AMD card);
  4. the phase switch's cost (a graph rebuild and a pool resize at the first
     decode after a prefill) — the fork rebuilds graphs (`llama-context.cpp:2042-2048`);
  5. the page move rate: pages move about once per page of tokens
     (`llama-kv-stream.h:37-40`), so the stable case is a few prefetched
     layers, not a full re-stream every round.

## Current state

**Window 226's finding, fixed** (`0123zz`, 2026-11-14): all 10 paged cases were
"not supported [OpenCL]" because 0122 put the page-row helper between the GQA
kernel's `intel_reqd_sub_group_size(16)` attribute and `kernel void`; the helper
took the attribute, the DPAS builtins failed, the optional program build left
the kernel null, and `supports_op`'s kernel clause was false for every pair. The
helper now precedes the attribute, the predicate is tested logic in
`ggml-opencl-fa-paged.h` (`fa-paged-predicate`, every clause flipped), and
`GGML_OPENCL_FA_PAGED_DEBUG=1` logs the clauses so a declined case names one.
**Window 228's two bugs, fixed:** the KV clause compared enum literals (`Q4_0`
is 2, not 4), declining exactly the agent's q8_0:q4_0 pair while f16 passed; the
header now names `GGML_TYPE_*`. The f16 case that was accepted crashed
(CL_EXEC_STATUS_ERROR) because the backend passed a device table tensor's base
and ignored the used row: the kernel read row 0, `INT32_MAX`, and a pool row of
2^31 rows went out of bounds. The used row is now passed with its byte offset
(`ulong table_off`), or a host row uploaded; the conversion kernel takes the same
offset. Recipe green `b3d290fe` (41 of 41); red `kv-stream-red6` `7683bf7b`
(the kernel clause dropped, 2 checks); a safe decline red is the KV clause's
literal `4`. The card's next run decides whether the kernel compiles and the
paged cases pass -- that is Claude's window, not run here.

**I4 built** 2026-11-14 (patch 0123z): a paged prompt's pages are gathered into
0037's f16 blocks by the conversion kernel (its table and ring args) and the
f16 prompt kernel runs with the carry -- the fork's gather form, the chunked
page-runs merge deferred; the backend forces that path for a paged prompt and
`supports_op` accepts up to 512 rows, so a stream-on prefill cannot slip to the
CPU. The decode path's table is uploaded to a device buffer (4 KiB a ubatch at
262k) and released after the queued command. Cases: paged prefill at 512 rows,
kv 8,192/4,096, q8_0:q4_0 and f16, 5 of 5 with `n_slots=16` on the CPU
(`measured-here`, plumbing). **Not compiled on a card** (the kernel text).

**I5 partial** 2026-11-14: `--llama-kv-resident` and its refusal at start,
`/props`'s `kv_stream`, the refusal cell enabled with its red arm
(`kv-stream-red5`, `381660e9`: the kill switch accepts every ask, recipe EXIT 8,
`kv-stream-refusal` failing all refusal checks); the decode phase budget is
refused (not built), the copy queue, the ready/consumed flags and the prefetch
are deferred and named above. Recipe green `294844e5` (40 of 40); red4
`487e82b1` for the pool-write cell; red5 `381660e9` for the refusal. The served
legs below are Claude's card window, not run here.

**I3 kernels and cases built** 2026-11-14 (patch 0122, cases 0123): 0016's
`flash_attn_gqa_dpas.cl`, with 0088's int8 Q K^T and 0089's staged 1-3 rows,
resolves K/V rows through the page table and the ring; the backend takes a
paged op's attended length from the table, routes it to that kernel from 1 row,
and `supports_op` accepts only the shapes the kernel now handles (sync tags
still declined, I5). Paged `test-backend-ops` cases: 10 of 10 OK on the CPU
(`measured-here`, plumbing only). **Not compiled or run on a card**: the
kernel text's first build is Claude's window. **The wiring landed** in 0121:
`cparams.kv_resident_size` (0 = off, arcint sets it only in I5), the cache's
validation, pool sizing, shrunk tensors, the stream's construction,
`prepare` in `apply_ubatch`, the page table and per-layer ring views built in
the graph and set on the fused FA node, filled in `set_input`; the structural
cell `kv-stream-cache` passes (recipe green `5886672e`: 39 of 39, EXIT 0). **A stream-on model run is not possible here**:
no GGUF is committed and arcint's llama engine throws at start without an
OpenCL card (`backend_llama.cpp:505-520`), so the behavioural evidence stays
`kv-stream-map` (green) with reds `kv-stream-red2` and `kv-stream-red3`
(`2d309ae6`, the one-page table shift: recipe EXIT 8, the four ubatches'
attention differ). Named deviations in 0121: a stream-on graph is never
reused (the fork compares `layout_id`; here `can_reuse` returns false, cost to
measure), and the table's device upload for OpenCL is I5's queue path.
Everything else as below.

**I2 built** 2026-11-14 (patch 0120): the stream class's page map, host store,
eviction/write-back and table, the device-free `kv-stream-map` cell green and
its red `kv-stream-red2` failing 6 checks; the deviations above. The
`llama_kv_cache` wiring and the served CPU arm are I3's prerequisite (the
engine requires a card at start). Everything else as below.

**I1 built** 2026-11-14 (patch 0119): the paged op, the CPU gather reference,
and the OpenCL backend's refusal; the device-free cell is `fa-paged`. The
review's two items are in: a compact mask (last `mask->ne[0]` keys, earlier
columns visible) equals the unpaged full-mask run, its red (the widening's
copy without the `col0` offset, branch `kv-stream-red-widen`) fails exactly
those two cases (recipe EXIT 8, `b4a8cd98`); and §4.3 names
where I5's probe refusal stops the scheduler's silent CPU placement; its cell
`kv-stream-refusal` is enabled in I5 with the red arm `kv-stream-red5`. Green: 6 of 6 cases, recipe
37 of 37 (EXIT 0). Red: ring mutation 3 of 4, widening mutation 2 of 2
compact-mask cases, recipe EXIT 8. Everything else is as below.

Design note only, 2026-11-14. Nothing built, nothing measured on a card. The
prior art read is `research-achyudh.md` §2 (`code`, `paper`); the fork is at
`b540a4b9b`; arcint is at 1.6.2 + the folded series. Owed: the operator's
decision on the order (I1..I5 as written), and the first card window for I3.

## Where it lives

The increments land in `contrib/llama.cpp/patches/`: a new llama-side patch
(the stream class + cache + graph inputs, folded into a `0000-`-style patch or
as new numbered patches after 0118, per the release's convention) and a
ggml/OpenCL patch under `0000-04`'s file set (`flash_attn_gqa_dpas.cl`,
`flash_attn_dpas2.cl`, the f16 conversion) — the patch README gains a section
per patch; DESIGN §7 and §3.9 gain a line when the state of play moves;
`backend_llama.cpp` and `config.cpp` carry the flags. The CPU test is under
`tests/` (the style of the existing device-free tests), the card cases in
`test-backend-ops` (`0000-12-tests.patch`'s place).

## Open questions for the next round

1. **Who decides the order?** I1-I5 are written as the safe order; I3 alone
   prices the decode kernel. If the operator wants the quant sooner, I2+I5 can
   be built behind the CPU backend first and the paged kernel later — the
   gate leg then waits for I3.
2. **The 1-3-row disposition** (§3.5): route to paged 0016 or refuse — the
   first is preferred and cheap (a dispatch threshold).
3. **The phase switch's graph rebuild** at a first decode after a prefill:
   whether arcint's scheduler can reuse the graphs for the decode pool (a
   different pool size changes the views) or must re-reserve each switch.
