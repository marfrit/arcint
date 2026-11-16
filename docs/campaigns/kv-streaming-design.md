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
  built -- the stream's page moves are `clEnqueueWriteBuffer` per layer on the
  in-order compute queue, which is **unhidden** (0032's second queue, 0042's
  after-graph placement and the prefetch are the overlap, and they need a card
  to validate); the -2 pool-does-not-fit path is the cache's construction
  refusal plus the prepare assert (a phase resize can be refused only after
  I5's queue). A start that asks for the decode budget or a device without
  paged FA is refused, not silently degraded.

**Review round, 2026-11-14 (`code`):** the per-layer move (0123zzzzz3) copies
each attention layer's streamed pages into its ring slot inside its own FA
dispatch, so a KV window beyond the pool no longer reads another layer's pages;
the cache's cross-layer refusal is gone (~`if (pages)` prepare path). An outside
review of the stream then found four defects, fixed in 0125 with the
device-free arms named there: the store indices map through
`llama_kv_stream::row()` in both K and V (beyond the pool a cell index would
write past the pool), with the state save/restore and `clear()` wired and KV
windows/streams/transposed V refused; one paged predicate (`ggml_cl_fa_paged_form`)
shared by `supports_op` and the dispatcher, carrying the rows, mask heads, op
params and the built kernels the dispatcher actually needs; the CPU plan's
widened mask reserved whenever `mask_col0 > 0`; and a stream-on graph whose
fused node is not `LLM_FUSED_OP_FLASH_ATTN` throwing instead of asserting, with
the refusal red switch no longer compiled into the served binary. The card
window 266 then found `GGML_MAX_SRC` at 10 with the three stream sources at
9/10/11: `set_host` wrote past `src[]` and every FA node read garbage as its
plan; 0126 raises it to 12 with a `static_assert`, and the layers cell reads the
node's sources back. Window 269 then found the stream's USM buffer filled through
its 64-byte placeholder (a device page fault in the cache's constructor) and the
op test's host store outside USM; 0129 clears the host bytes by `memset` and has
the dispatcher copy a regular host store on the device (and rebuild a whole-upload
SoA store to AOS). Window 273 then served right at 4096/65536/131072 and garbage
at 261000, the only streaming depth; 0129a fixes the CPU arm's V row stride (K's
stride was used for V, unlike the card's dispatch) and adds the per-request
stream line, and the new `test_kv_stream_deep` compares streamed attention with
the all-resident run at four streamed pages, two layers, q8_0:q4_0 and f16. The host store
Window 275 (`measured-here`) moved the defect onto the card's path: the full
suite aborted on the first 2-layer op case (`FLASH_ATTN_EXT(hs=256,...,nb=1)`),
whose ring is a whole-uploaded q8_0 tensor and therefore SoA -- `nrk == 1` failed
because the ring check saw two planes. 0129b moves the per-layer page copy AFTER
the ring's AOS reconstruction and writes into the reconstructed AOS buffer the
kernels read (no SoA plane is written), names nrk/nrv and the ring's buffer kind
before that check, makes the 2-layer cases reachable from `-o FLASH_ATTN_EXT`
(their last op is the joining ADD, so the op name must be the plain one), and
prints the per-request stream line at WARN so a window sees it without `-v` (a
route drops INFO). The same window's `-v` rerun read every dumped page table as
all zero while the stream's line named pages in slots (`measured-here`), and
a fresh short request after a streamed one answered wrong with nothing streamed:
0129b makes the table write use `ggml_backend_tensor_set` whenever the node does
not sit in host memory (a write through `dst->data` lands only there; the graph
input may be placed in the backend's device buffer), and 0129c prints the table's
buffer kind, `data`, extra and host-side bytes in the dump so the next window
says which write was lost. The stream line now prints at most twice per request
(the first ubatch, and the first that streams) instead of once per ubatch (54 in
one window), and the device-free `test_kv_stream_deep` gained an after-clear arm
(stream a six-page context, clear, serve a one-page request, compare with a fresh
all-resident run) and a table-vs-plan arm; both are green on the CPU path -- but the arm missed the
hybrid graph path (`llm_graph_input_mem_hybrid::set_input`), which is where the
served model's table was never filled; 0129d fills it there, measured on window
281 (258 distinct table values, 7,968 ring entries, needle found at 261k).
Round 8 (`measured-here`, windows 278-281): 0129b's ring check accepts a SoA ring
that carries its AOS buffer (`(nrk == 1 || rk_aos_buf)`), and 0129e releases the
host store's restore buffers once (window 280's `clReleaseMemObject -38`). The
2-layer op case then failed at the barrier (`-5`): 0129f checks every page copy's
offset+size against the buffers' `CL_MEM_SIZE` before enqueuing (aborting with
page, row, offsets and sizes), prints the copy geometry with
`GGML_OPENCL_FA_PAGED_DEBUG=1`, and fixes the case's own table (its ring entries
descended while the plan ascended, so the kernel read another pair's rows).
0129g logs attended/resident-attended/streamed for the request's prefix beside the
context totals, and makes `GGML_OPENCL_FA_PAGED_SYNC` OFF by default (window 279's
A/B arms were identical because the old default drained every layer).
Round 9 (window 283, `measured-here`): SYNC off by default passes the answer bar
at pool 3584 (4k 49.92 t/s, 65k 32.90, 261k 6.09, needle at 104k of 261k found,
Paris after; the drain costs 0.1 ms/step at 261k, so the 261k cost is not the
per-layer `clFinish`). 0129h prints what the FA kernel will READ under DEBUG --
this layer's table row's largest ring index and slot against the ring pages in
its view and the pool's pages -- and aborts if either points outside, because
window 283's `-5` surfaced after the copies returned clean and the copy bounds
cannot see kernel reads; it also prints a request-end line from `clear()` so the
last ubatch of a request (the 261k one) is logged. The ring's capacity question:
the second KV buffer measures 416 MiB, i.e. two `slots[]` tensors of 473 pages
each (2 x 473 x 425,984 B), so the double buffer for overlap exists already; the
GTT peak 6,722 MiB is the host store of all 1024 pages x 16 attention layers.
Overlap (item 7) has been built since round 10 (below).
**I5.2 — the overlap (round 10, `code` + window 284 `measured-here`).** Window 284
measured the price: 0.579 ms per streamed page per decode step (pool 3584 511.3
ms/step, pool 2560 602.8, 631 - 473 pages), ~11.8 GB/s, close to the link's
14.3 GB/s, so 274 of the 511 ms at 261k was page traffic and the rest compute.
If the traffic hid entirely behind the compute, ~274 ms/step, ~11 t/s (estimate).

- **Every page move today, by direction and when (`code`, round 10):**
  - host->device, per layer, at that layer's FA dispatch: the streamed pages of
    this attention layer from the USM host store into its ring slot
    (`ggml_cl_fa_kv_stream_pages`, `ggml/src/ggml-opencl/ggml-opencl.cpp:23574ff`,
    `clEnqueueWriteBuffer` on `backend_ctx->queue`); a regular (non-USM) host
    store, as the op test uses, is the same bytes device->device.
  - device->device, per layer: the page table's used row
    (`set_input_kv_pages`, 0123z) and the ring's AOS reconstruction
    (`ggml_cl_flash_attn_reconstruct_aos`) -- the latter only when the ring was
    uploaded SoA (the op test, not the served slots).
  - device->host: the host store's write-back when a resident page is evicted
    (`llama_kv_stream::evict` -> `ggml_backend_tensor_get` from the pool slot,
    `src/llama-kv-stream.cpp:150-168`, then `host_set_page`), and the stream's
    `assign()` read-back (`llama-kv-stream.cpp:142-148`). Both run inside
    `prepare()`: at prefill (`set_rows` stores the pages the prefill wrote) and
    when a page is being replaced; **not per decode step** with a steady page
    map. They stay synchronous (`clEnqueueReadBuffer` CL_TRUE) this round: their
    cost is not in the 261k decode step, so moving them to the copy queue would
    change nothing that has been measured, and the host store has to be
    consistent before any read-back (the drain orders it).
- **What the references do and what is followed (`code`, read at their
  checkouts):** Strata's chunk streamer keeps a copy stream per device and a
  ring of halves: `copy_ahead` waits a `freed[half]` event before refilling a
  half and records `filled[half]` after the copies; the compute stream waits
  `filled[half]` and records `freed[half]` after the kernels that read it
  (`src/dsv41/ds4_model.cu:4868-4888, 4934-4939, 11806-11811`). FreeToken's
  prefill double buffer: one staging buffer per layer parity, copies on
  `prefill_copy_stream` waiting the buffer's release event, a ready event
  recorded after the copies, the compute stream waiting it, the release event
  recorded on the compute stream after the GEMMs
  (`python/freetoken/moe/offload_cache.py:887-918, 1055-1068`), and the caller
  prefetches layer L and L+1 at L's start (`python/freetoken/layers/moe.py:390-395`).
  Both are the same choreography this build follows: two parities, one ready and
  one consumed event each, copies on a second in-order queue. The fork
  (`~/src/achyudh-ref src/llama-kv-stream.cpp:428-561`, `code`, read round 9)
  does the same with copier threads and per-layer atomics.
- **Deviations:** no copier threads and no per-layer atomics -- the host walks
  the graph in dispatch order already, so one layer ahead is enqueued from the
  dispatch itself and the two queues order the rest (`code`); the copies wait a
  marker of the compute queue taken when the copy is enqueued (0032's `gmark`
  pattern) rather than a per-layer fence, which also keeps the measured lesson
  (0042, `measured-here`: background copies pay only when enqueued with the
  graph's kernels) -- the copy of layer L is enqueued at L's dispatch, behind
  L-1's kernels. A deviation the fork cannot be blamed for: the fork's copier
  hid the move behind the previous layer's compute by a second queue as well; no
  DESIGN invariant rejects this form (it adds no cross-stream writer: a slot is
  written only by its own copy and read only by its own FA, ordered by the two
  events).
- **Built (0129l, `code`):** `GGML_OPENCL_FA_PAGED_OVERLAP=1`, default off. The
  copies go to a second in-order queue `ctx->fa_paged.q`; parity per dispatch in
  the graph (`fa_paged.next`, reset at `graph_gen` change); a copy waits the
  parity's `consumed` event (recorded on the compute queue after the kernels of
  the FA two dispatches earlier) and a compute-queue marker; after the copies a
  `ready` marker is recorded on the copy queue and, unless NOWAIT, the compute
  queue takes a barrier on it before the FA kernels; after the FA kernels the
  parity's `consumed` marker is recorded for the next copy. `synchronize` drains
  the copy queue. Unset: every copy stays on the compute queue exactly as
  0123zzzzz3 had it, so the served bytes are the same.
- **Window 286 (`code` + `measured-here`): 0129l is serial by construction.** Its copies sit
  in their own layer's FA dispatch and wait a marker of the compute queue taken there, so a
  copy begins only when the compute queue reaches that layer and the FA waits it: the old
  order plus a marker, a barrier and two flushes a layer. Measured: 0 gain, +8 ms/step at
  261k (511.4 unset, 519.5 overlap), the 65k slope unchanged (0.587 ms per page). Not a
  verdict on overlap (a materially different build, CLAUDE.md). 0129n gives the copies real
  lead: one layer ahead, enqueued at the previous layer's dispatch end, the graph's first
  two nodes at its first paged FA after a compute-queue marker (0042's placement), and only
  the parity's `consumed` event waited otherwise; C7 coalesces consecutive pages+rows into
  one command, counted under DEBUG.
- **0129q (`code`), 0129n read back:** the parity is keyed by the RING SLOT (the node's K ring
  buffer + `view_offs`, two remembered keys), not by the node's place in the chain: the
  stream's slots are `slots[ikv % 2]` and the MTP head's graph or a layer-skipping graph need
  not preserve that order (a third distinct slot aborts under DEBUG). No silent fallback: a
  node whose store is not USM or whose ring is not one region WARNs once a process, and the
  last node of the first graph with streamed pages prints "issued ahead / fell back" counts.
  `issue`'s out-of-range read on a one-node chain fixed.
- **Node 0's lead (`inferred` price):** its copies still wait a compute-queue marker taken at
  the first paged FA, so they wait the first GDN layers' kernels as well as the input uploads.
  Kept rather than moved to `graph_compute`: moving is the same marker one earlier, and the
  uploads are the only write on the compute queue that the copies must follow (0042); the
  first GDN layers write nothing the copies read. Price ~1/16 of the copy time (one of 16
  attention layers) per step, about 17 ms at 261k by window 284's 274 ms -- `inferred`.
- **Window 288 (`measured-here`, head 48e19f15):** the lead works. Q5 UD-Q5_K_M, B60, one
  process an arm.

  | arm | pool | 4k ms | 65k ms (t/s) | 261k ms (t/s) |
  |---|---|---|---|---|
  | old `bd68d80d` | 1024 | 74.0 | 173.1 (21.99) | - |
  | new, unset | 1024 | 73.6 | 160.8 (23.67) | - |
  | new, OVERLAP | 1024 | 74.1 | 119.9 (32.22) | - |
  | OVERLAP+NOWAIT | 1024 | 74.0 | 120.0 (31.72) | - |
  | new, unset | 3584 | - | - | 453.2 (6.78) |
  | new, OVERLAP | 3584 | - | - | 274.3 (12.07) |

  C7 coalescing alone: -12.3 ms at 65k, -58.2 ms at 261k (against 286's 511.4). C6 lead:
  -40.9 ms at 65k, -178.9 ms at 261k; 65k is 4.2 ms above all-resident 115.7, 261k lands at
  280 as estimated. DEBUG at 65k: 16 of 16 nodes take the lead, each 2 commands after
  coalescing for 43,024,384 bytes (101 pages x 425,984, one command for K and one for V),
  slots alternating. Answers: 65k identical opening in all arms, needle absent in the text,
  Paris after; 261k needle found and Paris after in both new arms. **Open item:** the 261k
  opening differs between unset ("...exec/backend_o") and OVERLAP ("...exec/flash_nex"),
  acceptance 173 vs 179. A different opening alone is within the card's per-forward
  nondeterminism (0.136 nats floor), so it is neither proof of a race nor proof against one;
  it stays open until `GGML_OPENCL_FA_PAGED_VERIFY=1` is clean in both arms. The NOWAIT arm's
  absent-needle wording differs (424 vs 411 tokens) -- also not a visible failure.
- **VERIFY replaces the op-level case for now (`code` + `measured-here` on the card):** the
  op suite cannot reach the lead path (plain host store, SoA 2-layer ring, window 288:
  "node 0 of 1 fell back"), so the exact integrity check runs on the served path instead:
  FNV over exactly the plan's streamed ring rows at FA j's queue position, against the host
  store pages (0129s). The design note counts it as the red evidence until a USM + one-region
  op case exists; the design note says the NOWAIT arm is red only if VERIFY reports
  mismatches.
- **No op-level evidence yet (`code` of the suite):** `test-backend-ops`' host store is a plain
  buffer and its 2-layer ring is SoA (window 283's DEBUG line: `nrk 2`), so 0129n's coalescing
  and lead both take the fallback there; the FA filter's pass count says nothing about them.
  A case that reaches the path needs USM host allocations and a one-region ring -- the next
  step; until then the served NOWAIT arm is the only red evidence for "the FA waits its copy".
- **Window 289b (`measured-here`, unitrace device timeline, 261k pool 3584, one cached 32-token decode):**
  unset one queue: span 6317.6 ms over ~16 graphs, kernels 3181.6 ms, page copies 2919.7 ms
  (41.8 GB at 14.32 GB/s), idle 209 ms -- ~199 compute + ~182 copies serial a step.
  OVERLAP: span 3586.6 ms, kernels 2972.8 ms, copy queue 2678.2 ms (38.36 GB at 14.32 GB/s),
  both busy 2208.9 ms, compute only 788.2 ms, copy only 469.3 ms, idle 110 ms -- ~198 compute
  + ~31 copy-only + ~7 idle = ~239 ms a step. Compute does not slow under concurrent DMA
  (flash_attn_gqa_dpas 9356 us vs 9148 us a call). Copy-only sits before the FIRST paged FA
  of every graph (28.0 ms, copy queue busy), i.e. node 0 (C8), and before each other paged FA
  0.4-1.3 ms (each ~160 MB copy ~11.2 ms is just longer than the compute between two
  attention layers): ~28 + 13 ms a step. FA dominates compute: 9.15 ms x 16 ~ 146 of ~198.
- **C8 (`code` + 0129u):** node 0's copies wait a compute-queue marker taken at the first
  paged FA, so they start after the graph's stems and first GDN layers; node 1's copies
  follow node 0's on one in-order copy queue, but FA 0 waits only `ready[slot 0]` -- the
  28 ms is that start delay plus node 0's ~11 ms, not node 1's wait. 0129u copies the next
  graph's node 0 pages at the last paged FA, waiting the slot's last reader's `consumed`
  event (two attention layers back), so they run under this graph's last FA, last layer,
  the head and the host's sampling; `before(0)` reuses them when this graph's node 0 plan
  rows equal the rows they filled and re-copies on a difference (counted). Valid by
  immutability: streamed pages never change in the host store (prepare() writes the store
  only through zero_host/host_set_page for pages it saves, llama-kv-stream.cpp:172-259) and
  new K/V go to resident cells (llama-kv-cache.cpp:1730,1765). No invariant conflict; the
  pool partition and slots are untouched, nothing is evicted for it. Estimated 239 -> ~211 ms.
- **0129v/0129w (`code` + window 290 `measured-here`):** the cross-step lead is keyed by
  (plan page, ring row) pairs and the issuing chain, not by the row sequence -- two plans of
  the same size share `[0..n-1]` while holding different pages, and the MTP head's graph is its
  own chain. Window 290: VERIFY=2's self-test fired (1 of 312 chunks, the flipped byte), and
  three lines were silent: the self-test tested its counter after the increment; the summary
  printed at the first graph with zero counters; and the request-end line never reached the
  stream because the served hybrid model's memory is `llama_memory_hybrid`
  (src/llama-model.cpp:3317-3324), whose `request_end()` was `llama_memory_i`'s no-op. It now
  forwards to the attention cache (hybrid, hybrid_iswa, kv_cache_iswa), and the request's
  totals -- issued ahead, fell back, verify nodes/bytes/mismatches, reused/re-copied -- print
  at the request's end through a weak symbol, so a build without the OpenCL backend links.
- **Window 292 (`measured-here`, 65k, c152135c + 0129w):** the verify is alive and green --
  A-self (VERIFY=2) printed "expect exactly 1 mismatching chunk, got 1" 64 times and the
  request totals read **1,040 nodes, 26,894,925,824 bytes, 64 mismatches**, one flip per graph
  seen exactly once; A-verify (VERIFY=1) read the same nodes and bytes with **0 mismatches**,
  Paris answered. That is 0129n's lead's integrity evidence at 65k (the 261k arms were still
  running). C8 (0129u/v) was dead: the tail that prefetches the next graph's node 0 and issues
  the next layer one ahead sits after a `return` in the split-decode branch the served decode
  takes (`ggml-opencl.cpp:25979` before :26102), so every node's copies went out in its own
  dispatch; 0130c calls it in that branch too and names every early return of the prefetch.
- **0129n-0129w decode: no consumed wait (`code`), OVERLAP decode timings before
  `71429597` void.** `consumed[p]` -- the event a copy into slot p waits before overwriting
  bytes the FA two layers back read -- is armed only in
  `ggml_cl_fa_paged_overlap_after` (`ggml-opencl.cpp:23960-23972`), and
  `overlap_graph` clears it at every new graph (`:23890-23905`). In the split-decode branch,
  which every decode takes, the tail call was unreachable until 0130c (`return` at :25979,
  call at :26102), so on a decode `consumed[]` stayed nullptr, `issue()` passed `nw = 0`
  (`:23940`), and a copy into slot p could start while the FA two layers back was still
  reading it: a write-after-read race. Prompt graphs were not affected -- a prompt FA ends at
  the function's tail, so `after` ran and armed the consumed marker. The read side was
  correct: each FA waited its own ready event, which is why VERIFY (a hash at the FA's start,
  blocking readback) read 0 mismatches -- it cannot see a write that lands *during* the FA.
  Window 288 (261k, 12.07 t/s), 289b and 292 are therefore **void as OVERLAP timings** until
  re-measured on `71429597`; window 288 and 292 both showed the alternate opening
  ("...exec/flash_nex" against "...exec/backend_o") at 261k in the OVERLAP arms, two windows
  running. The post-FA check (VERIFY=3: a second hash enqueued right after FA j's kernels,
  compared with the pre-FA one, non-blocking readback collected at the next graph's start and
  at the request's end) is built in 0130d: its device-free gate is green, the card number is
  owed; `GGML_OPENCL_FA_PAGED_NOCONSUME=1` drops exactly the consumed wait, its red arm (`code`).
- **Window 293b (`measured-here`, build `881f4361`):** the race fix pays where the copies are
  the critical path -- Q5 at 65k, OVERLAP against unset, 116.3 against 161.0 ms/step (-28 %),
  next-graph lead 105 issued / 105 reused, VERIFY=1 at 261k 4,032 nodes / 425.9 GB / 0
  mismatches. At 261k the OVERLAP arms open with "...exec/flash_nex" where window 288 and
  292's unset arms opened with "...exec/backend_o"; this window has no unset arm at 261k, so
  the 261k OVERLAP numbers are not comparable across windows yet. 0130a-c's 4-7 row V2
  kernel is a **measured negative**: +14.9 % at 240k no-paging (222.7 -> 255.9 ms/step) and
  +8.0 % at 261k OVERLAP (262.2 -> 283.3); the switch stays in, default off. 0130e puts
  0130a's depth cases into the perf list (window 293b's perf arms matched 0 cases), and
  0130f is the 0130b mechanism done right -- the V halves to private registers, stored after
  P V: `GGML_OPENCL_FA_DECODE_PRE=1`, default off, the card number owed (`code`).
- **Window 294 (`measured-here`, build 9856f644 = 0130d):** four Q5 arms, OVERLAP=1,
  VERIFY=3, greedy 32-token: B-red 261k pool 3584 and A-red 65k pool 1024 with
  `GGML_OPENCL_FA_PAGED_NOCONSUME=1`, B/A-green without it. **Every arm printed
  `post-FA changes 0` and no `post-FA changed` WARN**, red and green the same time (827.7 vs
  828.0 s at 261k, 113.4 vs 113.3 s at 65k), pre-FA verify clean (4,032 nodes / 425.9 GB at
  261k, 1,040 / 26.9 GB at 65k). So the red arm is green: either the race cannot land at
  these depths or the check cannot see it, and the race's absence is not proven yet.
  Also: window 293b's 261k B arms used pool 3584, not 1024; the kernel's K/V codes are
  KT 1 (q8_0) and VT 2 (q4_0), not KT 8 / VT 4 (`code`).
- **Why NOCONSUME=1 did not land, from the code (`code`):** every ordering that keeps a copy
  out of the slot an FA is reading, in the order that matters.
  - The copy queue is one in-order queue (`clCreateCommandQueue(ctx->context, ctx->device,
    0, &err)`, `ggml-opencl.cpp:23988`), and a node's copies are issued at `issue` time
    (`:24094-24135`). Copy k+2 is enqueued at `after(k+1)` (`:24199`), so it starts no earlier
    than copy k+1's marker; copy k+1 starts at copy k's marker, which is exactly FA k's start,
    because the FA waits `ready[parity] = copy k`'s marker (`before`, barrier at `:24189`).
  - The two slots alternate: copy k+1 fills the slot FA k+1 reads, copy k+2 fills FA k's
    slot. Copy k+2 reaching FA k's slot therefore needs copy k+1 shorter than FA k. Window
    289b's `measured-here` figures say the opposite: at 261k a layer's copy ~11.4 ms against
    the FA call's 9.15 ms, so copy k+1 ends after FA k does and copy k+2 launches into an
    idle slot. At 65k the step is 116 ms against ~48 ms of copies a step, the copies are
    not the critical path, and copy k+1 is still the longer side of the pair at the same
    K/V ratio (`inferred` from 289b's per-layer ratio). The hazard needs FA k to hold its
    slot LONGER than copy k+1 -- that is what 0130h's delay buys.
  - `wl[1]`: `issue`'s wait list is at most two events, `consumed[parity]` first and the
    `after_inputs` marker second (`:24054-24060`); NOCONSUME drops only the first, and the
    second (compute queue marker, so after the graph's stems) still bounds the first nodes'
    copies.
  - The cross-step prefetch, the one copy whose start coincides with a reader's start, keeps
    its wait unconditionally: `cl_uint nw = o.consumed[p] != nullptr ? 1 : 0;` and
    `cl_event wl[1] = { o.consumed[p] };` (`:24050-24051`), not gated on
    `ggml_cl_fa_paged_noconsume()`. 0129n-0129w's race was there: before 0130c that event was
    never armed, so the pre-0130c copy into the graph's node-0 slot could start exactly when
    the graph's last FA started reading the same slot.
  - VERIFY=3's own enqueues lockstep the host at the next graph's start:
    `ggml_cl_fa_paged_verify3_collect` calls `clWaitForEvents` on both readback events of
    every pending pair (`:24482`), and the post hash's readback was enqueued after that
    FA's kernels; so before the host dispatches the next graph it waits for the previous
    graph's last post hash -- i.e. its last FA -- to finish. That removes host lead across
    the graph boundary, which is where the prefetch copy would otherwise be enqueued while
    the last FA runs. (VERIFY=1's blocking readback at the FA's start does the same a node.)
  - `GGML_OPENCL_FA_PAGED_DELAY=<n>` in 0130h is the arm that then makes the intra-graph
    hazard reachable: an n-iteration spin on the compute queue between the split kernels and
    `k_merge`, so FA j's hold grows by the spin; the red needs n large enough that FA k's hold
    exceeds copy k+1's duration -- at 261k ~2.3 ms, at 65k ~0.5 ms (`inferred` arithmetic from
    289b) -- so `DELAY=1000000` (~1-2 ms if the loop is ~4 cycles at 1.5 GHz) is a first
    value and `5000000` a safe one; raise until the red runs. With the consumed wait on the
    same delay must read 0: `after` arms the marker on the compute queue (`:24198-24199`), so
    copy k+2 waits FA k's kernels, delay included.
- **C9 (`code`, analysis only, pending window 291's non-paged number):** the paged GQA decode
  kernel's launch is `ggml/src/ggml-opencl/ggml-opencl.cpp:25850-25856`: gws
  `{fa_gqa_sides*n_tiles*16, n_head_kv*n_batch, n_splits}`, and w289b's grid {160,4,256} is
  2 sides x 5 tiles x 16 lanes = 10 sub-groups a work-group (5 tiles because GQA 6 x 6
  tokens = 36 rows), 4 KV heads, 256 splits (the cap, ~1020 keys a split). Bounds, from the
  kernel: every work-group walks its split's keys; each key costs one `fa_gqa_page_row`
  lookup (an int load of the 1024-entry table, `key/ps` a shift for ps 256,
  `flash_attn_gqa_dpas.cl:166-172`) and a K row (1088 B q8_0) and a V
  row (576 B) -- 261k keys a layer is ~434 MB, the number w289b measured at 47 GB/s against
  a 456 GB/s peak, so the call is bandwidth/occupancy bound, not arithmetic bound; 1024
  work-groups at one a Xe core (0045's measured 20 a wave) is 51 waves, so the split count
  sets the wave count while kv_per_split sets the per-wave work. The merge
  (`:25875-25890`, one lane per float4) reads the partials: 12 heads x 6 tokens x 256 splits
  x 256 floats = 18.9 MB a layer, ~302 MB a step -- a tenth of the FA's own read. Compare
  the non-paged path when w291's 240k number lands; before it, 0097's split rule (quantized
  K/V wants 128-256 a split at 4k+, 256 cap) and 0045's wave quantization are the two
  measured anchors. No build until then; the change would follow llama.cpp CUDA's
  flash-decoding split (`ggml-cuda/fattn-common.cuh` `launch_fattn` /
  `flash_attn_combine_results`, `code`, cited in 0045) -- fewer, longer splits when the wave
  count is under the core count -- and Strata's paged decode attention would be read first
  (no proposal without that read).
- **Next lever, analysis (`code` from the references, `measured-here` from windows 284/288):**
  at 261k the step is copy-bound: 473 pages x 425,984 B x 16 layers = 3.22 GB a step; at
  14.3 GB/s USM that is 225 ms against 274.3 measured, with compute 237 ms. What lowers it:
  - **Page selection / sparse attention over streamed pages** -- Strata's resolve
    (`src/kernels/cuda/kv_stream.cu:100-165`) looks up only the blocks the call's ids touch
    and copies just the misses (`copy_kernel`, :167-180), with a clock-sweep slot choice;
    FreeToken keeps a paged KV pool in VRAM and attends selected tiles
    (`python/freetoken/attention/dsv41_sparse.py`, `kernel/triton/dsv41/sparse_attn.py`).
    For dense attention at 261k every page is touched, so selection saves nothing today; it
    becomes the lever if the served attention is sparse.
  - **Lower precision for streamed pages only** -- Strata streams K8V4/Q4/int8 pools
    (`kv_stream.cu:34-59`); our pages are already q8_0 K / q4_0 V. K at q4_0 would take a
    page from 425,984 to ~294,912 B: 2.23 GB, ~156 ms at 14.3 GB/s. That is the largest
    single cut and it is a precision decision, judged at the answer (CLAUDE.md), not here.
  - **Keeping pages resident across steps** -- the pool partition already does this for the
    resident pages; for dense attention the streamed set is read every step, so residency
    growth is the only cut, i.e. a bigger pool or fewer bytes a page (above).
  - **USM gather kernel** (round 11's option): the paged kernel reads the host store through
    the table, no copies, one enqueue a layer; the bytes over the link equal the copies' but
    the VRAM write of the staging copy is saved. **No reference does this** -- an arcint
    option. llama.cpp is not a precedent for it: with `--no-kv-offload` the KV buffers come
    from `ggml_backend_cpu_buffer_type()` (`src/llama-kv-cache.cpp:251-255`, `offload` false)
    and the graph then pins every node between the KV store and the attention output to the
    CPU backend (`src/llama-graph.cpp:3186-3190`: `ggml_backend_sched_set_tensor_backend(sched,
    cur, backend_cpu)`), so no device kernel reads KV over the link there.
  - **The 37 ms** between max(237, ~216 inferred copies) and 274.3: candidates are node 0's
    marker (one of 16 layers, ~17 ms `inferred`), the tail layer's copies starting only after
    its preceding `consumed` (one layer not fully hidden), 16 event waits + 16 barriers a
    graph, and the copy engine sharing with the input uploads (0042). A unitrace pass that
    sums `clEnqueueWriteBuffer` busy bytes/time and the gaps between kernel batches splits
    copy time from wait time per node; that is the measurement to run.
- **Red (0129l, development only):** `GGML_OPENCL_FA_PAGED_OVERLAP_NOWAIT=1`
  keeps the copies on the copy queue but drops the FA's ready wait. Expected:
  at 261k the kernel reads a slot mid-copy -- wrong answers (needle or Paris) or
  a failing check; if an arm reads right, the device finished the copies early
  and the arm proves nothing.
- **Connection to `--llama-kv-resident`:** the ring's double buffer is the two
  `slots[]` tensors (2 x n_ring pages, 384 MiB at pool 3584, window 284's
  416 MiB second KV buffer), so the overlap needs no new VRAM.

- The ring's capacity is `n_ring` pages = the configured context's non-resident
  pages (llama-kv-cache.cpp's construction from the pool), not two slots: the plan
  names one row per streamed page of the whole context, so the worst case is that
  complement and a static bound has to hold it; the price is its VRAM (473 pages,
  ~3 GiB at pool 3584), kept until a dynamic resize exists.

is a USM tensor and its page access goes through the slices entry points, never
`->data` arithmetic on an OpenCL buffer; the cache sets `stream_buft = buft`
only when the caller gave none, and the stream's construction throws when that
buffer type's device lacks the USM host entry point.

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

**Window 229's split, open:** f16 decode over a ring is correct; q8_0:q4_0
decode over a ring returns NaN (the no-ring q8_0 arm passes), and both types'
prefill conversion returns NaN. The predicate accepts every case, the table is
now read at its used row, and the row formula matches the test's table and
strides exactly (checked against pool and ring row counts). The backend now
logs, under `GGML_OPENCL_FA_PAGED_DUMP=1`, what a paged launch is handed --
`n_kv`, page size, the table row and its byte offset, the pool's/ring's row
counts and nb1/nb2 for K and V, and the table row's entries read back -- so the
next window names the wrong number rather than guessing. Recipe green
`31a3b121` (41 of 41); the red line is the ring-ignore kernel copy (Claude's).
Window 232's dump cleared the bindings: a failing q8_0:q4_0 ring case and the
passing f16 ring case at the same shape get the same table, the same pool/ring
row counts and the strides those tensors have; every table entry indexes inside
the pool or the ring. The failure is inside one *type path* (staged K for q8_0,
q4_0 V staging, or 0088's int8 b operand); f16 reads the ring per lane and
passes. Three isolation runs are named in README 0123zz
(`-DGQA_K_DIRECT`, `GQA_I8=0`, `F16KV_BLOCK=4096`), and `0123zzz` adds the pair
arms that name the defect's side in one run: `q8_0/q8_0` and `q4_0/q4_0` at a
ring (6 rows) and `q8_0/q8_0` prefill (512 rows) -- 3 of 3 on the CPU
(`measured-here`; recipe green `ff547dc0`, 41 of 41). A pass on `q8_0/q8_0`
with `q8_0/q4_0` failing names the q4_0 V side; both quantized pairs failing
names the K side. No kernel change this round; the fix is owed after those
lines.


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

**I5 built: the page moves are per layer** (patch 0123zzzzz3, 2026-11-14).
The window-255 defect is gone: `llama_kv_stream::prepare()` no longer copies
any page into the ring (`src/llama-kv-stream.cpp:prepare()` after the plan
loop; `code`), it only packs the streamed pages' ring rows into the `plan`
tensor `I32 [n_pages]` (a row `>= 0` is the ring row, `-1` a resident page) and
the table. The copies happen **inside the attention call of each layer**:
`ggml_compute_forward_flash_attn_ext_paged` (ggml-cpu, `code`) and
`ggml_cl_fa_kv_stream_pages` (`ggml-opencl.cpp`, `code`) read
`host_k/host_v` at the plan's pages and write them into the K/V ring slot
`row*page_size*row_size` before the kernels (OpenCL: `clEnqueueWriteBuffer` at
the top of `ggml_cl_flash_attn`, on the in-order compute queue and before the
SoA reconstruction). A slot is shared by every second layer (`ikv % 2`); layer
i+2 overwrites it only after layer i's commands were enqueued, so in-order
execution gives each layer its own pages. This is **form (a) as step one**:
correct, not hidden; the second queue, the events and the ready/consumed flags
are the overlap, unchanged and still deferred.

What changed against the fork, each with its reason:
- **Host store layout**: one USM host-memory tensor per attention layer and
  direction, `[n_embd_gqa, page_size, n_pages]`, allocated through
  `ggml_backend_opencl_set_alloc_host(true)` around the stream's own context
  and read/written by the slice entry points (`host_set_page/host_get_page`,
  `code`). Reason: a tensor, not a byte range, gives the graph a src the
  scheduler can place on the card backend without a copy; the fork's raw host
  pointers need the copier thread, which step one does not build. On a CPU
  buffer type the same class uses host memory, so the device-free cell runs
  the whole path (`code`).
- **No copier/publisher threads, no copy queue, no flags** (fork
  `src/llama-kv-stream.cpp`, `code`): the copy is enqueued inline per layer,
  so ordering is the queue's; the fork's `ready/consumed` is only needed when
  producer and consumer are different queues (`paper`, design §1.3).
- **Slots sized by the caller** (`n_ring = n_pages - n_slots`), not by the
  backend probe: the probe is I3's remaining item.
- **Prefill**: the copies are issued once per FA call for the plan's pages;
  the paged gather converts each page once (`code`), so bytes over the link
  equal the copies'. A prefetch ahead of the graph (0042's placement) is not
  in this step.
- **The refusal is removed** (0123zzzzz2 added
  `LLAMA_KV_STREAM_UNSAFE=1`): a budget below the context now serves instead
  of refusing (`src/llama-kv-cache.cpp`, `code`), because the defect it
  guarded is fixed; the start refusal stays where it belongs, arcint's
  `src/exec/kv_stream_refusal.h` (§4.3, probe and card rules).

Tests: `tests/test_kv_stream_layers.cpp` (arcint's `arcint-kv-stream-layers`)
is device-free: three layers, one pool page, two ring pages, one slot shared by
all layers; each layer's output is byte-equal to a contiguous attention over
its own 512 cells, and the slot ends with the last layer's pages
(`measured-here`, CPU backend). Red arm: the copy guarded off
(`if (false && plan)`) fails 4 checks, including all three layers' outputs.
`tests/test-backend-ops.cpp` (patch 0123zzzzz5) gains a two-layer case sharing
one ring and one plan, each layer with its own pool and host store, joined by
an ADD at decode and 6 rows; the CPU arm does the same per-layer copy, so a
backend that copies up front reads the last layer's pages and differs. **Not
compiled or run on a card**: the first OpenCL build is Claude's window.

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
- **0130p-0130r (`code`), window 296's two holes closed:** (1) every development switch
  prints its value once when first read (NOCONSUME, NOFIX, DELAY n, VERIFY mode,
  VERIFY_SELFTEST, TIMELINE) and the request-end summary counts what ran -- consumed waits
  dropped, cross-step waits dropped, delay kernels, self-test writes -- so a window's log
  says whether the red arm was red. (2) `GGML_OPENCL_FA_PAGED_TIMELINE=1` stamps the
  request's first split-decode graph on both queues with markers that carry the next
  command's waits: copy start/end, FA split start/end, merge end, the consumed marker and
  the post hash start per slot, plus the derived margin `copy(k+2).start - FA(k).split end`
  with its minimum and negative count, and the cross-step prefetch's copy start against the
  last FA's split end. The copy queue gets `CL_QUEUE_PROFILING_ENABLE` only with the switch
  on; the compute queue's profiling is the build's `GGML_OPENCL_PROFILING`, and its absence
  warns once instead of printing zeros. (3) `GGML_OPENCL_FA_PAGED_NOFIX=1` drops exactly
  the cross-step prefetch's consumed wait, 0130c's own site, and NOCONSUME keeps its
  meaning. Why window 296's red arms could not land (`code`): VERIFY=3's collect at the next
  graph's start calls `clWaitForEvents` on every pending pair's readback events, which were
  enqueued after that FA's kernels -- the host therefore waits the previous graph's last FA
  before dispatching the next graph, and the prefetch, issued inside that last FA's `after`,
  is ordered behind it by the host. The post-FA check cannot see this race by construction;
  the timeline margin is the non-blocking detector, and a post hash on the second queue is
  the other. The delay n follows from the timeline: FA split duration and copy duration per
  slot, n ~ (copy duration - FA hold) * 1.5 GHz / 2 iterations (`inferred`).

- **0130w-0130z (`code`), round 7's timeline defects:** the stamps' slot field held the
  node index (one margin pair per slot), the events carried no host time, and 0130h's spin
  sat after the readers so it lengthened no read. The events now carry node and slot, the
  submit time (clGetDeviceAndHostTimer at the base), and `DELAY_AT=splits` puts the spin
  before the split kernels; to widen the read the prefetch must wait, the FA to lengthen
  is the slot's last reader two layers before the graph's last FA.

- **0131c-0131h (`code`), round 8's two defects:** 0130p's markers were commands on both
  queues (window 299: 451.9 ms/step with the switch unset against 261.3/262.1 before
  0130p), so the timeline serialized what it measured; its stamps are now the commands'
  own events (k_split both split stamps, k_merge, the writes' first/last events, the
  consumed marker, the post hash), the base is the earliest device start among them, and
  the recording start enqueues nothing. VERIFY=4 is the non-blocking race detector: a
  reader's hash after its kernels and a re-hash behind the slot's next copy's ready event,
  both in a USM-indexed pair, compared only at the request's end -- VERIFY=3's collect at a
  graph's start was the host wait that ordered the prefetch behind the last FA. The
  TIMELINE on/off text difference (`code` for the unchanged dependencies, `inferred` for
  the cause) is either a copy landing inside a reader or the B60's per-forward floor;
  VERIFY=4 with SELFTEST=4 decides it on the card.
