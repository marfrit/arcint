# research-vram-arena — how the references own a card's memory, and what an arcint arena would take

Stage 2 of `vram-user-space.md`. Recon only: no build, no card window (the card hosts were down; every figure
below is read from source or from the campaign records, none is new). Evidence classes: `code` = read in source,
`path:line`; `paper` = a doc or a commit message says so; `inferred` = reasoning from the above, not checked.
Where a doc and the code disagree the code is cited.

Trees read: Strata `origin/HEAD` (`src/` is the CUDA/HIP tree, `sycl/` the Intel port); Infernix and FreeToken as
checked out; llama.cpp as arcint serves it = `llama-port` at `3bd50d9df` (pin + patches 0001-0056; 0057-0068 are
not in that tree, so `ggml-opencl.cpp` line numbers below are 0056-level and move by a few hundred lines in the
served series; 0068 is read from its patch file). Every `ggml-opencl.cpp:N` is that file under `ggml/src/ggml-opencl/`.

## 1. Result in five lines

1. **No reference owns the card as one allocation.** Strata makes a handful of large allocations at load (slot
   arena, dense pool, verify arena, MTP arenas, prefill buffers), sizes the biggest from `free - reserve`, and
   leaves a 1-2 GiB reserve for everything outside them (`code`, section 2.1). Infernix is the closest: arenas plus a
   VMM frame pool whose base never moves, with a lease that lends frames to the prompt path (`code`, 2.2).
   FreeToken has a budget ratio over torch's caching allocator and no arena (`code`, 2.3).
2. **What all of them share is the arcint slot loan's idea**: the prompt path's buffers are carved from the expert
   cache's own memory (Strata `prefill.cpp:1397`, Infernix `FrameLease`), so the kernel never sees a new allocation
   for the loan. arcint's loan does the opposite at the driver boundary: it frees a `cl_mem` and allocates another
   (`code`, 2.4), so the driver is asked each time.
3. **arcint's allocations outside ggml's buffers are few in kind and many in call sites**: one grow-only helper
   (`ggml_cl_buffer::allocate`, `:380`) behind ~40 named buffers and a tag map, the bank-stream ring, and
   per-call temporaries created inside ops (section 3).
4. **An arena fits arcint's backend unusually well**: the backend already addresses every tensor as
   (`data_device`, `extra->offset`) and carves sub-buffers by aligned offset; `data_device` is set from the buffer
   in exactly one place (`:11684`). The buffer-type route is a local change (section 4).
5. **Smallest first step that pays**: a counting allocator behind `ggml_cl_buffer::allocate` and the raw sites,
   then a side arena for the backend's own grow-only buffers and the ring, reserved once at load (section 6).
   One premise is unproved and decides the whole design: whether the xe kernel evicts a big allocation whole or in
   parts (section 5, risk 1; section 7, question 4).

## 2. The references

### 2.1 Strata (`code`, `origin/HEAD`)

| What | Where | How it owns the memory |
|---|---|---|
| Expert slot cache | `sycl/src/core/expert_cache.cpp:663` | One `malloc_device_guarded` of `want` bytes (`want` = slots x blob stride), then zero-filled (`:685-690`) so a read of an unfilled slot is deterministic. A too-large `want` is refused against `device_free_bytes()` with the numbers (`:625-640`). |
| Alternatives to that malloc | `expert_cache.cpp:645-657` | `--vram-elastic`: physical segments behind one address range (`open_segmented`); CUDA VMM `VmmRange` mapped chunk by chunk. `docs/VRAM_ELASTIC.md` (`paper`): only the slot cache shrinks or grows, on a request, to leave VRAM free for another program. |
| Dense weights | `sycl/src/program/generate.cpp:2986` (`pool_bytes`), `sycl/src/core/device.dp.cpp:415` (`DeviceArena`) | One allocation for the whole pool, "the thing that can fail late, so it happens once, here" (`device.dp.cpp:413`). |
| Verify scratch | `sycl/src/core/verify.cpp:595-615` | A **count pass**: `Bump count; carve(count)` runs every `take<T>()` as arithmetic, then one `malloc_device_guarded(count.used)` and the second pass carves the same layout. |
| Prefill buffers | `sycl/src/prefill/prefill.cpp:366-399` (`Alloc`), `:1397-1398` | `Alloc` is either a fresh allocation per `take`, or a bump allocator over a borrowed region (`o.base = borrow; o.cap = borrow_bytes`): the borrowed region is the cache's own slots, refilled after the prompt (`:2424`, `:2455`). `count_only` sizes it with a granule. |
| MTP | `sycl/src/core/mtp.cpp:217,309,356,393,468` | Separate arenas (dense, experts, state, a main arena). |
| Budget | `src/program/generate.cpp:6874,7005,7034` (sycl port: `sycl/src/program/generate.cpp:4694-4722`) | The cache is sized to `min(asked, free - vram_reserve - prefill - pipe_first)`; `--vram-reserve-mib` 1,024 / 2,048 on Intel (`docs/INTEL.md:41`, `paper`). The reserve is the budget for everything outside the arenas: the runtime's own surfaces, the display, other clients. |
| Free-memory figure | `expert_cache.cpp:69-80` | On xe with no `ext_intel_free_memory` the query reports 0; Strata falls back to total minus the process's own fdinfo and says it does not see other processes (`code`). |
| Aliased pages | `sycl/include/strata/sycl_queue.hpp:259-290` (`malloc_device_guarded`; the file has 290 lines); `docs/INTEL.md:155-157` (`paper`) | On xe a 32 MiB+ allocation can come back with two 2 MiB pages mapped onto the same memory; every big allocation is checked and re-made behind a spacer (<0.5 s at start). |

Reading: Strata is "few large allocations at load + a measured reserve + bump carving inside", not one arena. Its
answer to the eviction cliff is the reserve and the instruction not to ask for more than the card has
(`docs/INTEL.md:142`, `paper`): the same budget logic arcint's stage 0 builds, one level up.

### 2.2 Infernix (`code`)

- **Weights**: one `DeviceArena(plan.device_capacity_bytes)` (`src/artifact/materializer.cpp:362-378`), or, in
  overlay mode, the VMM home mapping of the evictable pool (`:366-374`). `DeviceBuffer` "owns exactly one cudaMalloc"
  and `DeviceArena` is "the suballocation primitive" (`src/core/arena.h:18`, `arena.cu:157`).
- **Workspace**: an arena sized from `workspace_bytes(...)`; a prefill call wider than `static_columns` "swaps in an
  arena of the chunk's size lent from expert frames" (`src/models/qwen4_exp/program/program_impl.h:634-638`).
- **Expert frames**: per-pool frame arrays in a VMM arena whose base never moves; `resize()` backs more frames chunk
  by chunk or releases the top chunks after evicting the lowest-scoring experts (`expert_residency.h:36-39`;
  `src/core/vmm_arena.h:1-8`, `vmm_arena.cu:18-43`: `cuMemAddressReserve` once, `cuMemCreate`/`cuMemMap` per chunk).
- **Lease**: `FrameLease{pool, first, count, DeviceSpan memory}` = "a contiguous run of one pool's frames serves
  another use of device memory (the prefill stream's ring and areas, the Vision window) and comes back. Lent frames
  hold no expert ... decode graphs reach frames through the tables, so lending invalidates none"
  (`expert_residency.h:135-142`).
- **Budget**: `VramBudgetSource` returns device free/total plus, on Windows, the process budget; "no policy"
  (`src/core/vram_budget.h:1-5`). `SpillGuard` (`vram_budget.h:79-89`, `vram_budget.cpp:301-310`) reads free memory
  before and after an allocation and calls it spilled when free fell by less than the bytes asked: an eviction
  detector from outside, the same idea as arcint's eviction guard (`prefill-slot-loan-design.md` §28b).
- Allocation is not wholly arena-only: staging is a `DeviceBuffer` (`expert_offload.cpp:174`).

Reading: Infernix is the reference mechanism for "the loan reassigns ranges inside memory the process owns". It
gets there with a growable VMM range; arcint's OpenCL has no VMM call (`inferred`; not found in the OpenCL
headers the backend includes), so the same shape needs a fixed arena.

### 2.3 FreeToken (`code`)

Python over torch. `engine/cache_budget.py:38-45`: the MoE and KV pools get `memory_ratio * baseline_free -
weights - fixed_cache`, the remainder is "CUDA-graph/activation headroom". Each bank is one `torch.zeros(..., device)`
(`moe/offload_cache.py:562-568`), sliced into per-layer views (`_slice_pools`, `:371`); torch's caching allocator
owns everything else. A runtime rebuild drops references and re-sizes (`engine/graph.py:255`, `engine.py:1116`).
Budget only; there is no sub-allocator to copy. (`docs/research-freetoken.md` is the earlier survey.)

### 2.4 llama.cpp / ggml as arcint serves it (`code`, `llama-port`)

- **Buffers**: `ggml_backend_buft_alloc_buffer(buft, size)` is `iface.alloc_buffer` (`ggml-backend.cpp:40`);
  a tensor list goes through `alloc_buffer_n`, which groups by buffer type and splits at `get_max_size`
  (plan/grouping `:58-92`, default `alloc_buffer_n` from `:94`). The OpenCL type sets `alloc_buffer_n = NULL` and `get_max_size =
  CL_DEVICE_MAX_MEM_ALLOC_SIZE` (`ggml-opencl.cpp:14955-14964`, `:14926`).
- **Compute buffers**: `ggml_gallocr` keeps a free-block allocator per chunk (`ggml-alloc.c:115-312`,
  `MAX_FREE_BLOCKS 256`), the chunk size is the type's max size, at most `GGML_VBUFFER_MAX_CHUNKS 16` (`:96`).
  A reserve that needs more frees the old virtual buffer and allocates a new one (`:925-944`): a free then a create
  at the driver, every time the shape grows.
- **KV, recurrent state, outputs, weights**: `alloc_ctx_tensors_from_buft` (`llama-kv-cache.cpp:285`,
  `llama-memory-recurrent.cpp:119`, `llama-model.cpp:1946`), output rows `llama-context.cpp:2411`.
- **The OpenCL buffer** (`ggml-opencl.cpp:14839-14905`): one `clCreateBuffer(CL_MEM_READ_WRITE, size)` per ggml
  buffer (when that device allocation fails, `alloc_buffer` falls back to `CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR`
  with only a `GGML_LOG_WARN`, `:14900-14907`: a failed VRAM allocation silently becomes host memory, exactly an
  "outside the sum" case, `code`); `get_base` returns the device alignment, not a pointer (`:11647-11650`); `init_tensor` makes an `extra`
  with `offset = data - base` and `data_device = ctx->buffer[0]` (`:11680-11685`). Quantized tensors are re-laid out
  into sub-buffers at `align_to(extra->offset + ..., backend_ctx->alignment)` (e.g. `:11944-11955`). The
  alignment is `CL_DEVICE_MEM_BASE_ADDR_ALIGN / 8` (`:8100-8102`); the load log prints it and the max allocation
  (`:7869-7873`).
- **A type-level mode** `ggml_backend_opencl_set_alloc_host(true)` (`:14839`) makes the next buffer USM host memory
  (the expert bank); the loan's lend bank uses it (patch 0068). Not VRAM.
- **The expert cache** allocates its slot tensors with `alloc_ctx_tensors_from_buft` (`llama-expert-cache.cpp:471`)
  and the gather mirror through `reserve_gather` (`:504` -> `ggml-opencl.cpp:15636-15640`): the load-time reserve is
  already a precedent for "size it once, at load, from a known worst case".
- **The loan** (patch 0068, `prefill-slot-loan-design.md` §1): the tail slot tensors live in their own buffer
  (`ggml_backend_buft_alloc_buffer_n`); a loan is `free that buffer -> sched_reserve at the prompt chunk -> run`,
  the return is `re-reserve resident -> re-allocate the tail buffer -> refill -> verify`. At the driver that is a
  `clReleaseMemObject` and a `clCreateBuffer` each way. Strata and Infernix do the same with no driver call.

## 3. Inventory: arcint's device allocations (Flash-Next on the B60 unless noted)

**A. Through ggml's buffers (the sum the loan's rule counted).**

| Allocation | Hook | Size / rule |
|---|---|---|
| Dense weights | `llama-model.cpp:1946` | the model's non-expert tensors; one `cl_mem` per buffer, split at the type's max size |
| Expert slot buffer | `llama-expert-cache.cpp:471` | the unit's slot budget (11,600 MiB, `paper`: design §24a); minus the lent tail while a loan runs |
| KV | `llama-kv-cache.cpp:285` | n_ctx-sized at load; 2.0 GiB q8_0 on the record (`paper`: design §24a) |
| Compute buffers | `ggml-alloc.c:432` | the scheduler's worst case; the QSA mask term `10.7 B x cell x token + 0.146 MiB/token` dominates, ~3.0 GiB at 8,192 x 20,480 (`paper`: §24a) |
| Output rows | `llama-context.cpp:2411` | `n_outputs_max` rows x vocab x 4 B, ~3.8 MiB (`paper`) |
| MTP draft context | its own context | 770 MiB at 131,072 (`paper`: §24a) |

**B. Outside ggml's buffers, inside `ggml-opencl.cpp` (the sum was blind to these).** `paper` (design §27): the
loan's rule printed ~4.7 GiB spare on a shape measured 42 MiB from the card's edge; the difference is this table
plus the runtime's own surfaces (kernel ISA, scratch/private, heaps).

| # | What | Where | Rule |
|---|---|---|---|
| B1 | `ggml_cl_buffer::allocate`: grow-only, **releases the old object, then creates the new one** | `:380-388` | `size = max(size)`; no copy, so callers re-fill |
| B2 | the named `prealloc_*` members (MoE activations `moe_qa/da/sa`, K-quant `kq_xq/xd/xs/xh`, `kq2_*`, `kq_fa/fs/fc`, `kq_pairs/tiles/ntiles`, `pair_perm`, `splitk_partial`, `f32_splitk`, `fa_partial`, `cumsum_tmp`, `kq_stage_q/d`, `exchange`) | declared `:770-847`, `:1061`, `:1071`, `:1112`; grown at the call sites, e.g. `moe_qa` `N*K` B (`:23467`), `kq_xq` `n_vals*2` B (`:27091`), `kq_xh` `n_vals*2` B (`:27120`), `fa_partial` `n_batch*n_head*n_q*n_splits*floats*4` (`:22148-22164`), `splitk_partial` `ksplit*ne01*4` (`:24580`), `exchange` per-slice bytes (`:16213`) | scale with the ubatch (`N`, `n_vals`) and the split count: they grow with the first large prompt, at depth, in a request |
| B3 | `scratch_bufs` tag map (`transpose_2d`, `fa_dpas2_q16`, mask / `kf` / `vf` / `cr` classes, the `get_tensor.*` staging) | `:804-805`, `:8280`, `:21384-21650`, `:13621-14686` | same grow-only helper |
| B4 | gather mirror `kq_stage_q/d` | `:15636-15640`; reserved at load by the cache | 187.7 MiB (`paper`: §24a) |
| B5 | bank-stream ring `bstream.ring` | `:1116-1125`, filled at the first prefill graph after the order is learnt (`:27189`) | `K x max bank tensor`, K = 4 default: 751 MiB (`paper`) |
| B6 | `fa.fd_partial` pool | `:22155-22158` | grows with `partial_size_bytes`, `clFinish` before each grow |
| B7 | **per-call temporaries** (create, use, release inside one op): dequantized parent for SOA restore (`:19778`, `:19846`, `:19930`, `:19986`), FA K/V/mask pads and block map (`:22017-22080`, `block_n * n_head_kv * n_batch * nb1`), non-Intel FA partial (`:22167`), padded B for GEMM (`:23225`, `:25137`, `:25870`, `:26458`: `K*N_pad*4`), `aos` / `buf_tq,td` (`:26831-26953`) | in the op | one `clCreateBuffer` per call, the driver sees each one |
| B8 | Adreno-only attention scratch and constants (`:20191-20213`, `:22533`) | not served on Intel (`inferred`: guarded by Adreno paths) | n/a |

Not VRAM and not in scope: the USM host bank, the 32 MiB `stage_ring` (`:11860-11870`), the lend bank
(patch 0068, USM host). Which B7 temporaries are reachable on the Intel path under the served flags is **not
traced** (`inferred` from op names only); the counting step in section 6 settles it.

C. **Not ours and not controllable**: the OpenCL runtime's kernel ISA, per-thread scratch, command buffers and
queues (`paper`: design §27 "the runtime's own surfaces"; the xe query's `used - fdinfo` is its measure, §28a).
An arena does not remove C; it turns the unknown B into one number and makes C the only unknown.

## 4. Design sketch: an arcint arena

### 4.1 Shape

One arena per device, owned by the backend context (one context per platform, patch 0023), made of `n` chunks:
each chunk one `clCreateBuffer` of at most `CL_DEVICE_MAX_MEM_ALLOC_SIZE` (the type already splits at that size).
Total = `card total - kernel-owned VRAM - runtime reserve`, taken once at load from the xe query
(`vram-user-space.md` stage 0), touched at load so that failure happens there (Strata zero-fills its arena,
`expert_cache.cpp:685-690`, `inferred` as the reason). Offsets come from a small free-list allocator (ggml's own
`ggml_dyn_tallocr`, `ggml-alloc.c:121-312`, is the model; arcint has under ~100 live ranges). Two ends: stable
ranges (weights, slots, KV, ring, preallocs) from the bottom, transient ranges (compute buffers, loaned slots) from
the top, so a loan's holes coalesce (`inferred`; Strata's borrowed region is one contiguous run, `prefill.cpp:1397`).

### 4.2 Hooks, outermost first

1. **ggml buffer type** (`ggml_backend_opencl_buffer_type_alloc_buffer`, `:14841`): instead of `clCreateBuffer`,
   `arena.alloc(size)` returns (chunk `cl_mem`, origin). The buffer context stores them and does not release the
   chunk; `free_buffer` returns the range. The USM-host buffer path (`:14855-14872`) gives `ctx->buffer[0]` a 64-byte placeholder
   `cl_mem` and never enters the arena (`code`). `init_tensor` sets `extra->data_device = chunk` (the **only** place it
   is set from the buffer, `:11684`; verified against 0057-0068's patch files, which add no such assignment, `code`) and `extra->offset = origin + (data - base)`. Everything downstream already
   adds `extra->offset` (`code`: the SOA carve at `:11944-11955`, the per-op sub-buffers such as `:9654-9658`, the
   the planar set_tensor write uses `e->offset + view_offs + offset`). 856 `data_device` uses exist and were not all read
   (`inferred`: they follow the same convention); a red case is a test-backend-ops run with the arena forced on and
   a non-zero origin. `data_device` must be the **root** chunk, not a per-buffer sub-buffer: the backend creates
   sub-buffers from it per op (`:9658`, `:22994`), and OpenCL forbids a sub-buffer of a sub-buffer (`paper`: spec).
   Set `get_max_size` to the chunk size so gallocr's chunking (`ggml-alloc.c:160-177`) never asks for a larger run.
2. **The backend's raw buffers** (section 3 B1-B6): `ggml_cl_buffer::allocate(context, size)` gets a sub-buffer of
   the arena (`clCreateSubBuffer`, REGION): a host-side object over memory the process already owns. The member
   stays a `cl_mem`, so every `clSetKernelArg(..., &buf.buffer)` is unchanged. `allocate` today needs only the
   context; the arena is found from it (a per-context map, as `g_bstream_ctxs` already is, `:11225`).
   Grow = new range + release old; to avoid fragmentation, reserve the high-water at load (hook 4).
3. **Per-call temporaries** (B7): become tagged grow-only scratch (the pattern B3 already uses) or are served from
   the transient end and freed at op end. This is the only part that edits op code.
4. **Load-time reserve**: Strata's count-then-carve (`verify.cpp:595-615`): run the worst-case graph once (the
   scheduler's `graph_reserve` at the max ubatch exists; the loan's `sched_reserve` at the prompt chunk is patch 0068, and
   patch 0052 is `--llama-reserve-cells`, a cell cap on the reserve) with the allocator in
   counting mode; the counted high-water per tag is reserved once. B4 and B5 already have a load-time size (the
   largest bank tensor is known at cache build); today the ring waits for the first prefill.

### 4.3 What the slot loan becomes

`free tail buffer` = `arena.free(range)`: no `clReleaseMemObject`, no driver call. The prompt's compute buffer is
`arena.alloc(size, transient)` and lands in the lent range (top end, coalesced). The return frees the compute
range, re-takes the same range for the tail slots, and the refill runs as now. The loan's `first_touch` term and
the "outside the sum" error go away for everything in B, because B is inside the arena and counted by it; the
remaining rule is `arena.free_bytes() >= need`, a pure in-process figure. The card-wide `used` query shrinks to
a startup sizing and a watchdog on C. Tensor pointers survive as today (§1 of the loan design): only `buffer`,
`data`, `extra->offset` move. This is Infernix's `FrameLease` and Strata's borrowed `Alloc`, minus their
frame-granularity.

## 5. Risks

1. **Eviction granularity (decides the design).** TTM evicts buffer objects, not byte ranges (`inferred`: the
   campaign's observed ~15 GiB moved to GTT is a set of whole objects). One 22 GiB object under pressure from a
   second client, or from the runtime's own late allocation (kernel ISA, scratch for a newly compiled kernel,
   queue buffers), could be evicted **entirely**: worse than today's many small objects. The arena therefore does
   not remove the reserve, it relocates it: size = total - (measured `used` beyond the process at the high-water of
   C) - margin, and the only allocation left after load is the runtime's. Chunking at a few GiB (the max-alloc
   split) limits the blast radius; whether that helps depends on the kernel and is question 4. The `dmem.min`
   stage (vram-user-space stage 3) is the kernel-side complement.
2. **Max allocation and 64-bit addressing.** `CL_DEVICE_MAX_MEM_ALLOC_SIZE` caps a chunk. Past 4 GiB a kernel must
   be compiled for 64-bit addressing (Strata adds `-ze-opt-greater-than-4GB-buffer-required` to its JIT,
   `docs/INTEL.md:152`, `paper`); arcint's >4 GiB slot buffers already run (`paper`), `ggml-opencl.cpp` has no such
   option string (`code`, grep), so the runtime must be deciding it. A bigger chunk is the same question.
3. **Alignment.** Sub-buffer origins must be multiples of `CL_DEVICE_MEM_BASE_ADDR_ALIGN` (`paper`: spec,
   `CL_MISALIGNED_SUB_BUFFER_OFFSET`); the backend already aligns every origin with `align_to(.., alignment)`
   (`:11944` and ~40 more) and the Intel 2D-block kernels want `alignment % 64 == 0` (`:2505`, `:4563`). The arena
   rounds every range to `max(alignment, 256)` (Strata rounds takes to 256 B, `prefill.cpp:376`) and keeps the
   chunk base as returned by the runtime.
4. **Intel USM versus `cl_mem`.** Device USM (`clDeviceMemAllocINTEL`) would give raw pointers and free sub-
   allocation, but the kernels take `cl_mem` plus a `ulong` offset; `clSetKernelArgMemPointerINTEL` (`usm_set_arg_ptr`, member
   `:1068`, resolved `:8208`) is used by the host bank (`bctx->usm_set_arg`, `:27281-27291`, `:30353`, `:30367`), by the
   doorbell and pub-back pointers (`:15591`, `:15624`) and by the `g_kq_single_ptr` argument (`:30443-30444`), `code`. Rewriting ~800 kernel-argument sites is not a first
   step. `cl_mem` chunks + sub-buffers keep the kernels as they are. Cost: sub-buffers add host objects and a
   `clSetKernelArg` on a sub-buffer; decode is host-bound, the per-launch floor being A770 2.2-2.6 us and B60 0.6-0.9 us (`paper`:
   `docs/campaigns/llama-engine-kernel-gap.md:323`), so
   question 6 measures it. **Sub-buffer count** has no stated limit in the spec beyond memory; the SOA split
   already makes 2-5 per tensor.
5. **Fragmentation.** Grow-only `allocate` is a free-then-create (B1) and gallocr reallocates on every growth
   (`ggml-alloc.c:935-944`); in an arena that is free-then-alloc and fragments unless the high-water is reserved
   up front and transient ranges live at the other end. Strata's answer is the count pass; adopt it.
6. **Xe aliased pages.** Strata saw big xe allocations with pages mapped onto the same memory and wrote
   `malloc_device_guarded` (`docs/INTEL.md:155`, `paper`; L0/SYCL, not shown for OpenCL). An arena makes one very
   large allocation the single point of failure for it. Question 5.
7. **Exclusivity.** The arena commits the card. A unit already holds its card permanently (`paper`); the image and
   TTS services stop or switch units through the manager, so nothing new is lost, but a window that shares the card
   with another tenant (stage 3) must keep the arena below the tenant's `dmem` floor.
8. **Lazy population.** Whether a `clCreateBuffer` of 22 GiB makes the memory resident at creation or at first use
   is unknown (`inferred`: a first-use population would make the arena's first prefill the same cliff, only once).
   Touch at load (4.1) and read the fdinfo resident figure to see (question 3).
9. **Accounting stays honest.** llama's memory breakdown asks buffer sizes (`llama-model.cpp:2528`); arena-backed
   buffers report their own size, so those figures stay valid; the arena's total is added next to them.

Not adopted from the references, with the reason (the README's "scoping out is a deviation" rule): the **elastic
VMM range** (Strata `--vram-elastic`, Infernix `VmmArena`). Reason: OpenCL has no virtual-memory call
(`inferred`); Level Zero has `zeVirtualMemReserve`/`zePhysicalMemCreate`/`zeVirtualMemMap` but a `cl_mem` cannot be
made from them in this backend. It is the parts that make Infernix's frame pool grow and shrink in place, so the
operator is asked (section 7, question 9); the fixed arena above covers the loan, not elasticity.

## 6. The smallest first step that pays

**Step A: count (no behaviour change).** A tag, a live-bytes and a high-water counter in `ggml_cl_buffer::allocate`
and a wrapper for the raw `clCreateBuffer` sites of B7, one log line per tag at the end of a request, and the same
sum exported through a proc address so the loan's rule can read it (`ggml_backend_opencl_extra_bytes`). It
produces the inventory column section 3 lacks (the sizes of B2, B3 and B7 are rules, not measured numbers) and
settles which B7 sites are live on the Intel path. Red case: a mutation that leaks one allocation must show in the
counter. One served long prompt on each card.

**Step B: the side arena.** One `clCreateBuffer` at load, sized from step A's high-water of B1-B6 plus margin; the
helper hands out sub-buffers (hook 2). It is `ggml-opencl.cpp`-only: no buffer-type change, no llama change, the
slot loan untouched. Why it pays first: B is exactly what grows in a request, at depth (the ring's first touch, the
preallocs' first large ubatch); today those are driver allocations made when the card is fullest, and they are why
the loan's rule needs the `first_touch` term and a measured allowance. After B they are one allocation made at
load, and the loan's remaining unknown is C. Gate (`vram-user-space.md`): answers at the answer-level bar, zero GTT
growth beyond the host banks in the depth legs, no xe resets, prefill and decode inside the run-to-run spread.

Then C (ggml buffers through the arena, hook 1), then D (the loan as range reassignment, 4.3). C and D wait for
question 4.

## 7. Open questions for a measurement

1. **The two numbers the load log already prints**: max mem alloc and mem base align, on the B60 and the A770
   (`ggml-opencl.cpp:7869-7873`). Sets the chunk count and the alignment.
2. **The B inventory at depth** (step A): per tag live and high-water, Flash-Next at 103k and the agent at 245k,
   cold and warm; which B7 temporaries run on the Intel path.
3. **Creation cost and residency of a ~22 GiB `cl_mem`**: time, `drm-resident-vram0` immediately after create,
   after a fill, and after the first kernel.
4. **Eviction granularity** (decides section 5, risk 1): an arena at `total - 1 GiB` in one process, a second
   process allocating 2 GiB; read both processes' resident-VRAM and GTT. Also read the xe/TTM eviction path in
   the DKMS tree (owed; the tree is on the card host).
5. **Aliased pages on an OpenCL big allocation**: Strata's `arena_alias_check` run against arcint's slot buffer
   and a candidate arena chunk (`sycl_queue.hpp:168` `arena_alias_check`, `:259-290` `malloc_device_guarded`).
6. **Sub-buffer overhead on decode**: Flash-Next and the coder at their served settings, buffers as sub-buffers
   of one chunk against plain buffers, same window, per `CLAUDE.md` (the host gap decides it).
7. **The reserve left for C**: `used - arena - other_clients` after load, after a depth prompt and after a
   recompile-forcing change (a new kernel specialization), to size the margin.
8. **Whether the lent range really coalesces**: a loan cycle with the arena's free list logged (ranges, holes),
   including the MTP context and a second concurrent request.
9. **Operator decision**: scope out the elastic VMM range (needs Level Zero, section 5 last paragraph) or open it as
   its own campaign with Infernix `vmm_arena.cu` as the reference.
