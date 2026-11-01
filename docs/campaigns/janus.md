# janus — arcint switching the model it serves

**Open** (2026-11-01). Recon only: this document reads the code and the
references and prices the options. No card window was run for it, no build,
no service touched.

Charter: today a model switch is a **unit switch** — one process per served
model, the service manager stops one unit and starts the other on the same
card. The operator's question is what a switch could cost instead: keep a
model's prepared weights in VRAM, host RAM or on the NVMe and promote it to
serving fast (standby tiers).

Evidence classes used below: `code` (read in source or a patch), `paper`
(published text or product documentation), `measured-here` (a number this
project's record carries; where the number comes from the operator's notes it
is written `measured-here (operator record)`), `inferred` (reasoned from the
code and sizes, not run).

Line numbers are from the tree this repository builds: arcint's own
`src/` as of this branch, and the pinned llama.cpp with the series applied
(1.2.6-era tree, patches 0001-0063; a few hundred lines may drift with
0064-0072 — the named functions are the anchors). External references were
read on 2026-11-01: llama.cpp at the pin, llama-swap, vLLM and ollama from
their source (revision named at each use), LM Studio from its documentation,
and Strata and FreeToken from copies of the cited files of the public
repositories `github.com/Niko1221/Strata` and
`github.com/FlashML-org/FreeToken` (`code`).

## Charter

- **A switch is a unit switch today.** One arcint process per served model
  (`code`: one `--gguf`, one architecture admitted, DESIGN §2, §4.4); the
  unit files exclude each other on the card (`code`: the units' card-exclusion
  directive), so the predecessor must exit before the successor
  starts. There is no in-process model switch and no second main model.
- **Measured cost of that switch** (`measured-here (operator record)`): a cold
  load takes the agent 74-75 s, the coder 57-71 s, the two-card configuration
  ~135 s; **more than 95 % of it is initialisation, not I/O.** These are the
  numbers this document tries to break down; they are not re-measured here.
- **The window ends at the listening line** (`code`: `src/main.cpp:154`,
  after the backend is constructed). Every phase named in §1 sits between
  process start and that line, except §1 phase 9, which is the first
  request's cost.
- **The operator's idea**: standby tiers — a model's prepared weights held in
  VRAM, host RAM or on the NVMe, promoted to serving on demand.

## Reference to follow

There is no single reference engine; the reference split is:

- **llama.cpp's own server** at the pin: a **router mode** with one child
  process per model, LRU unload at `--models-max`, and an idle **sleep**
  that destroys the model and reloads it on the next request (`code`:
  `tools/server/README.md:1843-1900, 228-231`; `tools/server/server-models.cpp:1107-1130`;
  `tools/server/server-context.cpp:1058-1080`; `common/arg.cpp:3831`).
  Measured effect in its own repository: none cited; the reference here is
  the mechanism, not a number.
- **vLLM's sleep mode** (`code`): `/sleep?level=1|2` and `/wake_up?tags=`;
  level 1 offloads weights to host RAM, level 2 discards them and reloads
  from the model source on resume (`vllm/device_allocator/sleep_mode_backend.py`,
  the `SleepModeBackend.suspend` docstring; `vllm/v1/worker/gpu_worker.py:236-292`;
  `vllm/entrypoints/serve/dev/sleep/api_router.py`). The levels rest on CUDA
  virtual-memory mappings: the allocator reserves the addresses and maps the
  physical backing (`cuMemCreate`/`cuMemMap`, `csrc/cumem_allocator.cpp:169`),
  and level 1 **unmaps that backing while the pointers stay valid**
  (`cuMemUnmap`, `:254`), remapping it under the same pointers on wake
  (`create_and_map`, `:169`), so no tensor is rebound. That is the shape of the operator's warm standby;
  §2(c) prices why OpenCL cannot copy it unchanged.
- **Strata** (`code`, `github.com/Niko1221/Strata`, the cited files read as
  copies of it): an idle-unload tier and a lazy start — `--idle-unload N`/
  `idle_unload_s`, the model unloaded after N s without requests and reloaded
  on the next one, and `--lazy`/`lazy_load` plus `POST /load` to load ahead of
  a request (`docs/DETAILS.md:330, 336, 346-348, 449, 473`); the elastic-VRAM
  page's "unload, restart and crash" says a restart is a fresh sizing from what
  is free *then* (`docs/VRAM_ELASTIC.md:95-103`). Its expert arena is a
  **native pack read once at startup into a `PinnedArena`** (host RAM,
  `cudaHostRegister` best-effort), or optionally mapped read-only from
  `experts.bin` instead of read (`STRATA_ARENA_MMAP=1`, `docs/DETAILS.md:247-248`;
  `ArenaExpertSource`, `include/strata/core/expert_source.hpp:1245`, the
  read-once comment at `:1242-1243`, the mapping member at `:1305`) — the
  reference for (b)2's on-disk prepared artifact and a data point for (c)'s
  pinned host tier: the comment at `:1237-1240` records ~19 GB/s in the engine
  reading `MapViewOfFile`, against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
  reading the same bytes with `std::fread` into an ordinary heap arena — the
  `PinnedArena` is the consequence, not the measured arm (`code`, Strata's own
  measurement). A warm restart is considered explicitly: `experts_unbuffered`
  decides whether the expert files are better read unbuffered so that "a warm
  restart after an idle unload never gets slower"
  (`include/strata/core/pinned.hpp:169-180`). Its management protocol has a
  supervised `runtime.restart` with `allow_source_switch`
  (`docs/MANAGEMENT_PROTOCOL.md:44-45`) — option (a) with a supervisor.
- **FreeToken** (`code`, `github.com/FlashML-org/FreeToken`, the cited files
  read as copies of it): `ft-daemon` owns exactly one `ft serve` child (spawn,
  kill, orphan detection; `start()` is idempotent, a live child with different
  arguments is a 409 Conflict), and `switch()` composes stop then start
  (`python/freetoken/daemon/serve_manager.py:1-16, 41, 165, 269, 354, 431`);
  `POST /engine/switch` is the endpoint (`python/freetoken/daemon/app.py:255-270`).
  Option (a) with a supervisor, one child, no weight tier.

The prior survey of the same ground is `research-agent-lanes.md` (llama.cpp,
vLLM, SGLang, ExLlamaV3, ollama, MLC-LLM at source, 2026-10-08) and
`research-cold-start.md` (kernel-compilation caches across stacks,
2026-09-05); both are read here and their evidence classes are kept.

## Gate

No gate is set for a change yet — this is recon. The **decision measurement**
of §4 has a bar (its arms and the bar are stated there, and it must be able to
fail). If any option introduces a copy or a transformed artifact of the
weights, `DESIGN §3.4` point 2 applies unchanged: integrity checks on copied
data stay exact, while correctness is judged at the answer (facts, needle,
acceptance task) within the answer-level bar.

---

## 1. Where the 57-75 s go, per phase

The code does not timestamp the load; it prints one total and a handful of
sub-lines. The phase list below is therefore `code` + `inferred` about their
share; §1.2 names the lines and probes that decide the share in a window.

### 1.1 The phases between process start and the listening line

| # | phase | what runs | evidence | class | per load | can be kept |
|---|---|---|---|---|---|---|
| 0 | process start | dynamic linking, OpenCL platform/device discovery, `ggml_cl_init` on first use | `code`: `ggml/src/ggml-opencl/ggml-opencl.cpp`, `ggml_cl_init`; `src/main.cpp:60-110` | CPU | yes | no (address space) |
| 1 | GGUF metadata | `gguf_init_from_file`, hparams, tensor table, optional split shards | `code`: `src/llama-model-loader.cpp:533` (ctor), `:726` (the `loaded meta data` line) | CPU, small | yes | metadata is on disk already |
| 2 | file read / mappings | `init_mappings`; `AUTO` load mode resolves to **no mmap** for the OpenCL device (no `mmap_support`), so tensors are read and uploaded one at a time; `--llama-cpu-moe` sets `MMAP`+`lazy` for the host bank | `code`: `src/llama-model.cpp:1577-1585`, `:1604` (`loading model tensors`); `src/llama-model-loader.cpp:1414`; `src/llama-mmap.cpp:492-493`; `src/exec/backend_llama.cpp:586-604` | I/O + page cache | yes | page cache |
| 3 | OpenCL kernel programs | `cl_program_cache_init` then `load_cl_kernels` builds one program per kernel source (195 `.cl` files in the pinned tree); each is either loaded from the on-disk cache or `clBuildProgram`-ed from source | `code`: `ggml/src/ggml-opencl/ggml-opencl.cpp:14951-14961` (alloc_buffer → init + `load_cl_kernels`), `:1788` (`load_cl_kernels`), `:1599/1608` (`clCreateProgramWithSource`/`clBuildProgram`), `:1643/1662` (cache look-up); `ggml/src/ggml-opencl/cl-program-cache.cpp:307` (`cl_program_cache_init`), `:184` (default dir), `:358-391` (MISS/HIT) | **compile** | yes | **disk** (llama.cpp cache + the runtime's own binary cache) |
| 4 | tensor upload / transforms | `load_all_data` stages each tensor (the async pinned path is dead on OpenCL — the caps are false — so the per-tensor host buffer is what runs) and writes it to VRAM; low-bit types (IQ2_XXS/IQ2_XS/IQ2_S/IQ3_S/Q2_0) are **planar-converted on the CPU** field by field before a blocking write; IQ4_NL gets a scale/quant split | `code`: `src/llama-model-loader.cpp:1556`, `:1705-1815`; `ggml/src/ggml-opencl/ggml-opencl.cpp:11916-11921` (`ggml_cl_lb_planar`/`ggml_cl_lb_convert`), `:12009-12027` (`set_tensor`, the planar branch), `:11942-11952` (`ggml_cl_usm_set_tensor`'s IQ4_NL split); Q4_K/Q8_0 fall through to the plain write at `:13679-13686` | **copy + CPU** | yes | VRAM image; a pre-transformed file on disk |
| 5 | expert cache init | `llama_expert_cache::init` reads **every expert of every MoE layer** from the GGUF, writes the slot tensors (VRAM) and the bank tensors (pinned USM host), builds the index tables, and runs copy/table checks; the slot loan adds a `lend bank check` that re-reads every lendable row | `code`: `src/llama-model.cpp:2069`; `src/llama-expert-cache.cpp:261` (init), `:546-612` (the fill, the `filled in %.1f s` line at `:606`); the check `:596-599`; the loan's check `prefill-slot-loan-design.md:278` | **CPU + copy** | yes (Flash-Next only) | the host bank already is; the slots must be re-uploaded |
| 6 | drafter | `--llama-mtp-gguf` / `--llama-dflash-gguf` load a **second model** (its own phases 1-4) and a second context | `code`: `src/exec/backend_llama.cpp:845-891` | as 1-4 | yes | as 1-4 |
| 7 | context | KV/recurrent allocation, `sched_reserve` (worst-case prompt and decode graphs, compute buffers), per-lane windows | `code`: `src/exec/backend_llama.cpp:753-820`; `src/llama-context.cpp:562` (`sched_reserve`), `:740` (timer), `:885` (`reserve took %.2f ms`) | CPU + alloc | yes | no (re-init) |
| 8 | server | chat template, router/limits, slot pools, then the `listening` line | `code`: `src/exec/backend_llama.cpp:909-1120`; `src/main.cpp:120-154` | CPU, small | yes | no |
| 9 | first request | graph build and **first use of each kernel** — the load path runs **no warm-up forward** | `code`: `src/llama-cparams.h:53` and `src/llama-context.cpp:213` (the flag exists, defaults off), no `warmup` call anywhere in arcint's `src/`; contrast llama.cpp's own server `--warmup` (`common/arg.cpp:1971`) | compile + CPU | yes | disk caches make it cheap; a warm-up pass would move it inside the load |

Note on phase 9: this is the *current* libllama path. The pre-warm forward
`research-cold-start.md:111` records as arcint's (`load_paged()`) belonged to
the **retired OpenVINO engine**; `load_paged` is not in this tree (`code`).

What this says about standing by:

- **Keepable on disk, once**: phase 3's program binaries (already cached,
  but into a **default temp directory** — `cl-program-cache.cpp:184`, i.e.
  `$TMPDIR/llama.cpp/cl-cache`; the units keep a shared temp directory rather
  than a per-process one, which is what makes that survive a service restart,
  `code`), and — if built — a pre-transformed
  weight file for phase 4's planar types (Flash-Next). Metadata (phase 1) is
  free.
- **Keepable in host RAM**: the MoE bank (already pinned USM), the page cache,
  and a lend bank.
- **Must rerun per process**: 0, 1, the buffer creation and every H2D copy of
  4 and 5 (unless the weights live in a surviving process), 6, 7, 8, and the
  first-request tail 9.
- A standby process (option c) is precisely the proposal to move 4 and 5's
  copies out of the critical path by keeping the bytes in host memory.

### 1.2 What the code already logs, and the probes the window should read

Existing lines (all on stderr, tag `lgc`, no timestamp of their own — the
service manager's journal supplies one, `code`: `src/util/log.cpp:16-60`):

- total: `lgc  load: llama.cpp <desc> (<arch>, <GiB>) on ..., in <T> s; ...`
  (`src/exec/backend_llama.cpp:1108-1111`). `T` is measured from before
  `llama_model_load_from_file` (`:606`) through the context, MTP and mmproj
  setup, i.e. phases **1-7**.
- phase 1: `llama_model_loader: loaded meta data with ...` (`:726`);
  phase 4: `load_tensors: loading model tensors ... (load_mode = ...)` and
  `load_tensors: ... model buffer size = ... MiB` (`src/llama-model.cpp:1604`,
  `:1995`); phase 7: `llama_context: reserve took <X> ms, sched copies = N`
  (`src/llama-context.cpp:885`); phase 5: the expert cache's
  `... filled in <T> s` (`src/llama-expert-cache.cpp:606`); the slot loan's
  `lend bank check ... (<T> s)` while the loan is on
  (`prefill-slot-loan-design.md:278`).
- phase 3: `ggml_opencl: loading OpenCL kernels` plus, with
  `GGML_OPENCL_KERNEL_CACHE_DEBUG=1`, one `HIT`/`MISS`/`SAVE` line per
  program and a running tally (`cl-program-cache.cpp:282-300`). The cache's
  own enable line is `ggml_opencl: kernel cache enabled at '<dir>'`
  (`:339-342`), and the `ggml_opencl: kernel cache disabled ...` line when the
  directory cannot be used is an INFO/`fprintf`, not a WARN (`:313-334`).
- **No new probe is needed.** The lines above, timestamped, bound every
  phase: `loaded meta data` opens phase 1, `loading model tensors` phase 2/4,
  `loading OpenCL kernels`/`kernel cache enabled at` phase 3, `constructing
  llama_context` (`src/llama-context.cpp:183`) closes the model load and opens
  phase 7, `reserve took` closes the reserve, the drafter's own `loading model
  tensors` bounds phase 6, `filled in` closes phase 5, and `listening on` the
  total. A temporary stamp is justified only where a run shows a gap between
  two consecutive lines.
- To separate reduce/reuse from transfer in phase 4: `/proc/<pid>/io`
  (`read_bytes`, `rchar`) across the load, and a `unitrace -d --opencl`
  summary against the load (the tool the loan and LEO rounds used,
  `code`: `vram-user-space.md`). On the CPU side, `strace -c -f` over the
  load counts the `pread`s and the time in them; `perf` attributes the
  planar `memcpy` loops (`ggml_cl_lb_convert`) and the program build.
- Bar for the reading itself: the phase times must add to the measured total
  within the window's spread, and the run must be repeated in the reverse
  order (`CLAUDE.md`: a test that cannot fail measures nothing). §4 states
  the decision arms.

### 1.3 Current state — the measured anchor that already exists

Flash-Next's own load was measured with the 1.2.0 package
(`measured-here`, `docs/campaigns/flash-next-llama-engine.md:1719-1723`):
**55 s cold, 35 s warm**, with 43.8 of the 70.6 GiB of shards resident after
a cold load and **0 major faults** in every answer. Two facts follow:
the page-cache share of the load is finite but real (~20 of those 55 s, ~36 %,
there), and even fully warm the same model needs 35 s — which for Flash-Next
is the expert cache's fill and the program/context work. **The operator's
">95 % initialisation, not I/O" figure belongs to the dense units** (the agent
and the coder); it cannot be carried across to Flash-Next, whose own measured
pair splits as above. The agent and the coder have no expert cache; their
74-75 s and 57-71 s therefore cannot be the expert fill, which points at
phase 3 (program build) and the re-init phases as the candidates the window
has to separate. This is the strongest inference in this document and the
first thing the measurement must confirm or kill.

---

## 2. The options, priced

All costs below are `inferred` unless a `measured-here` is named. VRAM figures
from the record, each with its own source: the agent's 24,195 MiB peak at
245,760 context (`measured-here`, `docs/campaigns/agent-context.md:139`;
DESIGN §3.3 carries a different, earlier point, 24,311 MiB at 138k on 0.7.12),
Flash-Next IQ3_S 23,478 MiB (`measured-here`, DESIGN §7.2), the coder 15,971
of 16,384 MiB (`measured-here`, DESIGN §3.3).

### (a) Today's unit switch

- **Mechanism** `code`: one process per model, card exclusion between units;
  stop, then start; no overlap. This is the baseline.
- **Switch latency** `measured-here (operator record)`: 74-75 s (agent),
  57-71 s (coder), ~135 s (two cards), plus the service manager's stop.
- **Memory**: one model resident. Host RAM holds the page cache and, for
  Flash-Next, the bank.
- **What breaks**: every in-flight request and its KV/recurrent state is
  lost; no request is served during the switch; the card is idle.
- **Invariant**: none touched. It is today's configuration.

### (b) Faster cold loads (program cache, pre-transformed weights, a staged upload)

- **Mechanism**:
  1. **Program cache**: keep `cl-program-cache`'s directory off a temp path
     that can be cleared (a persistent directory under the service's state
     dir), and confirm the HIT path actually runs in production. The
     runtime's own binary cache sits one layer below
     (`paper`/`code`: `research-cold-start.md`, NEO `NEO_CACHE_*`); both
     must be warm for a compile to cost nothing. This is the only item of
     (b) that is a configuration change.
  2. **Pre-transformed weights on disk**: write the phase-4 transform once,
     keyed by (source, type, kernel field layout), and read it instead of
     converting on every load. **It buys nothing for the two models the
     operator measured**: the transforms exist only for the **planar** low-bit
     types (IQ2_XXS/IQ2_XS/IQ2_S/IQ3_S/Q2_0, `ggml_cl_lb_planar`,
     `ggml-opencl.cpp:11916-11921`) and for the IQ4_NL scale/quant split
     (`:11942-11952`); `set_tensor` has no branch for Q4_K_M or Q8_0, and the
     agent and the coder are Q4_K_M — they fall through to the plain write
     (`:13679-13686`). It helps **Flash-Next's planar types** (and its
     n-gram/PLE table) only. Its reference is Strata's prepared on-disk arena
     pack (`STRATA_ARENA_MMAP`, above).
  3. **A staged/pinned upload**: the loader's async pinned path exists but is
     **never taken on OpenCL** — the device reports `async = false`,
     `host_buffer = false`, `events = false` (`code`:
     `ggml-opencl.cpp:15116-15122`), so `load_all_data` falls to the
     per-tensor staging buffer (`src/llama-model-loader.cpp:1806-1810`).
     Giving OpenCL a pinned host buffer, events and an async write is
     **backend work** (a buffer type + the device caps), not a loader
     setting.
- **Switch latency** `inferred`: item 1 removes the program-cache share if
  phase 3 is the owner (the §1.3 hypothesis); items 2 and 3 remove CPU/planar
  and staging work for the models they apply to. The measurement in §4
  decides; nothing here claims a number.
- **Memory**: one model resident, as today. Disk cost of item 2: one
  transformed copy per **MoE** model (Flash-Next's shards are 70.6 GiB),
  unless the transform is keyed and paged.
- **What breaks**: nothing structural. A transformed file must be
  invalidated when the kernel field layout or the quant changes, and an
  interrupted write must not become a half-file (write temp + rename, as
  `cl-program-cache.cpp:253, :460` already does for its blobs).
- **Invariant**: none. The transformed copy is a copy: `DESIGN §3.4` point 2
  keeps the integrity check exact; the answer-level bar is unaffected by a
  lossless transform, and an arm that skips a field must be caught by a
  byte check, not by the answers.

### (c) A warm standby process per model, weights in host RAM, a promote that does not exist yet

- **The reference and its limit.** vLLM's level-1 sleep keeps the model's
  **pointers** valid while the physical backing is unmapped, because the CUDA
  VMM allocator owns the mapping (`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/
  `cuMemUnmap`, `csrc/cumem_allocator.cpp:169-268`) and can map the same virtual
  address again in `wake_up`. **OpenCL has no such mechanism and llama.cpp has
  none either.** `cl_mem` has no address-preserving re-backing, and after load
  every tensor is bound to a `cl_mem` plus an offset in `tensor->extra`
  (`code`: `ggml-opencl.cpp:12009` and the plain path at `:13679-13686`); there
  is no buffer migration. So the vLLM shape is the **reference for the state
  machine**, not a mechanism to port. That holds for the XPU port too, which
  is not a CUDA-only route: `XpuMemAllocator` discards and unmaps XPU physical
  memory and remaps it to the same reserved virtual address
  (`vllm/device_allocator/xpumem.py:112-118`; named by `sleep_mode_backend.py:115`).
  Both own their allocator's virtual memory; arcint's OpenCL backend does not,
  and the SYCL/Level Zero path that does own its VM is dead under xe on this
  host (`measured-here`, DESIGN §2: SYCL aborts under xe with a memcpy abort;
  `research-vram-arena.md` §8.1: native OpenCL cannot see an L0 range, two
  drivers and two VMs).
- **What a promote therefore costs.** New device buffers for the promoted
  tensors, a rebind of every tensor's `extra` (the backend's `data_device` and
  `offset`), and then a re-reserve. The nearest existing machinery is the
  expert cache's own slot/bank rebinding (the `pair_bank`/`exchange` entry
  points of patch 0021, used by `llama_expert_cache`), but it moves fixed-size
  slices between an already-open bank and slots, not a whole model. **This is
  new backend code** (a promotable buffer type + a rebind pass), and it is the
  first price of (c), before any byte moves.
- **Mechanism, if built**: hold the model's tensors in a **pinned USM host
  buffer** — the allocator path the expert cache's bank already uses
  (`code`: `ggml_backend_opencl_set_alloc_host`, `ggml-opencl.cpp:14939-14975`,
  called around the bank allocation at `llama-expert-cache.cpp:471-474`). On
  promote: create the device buffers, copy, rebind, re-reserve (or, for the MoE
  bank, keep reading over the link exactly as a decode miss does today,
  `code`: DESIGN §3.9's `bank` row, which needs no promote at all).
- **Switch latency** `inferred`: **not a 1.2 s copy** — that figure is the
  H2D bytes alone (agent ~16 GiB over the B60's 13.9-14.3 GB/s ≈ 1.2 s; coder
  14.6 GB over the A770's 1.8 GB/s ≈ 8 s; both links `measured-here`, DESIGN §2
  table). Until the promote path exists, the switch is the cold load plus the
  new work; once it exists, the floor is the copy **plus** the rebinding pass
  and the re-reserve.
- **Memory**: an idle standby must hold ~0 VRAM — an idle process holding VRAM
  is what the kernel evicts under pressure (`measured-here`, the vram-user-space
  cliff of 2026-10-11). It holds the model in host RAM: ~16 GiB (agent),
  14.6 GB (coder), or Flash-Next's bank 20-40 GiB. The **binding resources are
  host RAM and pinned memory.** DESIGN §2 sizes the serving container at 52 GiB
  (96 GB is the planned host total), and the bank is pinned USM; a pinned
  standby competes with the served model's own bank. So at most one dense model
  or one MoE bank fits beside a served model.
- **What breaks**: the KV cache, recurrent state, checkpoints and the MTP
  context die with the context and are rebuilt (`code`: checkpoints are
  in-process only, DESIGN §3.4/§8.4; `kv-checkpoint-restore` is the open
  campaign). The expert cache's slot placement and adapt counters restart (it
  has an off-by-default save/load, patch 0070; `--llama-expert-cache-save` is
  opt-in, `code`: `docs/campaigns/expert-foresight.md` item 14). Cached prompts
  are lost.
- **Invariant**: **§3.3 is not broken by a zero-VRAM standby** — its text is
  that the serving peak is sized by measurement, not by an admission model, and
  an idle standby does not size a serving peak. **§4.3 is not broken either**;
  its text is that the context is configuration decided once by measurement, not
  "a reservation". The real constraints are host RAM and pinned memory
  (`DESIGN §2`), and the vram-user-space rule the moment VRAM is held idle. If
  a host-RAM standby beside a served model does not fit DESIGN §2's budget, that
  is the conflict to put to the operator (§5).

### (d) One process with several models resident when they fit

- **Mechanism**: several main models in one address space, routed by the
  request's `model` field, each with its own context/KV and — for MoE — its
  own expert cache. arcint already loads several model objects in one
  process for the drafter (`code`: target + MTP/DFlash, `backend_llama.cpp:845-891`),
  so the llama.cpp side is mechanically reachable; what is missing is
  routing and per-model admission.
- **Switch latency** `inferred`: ~0 when the target is already resident;
  otherwise it is (b) or (c) for the newly loaded one, while the resident
  one keeps serving.
- **Memory**: the sum. On these cards the sum does not fit: the B60's agent
  (24,195 MiB peak) and Flash-Next (23,478 MiB) each nearly fill 24,480 MiB.
  "When they fit" is therefore a real condition, not a formality, and for
  the two B60 language models it is never met. A smaller pairing (a model
  that peaks under half the card) could.
- **What breaks**: `DESIGN §3.10`'s "one copy of the weights" is per
  process/lane, not per set of models; the expert cache is per card and its
  budget must be split; the prefill slot loan assumes one cache and one lane
  (`code`: the loan refuses `lanes > 1`, `prefill-slot-loan-design.md` §6);
  the request `model` field becomes binding for several models
  (`DESIGN §4.2`, which today binds it to a lane).
- **Invariant/constraint**: the VRAM sum (a second serving peak to size,
  DESIGN §3.3's rule applied twice), `§3.10`'s one copy of the weights,
  `§4.2`'s name binding, and the vram-user-space campaign. The loan's
  one-cache assumption is the immediate code obstacle, not an invariant.

### (e) The lanes approach revisited

- **Mechanism, as built**: **one process, one model, one KV pool, a context
  per named lane**; the request's `model` field picks the lane
  (`code`: DESIGN §3.10, `docs/campaigns/lanes-agent-subagent.md`). Lanes
  are not several models.
- **What the test drive found** `measured-here`: a lane's decode followed the
  *pool's* fill, not its own (subagent 14.2 t/s with the agent lane at 130k,
  against 32.1 alone); patch 0025 (a strided f32 copy kernel) and 0026 (a
  cell window per lane) fixed the interference, and the same probe then read
  32.1 / 32.0 / 32.0 t/s at ~20k / ~64k / ~118k of the other lane, with both
  lanes batched at 21.6 + 21.7 t/s (patch 0028/0029). The campaign stayed
  open for one decode across lanes; the production agent runs one lane
  (`code`: DESIGN §3.10). With Flash-Next the lanes leg evicted ("aborted,
  numbers void"), and Flash-Next's indexed memory refuses cell windows
  (`code`: `lanes-agent-subagent.md`, "Flash-Next with lanes").
- **For Janus**: lanes are a state-sharing lever between contexts of one
  model, not a switching lever between models; using them for switching is
  option (d) with the lane machinery's routing and pool. What transfers is
  the *routing by name* and the `n_ctx`-per-name contract (`DESIGN §4.2`),
  and the lesson that a shared pool makes one occupant's speed depend on
  another's fill.
- **Invariant touched**: `§3.10`, `§4.2`, and the Flash-Next indexed-memory
  limitation, which is a `code` fact, not a decision.

### Summary of the price

| option | switch latency | VRAM | host RAM | main loss | invariant |
|---|---|---|---|---|---|
| (a) unit switch | 57-135 s (`measured-here`, operator record) | one model | page cache | KV, in-flight requests | none |
| (b) faster cold load | the cacheable share of that (`inferred`) | one model | page cache + disk copy | nothing structural | none (§3.4 point 2 stays) |
| (c) warm standby | new promote code, then H2D + rebind (`inferred`) | ~0 idle, peak when serving | the model bytes, pinned | KV, checkpoints, adapt state | host RAM + pinned (§2); vram-user-space if VRAM is held |
| (d) several resident | ~0 on a hit | the sum | the sum | nothing on a hit | the VRAM sum, §3.10, §4.2 |
| (e) lanes | not a switch | one model | one model | — | §3.10, §4.2 |

---

## 3. How others switch models

Read at source where the source is public; product documentation is marked
`paper`.

- **llama.cpp's server, at the pin** (`code`, read in the pinned tree):
  - **Router mode**: launch without a model, `--models-dir`/`--models-preset`,
    one **child process per model**, and `--models-max N` (default 4, 0 =
    unlimited). On a request for a model past the limit it evicts the **LRU**
    among ready, non-busy instances (`tools/server/server-models.cpp:274-295`,
    `:1107-1130`; `tools/server/README.md:228-231, 1843-1900`). The router's
    instance states include **SLEEPING** (`tools/server/server-models.h:25-38`).
  - **Idle sleep**: `--sleep-idle-seconds` (`common/arg.cpp:3831`); when it
    fires, the server **destroys the model and context** and emits
    `SERVER_STATE_SLEEPING`; the next request **reloads** the model
    (`tools/server/server-context.cpp:1058-1080`). A standby *process* whose
    weights are **not** kept: a cold load on wake, not a promote.
  - So llama.cpp's answer to "how to switch" at the pin is: processes + router
    + LRU, with an idle sleep that reloads. No weight tier in host RAM, no
    fast promote. (`research-agent-lanes.md` tabulated the router at an earlier
    pin as `code` with the same shape.)
- **llama-swap** (`code`, read at `main`, 2026-11-01): the proxy above
  llama-server generalised. Each model entry is a **command** started on
  demand and stopped on TTL or eviction (`internal/config/model_config.go:139-160`,
  `:12, :191-206`); readiness is a health check (`checkEndpoint`,
  `healthCheckTimeout`, `internal/process/process_command.go` start/WaitReady);
  groups decide eviction: `swap` stops same-group siblings, `exclusive`
  stops other groups' non-persistent members (`internal/router/group.go:59-97`);
  the config exposes `ttl`/`ttl: -1` for the global default and `cmdStop`
  (`docs/config.example.yaml:131-141, 433-441, 644-662`). Mechanism: an OS process
  switch, i.e. option (a) with a scheduler in front. No weight tier.
- **vLLM sleep mode** (`code`, read in a vLLM tree at `d1922cb`):
  `POST /sleep?level=1|2` and `POST /wake_up?tags=weights,kv_cache`
  (`vllm/entrypoints/serve/dev/sleep/api_router.py`). The backend contract
  states the levels: **level 1 offloads weights to host RAM (restorable
  in-process); level 2 discards them (reloaded from the model source on
  resume)** (`vllm/device_allocator/sleep_mode_backend.py`, `SleepModeBackend.suspend`
  docstring). The move is a CUDA VMM move: `sleep(offload_tags=("weights",) if
  level == 1 else ())` takes a **pinned** CPU backup of each allocation (at this
  pin `torch.empty(..., pin_memory=PIN_MEMORY)` at `cumem.py:264-271`; upstream
  `main` uses a `create_cpu_backup` buffer via `torch.frombuffer`) and
  `cuMemUnmap`s the physical backing while the virtual address stays reserved
  (`csrc/cumem_allocator.cpp:169-268`); `wake_up` maps the backing again under
  the same pointers and copies the CPU buffer back (`cumem.py:327-356`). Level 2 additionally saves named buffers to
  CPU in `gpu_worker.sleep` before discarding
  (`vllm/v1/worker/gpu_worker.py:236-269`), so the version at this pin keeps a
  CPU copy even for level 2 — the docstring and the code differ in emphasis,
  and the code is what runs. The mechanism has an **XPU port**: `CuMemBackend`'s
  docstring names `XpuMemAllocator` on XPU (`sleep_mode_backend.py:115`), and
  `XpuMemAllocator`'s says sleep offloads or **discards and unmaps XPU physical
  memory**, wake-up **remaps it to the same reserved virtual address**
  (`vllm/device_allocator/xpumem.py:112-118`). **This is the reference shape
  for option (c); §2(c) prices why OpenCL cannot copy the mapping part of it.**
- **Ollama** (`code`, read at `main` commit `c1db5f9`, 2026-11-01): a model is
  a runner process; each request carries `keep_alive`, which becomes the
  runner's `sessionDuration` (`server/sched.go:555`) and arms an expiry timer
  (`:429-452`; the immediate-expiry path at `:383-391`); `0` unloads when idle.
  A runner whose runner options differ is **reloaded**, and `needsReload`
  compares `NumCtx` after `effectiveContext` (`server/sched.go:1423-1470`) —
  an explicitly set `num_ctx` reloads, an auto-derived one does not.
  `OLLAMA_MAX_LOADED_MODELS` defaults to 3 runners a GPU
  (`defaultModelsPerGPU = 3`, `:121`) and evicts to make room (`:269-388`).
  Mechanism: process paging with a keep-alive TTL; a context change forces a
  full load.
- **LM Studio** (`paper`: product documentation; the source is closed, so no
  file or line exists to read): it loads a model on demand for a request and
  keeps it for a configurable idle period, with a "just-in-time" loading
  posture and one loaded model per model key. The prior survey classified it
  as `docs` (`research-agent-lanes.md`). No number is claimed here.
- **Strata and FreeToken** (`code`, the two project references — paths and
  lines in "Reference to follow" above): Strata pairs an **idle unload** and a
  **lazy start** (unload after N s, reload on the next request; `POST /load` to
  load ahead) with a **supervised restart** (`runtime.restart`,
  `allow_source_switch`), so its switch is option (a) under a supervisor; its
  prepared-artifact mechanism is the **arena pack**, read once at startup or
  mapped read-only (`STRATA_ARENA_MMAP`). FreeToken's `ft-daemon` owns one
  `ft serve` child and `/engine/switch` composes stop+start — option (a) under
  a supervisor with no weight tier. Neither carries a host-RAM promote.
- **Dial moves nothing**: Ollama has no weight tier either; llama.cpp's sleep
  discards and reloads; Strata and FreeToken restart; only vLLM carries a
  host-RAM weight tier with an in-process remap, on CUDA and XPU over those
  allocators' own virtual memory. Its shape is not reachable from OpenCL here
  (§2(c)). That asymmetry is why (c) follows vLLM (and pays for new backend
  code) and (a)/(d) follow llama.cpp/llama-swap/the two project references.

Where paper and code disagreed: vLLM's sleep-mode documentation calls level 2
a discard, and the level-2 path at this pin still clones the buffers to CPU
before the allocator frees them; the code is what runs, and the doc's value
for Janus is the level-1 mechanism, which both agree on.

---

## 4. Recommended first step: one card window that decides between the options

**The measurement**: the load phases of §1.1, one fresh process a leg
(`DESIGN §5.2`), the unit's flags, the shipped log lines under timestamps. No
engine probe is needed: `loaded meta data`, `loading model tensors` (with its
`load_mode`), `loading OpenCL kernels`, `kernel cache enabled at '<dir>'`,
`model buffer size`, the expert cache's `filled in <T> s`, `reserve took <X> ms`,
the `in <T> s` total and `listening on` already bound every phase of §1.1.
`tools/fn_sweep.py` is the harness: a fresh process from the unit's flags,
`--env K=V` for the cache switches, a wait for `listening on` and the printed
`load N s`, `--reverse` for the answer order (`code`:
`tools/fn_sweep.py:72, 97, 171-196`). The run itself is wrapped in `ts` (or
fn_sweep stamps its own leg lines — a harness touch like the `/proc/<pid>/io`
one); a timestamper applied to the finished leg log cannot stamp after the
fact. The stamps turn the shipped lines into the phase boundaries:
`constructing llama_context` (`src/llama-context.cpp:183`) is the
first line of the context phase, i.e. it closes `T_load`; `reserve took`
closes `sched_reserve`; the drafter's own `loading model tensors` and context
lines bound `T_draft`.

**Arms** (same model, same flags, the served configuration):

1. **A — fresh, empty program caches**: `GGML_OPENCL_KERNEL_CACHE_DIR` a new
   empty directory, and the runtime's own cache controlled too (`NEO_CACHE_DIR`
   a fresh empty directory or `NEO_CACHE_PERSISTENT=0`; `paper`/`code`:
   `research-cold-start.md`). **Not** the directory "as it is on the host":
   after any production run that directory is warm, so the compile share would
   read 0 by construction.
2. **B — the program caches warm, the pages cold**: the same run again with the
   same directories, which A has now filled.
3. **C — both warm**: the model file read once before the start (its page
   cache warm), the program caches warm.

Order A, B, C, A, B, C, reversed, with `GGML_OPENCL_KERNEL_CACHE_DEBUG=1` and
`NEO_CACHE_STATS=1` on every leg. Models: the agent (B60) and Flash-Next (B60)
if the window allows; otherwise the agent plus the coder (A770), the pair that
prices both H2D links.

**What (c) still pays, even after a promote path is built** (so its saving is
not read as the whole load): the per-tensor H2D copy back to VRAM, the KV
buffer clear (`llama-kv-cache.cpp:293, :403`), `sched_reserve`
(`llama-context.cpp:562-885`), the MTP context, and — new work — the device
buffers and the rebind pass of §2(c). It does **not** have to pay the GGUF
parse, the per-tensor `read_buf` staging (`llama-model-loader.cpp:1806-1810`),
the read itself or the planar conversion — and a **live** standby skips
`T_compile` and `T_pages` too, because its process already holds the built
programs and has read the file. Its saving is therefore the cold load minus
what it still pays, not `T_reduce` alone.

**The bar, and how it can fail**: read the phase probes, not one residual. Let
`T_load` = start → after `llama_model_load_from_file`, `T_ctx` = after the
context, `T_draft` = after the drafter, `T_rest` = total − those. Inside
`T_load`, `T_compile = T_load(A) − T_load(B)` and
`T_pages = T_load(B) − T_load(C)`, and
`T_reduce = T_load − T_compile − T_pages` (the reduction, staging and planar
work — the part (c) removes). Then:

- if `T_compile` is more than half of `T_load`, option **(b) item 1** (a
  persistent program cache) is the first step;
- else if `T_reduce` is more than half of `T_load`, (b) item 2/3 is the first
  step where it applies (planar: Flash-Next only), and (c) removes this share
  but only after the promote work of §2(c) exists;
- else if the copy-back that (c) still pays (`~model bytes / H2D link`,
  `inferred`) plus `T_ctx + T_draft + T_rest` already meets the operator's
  acceptable latency while the cold load's total does not, **(c)** is the
  candidate (a live standby skips `T_compile` and `T_pages` as well as
  `T_reduce`, so it is measured against the whole load, not against `T_reduce`);
- if none dominates, the load is spread and this document is rewritten with
  the measured shares before any option is chosen.

- **Red cases that must fail**: a leg with `GGML_OPENCL_KERNEL_CACHE_DIR=off`
  and `NEO_CACHE_PERSISTENT=0` must show the A-like time even against a warm
  directory (else neither cache is what moved); a leg with the model file
  guaranteed cold (the page cache dropped for that file only — not a global
  `drop_caches` on a serving host) must show the B-vs-C difference.
- **Tool change, not engine change**: `/proc/<pid>/io` deltas (or the load's
  `read_bytes`) need a small addition to `tools/fn_sweep.py`; it is a harness
  change and is named here as such.
- **No GPU work in this document.** The window is one of the operator's card
  windows; when it runs, `docs/sop-card-window.md` applies (sampler before the
  leg, card identity by PCI id, the zombie sweep).

**What the window must not do**: no engine change, no unit edits, no second
process on a card.

---

## 5. Open questions for the operator (the scope decision is his)

1. **Which models must switch, with which?** The B60's agent↔writer and
   agent↔Flash-Next are the pairs the units' exclusion already encodes; the
   coder lives alone on the A770. Is the target pair the B60's, or does a
   second card's pair matter?
2. **On which card?** The B60 has a 13.9-14.3 GB/s H2D link and the A770
   1.8 GB/s (DESIGN §2); the H2D floor of a promote differs by ~8x between
   them, so the card decides whether a single-digit-second promote is even
   arithmetically reachable — before the new backend work of §2(c) is priced.
3. **How often, and at what granularity?** A switch per request, per session,
   or a few times a day? The unit switch's 57-135 s is tolerable at the last
   and not at the first; a per-request switch makes (c) or (d) the only
   candidates, and (c) still pays the context rebuild.
4. **Acceptable latency**: what is the number a switch must meet — seconds,
   or the current minute? The §4 bar is stated to produce that number.
5. **Host RAM for standbys**: option (c) needs the model's bytes resident
   (~16 GiB dense, 20-40 GiB for Flash-Next's bank) beside the served model.
   DESIGN §2 sizes the serving container at 52 GiB (96 GB is the planned host
   total), and the bank is pinned USM. How many standbys may hold host RAM,
   and may they be pinned (a pinned standby competes with the served model's
   own bank)?
6. **May a switch lose the KV and the checkpoints?** Today it does; (c) and
   (d) either accept that or the `kv-checkpoint-restore` campaign becomes a
   prerequisite. That is a scope decision, not an implementation detail.
7. **Is the scope (b) only, or is the new backend work of (c) in scope?** (c)
   is the option that needs a promotable buffer type and a rebind pass that do
   not exist — not an adjustment of an existing mechanism. `CLAUDE.md`
   requires a reference mechanism that conflicts with the record to come here
   priced, and §2(c) prices it. (b) stays inside today's code.

---

## Where it lives

- Load path: `src/exec/backend_llama.cpp` (`t_load` at `:606`, the `in %.1f s`
  line at `:1108`), `src/main.cpp:154` (the listening line),
  `src/exec/llama_spec.cpp` (the MTP context).
- Engine side (the pinned llama.cpp + series): `src/llama-model.cpp:1565-2080`,
  `src/llama-model-loader.cpp:1556-1815`, `src/llama-mmap.cpp:492`,
  `src/llama-context.cpp:562-885`, `src/llama-expert-cache.cpp:261-612`,
  `ggml/src/ggml-opencl/ggml-opencl.cpp:11916-12050, 14939-14975, 1788-1830`,
  `ggml/src/ggml-opencl/cl-program-cache.{h,cpp}`.
- Campaigns this document leans on: `lanes-agent-subagent.md`,
  `prefill-slot-loan.md`, `prefill-slot-loan-design.md`, `vram-user-space.md`,
  `research-cold-start.md`, `research-agent-lanes.md`,
  `flash-next-llama-engine.md`.
- DESIGN: §2, §3.3, §3.4, §3.9, §3.10, §4.2, §4.3, §5.2, §8.4.
- The switching references: llama.cpp router/sleep (pin tree), llama-swap
  (`main`, 2026-11-01), vLLM (`d1922cb`), ollama (`main` `c1db5f9`,
  2026-11-01), LM Studio (`paper`), Strata (`github.com/Niko1221/Strata`),
  FreeToken (`github.com/FlashML-org/FreeToken`).
