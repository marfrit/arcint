# Cold start and kernel-compilation caching across engines — research for static-partition-cold-start

Scope: how other GPU inference stacks separate and address first-process warming, with emphasis on the layers arcint actually sits on (OpenVINO GPU plugin over OpenCL, oneDNN for the MoE GEMMs, Intel's compute-runtime/IGC underneath). All URLs accessed 2026-09-05.

## OpenVINO GPU plugin: `cache_dir` and `CACHE_MODE`

- What's cached: `ov::cache_dir` (a `Core` property) makes `compile_model()` write a per-model blob to disk keyed on the model + device + build environment; a hit skips re-running the plugin's graph-compile pipeline (layout selection, fusion, kernel selection and JIT of the GPU kernels for that specific static graph) on the next call for the same model.
- Knobs: `ov::cache_dir(path)`; `CACHE_MODE` (GPU-plugin- and IR-specific) — `OPTIMIZE_SIZE` (weightless cache, topology only, weights re-read from the original model file) or `OPTIMIZE_SPEED` (topology + weights baked in, larger blob, one fewer read at load).
- Persists: across processes and reboots, on the filesystem at `cache_dir`, keyed to the exact static graph — does not survive a graph-shape or model change.
- Measured delta: not quantified in the docs surveyed; framed as removing the compile-pipeline cost entirely on a hit, not as a percentage.
- License / source: Apache 2.0. OpenVINO Model Caching Overview and GPU Device docs, docs.openvino.ai/2025.
- Not covered, load-bearing here: `cache_dir` caches the compiled-model artifact, not an oneDNN artifact — oneDNN's own primitive JIT (what `created_onednn_kernels` counts) is a separate cache one layer down (next section), and a `cache_dir` hit does not imply oneDNN's primitives were reused; it only proves the OpenVINO-level compile pipeline was skipped. arcint's paged graph already runs with `cache_dir` off by design (`backend_ov.cpp` ~2694-2697, `ov::cache_dir("")`) — this mechanism is out of scope as a lever unless that design decision is itself revisited, which is out of this campaign's scope per `static-partition-cold-start.md`.

## oneDNN: primitive cache (in-memory, automatic) vs. persistent cache (on-disk, NOT automatic)

Two distinct mechanisms, easy to conflate.

- **Primitive cache** — process-global, in-memory LRU cache of already-JIT'd primitives, keyed on primitive kind/params/attributes/ISA.
  - Persists: within one process only; a fresh process starts empty and re-JITs everything — consistent with `created_onednn_kernels=325` recurring per process regardless of history.
  - Knobs: `DNNL_ENABLE_PRIMITIVE_CACHE` (build-time CMake option; `OFF` forces `DNNL_PRIMITIVE_CACHE_CAPACITY=0`). Automatic and on by default, no code changes needed.
  - Source: oneDNN Primitive Cache dev guide, docs.oneapi.io / uxlfoundation oneDNN docs. Apache 2.0.
- **Persistent cache** — not a transparent on-disk cache the library manages for you. It is an API the application must call: get a cache-blob ID via `primitive_desc::get_cache_blob_id()`, get the compiled blob via `primitive::get_cache_blob()`, store it yourself, and reconstruct the primitive later from a primitive_desc + that blob via the matching constructor.
  - Persists: across processes and reboots, but only if the calling application does the serialization/storage itself — there is no ambient env var that turns this on.
  - Invalidation: oneDNN version and git commit hash are baked into the blob ID, so a oneDNN upgrade invalidates every stored blob.
  - Source: Intel oneDNN Developer Guide, "Persistent Cache" (2024.1), intel.com. Apache 2.0.
- Load-bearing distinction: unless arcint (or the OpenVINO GPU plugin on its behalf) explicitly calls this API and owns the blob storage, oneDNN gives you nothing persistent across process boundaries — the per-process 325-kernel JIT is the expected, by-design behavior of an unmodified oneDNN integration, not a bug in oneDNN. Whether the GPU plugin does this wiring internally is unconfirmed by docs in this pass and is exactly the kind of plugin-internals question the recon step should answer by reading code.

## Intel compute-runtime (NEO): on-disk OpenCL/L0 program cache — one layer further down, and automatic

- What's cached: compiled GPU program binaries, on disk, transparently, for any caller going through `clBuildProgram`/its Level Zero equivalent — including oneDNN's own OpenCL kernels, since oneDNN's GPU backend for this stack is built on the OpenCL runtime. This is the layer most likely to already be doing something for arcint today without any code change.
- Knobs: `NEO_CACHE_PERSISTENT` (enable/disable on-disk caching), `NEO_CACHE_DIR` (path; falls back to a per-OS default, disabled entirely if unset and no writable default exists), `NEO_CACHE_MAX_SIZE` (bytes, default reported at 1 GiB, LRU eviction past that), `NEO_CACHE_STATS` (hit/miss instrumentation), and a debug trace `BinaryCacheTrace` (dumps hash attributes for cache-miss forensics).
- Persists: across processes and reboots, on disk, keyed on source + build options + IGC/compiler version/registry state + device identity — so an IGC or driver update invalidates it.
- Measured delta: not quantified in this doc; qualitatively "compiling required only the first time."
- License / source: MIT. intel/compute-runtime `programmers-guide/COMPILER_CACHE.md`, github.com/intel/compute-runtime.
- Open item: default enablement state and exact default `NEO_CACHE_DIR` were not confirmed to file-and-line precision in this pass — worth reading the source directly rather than inferring from docs, since it gates the single most plausible "why is process 4 already warm" mechanism.
- IGC itself has no persistent-cache knob of its own — it's the thing compute-runtime's cache exists to avoid re-invoking. One number worth carrying into the campaign's math: a commonly cited rule of thumb for OpenCL JIT overhead is 0.1-1 s per kernel, first compile only (Karl Rupp's OpenCL JIT benchmarks, karlrupp.net, 2016 — old data, order of magnitude only, not vendor-specific to this stack, flag as illustrative not authoritative). At 325 kernels/process that is 32.5-325 s, which brackets the observed 215-585 s range at its high end and is at least consistent with "JIT of 325 kernels" being a first-order contributor — not proof of it, since the range is wide enough to also leave room for the page-cache/first-use-fill hypotheses.

## SYCL kernel-bundle cache — a sibling mechanism, likely not on arcint's path

- Knobs: `SYCL_CACHE_PERSISTENT=1` (default off), `SYCL_CACHE_DIR` (path), `SYCL_CACHE_MAX_SIZE` (MiB, default 8192, LRU), `SYCL_CACHE_THRESHOLD` (days, default 7), `SYCL_CACHE_IN_MEM` / `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD` for the in-memory tier.
- What's cached / not: compiled device binaries keyed on source + included files + build options + device identity (self-invalidating on a driver bump); does not cover programs built from source strings at runtime or produced by linking multiple modules.
- Persists: across processes and reboots, on disk, only when explicitly turned on (default off).
- License / source: Apache 2.0 w/ LLVM exceptions. intel/llvm "A brief overview of kernel and program caching mechanism", intel.github.io/llvm.
- Relevance to arcint: this sits in the SYCL/Level-Zero runtime, parallel to, not underneath, the OpenCL path arcint uses per this task's framing. Unless some component quietly goes through SYCL/L0 instead of OpenCL, rule this out rather than try it.

## llama.cpp — Vulkan, SYCL, OpenCL backends

- Vulkan backend ships a `vulkan-shaders-gen` build-time tool compiling its GLSL compute shaders to SPIR-V ahead of time and embedding them in the binary — no per-process runtime shader JIT to warm up for the bulk of its pipelines, only fast, driver-cached in-memory pipeline-object creation at startup.
- SYCL backend targets Intel GPUs including older Arc parts; the OpenCL backend is explicitly documented as legacy ("either needs to be reimplemented properly... or will eventually be dropped"), with no first-class pipeline-cache persistence story found for it — docs steer users to Vulkan or SYCL instead.
- Measured delta: none found for cold start specifically.
- License / source: MIT. ggml-org/llama.cpp `docs/backend/SYCL.md`, GitHub discussions #5138, #10879.
- Transferable idea: AOT-compiled shaders sidestep the problem entirely — the same principle as OpenVINO's `cache_dir`, but applied at the Vulkan-shader layer instead of oneDNN's.

## ipex-llm on Arc — explicit warm-up guidance, no numbers

- Guidance: Intel's own IPEX-LLM docs recommend running a warm-up pass before real generation on first use of a model on an Arc/iGPU device, noting first-run compilation can take "several minutes" on Windows for first-time Arc/iGPU runs.
- Measured delta: qualitative only ("several minutes"), no controlled cold/warm numbers published.
- License / source: Apache 2.0. ipex-llm docs (readthedocs) and `docs/mddocs/Quickstart/*`, github.com/intel/ipex-llm.
- Relevance: the closest documented same-hardware precedent for arcint's own JIT hypothesis — Intel's own stack, same GPU family, independently reaches for "pre-warm with a dummy pass" rather than a persistent-cache fix.

## vLLM — `torch.compile` cache + CUDA-graph warm-up

- Two independent cold-start costs, both with public mitigations.
- (1) `torch.compile` Inductor cache: `TORCHINDUCTOR_CACHE_DIR` (PyTorch env var; vLLM defaults its own compile-artifact cache to `~/.cache/vllm/torch_compile_cache/`), disable/debug switches `VLLM_DISABLE_COMPILE_CACHE=1`, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. Persists across processes and, if the path survives, across container restarts.
- (2) CUDA-graph capture: not persisted as a first-class artifact — re-captured per process at server start as an explicit warm-up pass over a fixed list of batch sizes.
- Measured delta: community posts (Tensorfuse, RunPod) report cutting cold start from roughly 5 min to 90 s and 294 s to 82 s via a persistent `TORCHINDUCTOR_CACHE_DIR`, restricting captured batch sizes to the ones actually served, and a few dummy requests before accepting traffic — vendor/blog numbers on NVIDIA hardware, not peer-reviewed; read as "order of magnitude possible," marketing-adjacent, not a controlled benchmark.
- License / source: Apache 2.0. vLLM docs "torch.compile integration", docs.vllm.ai; Tensorfuse and RunPod engineering blogs (2026).
- Relevance: CUDA graphs have no OpenVINO-GPU/oneDNN analogue, but the pattern — explicit warm-up requests run before the server accepts traffic, at each shape class actually served — is the direct analogue of this campaign's "pre-warm" lever.

## SGLang — configurable warm-up functions

- Mechanism: runs a warm-up forward pass (or several, in reverse batch-size order, largest first) before capturing each CUDA graph size; exposes a `--warmups` CLI flag (CSV list of named functions defined in its own `warmup.py`) to run arbitrary extra passes before the server opens for traffic.
- Persists: nothing found documented across process restarts — a pure per-process warm-up model, the same shape as ipex-llm's guidance and vLLM's default posture.
- License / source: Apache 2.0. sgl-project/sglang `server_args.py`, LMSYS blog "Advanced CUDA Graph Techniques in SGLang" (2026), github.com/sgl-project/sglang.

## TensorRT-LLM — ahead-of-time engine build, but warm-up still recommended

- Mechanism: `trtllm-build` compiles a model into a `.plan` file autotuned and kernel-selected for one specific GPU SKU/dtype/shape-range combination, entirely offline — the most AOT-committed of the surveyed engines.
- Despite that, production guidance still recommends 10-20 dummy requests after engine load before accepting real traffic, because some kernel selection/JIT is deferred to first execution even from a built plan.
- Measured delta: the "10-20 dummy requests" figure is vendor-blog-sourced (Spheron, 2026), not found in NVIDIA's own docs in this pass — flag as unconfirmed-primary.
- License / source: Apache 2.0 core / proprietary kernel components. nvidia.github.io/TensorRT-LLM.
- Relevance: the strongest available precedent that AOT compilation and explicit pre-warm are complementary, not substitutes — even the engine that pushes the most work ahead-of-time still warms up.

## ExecuTorch / MLC-LLM — AOT graph export, same caveat as TensorRT-LLM

- Both compile ahead of time (ExecuTorch via `torch.export` to a fixed execution graph; MLC-LLM via TVM Unity to target-specific generated code) and both avoid the bulk of runtime JIT by construction.
- Measured delta: no first-run warm-up numbers found for either in this pass.
- License / source: ExecuTorch — BSD; MLC-LLM — Apache 2.0. llm.mlc.ai TVM install docs; secondary ML-runtime survey coverage.
- Relevance, and the caveat: the design intent in both is that AOT export removes the need for the pre-warm pattern entirely — a different tradeoff than arcint's, since a full AOT rebuild per shape/depth/precision combination is far more build-cost than kernel-cache reuse or a runtime pre-warm pass, and a rebuild-per-configuration approach is out of scope per the campaign's own "no change to the static partition's ranking rule."

## Separating JIT cost from page-cache cost — how others measure it

The common technique is first-request vs. second-request timing at fixed input, with an explicit cache-drop between cold trials when the target is page cache rather than JIT: `echo 3 > /proc/sys/vm/drop_caches` (root, the correct control for invalidating Linux's own page cache) and per-file residency checks via `fincore` (deprecated but still common) or its maintained descendants `vmtouch`/`pcstat`, which read a file's page count against `mincore()`/`/proc/<pid>/smaps` residency.

A directly relevant gap for this campaign's own hardware: ZFS's ARC is a separate, kernel-space cache that does not populate the Linux page cache the way a page-cache read does — files served from ARC will not show as resident to `fincore`/`vmtouch`, which only see the VFS page cache, not ZFS's own structures. If the model-weight or oneDNN-cache files this campaign instruments live on a ZFS dataset, a `fincore`-based "is this warm" check can read cold while the data is served instantly from ARC — the check needs `arc_summary`/`arcstat` (or simply timing the read) alongside or instead of `fincore` on such a path.

Sources: Baeldung "Dropping Page Cache in Linux", Percona blog on `fincore`, OpenZFS GitHub issue #16978 and community write-ups on ARC vs. page cache (2025-2026).

## What transfers to arcint

Ordered isolation plan for the campaign's entry criterion (owner not yet separated), cheapest and most diagnostic first:

1. **Confirm compute-runtime's on-disk program cache state on the test hardware** — read `NEO_CACHE_PERSISTENT`/`NEO_CACHE_DIR` at the process environment actually used to launch arcint, and check whether the cache directory has non-trivial contents and is growing after a tier-ON run. Five minutes, no code change, and it directly discriminates: if the cache is already on, populated, and growing, the recurring 325-kernel JIT cost is a cache-miss problem (hash instability per process — check whether the launch path embeds anything process-variant, e.g. a PID or timestamp, into a compile flag) rather than an absent-mechanism problem, which changes the fix from "add a cache" to "fix why the existing cache doesn't hit." If it is off or unwritable, that alone may be the whole defect. Expected move: process 2..N's load time should collapse toward tier-OFF's 30-45 s once this is confirmed on, populated, and hitting.

2. **Fixed-`created_onednn_kernels`, fresh-vs-warm-page-cache process pair** — the design note's own proposed experiment: run a process, record `created_onednn_kernels=325` and load time, then immediately run a second process with the OS page cache for the weight/partition files forced cold (`drop_caches` or an equivalent eviction, checked with `vmtouch`/`fincore` rather than assumed, and via `arcstat`/timing if any of those paths are ZFS-backed) while compute-runtime's own on-disk cache from step 1 stays warm. If load time still collapses to near-fast, the page-cache-of-weights hypothesis is ruled out as the dominant term and the kernel cache is confirmed as owner; if load time stays slow, page cache of the weight/partition data is implicated and step 1's fix alone will not close the gate.

3. **First-use `reserve_pinned` fills, isolated last** — only after 1 and 2 fail to fully explain the gap: instrument `bind()`'s `reserve_pinned` pass (patch 0018) with its own timer, run it standalone against a fixed-size partition with both the kernel cache and page cache pre-warmed by steps 1-2, and see what time, if any, remains unaccounted for. This is the only one of the three that requires a new counter rather than an existing one (`created_onednn_kernels` and OS-level cache tools cover the other two), which is why it is checked last.

Candidate levers, once an owner is named, each evaluated against DESIGN §3.4's history-independence invariant (both are per-process/seed-independent by construction, so neither should threaten it, but confirm at the review step by checking the loaded blob or executed warm-up path does not vary with which requests a process has already served):

- **Persistent kernel cache.** If step 1 shows compute-runtime's cache absent or missing hits, the fix is operational configuration at the arcint/launch-environment level (`NEO_CACHE_PERSISTENT=1` plus a stable, writable `NEO_CACHE_DIR` shared across the fleet's processes for a given card+driver+model combination) — no plugin code change implied. If instead the miss is because oneDNN's own in-memory cache is process-local by design (expected) and the OpenCL-level cache is not the whole story, the next-cheapest lever is wiring oneDNN's own persistent-cache API (`get_cache_blob`/blob-constructor) — plugin-internals work, out of arcint's own source unless the GPU plugin already exposes a hook for it, which recon should check before assuming upstream work is needed.
- **Explicit pre-warm pass.** If the owner is first-use fills (`reserve_pinned`) or an oneDNN cache-miss not worth chasing upstream, the direct lever is a dummy-request pass run once per process at start, before the first real request is admitted — the pattern ipex-llm, vLLM, SGLang and TensorRT-LLM all converge on independently. For arcint this means one synthetic prefill+decode pass per shape class actually served (at minimum the reference cell's own prompt length and KV precision) run inside process start-up and excluded from the first-request timing the gate measures — cheap to build, adds fixed latency to every process start (including the already-fast ones) unless gated behind the tier-ON condition specifically, and does not by itself reduce total warming work, only relocates it off the first customer request.
