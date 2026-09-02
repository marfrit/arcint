# M9 — expert offload v2 as a plugin patch series (design only, nothing built)

Every claim is cited to DESIGN §7.0.2s/§7.0.2t, to a file:line verified in the 2026.4-dev plugin tree, or marked **[to measure]**.

## 0. Recon, sharpened

- `plugin/ops/moe_offload_constant.cpp:65` uses the two-arg `allocate_memory(upload_layout, false)` → `get_lockable_preferred_memory_allocation_type` (`runtime/engine.cpp:110`). `supports_allocation` (`engine.cpp:97-101`) returns **false for `usm_shared` unconditionally**, so the branch is not "shared, else host" — on any card it is always `usm_host`. The pool is in GTT *by construction*, which is the mechanism behind §7.0.2t's `drm-total-gtt` 13.17 GiB vs `vram0` 6.69 GiB while serving.
- The slot buffers *are* the data-node memories: `bind_weights_on_first_exec` (`moe_3gemm_swiglu_opt.cpp:678-695`) fills `instance._weights` from `input_memory_ptr(WEIGHT_0/…)` — exactly the `desc.memory` built above. One allocation-type change moves the whole pool.
- `constant.cpp:132` already skips the `mem_lock` upload when `partial_upload.enabled`, and passes `skip_device_transfer=true` to `cldnn::data` (`:169`). No host lock at build time today.
- Miss path `expert_weight_providers.cpp:59-115` → `fill_weights_memory` (`moe_otd_runtime.cpp:242-254`): blocking disk read into a **function-local `std::vector<uint8_t> payload`**, host transpose, `copy_from(exec_stream, payload.data(), …, blocking=true)`.
- Both our cards set `use_onednn` (XMX/immad), and `execution_config.cpp:344` then forces `QueueTypes::in_order`. This decides P2's wait.

## 1. Patch plan (`patches/000X`, PR-shaped per §1.1, one concern each)

**P0 `0004-moe-otd-eviction-and-alloc-counters` (~15 lines).** `OtdPerfCounters` (`moe_otd_runtime.hpp:24`) gains `evictions`, `staging_bytes`, `device_slot_buffers`/`host_slot_buffers`; `[OTD_PERF]` prints them; `PartialUploadLogState::log` (`moe_offload_constant.cpp:80`) prints the chosen allocation type per constant. First, because it is the red-first observable for everything below and it decides whether P3 exists.

**P1 `0005-moe-otd-device-resident-slot-pool` (the core).** Pass an explicit allocation type at `moe_offload_constant.cpp:65`: `usm_device` while a per-model device-slot budget has room, `usm_host` after (two-tier, §3). Correctness, by grep and not by opinion:

- **`lock()` on a slot buffer is the sharp risk.** `gpu_usm::lock` (`ocl_memory.cpp:521-547`) does *not* throw on `usm_device` — it `allocateHost(_bytes_count)` and blocking-memcpys `_bytes_count`. The slot memory is a `reinterpret_buffer` of a *smaller* physical allocation onto the *full* constant layout (`moe_offload_constant.cpp:78`), so `_bytes_count` is the full constant size. Today a stray lock is a cheap pointer return; under P1 it becomes a multi-GiB host allocation plus a device read past the allocation — silent, not a crash. Owed before the patch: enumerate every `mem_lock`/`lock(`/`buffer_ptr()` reachable from a `data` node's attached memory (program passes, const folding, model-cache serialization, `data_node::get_attached_memory` users). The existing weightless-cache argument (`:66-73`) covers serialization only. Add a debug-build guard on the slot memories so a future caller fails loudly.
- **`buffer_ptr()` is safe.** `set_otd_weight_pointers`/`prepare_weight_pointers` (`:659-725`) keep `memory::ptr`; the only `buffer_ptr()` uses are the enqueue_memcpy dst (`ocl_memory.cpp:612`) and kernel args — both legal for device USM.
- **`copy_from(host ptr)` is safe.** `ocl_memory.cpp:603-618` is `enqueue_memcpy(device_dst, host_src)`; no lockable dst required.
- **Allocation failure.** `allocate_memory` throws at compile time, so an over-budget pool would kill the load with a cl error rather than degrade. Decision: **per-buffer fallback to `usm_host` plus a counter**, never fail. The fit already reserves against a measured figure and can correct `n_ctx`; a hard failure would make offload unusable on the 16 GiB card — the card M9's exit criterion names.

**P2 `0006-moe-otd-async-batched-slot-upload`.** Group the per-expert, per-tensor copies of one `try_acquire_simultaneous` into a batch, `blocking=false`, one wait at the boundary.

- **Lifetime is the blocker, not the flag.** `payload` is a loop-local vector freed each iteration (`moe_otd_runtime.cpp:~222-254`); a non-blocking `enqueue_memcpy` may read it after return — use-after-free. P2 therefore *requires* a provider-owned staging arena: a ring of `usm_host` buffers recycled only after their event completes, with `maybe_transpose_scale_zp` writing into the arena. This is the M9 row's "pinned staging" and it is not optional.
- **Where the event is waited.** The last point before the impl enqueues the GEMMs: end of `on_before_batched_gemv`/`on_before_prefill` (`moe_3gemm_swiglu_opt.cpp:727-800`), just before `set_otd_weight_pointers`. On our cards the queue is in-order (`execution_config.cpp:344`), so program order already stops a kernel reading a half-filled slot and the wait exists only to recycle staging. The patch must not depend on that: gate on `stream.get_queue_type() == QueueTypes::in_order`, else collect events and wait them at that same boundary.
- **P2b, its own patch and its own A/B:** `on_before_batched_gemv` opens with an unconditional `stream.finish()` (`:736`) on the offloaded path — a full queue drain per MoE layer per token (×40). The blocking `topk_id->copy_to` that follows already syncs on an in-order queue, so the drain looks redundant, but it is a distinct ordering change. Worth **[to measure]** only after P1.

**P3 hot-set pinning — conditional on P0, not scheduled.** `LRUCache` (`lru_cache.cpp:38-70`) is a plain recency list; pinning would be an eviction-exempt set keyed by cumulative hit counts, sized by a config fraction. Reason to hold: at ratio 10 resident capacity is ~90% of experts per layer, so a miss is almost certainly *compulsory*, not a capacity miss — §7.0.2s's 74% came from a 16-token probe, i.e. warm-up. Prediction: **P0's `evictions` reads ≈0 at ratio 10 [to measure]**. If so, P3 cannot move the hit rate there. P1 makes it further marginal: pinning pays because re-acquiring a resident expert is expensive, and P1 removes the per-token PCIe stream that makes it expensive. Decision rule fixed now: write P3 only if P0 shows evictions > 5% of acquisitions at the ratio under test.

## 2. Interaction with M7's fit — which world are we in

The two ledgers disagree by design today: the plateau probe (`backend_ov.cpp:2424-2470`) reads 0.11–0.14 GiB device while the config/IR figure (`:2495-2530`) reads 12.01 GiB host, and `:2542` already logs that a device figure under 5% of the host estimate is the expected shape. With P1 the same arithmetic measures the true device pool — no new formula, the probe simply lands elsewhere.

**The check.** Primary: an fdinfo split across the probe — Δ`drm-total-vram0` vs Δ`drm-total-gtt` over the plateau loop (arcint already reads fdinfo for `device_resident_bytes`; gtt is one more field). Secondary, no new plumbing: `device_share = slot_pool_probe / slot_host_bytes`. Unpatched: ≤0.05 (measured 0.011 at ratio 20, 16 GiB card). Patched: ≈ the device-cap fraction, ≈1.0 uncapped. A `device_share` in neither band means a build mismatch — refuse, do not warn.

**Reservation lines.**
- Unpatched: `expert slots 0.13 (probe)` charged; `expert slots host-side: 12.01 GiB (GTT, source: ir)` informational. Unchanged from §7.0.2t.
- P1 uncapped: `expert slots 12.01 (probe, device-resident)` charged; host line prints `0.00 GiB (GTT) — pool is device-resident`, so the absence is stated rather than implied.
- P1 two-tier: `expert slots 8.00 device (probe, cap 8.00) + 4.01 host (GTT)`, only the device half entering `FitTerms::slot_pool`.

## 3. The partially-fitting pool — two-tier is P1's core, not a follow-up

At ratio r the pool is `num_expert×(100−r)/100` slots per MoE layer; the 35B at ratio 20 prices ~12 GiB against a 16 GiB card already holding ~6.7 GiB of weights and graph (§7.0.2t). It does not fit, and a P1 that only fits on the 24 GiB card cannot be measured on the card the exit criterion names. So two-tier is P1's core.

**Granularity is forced by the data model.** One constant is one memory object covering all slots of one (layer, tensor); it cannot be half device. The tier is therefore per-constant, decided in `try_prepare_partial_upload` by charging `upload_layout.bytes_count()` against a running device budget in program order — earlier MoE layers land in VRAM, the rest in GTT. Whether program order is the right selection or it should be uniform-by-stride is **[to measure]**; program order is the smallest thing that works.

**Cap selection — the fit and the pool negotiate, in this order:**
1. arcint computes the analytic pool from the existing host-side ledger (`slot_pool_from_ir`, else the config formula at `backend_ov.cpp:2495-2530`).
2. Subtract from device total: weights+graph, drafters, activations, margin, `lanes×slab`, and KV bytes for a target `n_ctx` (the explicit one, or a floor when omitted). The remainder is the spare.
3. `cap = min(analytic_pool, spare × f)`, `f` **[to measure]**, exposed as `--expert-device-slots-mib` defaulting to `auto`, plumbed to the plugin as a config property read in `try_prepare_partial_upload`.
4. Compile. The plateau probe now *verifies* the cap instead of discovering an unknown, and the existing allocate–audit–replay loop corrects `n_ctx` by the measured overshoot exactly as today.

Step 3 is analytic and step 4 is measured, deliberately: the cap must exist before compile, and nothing before compile can be a measurement.

## 4. Pre-registered measurements

Rig: **A770 16 GiB, 35B q4, u8 paged KV, one lane, explicit `--n-ctx 32768`** (explicit, to hold the fit out of the comparison), `--offload-ratio 0/10/25/50` — the §7.0.2s sweep rerun. Decode probe **128 tokens, not 16**: the 16-token probe is warm-up-dominated and its 74% hit rate is a cold-start reading. Per run report decode t/s, `[OTD_PERF]` (hits, misses, evictions, tensor_load_count, disk_io_ns, transpose_ns, gpu_copy_ns), and fdinfo `vram0`/`gtt` at load **and at end of run**.

Red baseline to reproduce first: 1.9–2.3 t/s at every nonzero ratio, 309 µs/tensor, 13.1 s gpu_copy per 16 tokens. One reconciliation is owed on it — 13.1 s ÷ 309 µs ≈ 42.4k tensor loads for 16 tokens, against ≈12k implied by 26% of (16 × 40 layers × top-8) × 9 tensors. Read `tensor_load_count` directly rather than inferring it; if they do not reconcile, 309 µs is per something other than a tensor and §7.0.2s needs a correction, not a patch.

| patch | expected observable | sends it back |
|---|---|---|
| P0 | `evictions` and per-constant alloc type appear; no t/s change | any t/s delta — an instrument that costs is not an instrument |
| P1 | fdinfo flips: `vram0` up by ≈the cap, `gtt` down by the same; decode ≥2× the 1.9–2.3 baseline at ratio 25; miss-path latency (gpu_copy_ns ÷ tensor_load_count) falls, the dst no longer being host-mapped | Prüfstand not 10/10; output not byte-exact vs ratio 0 at the same seed; any `lock()` on a slot buffer; a fit refusal on a config that loaded before |
| P2 | `gpu_copy_ns` leaves the critical path — per-infer stage-acc median falls while gpu_copy_ns stays similar; `staging_bytes` bounded and constant across infers | output not byte-exact vs the P1-only build; stage-acc median unchanged (then the copies were never critical and the patch is unjustified) |
| P2b | per-token host time falls by the removed drains | any ordering defect, or no change |
| P3 | hit-rate delta at a ratio where evictions > 0 | delta < 2 points → drop |

M9's exit permits closing with "a profile naming the next blocker" instead of ≥8 t/s. If P1+P2 flip the fdinfo split and decode still sits near 2 t/s, that profile *is* the deliverable and M14 (expert FFN computed on the host) is the named successor — say so rather than iterating.

## 5. Red-first: patched vs unpatched, decided at load

Not an end-of-run number, so a wrong build cannot masquerade:
1. P0's `[OTD_PERF]` gains `evictions=`; the partial-upload log gains `alloc=usm_device|usm_host` per constant. An unpatched build prints neither token; `grep alloc=usm_device` is the build check.
2. arcint asserts the fdinfo split at load: with P1 active, Δ`drm-total-vram0` across the plateau probe must be ≥90% of the requested cap **[to measure — placeholder until one probe run sets the threshold]**. An unpatched binary moves `gtt` and trips it.
3. Negative control for P2: an env kill-switch forcing `blocking=true` on the patched binary must reproduce 309 µs/tensor. Same binary, both readings.

## 6. Risks

- **The OOM shift runs the other way, and that is the risk.** ~12 GiB moves out of GTT *into* VRAM: host pressure falls, device pressure rises on a card also holding weights, KV, and — on the other card — a co-tenant service. The two-tier cap is the mitigation, the fit's audit-replay loop the backstop. Disk reads remain, so the weights-file page cache still matters.
- **Driver migration.** The KMD may evict `usm_device` back to system memory under pressure, silently restoring the old performance mid-run. Invisible at load — hence fdinfo at end of run too; a falling `vram0` is a result, not noise.
- **`patches/0003`.** It edits the same `scratch_buffers` struct and the same `on_before_*` region (`moe_3gemm_swiglu_opt.cpp:697-800`) P2/P2b touch. Order fixed: 0003 first, P2 rebased onto it, and P2 must not resurrect the per-infer subbuffer churn 0003 removed.
- **Fusion-impact obligation** (0.3.0 ground rule 2; DESIGN §8). P1/P2 write no kernel, but P1 changes the allocation type of constants feeding `MOECompressed` — exactly the class of change that can flip a graph-optimiser decision about whether a node is a lockable constant. Owed with the patch: a compiled-graph dump before/after showing §7.0.2u's ×40 `moe_3gemm_fused_compressed`/`moe_router_fused` counts unchanged, plus an end-to-end profile. A copy micro-benchmark does not discharge it.
- **Supply chain** (§1.1). Four more patches deepen the structural dependency on a hand-built x86_64 plugin package. Each stays PR-shaped and names its upstream discussion; P1 in particular is a plausible upstream fix rather than a local hack and should be offered as one.
