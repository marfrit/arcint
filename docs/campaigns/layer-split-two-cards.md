# layer-split-two-cards — one model across the B60 and the A770, a layer range and an expert cache per card

**Open** (operator, 2026-10-07).

## Charter

Run Qwen3.8-Flash-Next on the libllama engine across both Arc cards. Each card
runs a contiguous range of layers, keeps the KV of its layers and an expert
cache for its own layers, and the host bank holds every expert once for both.
The aim is the expert hit rate and the context that one card's VRAM cannot
hold together. On the B60 alone, each context step costs expert slots:

- 65,536 tokens leave 13,700 MiB of slots;
- 98,304 leave 13,000;
- 131,072 leave 11,200;
- 262,144 leave 6,500.

All four are `measured-here`, B60, f16 KV, MTP 2.

## Reference to follow

Strata's layer split (`~/src/Strata-ref`, `docs/MULTI_GPU.md`; the mechanism
at source in the recon section below):

- **Layers:** contiguous layer ranges per card. The last card also runs the
  output head and the MTP draft layer.
- **Expert caches:** each card fills an expert cache for its own layers from
  the profile. The expert arena in host RAM is shared, pinned once.
- **Hand-off:** one per verify window through pinned host memory, a few
  hundred KB. No peer-to-peer access, so an x4 link works, and each card's
  PCIe share is probed on its own link.
- **Prompts:** prompt chunks flow through the cards in turn. While a later
  card reads chunk c, the first already reads chunk c+1.
- **`auto`:** tries every placement and keeps the one whose caches hold the
  most of the expert profile, the hottest pairs weighted most.

Its measured effect (`paper`: `docs/MULTI_GPU.md`, the Coder on an RTX 5080 +
RTX 3090 at 32K, bench/results/2026-09-29-layer-split):

- prompts +18-20 %;
- decode on par with the faster card alone, and ahead on code once both
  caches hold nearly every routed expert;
- a third, slower card made it slower: every extra card costs its own round
  per window.

## Gate

On the record before the work starts.

- **Answers:** CLAUDE.md's answer-level bar, against the one-card arm at the
  same context:
  - the capital, the 20k needle and the long answer stay right;
  - the acceptance task battery passes;
  - mean KL against the one-card arm's logits on the same window is at most
    0.03 nats above the one-card arm's own run-to-run KL on the B60.
- **Decode:** at 98,304 tokens of context, the split's decode on the 20k
  needle and the long answer exceeds the one-card arm's beyond the run-to-run
  spread (B60, MTP 2, f16 KV, served through arcint).
- **Prefill:** at 20k it does not fall below the one-card arm beyond the
  spread.
- **The A770's misses:** it sits on PCIe 3.0 x4 through the chipset (1.83
  GB/s, `measured-here`), so its share of misses over the link is reported
  next to the rate.

## Current state

- **One-card baseline** (B60, `measured-here`, arcint 0.6.0):
  - 32,768 tokens with 14,500 MiB of slots: prefill 422-431 t/s at 20k,
    decode 33.0-34.8 t/s with MTP 2.
  - 98,304 tokens with 13,000 MiB: to be measured.
- **Eviction** (`measured-here`, B60, 2026-10-07): a post-load VRAM margin
  of 645 MiB or less is not enough. The first request allocates more, xe
  evicts ~15 GiB of the process's buffers to host memory, and they stay
  there: 24 prompt tokens in 49 s, 0.4 t/s decode. The split must keep each
  card's margin above what a request allocates late.
- **What a request allocates late** (`measured-here`, B60, 98,304 tokens of
  context, 13,000 MiB of slots, residency sampled every 0.5 s):
  - the first request keeps +880 MiB;
  - a 20k prefill transiently needs ~3 GiB more than that. VRAM peaked at
    24,294 MiB with 147 MiB free on the card, then xe moved ~3.5 GiB to host
    GTT, where it stayed.
  - A 13k prompt in that state verified at 4.5 s a round instead of ~25 ms.
  Strata subtracts the prompt path's buffers before sizing a card's expert
  cache (`code`: G:2286-2307). Patch 0021 sizes the slots from the flag
  alone, so the prefill's buffers are not in its budget. Each card's budget
  in 0024 reserves them.
- **arcint** selects one OpenCL device (`code`: `src/exec/backend_llama.cpp`,
  `GGML_OPENCL_PLATFORM` plus `GGML_OPENCL_DEVICE=0`).
- **Patch 0021's expert cache** is process-global, one slot pool (`code`:
  `contrib/llama.cpp/patches/0021-expert-cache-slots-and-usm-bank.patch`).
- **Patch 0023** (one context a platform, `GGML_OPENCL_DEVICES`), the coder
  Q4_K_M across both cards (`measured-here`, 2026-10-07): the op tests per
  card equal each card alone; the answer is right. With `-ts 3/2`:
  - pp4096 2,945 t/s against the B60 alone's 2,470 (+19 %);
  - pp512 2,210 against 2,618;
  - decode 57.5 against 77.2 t/s.
  A model that fits on one card loses decode to the split; Strata has
  `--split-skip-if-fits` for that case. Flash-Next does not fit; its gain
  is the second card's expert slots (0024).
- **The B60 unit** at 98,304 tokens with 11,500 MiB of slots: no eviction
  (GTT stayed at the bank, 947 MiB free at the peak). 20k needle right,
  prefill 410 t/s, decode 21.0 t/s on the long answer.
- **Two OpenCL platforms** (`measured-here`, Intel compute runtime
  26.27.39122.11): the runtime lists the B60 and the A770 as two platforms,
  one device each. A cl_context spans one platform. A USM host allocation
  belongs to one context.

## Recon: the reference at source

Strata, `~/src/Strata-ref`, all `code`. G = `src/program/generate.cpp`,
V = `src/core/verify.cpp`, P = `src/prefill/prefill.cpp`.

- **Path:** the split is in the CUDA/HIP path. This checkout carries no SYCL
  sources.
- **Stages:**
  - one `GpuStage` per card (G:690-708);
  - a chain of `Verifier`s (verify.hpp:111-118, G:4139-4170);
  - a chain of `Prefill`s (prefill.hpp:107-114);
  - the single-GPU token graph is off under a split (G:3635).
- **KV:** each stage keeps the session (KV, QSA, GDN state) of its own layers
  only, at full context length (session.hpp:49-56, 94-98; G:2466-2467).
  `docs/MULTI_GPU.md` says "the full context on every card"; the code wins.
- **Dense weights:** each later stage carries a full copy of them (G:2184-2185).
- **Head and drafter:** the head is on the last stage (G:2471-2474). The MTP
  drafter is on the last stage's device and reads only its final residual
  (G:2498-2500, verify.hpp:142).
- **Expert cache per stage:** filled from that stage's share of the profile,
  hottest first, into the room left after the stage's session, its prompt
  buffers and the reserve (G:2286-2307, 2860-2906). One host residency table
  holds slot indices local to the owning stage's cache and is replicated to
  every device (G:3595-3620).
- **Misses:** the CPU pool is shared and routed by layer to the owning stage
  (`SplitDrive`, G:662-685).
- **Link probe:** the PCIe share is probed per device (G:2220-2227).
- **Swaps:** adaptive swaps go to the owning stage's cache on its own stream
  (G:4462-4481).
- **Arena:** pinned once with `Portable|Mapped` (`src/core/pinned.cu:319`).
- **`auto`:** minimises a predicted decode-window time,
  `miss_ms·(1−held) + Σ_stage layers·layer_ms` (G:2330-2391), and tries every
  placement for two cards. `docs/MULTI_GPU.md` says "the most of the profile
  held"; the code wins.
- **Decode hand-off:** ~51 KB a token (the residual, the pending write and
  the inject) through mapped host memory, once per verify window (V:815-820,
  421-425). The stages run serially; the async commit is off under a split
  (G:2169).
- **Prefill hand-off:** a one-chunk pipeline through two host buffers a
  boundary; stage 1 reads chunk c+1 while stage 2 reads chunk c
  (P:526-530, 1866-1882).
- **Measured there** (bench/results/2026-09-29-layer-split, `paper`), 5080 +
  3090, Coder, 32K:
  - K=26: prompts 2,039 / 2,357 t/s against 1,726-2,017 / 1,970 alone;
    decode 83.8 / 109.7 against 83-87 / 88-105;
  - short prompts (2K) slower on the split;
  - a third card (a 2080 Ti) slowed it.

Our llama.cpp pin (bed0a85, patches 0001-0022), `code`:

- **Device selection:** ggml-opencl probes one platform and makes one
  shared context (`ggml_opencl_probe_devices`). arcint selects one device
  (`src/exec/backend_llama.cpp:139-148`).
- **One-context assumptions:**
  - kernels are built with `clBuildProgram(p, 0, NULL, ...)` for every
    device in the context;
  - `supports_buft` accepts any buffer of the same context;
  - `get_max_size` caches one value in a static (the A770's single
    allocation is capped at 4 GiB).
  Separate contexts per card make the first two per-device by construction.
  The static stays to fix.
- **Cross-device copies:** the backend declares no `cpy_tensor_async`, so
  the scheduler copies through host memory (`ggml_backend_tensor_copy`).
- **Sync between backends:** `graph_compute` calls
  `sync_with_other_backends` for every node. That is a decode cost to
  bound.
- **Layer placement:** llama.cpp's layer split puts layers, their KV and
  recurrent state on `dev_layer(il)`, and the output layer on the last device
  (`llama-model.cpp:1557-1614`, `llama-kv-cache.cpp:216-218`). CPU buffer
  overrides win over the layer's device.
- **Patch 0021 is one-card by construction:**
  - its cache takes the first GPU layer's device and skips layers on any
    other (`// one card`);
  - one slot buffer, one bank, one budget, one stats struct;
  - `exchange_slices`' static scratch and the `keep` ring are shared
    statics;
  - the routed-ids readback finishes on one queue.
- **Kernel choice per card** follows the device's sub-group sizes per backend
  context (`sg8`, the IQ4 codebook select), so each card gets its own
  kernels once each card builds its own programs. The `GGML_OPENCL_*`
  tuning knobs are process-wide.
- **MTP:** the MTP GGUF is loaded as a second model with default params
  (every visible device). It must be pinned to the last device, next to the
  output head.

## Design

- **ggml-opencl, one context a card** (patch 0023): probe the GPU devices of
  every Intel platform, with a context, a queue and programs per device and a
  distinct device name. Fix the remaining statics per device: `max_size`, the
  flash-attention `failed[]`, the scratch buffers. Bound
  `sync_with_other_backends` to the devices the graph actually crosses.
- **Expert cache per card** (patch 0024): one cache per device, each over
  the layers that device runs.
  - Its own slot buffer (split under 4 GiB on the A770), its own USM bank
    in its own context holding exactly its layers' other experts, its own
    budget, residency table and stats.
  - The routed-ids readback and the swaps run per queue.
  - Budget: Strata sizes each card's cache after the session, the prompt
    buffers and a reserve are placed (G:2286-2307). llama.cpp loads the
    model, and with it the cache, before the context allocates KV and
    compute buffers. So the budget per card is the operator's value
    (`--llama-expert-cache A,B`), set from the measured late allocation:
    ~0.9 GiB kept after the first request plus ~3 GiB during a 20k
    prefill (B60). Deviation recorded 2026-10-07; a reservation computed
    at context creation is the follow-up.
  - Strata's arena is one pinned region read by every stage. Here each card
    reads only its own layers' bank, which is the same access pattern, in
    two allocations, because a USM allocation cannot span two platforms.
- **arcint:**
  - `--device GPU.0,GPU.1` and `--llama-layer-split K`: llama.cpp's
    `LLAMA_SPLIT_MODE_LAYER` with `tensor_split` set from K;
  - the expert budget per card: `--llama-expert-cache B60_MiB,A770_MiB`;
  - the MTP model pinned to the last device;
  - the expert profile split by layer.
- **Order:**
  1. 0023 with a dense model across both cards (llama-bench `-sm layer`)
     against each card alone;
  2. 0024 with Flash-Next at K chosen by Strata's window-time model, the A770
     at 1.83 GB/s, so its share of misses is weighted by its link;
  3. served, against the one-card gate above.

## Where it lives

`src/exec/backend_llama.cpp` (device selection, model and MTP contexts),
`contrib/llama.cpp/patches/` (0021 the expert cache, 0022 the Intel kernels).
