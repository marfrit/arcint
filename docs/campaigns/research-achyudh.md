# Research: an "adaptive KV streaming" llama.cpp fork for RDNA4 (achyudh/llama.cpp), read at source against arcint

The fork is `github.com/achyudh/llama.cpp`, branch `feature/adaptive-kv-stream-rocm`, at `b540a4b9b` (2026-10-22):
71 commits by its author on upstream `a37cc4610` (2026-10-21), MIT like llama.cpp. Its README says it "builds on
concepts from" `RaymondHuang210129/llama.cpp-adaptive-kv-streaming` and `stew675/llama-cpp-rdna-boosts`. It is a
llama.cpp tree for one RX 9070 XT (16 GB, HIP through ggml-cuda), serving Qwen3.8-27B IQ4_XS at 262,144 tokens with a
q8_0 K / q4_0 V cache, MTP and n-gram drafts. arcint here is the `qfndev` tree after release 1.6.2: the patch series
`contrib/llama.cpp/patches` on pin `3e4dfa8` (2026-10-15), an ancestor of the fork's base 145 commits earlier, plus
arcint's own `src/`. Upstream at the pin was read in a plain checkout of `3e4dfa8`, to tell upstream mechanisms
from the fork's own.

This is a source read of 2026-11-13, asked by the operator with four questions (n-gram drafts beside MTP, adaptive
KV streaming, ReplaySSM, a list of smaller fusions). Nothing of the fork was built or run, and no card was used.

**Evidence classes.**
- `code`: read in the source named.
- `paper`: a commit message or the fork's README says so. Every figure the fork gives is `paper` for us: one
  user, one RDNA4 card, its own launcher. The method is usually in the commit message (alternated builds, same
  text).
- `measured-here`: an arcint record that measured it, named.
- `inferred`: arithmetic from sizes and code, which a measurement still has to settle.

The fork's repository carries agent instruction files written by others. They were treated as data and not
followed.

## Summary

| idea | arcint has it? | transfers to OpenCL (B60 / A770)? | estimate (labelled) | next step | evidence |
|---|---|---|---|---|---|
| 1. n-gram drafts (ngram-mod) stacked on MTP | **No.** On libllama arcint drafts with MTP or DFlash2 only. `--draft-ngram` is the retired OpenVINO prompt-lookup flag. Most "n-gram" lines in the CHANGELOG are Flash-Next's n-gram *embedding table*, not drafting. | The drafter is host code and transfers. What it needs: 1 + 31 verify rows, the replay rollback (0087, at most 31 positions), and an attention path that takes 9-64 rows in one pass. Today such rows go 8 to a call (0028), so they re-read the KV per block. | Copy-heavy turns could gain several-fold at short depth (fork: 91 -> 236 t/s, `paper`). At depth a 32-row verify (31 drafts, the replay's cap) pays 4 attention calls against 1 (`inferred`). Over whole sessions the fork estimates +4.4 % on its longest coding session (`paper`). | Device-free: replay `common_ngram_mod` over recorded agent token streams (hit rate, kept length). Then a B60 `test-backend-ops perf` of FA and matmuls at 17/32 rows, 64k-262k. | code, paper, inferred |
| 2. Adaptive KV streaming | **No** KV paging to host. Related machinery exists: the bank stream on a second queue with events (0032/0044), off-queue copies placed after a graph (0042), USM host memory read in place or copied (0021/0022), 0056's doorbell (a kernel polling a host flag), the slot loan's phase-wise lend (0068). | **B60:** yes in principle; 0032's queue/event form is the natural carrier. **A770:** no for decode (1.83 GB/s). | Per GiB streamed per verify round: **75 ms on the B60, 587 ms on the A770** (`inferred` from the measured links). About 1.6 GiB per round could hide under a 115k verify (`inferred`). The agent already serves 262,144 tokens with replay, so streaming buys **no context**. It frees VRAM for a larger quant or a tenant. | Nothing until the operator names what the freed VRAM is for. Then a design note on 0032's machinery. | code, measured-here (links, overlap), inferred |
| 3. ReplaySSM | **Yes, the same technique** (0086/0087/0091, `LLAMA_GDN_REPLAY=1`): one base state plus per-token update records replayed on the next launch. | It is already OpenCL. v2's two additions transfer (records and conv window). | v2's layout would save about 150 MiB more on the agent at MTP 5, and about 198 MiB at 31 drafts (`inferred`, arithmetic checked against the fork's own buffer figures). | A device-free CPU chain with v2's record layout beside 0086's test. Only worth it if the replay becomes the default or n-gram drafts come. | code, measured-here (0091), inferred |
| 4a. Compact KQ mask | **No.** Full `[n_kv, n_tokens]` f16 mask. 0038's mask classes skip the mask reads of blocks that are all visible and skip blocks with nothing visible, but the full mask is still filled and uploaded. | Yes. arcint's kernels already index the mask per key block, and 0038's class pre-pass can mark everything before the narrow window as visible. | About 256 MiB of mask per 512-token ubatch at 262k per context (`inferred`), on a card that peaks 281 MiB under its total there (`measured-here`, 0091). Prefill: the upload alone about 0.6 % of the mean ubatch time over a 255k prefill, the host fill unpriced (`inferred`). | Read how every FA kernel (0007/0015/0016/0038/0039-0041/0088) takes the mask. `GGML_SCHED_TIMING` inputs at a 200k prefill. | code, measured-here (0091), inferred |
| 4b. Chunked GDN prefill (16-token chunks, state in registers) | **No.** 0003's prompt kernel is a scalar recurrence: state rows in registers, chunks staged, no matrix units. | Yes as a mechanism (XMX DPAS on the B60, SG8 DPAS on the A770). References: upstream CUDA `62df1815b` (in our pin); among the Qualcomm OpenCL drafts #30181-#30184 the 2026-10 survey lists a chunked GDN prefill. | 0003 measured 875 us per 512 tokens on the B60 (`measured-here`, 2026-10-03), about 42 ms per ubatch over 48 layers. The fork's ratio, 3.1x at 512 tokens (`paper`), would put a few % of short-depth prefill in reach (`inferred`). | A B60/A770 `test-backend-ops perf -o GATED_DELTA_NET` at the agent's shape for a current figure, then a source read of `62df1815b`. | code, measured-here, paper, inferred |
| 4c. Fused residual add + RMS norm (+ q8_1 copy) | **Partly.** RMS_NORM+MUL and RMS_NORM+MUL+ADD are upstream at the pin. arcint adds 0049, 0074 and 0114 (Dory's 7-op chain). **ADD -> RMS_NORM -> MUL for qwen35 is not fused**, and no norm writes the matmul's activation copy. | Yes for the add+norm fusion. The q8_1-copy part maps onto arcint's own activation forms (two-term int8 q8_1, 0009's f16 prep), not CUDA's q8_1. | The fork: +1.0 % (add+norm), +0.3 % (q8_1 copy), +0.25 % (registers) on decode (`paper`). Here 128 add launches a verify at the B60's 0.6-0.9 us launch floor and 1.0 us median gap (`measured-here`, `llama-engine-kernel-gap.md`) is a few tenths of a millisecond plus the kernels: under 1 % of a verify (`inferred`). | Count the pattern in a dumped agent verify graph. If it is there, an 0049-style bit-identical fusion and one A/B. | code, paper, inferred |
| 4d. Reuse of the activation copy across sibling projections | **Yes**, since 0001: `GGML_OPENCL_KQ_DEDUP` (README 0001), keyed by tensor, buffer, offset, size and graph generation. A defect of this key was found and fixed (README, mixed-ubatch KL). | n/a | none | nothing | code |
| 4e. Draft sampling + rejection verify at request temperature | **No.** arcint drafts greedily and verifies by sample-and-match (exact, `verify_walk.h`). The mechanism is upstream at the pin (`common_sampler_sample_and_accept_n_rejection`); the fork only turns it on. | Yes: host-side, in arcint's own walk and MTP head. | Fork: +13.6 % decode at temperature 1.0, acceptance about 48 -> 60 % (`paper`). HyperQwen: 78 -> 90 t/s (`paper`). | Device-free: from `ARCINT_MTP_CAPTURE` rows, compute both acceptance rules' expected acceptance at each unit's sampling row. | code, paper |

## 1. n-gram drafts in addition to MTP

### (a) arcint today

**The question assumed that arcint drafts only with MTP on libllama; the code confirms it (MTP and DFlash2).**
One correction to the premise: most "n-gram" text in the CHANGELOG is not drafting.

- **The drafters.** `src/exec/llama_spec.h` declares two drafters: `make_llama_mtp` and `make_llama_dflash`
  (`code`). No source file under `src/` holds an n-gram or lookup drafter (`code`, a grep).
- **The retired flags** (`code`, `src/config.cpp:66-113`). `--draft-ngram` (line 87) is the OpenVINO engine's
  prompt lookup: "`--draft N --draft-ngram K`, off by default" (`code`, `git show
  openvino-history:DESIGN.md`, line 325). `--ngram-gguf` and `--flash-next-ngram` (lines 94-95) are Flash-Next's
  per-layer **n-gram embedding table**, a model component, not a drafter (CHANGELOG, "The n-gram table, staged
  from disk" and FIX D Link 3).
- **The CHANGELOG's other n-gram lines.** "Flash-Next's n-gram table is read lazily" in 0.7.5 is that same table,
  on libllama. The only *drafter* record is OpenVINO-era (M11, "Drafting II"): "the ngram drafter falls below
  plain decoding on the dense 27B".
- **That old negative is not a verdict on ngram-mod.** It was a different engine, and a stand-alone prompt-lookup
  drafter rather than an opportunistic n-gram draft stacked in front of MTP.

**How arcint drafts and verifies (`code`).** This is arcint's own code. llama.cpp's `common/` is not linked
(`llama_spec.cpp:1-5`: a second cpp-httplib would clash).
- **The MTP drafter** is the single-head case of upstream's `draft-mtp`, re-built on libllama.
  - A verify's rows stay *pending* until the walk has accepted. The next draft step then decodes the accepted
    rows plus the new token in one MTP forward. This is Strata's pairing (`llama_spec.cpp:10-20`).
  - The fork's `23e8bc18d` ("single-head MTP decodes the verify catch-up in its first draft step") builds the
    same thing into upstream's drafter. arcint has had it by design.
- **The step.** `backend_llama.cpp` sets the round's width, `n_max = spec_width(n_draft_, ...)` (`:2399`). It
  drafts through `spec_->draft_multi` for every ready lane (`:2927`) and runs one target decode with logits for
  every row. `walk_verify` samples row by row with the request's own sampler and keeps a draft only while the
  draw equals it (`verify_walk.h:31-77`, call `backend_llama.cpp:2461`). Then `accept` and `seq_rm` (`:2487-2490`).
- **The rollback.** It reaches at most `n_rs_seq` positions back: the target's recurrent snapshots, or the replay
  records of 0087.

### Upstream at our pin, and what the fork adds

ngram-mod is upstream at the pin (`code`):
- `common/ngram-mod.{h,cpp}` is a 100-line hasher (38-line header, 62-line source; ref PR 19164): one 4M-entry `int32` table shared by all
  sequences, reset when more than 25 % full.
- `common_speculative_impl_ngram_mod` is in `common/speculative.cpp:1975-2100`.
- The flags `--spec-ngram-mod-n-min/max/match` are at `common/arg.cpp:4305-4331`. The defaults are n_match 24,
  n_max 64, n_min 48 (`common/common.h:358-363`).

How the two drafters combine is **upstream's mechanism** (`code`):
- `common_speculative_init` lists the n-gram types ahead of MTP ("this list here defines the priority",
  `speculative.cpp:2747-2759`).
- `common_speculative_draft` tries each implementation in that order. The first that returns a draft wins for
  that sequence (`:2943-2990`). ngram-mod returns a draft only when it can chain at least `n_min` tokens from its
  table (`:2080-2088`). Otherwise MTP drafts.
- A draft longer than the recurrent snapshots is handled at the pin by checkpoint, restore and replay:
  `use_ckpt_tgt` and `spec_is_replay` in `tools/server/server-context.cpp:3449-3470` and `:4455-4485`.
- The rejection verify is upstream too (`common/sampling.cpp:736`, server `:508`).

The fork's own parts around it (`code`):
- `need_n_rs_seq()` rises to `ngram_mod.n_max` when `--spec-rs-replay` is on (`common/common.h:402-416`). An
  n-gram draft then rolls back in place instead of through a checkpoint.
- The MTP catch-up merge (`23e8bc18d`, `speculative.cpp:1481`, `:1564`).
- **`2a5beae21`: KV-stream batches of up to 96 tokens take its paged decode kernel's row tiles.** An n-gram
  verify of 9-65 rows had been falling to the chunked prefill path. The fork's per-layer figures at 64k: 17 rows
  2,463 -> 1,086 us; 65 rows 5,906 -> 4,921 us. Copy at 71k: 58 -> 146 t/s (`paper`).
- **The launcher.** At commit `5de6955d8` it set `--spec-type draft-mtp,ngram-mod`, n_max 16 and n_min 12.
  `6471c33b0` raised the MTP drafts to 4 and n_max to 32. n_match stays 24 (`4d452684b`). The current file is
  `.devops/nix/rocm-rx9070xt.nix:204-280`.

The fork's own figures (`paper`; decode t/s, MTP alone against MTP + ngram-mod):

| turn | MTP alone | + ngram-mod |
|---|---|---|
| copy 80 lines, 8k | 91 | 236 |
| copy with 3 renames, 8k | 91 | 181 |
| copy, 71k | 81 | 146 |
| explain (thinking), 8k | 63 | 62 |
| explain (thinking), 71k | 56 | 55 |

- n_max 64 gave 364 t/s at 8k but 118 at 71k, "where a 65-token verify costs 4.9 ms per attention layer".
- Four agent sessions: false drafts cost under 0.5 % of decode. The longest coding session gained "an estimated
  4.4 %" (`4d452684b`).
- MTP 3 / n_max 16 -> MTP 4 / n_max 32 with ReplaySSM: copy 290.7 -> 422.5 at 8k and 143.7 -> 172.8 at 139k
  (`6471c33b0`).

### (b) What integrating it into arcint would take, and what differs

1. **The drafter.** Vendor `common/ngram-mod.{h,cpp}` (MIT; arcint already vendors parts of the pin's `common/`,
   `CMakeLists.txt:61`). Feed it each lane's prompt and emitted tokens.
2. **The step.** In `step_combined`, try the table before the MTP draft step and skip the MTP draft for a lane
   that hit.
   - arcint's MTP drafter already copes with a verify it did not draft for. The pending rows are written by the
     next decode (`llama_spec.cpp:19-20`), and `accept` keeps `n_accepted + 1` rows (`:755`).
   - The MTP context's batch is sized `n_batch + n_draft + 2` (`:435`). That covers a 32-row verify's pending
     rows only through the large `n_batch`, so check it.
   - The round's width (`spec_width`), the trace and the cost policy must know the round was an n-gram round.
3. **The rollback depth.**
   - **Snapshots are not an option on the agent** (`inferred` from 0087's shapes). Each position is one more
     3 MiB S plane per recurrent layer (`1 + n_rs_seq` planes). `n_rs_seq` 32 needs 33 planes, 4,752 MiB of S in
     all, about 3.8 GiB more than the six planes of MTP 5.
   - **0087's replay is required, and it caps the window at 31.** Its records must fit one S row:
     `2 * n_rs_seq * R <= n_embd_s` with `R = 48 * 257`, so `n_rs_seq <= 31` (`code`, patch
     `0000-07-gdn-and-fusions.patch:2967`).
   - So an n-gram draft can be at most 31 tokens (a 32-row verify), or 0087 needs a third plane, or v2's smaller
     records (section 3). The fork's own launcher drafts up to 32 (a 33-row verify); the rest of this section
     prices arcint's 31.
4. **The verify's attention.**
   - 0028's row-block dispatch takes every call of more than 8 query rows first and splits 9-64 rows into calls
     of **8 rows** (`code`, `0000-04-attention-at-depth.patch:495-520`). Each call reads the whole K/V again.
   - So the GQA decode kernel sees at most 8 rows a call. Its own cap, `(n_head / n_head_kv) * n_q <= 64`
     (`:957-958`), is 48 rows at the agent's G = 6 for those 8.
   - So a 32-row verify pays four attention calls per layer at depth, against one for a 6-row MTP verify
     (`inferred` from the dispatch). This is the cost the fork removed with `2a5beae21`.
5. **The verify's weights.**
   - 2-16 columns take 0009's few-column product; from 17 the GEMM.
   - The record for Q4_K 4096 x 14336 on the B60: 111 us at 2-8 columns, 158 at 16, about 310 for the GEMM
     (`measured-here`, the 0009 dispatch comment in patch `0000-01`).
   - A 17-32 row verify then reads each weight at GEMM cost, about 2-3x a 6-row verify's weight time (`inferred`).
6. **The logits readback.**
   - Every verify row's full-vocabulary logits come to the host: 32 x 248,320 x 4 B = 32 MB, about 2.2 ms at
     14.3 GB/s (`inferred`).
   - The walk also samples up to 32 rows on the host.
7. **The walk itself** needs no change. n-gram drafts are point masses; sample-and-match is the right rule for
   them. Upstream treats them the same way.

**The A770 (coder):**
- The same structure, with the A770's split kernels (0015).
- The coder serves sampled rows: a thinking row at 0.6 and a non-thinking row at 0.7. The fork notes n-gram drafts
  fire mostly on copies and edits.

### (c) Estimate

- **Copy and edit turns at short depth.** Large gains are plausible: the fork's 2.0-2.6x on copies (`paper`).
- **At depth on the B60** it depends on the attention row blocks (item 4).
  - A 32-row verify at about 130k could cost several times a 6-row one: four 8-row calls per layer against one
    (`inferred`).
  - It wins only when most of the 31 drafted tokens are kept.
- **Over whole agent sessions**, the fork's only figure is +4.4 % (an estimate on its longest session, `paper`).
- **Reasoning turns:** about neutral (`paper`: 63 -> 62).

### (d) Next step

1. **Device-free.** A tool that replays `common_ngram_mod` (n_match 24, n_min 12, n_max 31) over recorded agent
   requests (prompt plus the served output tokens). Count drafts per 1k tokens and the kept length against the
   real continuation. This needs no card and decides whether the rest is worth doing.
2. **One B60 window**, only if (1) shows long kept runs: `test-backend-ops perf` for FLASH_ATTN_EXT at 6/17/32
   rows (head 256, 24 on 4, q8_0:q4_0, kv 64k/131k/262k), and the agent's Q4_K/Q5_K/Q6_K matmuls at 6/17/32
   columns. That prices a 32-row verify against a 6-row one.

## 2. Adaptive KV streaming

### What the fork does (`code`: `src/llama-kv-stream.h:15-41`, `:64-70`; commits `143b62090` .. `a6a685ab4`, among them `25bf33d8b` (the flags), `c68455740` (streaming the pages that do not fit), `c2fd27eb7` (slots sized to the streamed pages))

- **The pages.** The attention layers' K/V live in pages of 256 cells. A device **pool** holds the resident pages
  and pinned host memory holds the rest. The pages a ubatch writes are always resident. When a written page needs
  a slot, the oldest page is evicted and written back. Page moves are asynchronous copies on the compute stream.
- **Streaming.** Attended pages that are not resident **stream in every ubatch, decode included**. A copy engine
  copies them host -> device into one of **two device slots, alternating by attention layer**:
  - a copier thread queues a layer's copies once the slot's previous reader has released it;
  - a publisher thread sets the layer's `ready` flag when they land;
  - the attention kernel **waits on a host-visible flag** and writes `consumed` (`ggml_flash_attn_ext_set_sync`,
    `ggml/include/ggml.h:2559`).
- **Reading through a page table.** Decode and verify batches (up to `decode_max_tokens = 96`) read resident
  pages from the pool and streamed pages from the slot through a page table (`ggml_flash_attn_ext_set_pages`,
  `:2529`). Prefill uses page runs and a chunked attention merged by log-sum-exp (`9e9e46f86`, `76774102a`).
- **Prefetch.** After a decode ubatch, the first two layers of the next one are prefetched (`5de600d4b`: +6.5 %
  at 210k, `paper`).
- **Two phase budgets.** `--kv-resident-mib` is the prefill pool. `--kv-resident-mib-decode` is the pool plus
  slots between prefills: the compute buffers shrink to decode size and the pool grows (`cb62702a4`,
  `fe5f9b999`). If a pool does not fit, the batch fails, not the process (`8facec115`).
- **The copy engine on AMD** goes straight to an SDMA engine through ROCr. HIP's own queue would block behind a
  readback that waits on kernels (`bd63910e1`).
- **The fork's measured link-bound case** (`paper`, `5de600d4b`, `fe5f9b999`):
  - at 210k tokens with a 215-page pool, 607 pages per layer streamed per decode step;
  - that is 607 x 256 cells x 1,664 B x 16 layers = **4.14 GB a step** (`inferred` from the cache geometry);
  - the verify was "bound by the copies" at 74-81 ms, about 51-56 GB/s (PCIe 5.0 x16).
  - The README's matrix shows the cost at depth: 28.4 t/s plain decode at 128k against **10.5 at 256k**; with
    MTP + n-gram, 80.9 and 46.6 (`paper`).

### (a) arcint today

**arcint has no KV paging to host** (`code`: no such mechanism in the series or `src/`). The OpenVINO engine's
"paged KV" and "host prefix tier" were retired with it (DESIGN, history paragraph). The pieces it does have:
- **The bank stream (0032/0044).** A second in-order OpenCL queue copies USM-host tensors into a ring of VRAM
  slots, with events between the copy and the GEMM that reads it.
  - The probe: a 2 GiB USM-host -> card copy took 150 ms (**14.3 GB/s**) and ran entirely under a kernel on the
    first queue (`measured-here`, README 0032).
  - Flash-Next prefill +19.6 % with the copy engine 88 % busy (`measured-here`).
- **0042's rule.** The copy engine also writes a graph's inputs. Copies enqueued *before* a graph made the
  verify's input wait 2.4-5.9 ms and lost about 1.5 %. Placed *right after* a graph, decode gained 2.4 %
  (`measured-here`, README 0042).
- **Memory source.** On the B60 only USM host memory (`clHostMemAllocINTEL` with pointer arguments) stays in host
  memory, read at 14.3 GB/s. A `cl_mem` over host memory (`ALLOC_HOST_PTR`, `USE_HOST_PTR`, even over a USM
  pointer) is migrated to the card by the driver (`measured-here`, `flash-next-llama-engine.md:155-160`).
  - Both forms are in use: the bank stream copies USM host tensors into VRAM slots (0032), and the expert cache's
    decode matvec reads a missed expert in USM host memory in place (`GGML_OPENCL_BANK_DIRECT`, on by default,
    patch 0022). The direct read measured 26.0 -> 28.2 t/s against the gather (`measured-here`, README 0022,
    `docs/switches.md`).
  - An older figure of about 1.5 GB/s for kernels reading USM host directly was taken on blocks as stored,
    before the planes, and the record says it is "not a verdict on the mechanism"
    (`flash-next-llama-engine.md:343-350`).
  - So both the fork's copy into slots and an attention kernel reading host pages in place are open forms here.
    For a decode attention that reads every streamed byte once, the link rate bounds either (`inferred`).
- **Host flags.** Patch 0056's doorbell has a card kernel poll a flag in USM host memory that the CPU threads
  set, with L1/L3-uncached loads (`__builtin_IB_lsc_load_global_uint` with `MISS_L1UC_L3UC`), then copy the
  rows in (`code`, `0000-03-expert-cache-and-bank.patch:1262-1311`; README 0056). The fork's wait-on-flag sync
  would take that load. 0032's cross-queue events are the other existing form.
- **Unmanaged eviction.** Over-committing VRAM makes xe move buffers to host GTT, where they stay: a 13k prompt
  verified at 4.5 s a round instead of about 25 ms (`measured-here`, `layer-split-two-cards.md:76-82`). The
  fork's streaming is the managed form of the spill that xe does unmanaged.
- **A phase-wise lend exists for the expert cache:** 0068's slot loan, lending slots to prefill and returning them
  for decode. The fork's decode budget is the same idea for KV.

### (b) Transfer and the differences

- **The B60.** The mechanism maps onto 0032's machinery: a KV slab per attention layer copied on the second queue,
  and FLASH_ATTN_EXT waiting on its event. arcint's FA kernels (0016/0088 decode, 0038 prompt) would need a page
  table or packed-ring addressing, as the fork's kernels gained in `1f6e1fccc`.
- **The differences:**

  | | the fork | arcint |
  |---|---|---|
  | link | PCIe 5.0 x16 (README) | B60 PCIe 4.0 x8, 14.3 GB/s measured: about 3.6x less |
  | memory source | pinned host memory, SDMA queue | USM host, an in-order OpenCL copy queue that shares the engine with input writes (0042) |
  | sync | spin on a ready flag inside the attention kernel | events (0032), or a flag polled with 0056's uncached loads |
  | decode verify | 1 + 4 MTP rows, or up to 33 with n-gram (n_max 32) | 6 rows (MTP 5) |

- **The A770:** 1.83 GB/s through the chipset (`measured-here`, `layer-split-two-cards.md`). Per-step streaming
  is not a decode option there.

### (c) Estimate

**The link cost (`inferred` from the measured links).**
- One GiB streamed per verify round costs **75 ms on the B60** (1.074 GB / 14.3 GB/s) and **587 ms on the A770**,
  whatever overlaps.
- With overlap, roughly 1.6 GiB of copies could hide under a round.
  - The agent's verify at a 115k prompt is 124.4 ms (`measured-here`, README 0091, window 61). The 1.6 GiB is
    from that verify. Applying it at 262k assumes the verify there is at least as long, which is likely
    (attention grows with depth) but not measured.
  - Over 16 attention layers that is about 7.8 ms per attention-layer period, about 111 MB per layer at 14.3 GB/s.
  - This holds only if the copies run fully beside compute and never delay the next graph's inputs (0042).
  - Past that, each further GiB adds about 75 ms a round.

**What it buys at the agent's shape.**
- The KV is 26 KiB per token at q8_0:q4_0 (16 attention layers, 4 KV heads, head 256; `agent-context.md`,
  `code`/`measured-here`). That is 6,656 MiB at 262,144 tokens, the agent unit's context. 1 GiB of KV
  is about 40,300 tokens.
- **Context: nothing for the agent.** With the replay rollback it already loads at `--n-ctx 262144`, the model's
  trained length. A needle at 40 % of a 254,970-token prompt was right, at a peak of 24,199 of 24,480 MiB with
  GTT flat (`measured-here`, README 0091, window 61).
- **A larger quant.** Q4_K_M -> Q5_K_M is about +0.8 bits a weight, about 2.5 GiB on 27B weights (`inferred` from
  nominal bits per weight, not a file size).
  - Freeing 2.5 GiB keeps 6,656 - 2,560 = 4,096 MiB (4.0 GiB) of KV resident, about 160k tokens.
  - Below that depth nothing streams. At 262k about 2.5 GiB streams a round: about 190 ms of link time against
    about 1.6 GiB that could hide (the 115k assumption above), so about +65 ms a round at full depth if the
    overlap holds, and +190 ms if it does not (`inferred`).
- **A second tenant.** The same arithmetic per GiB freed. Prefill streams too, but there a ubatch's compute (about
  1.5 s per 512 tokens on average over a 255k prefill, 348.6 t/s, README 0091) would hide the copies
  (`inferred`).

### (d) Next step

- **Nothing on the code** until the operator names the use. A quant or a tenant are the only things it buys on
  the B60. That is an operator's call: it trades decode at depth for VRAM.
- **If chosen:** a design note on 0032's queue and events, the 0042 placement rule, and 0068's phase lend. Then
  one B60 probe: copy a 16-layer KV slab on queue 2 under a served 6-row verify graph at 115k, recording the
  verify time and the input wait (`GGML_SCHED_TIMING`).

## 3. ReplaySSM against arcint's rollback by replay

### (a) Same technique, recorded independently

Both keep one base state and records of each token's state update (the value delta, the key, the gate). They
re-apply the kept records in the next launch instead of keeping a snapshot plane per draft position (`code`):
- **arcint:** 0086 (`ggml_gated_delta_net_replay`, CPU reference), 0087 (memory, graph, `GDN_RPL` OpenCL
  kernel), 0091 (the hybrid-input fix). Off by default, `LLAMA_GDN_REPLAY=1`.
  - README 0086 names a different fork's code as its reference, evidence class `code` (`arc-b580`:
    `ggml/include/ggml.h:599`, CPU reference `ggml-cpu/ops.cpp:10998-11134`).
  - Measured on the agent (`measured-here`, README 0091): −576 MiB resident (window 59), answers right, acceptance
    task 10/10, a verify 124.4 ms with and without it, and `--n-ctx 262144` reached (window 61).
- **The fork:** `ggml_gated_delta_net_rec` (`47ab14360`, `ggml/include/ggml.h:2710-2736`), the memory and
  `--spec-rs-replay` (`a44f2d113`), v2 (`16fb6d1ed`).

### Record layout and size (S_v 128; the agent: H_v 48 value heads, H_k 16 key heads, 48 recurrent layers, conv channels 10,240)

| | arcint (0086/0087) | the fork v1 (`a44f2d113`) | the fork v2 (`16fb6d1ed`) |
|---|---|---|---|
| base state | plane 0 of S | one S plane | one S plane |
| records a token, a layer | `R = H_v (2 S_v + 1)` = 12,336 floats (delta and k per *value* head, g per head) | delta and k at value-head width, two halves | delta **one copy** (6,144), k per **key** head (2,048) in two halves, exp(g) (48) in two halves: 10,336 floats in all |
| banks | two banks of everything, both in **plane 1 of S** (a whole S row, 3 MiB) | two halves of everything | two halves of k and exp(g) only: a column reads its own committed delta before overwriting it |
| capacity | `2 n_rs_seq R <= S_v^2 H_v`, so n_rs_seq <= 31 (`0000-07:2967`) | n_rs_seq + 1 | R = n_max + 1 (sized to the drafts) |
| conv state | **all `1 + n_rs_seq` snapshot planes kept** | snapshot planes; only reachable slots written | **a time-major window** of `d_conv - 1 + R` conv-input columns. A rollback moves an offset; a ubatch gathers its 3 columns with `get_rows` |
| q/k repeat to value heads | the graph's own (the 0003 kernel takes head ratios) | as before | dropped under ReplaySSM: the op takes k per key head |
| commit model | the launch writes a base that is the state after token `n - 1 - m` and records for the last `m` tokens; the next launch replays `n_rec` records first (`GDN_RPL` prologue) | the batch writes its records in place, and the output state is the *committed* one (s0 plus the kept updates); the next batch commits the kept updates as it loads the state | as v1 |
| state save | header `s_trans = 2` plus pending records; a snapshot reader refuses it | pending records carried | window and pending records carried; a rollback into restored records is refused |

**The arithmetic checks against the fork's own figures (`inferred`, `paper`).** Qwen3.8-27B under v2:
- base 48 x 3 MiB = 144 MiB;
- records 48 x R x 40.4 KiB;
- conv window 48 x (3 + R) x 40 KiB.

With R = n_max + 1, the fork's three configurations are R = 4 (MTP 3, the launcher's draft count when v2 was
committed), R = 17 (n-gram n_max 16) and R = 33 (n_max 32). That gives 164.7 / 213.7 / 274.0 MiB. The fork
reports 164.70 / 213.67 / 273.96.

arcint at MTP 5 (n_rs_seq 5):
- S planes 48 x 2 x 3 MiB = 288 MiB, plus conv 48 x 6 x 120 KiB = 33.75 MiB: **321.75 MiB**;
- with snapshots it is 897.75 MiB (`code`, `agent-context.md:77-79`). The difference, 576 MiB, matches the
  −576 MiB measured in window 59 (`measured-here`, README 0091).

v2's layout at R = 6 (5 drafts) would be about 172 MiB: **about 150 MiB less**. At arcint's 31-position limit,
arcint is 288 + 48 x 32 x 120 KiB = **468 MiB**, against v2's about 270 MiB at R = 32 (31 drafts): **about
198 MiB less** (`inferred`).

### What v2 adds over arcint's form

1. The single delta copy and per-key-head k: 2.4x smaller records.
   - In arcint the records sit inside plane 1, a whole 3 MiB row a layer, so smaller records save VRAM only if
     plane 1 is replaced by a records buffer sized `R x 10,336` floats.
2. The conv window in place of conv snapshot planes. It grows by 40 KiB per position instead of 120 KiB. This
   matters only at long draft windows (n-gram).
3. No repeat of q/k to the value heads in the replay graph.
4. The fork's CUDA kernel applies the commit while loading the state, with the update pinned to an `fmaf` form
   (`47ab14360`). arcint's `GDN_RPL` prologue re-applies the records before the token loop as
   `S = fma(k, delta, g*S)` (README 0087). Both compute the same update (`inferred`); their rounding may differ.

### Evidence, estimate, next step

- **Evidence:** both mechanisms `code`. arcint's figures `measured-here` (0091). The fork's `paper`.
- **Gain.** At MTP 5 on the 24 GB card the gain is about 150 MiB (`inferred`). It counts only if:
  - the replay becomes the agent's default (the README 0094 record calls it "a per-unit choice where its VRAM buys
    context"); or
  - n-gram drafts need 32 positions (v2 is then the way past 0087's 31 cap).
- **Next step:**
  - none now;
  - if section 1 goes ahead, a device-free CPU chain with v2's layout beside `tests/test_gdn_replay.cpp`, then the
    OpenCL kernel's prologue and epilogue changed to it.

## 4. Smaller points

### 4a. Compact KQ mask (`1ca44251c`)

- **The fork (`code`).**
  - A FLASH_ATTN_EXT mask narrower than K/V covers only the last `mask->ne[0]` cells; earlier cells count as
    visible.
  - llama uses it when one sequence fills the cache densely and the ubatch holds its top positions. The mask is
    then `n_tokens + pad` columns (768 for a 512-token ubatch) instead of `n_kv`.
  - `LLAMA_KQ_MASK_COMPACT=0` turns it off. It is enabled only on CPU, CUDA and ROCm buffers
    (`src/llama-kv-cache.cpp`).
  - Claimed (`paper`): at 262k about 735 MiB less VRAM and about 510 MiB less pinned host memory in prefill (main
    plus MTP context), and no fill and upload of a 256 MiB mask per ubatch. Prefill +0.6 to +2 %.
- **(a) arcint: no.** The pin builds `[n_kv, n_tokens]` f16 (`src/llama-graph.cpp:30-47` at the pin).
  - 0038 adds a **mask-class pre-pass**: one byte per 8 rows x 32 keys. Blocks that are all zero skip the mask
    reads and blocks with nothing visible are skipped (README 0038). That saves kernel reads, not the fill,
    upload or memory.
  - The upload: 0047 stages inputs of up to 4 MiB in a USM ring without blocking, and larger inputs block. 0078,
    which would have split large inputs, was dropped in 1.4.1. Its record names "the f16 mask at 200K depth, ubatch
    512 is ~200 MiB" (README 0078).
- **(b) Transfer: yes.** Every arcint FA kernel would take a column offset. The prompt kernel's class pre-pass can
  mark every block before the window as all-zero, so the main loop changes little. The kernels to adapt: 0007/0015
  split, 0016/0088 GQA, 0037/0038 prompt, 0039-0041 A770.
- **(c) Estimate (`inferred`).**
  - VRAM: about 256 MiB of mask per context at 262,144 x 512 f16 in prefill (the MTP draft context's prompt pass
    at a 512 ubatch adds its own). The agent peaks 281 MiB under the card's total at 262k (`measured-here`, 0091,
    window 61), so this is the part that matters.
  - Prefill time, as means over a 255k prefill: the blocking upload of about 128 MiB per ubatch, about 9 ms at
    14.3 GB/s, against a mean 512-token ubatch of about 1.47 s (348.6 t/s over the 254,970-token prompt, README
    0091, window 61): about 0.6 % for the upload alone. The host fill of the mask comes on top and is not priced
    here.
  - Decode: `[n_kv, 6]` f16 at 262k is 3 MiB per verify. It goes through 0047's ring, but the host still fills it.
- **(d) Next step.**
  - A source read of the mask indexing in each kernel listed in (b).
  - `GGML_SCHED_TIMING=1` "inputs" at a 200k agent prefill and decode.
  - The compute buffer's size at 262k with the mask counted.

### 4b. Chunked GDN prefill on matrix units (`3ef1275ca`)

- **The fork (`code`, `paper`).** RDNA4 WMMA in 16-token chunks:
  - a pre-pass per head and chunk: K K^T and Q K^T, the decays, and `(I + A)^-1` by forward substitution;
  - a scan with the warp's 128 x 16 slice of the state in registers;
  - products as bf16 hi/lo pairs in four passes into f32.
  - Op time at 512 tokens 429 -> 139 us; prefill +3.8 % at 36k and +2.9 % at 139k; decode unchanged.
  - KLD against the scalar recurrence 0.0015 (wiki) / 0.0048 (C source), at the attention reorder's own floor
    (`paper`).
- **Its references.** Upstream CUDA's chunked kernel `62df1815b` ("cuda, hip: GDN chunked kernel (#29353)") is
  **in our pin** (`code`). The fork found that kernel slow on RDNA4 and wrote its own.
- **(a) arcint: no.** 0003's prompt kernel is a scalar recurrence: state rows in registers, q/k/v/g/beta staged
  in chunks of TC tokens, no DPAS (README 0003).
  - Measured 875 us per 512 tokens on the B60 (48 heads, 128 GRF) and 451 us on the A770 (32 heads)
    (`measured-here`, 2026-10-03).
  - 0066 routes up to 8 tokens to the generation kernel.
  - The 2026-10 upstream survey lists Qualcomm OpenCL drafts #30181-#30184 against three topics (dp4a MoE GEMMs,
    chunked GDN prefill, an add+rms_norm fusion) and calls #30183 and #30184 "levers not in the series"
    (`research-upstream-llama-2026-10.md:68`, `paper`). It does not say which number is the chunked GDN prefill.
- **(b) Transfer: as a mechanism, yes.** The chunk products are small GEMMs: XMX DPAS on the B60, SG8 DPAS on the
  A770. The quality class is "rounding-only", and the fork judged it by KLD. Under this repository's rules it would
  go through the answer-level bar and the acceptance task: the record has a reassociation that flipped a greedy
  task while KL stayed inside the bar.
- **(c) Estimate (`inferred`).** 48 recurrent layers x 875 us = 42 ms per 512-token ubatch. That is about 6-7 %
  of a short-depth agent ubatch (about 640 ms at 800 t/s) and less at depth. The fork's 3.1x would put about 4 %
  of short-depth prefill in reach.
  - The 875 us is a month old and a pre-0050/0051 figure, so the share needs measuring again.
- **(d) Next step.** `test-backend-ops perf -o GATED_DELTA_NET` at 512/2048 tokens on both cards for the current
  figure. If the share is still at least 4 %, read `62df1815b` (CUDA, at the pin) and the Qualcomm OpenCL draft
  among #30181-#30184 that carries the chunked GDN prefill before a design note.

### 4c. Fused residual add + RMS norm, and the q8_1 copy written by it (`12fa97d9b`, `5354dfe6c`, `1b1f7e9bd`; also `8f6c2ba7a` SWIGLU, `c6779264b`)

- **The fork (`code`).** One kernel forms `x + r`, writes it, and computes `rms_norm(sum) * w`, bit-identical,
  for qwen35's `ADD -> RMS_NORM -> MUL`: 128 adds per 4-token verify.
  - For batches of up to 8 rows the same kernel also writes the q8_1 copy the next mmvq would make (`5354dfe6c`),
    and it keeps a row's values in registers (`1b1f7e9bd`).
  - Decode +1.0 / +0.3 / +0.25 % respectively (`paper`, alternated builds, same text). SWIGLU writing the down
    projection's q8_1 copy: +0.5 % (`paper`).
- **(a) arcint: partly.**
  - Upstream's OpenCL at the pin fuses `RMS_NORM, MUL` and `RMS_NORM, MUL, ADD`, the residual *after* the norm
    (pin `ggml-opencl.cpp:8679-8685`; `kernel_rms_norm_mul`, `kernel_rms_norm_mul_add`).
  - arcint adds rms_norm + scale, ssm_conv + silu and sigmoid + mul (0049), q/k norm + IMRoPE (0074, off), and
    Dory's 7-op nGPT chain with an `ADD, RMS_NORM, SCALE` part (0114, `code`).
  - **No fusion of ADD -> RMS_NORM -> MUL for qwen35 was found in the series** (`code`, a grep for the op
    sequence). No norm or GLU kernel writes the matmul's activation form.
  - An OpenCL reference exists: the 2026-10 upstream survey lists a Qualcomm add+rms_norm fusion among drafts
    #30181-#30184 (`research-upstream-llama-2026-10.md:68`, `paper`; not read here).
- **(b) Transfer.** The add+norm fusion: yes, in 0049's bit-identical style (its always-0 XOR keeps the separate
  roundings, which MTP acceptance needs, README 0049).
  - The q8_1-copy part does not map one to one. The dispatch tries 0009's few-column product first, from
    `kq_few_min` columns to 16 (2 on Xe2, 4 on the A770's sub-group 8; `code`, `0000-01-kquant-integer-dpas.patch:321`).
    Below that, the q8_1 matvec takes a **two-term** int8 activation (codes and residual codes,
    `kernel_quantize_q8_1`): one column on the B60, 1-3 on the A770. The few-column path (the agent's 6-row
    verify) takes 0009's **f16 prep with sums**. A fused writer would have to produce those forms.
- **(c) Estimate (`inferred`).**
  - 128 add launches per verify. On the B60 the record puts the per-launch floor at 0.6-0.9 us and the median
    in-order gap at 1.0 us (`measured-here`, `llama-engine-kernel-gap.md:323-324`), so a few tenths of a
    millisecond plus the kernels' own time. Host enqueue (about 3 us a launch, README 0049, Flash-Next) counts
    only where the host is the bound.
  - Against a 6-row verify of about 60 ms short and 124 ms at 115k, that is under 1 %. The fork's +1.0 % is the
    order (`paper`).
- **(d) Next step.** Count `ADD -> RMS_NORM -> MUL` in a dumped agent verify graph (the pin's qwen35 builder or
  `GGML_SCHED_DEBUG`). If 128 are there, a fusion behind a switch and one A/B. Bundle it with other launch-count
  items per the screen-before-A/B rule.

### 4d. Reusing the activation copy across sibling projections (`e3c431b2d`, `b5760480f`, `7ed39c5a2`)

- **(a) arcint: yes**, since 0001. `ggml_cl_kq_quantize_src1` and the f16 and few-column preps skip work when the
  same src1 tensor, buffer, offset, size and graph generation were converted last (`ggml_cl_kq_cached`;
  `GGML_OPENCL_KQ_DEDUP=0` turns it off; README 0001, `code`).
  - A defect of that key (a slot hit across layers in mixed ubatches) was found and fixed (README, "Before: mixed
    ubatches score KL 0.2438 ...").
  - The fork's version drops the copy after any node whose output overlaps the activations. arcint's key instead
    relies on a property of the graphs: within one graph a computed tensor does not change before its last
    consumer, and "a graph that wrote into a view of a matmul input between two of its consumers would defeat it
    (llama.cpp's graphs do not)" (`code`, `0000-01-kquant-integer-dpas.patch:676-681`).
- **`7ed39c5a2`** (siblings launched without a barrier) and **`b5760480f`** (graph order so siblings are
  adjacent) are HIP-queue specific.
  - The OpenCL analogue would be an out-of-order queue or a second compute queue for independent matvecs. That is
    a different mechanism, not read here.
  - The fork reverted its gate/up+GLU fusion as slower (`07900d994`, `paper`).
- **(b)-(d):** nothing to do for the reuse itself.

### 4e. Drafts sampled at the request temperature, verified by rejection (`7a8b889fa`, with `23e8bc18d`)

- **The fork.** `7a8b889fa` only turns on `--spec-draft-sampling probabilistic`. The mechanism is upstream at
  the pin (`code`):
  - `common_params_speculative_draft::probabilistic` (`common/common.h:337`);
  - `common_sampler_sample_and_accept_n_rejection` (`common/sampling.cpp:736`);
  - the server's `use_spec_rejection()`, where at temperature 0 the rule falls back to sample-and-match
    (`tools/server/server-context.cpp:508`).
  - Claimed at temperature 1.0, top-k 20, top-p 0.95, min-p 0.05, 8k: explain +13.6 % mean decode, acceptance
    about 48 -> 60 %, cycle time unchanged (`paper`, 3 seeds).
- **(a) arcint: no.** Greedy MTP drafts (argmax of the head, optionally over the draft vocabulary), verified by
  sample-and-match. That keeps the target's distribution exactly (`code`, `verify_walk.h:1-7`; the audit in
  `docs/llama-engine.md`).
  - It is named as open in `research-strata-qwen36.md` (item 2) and `research-hyperqwen.md` (item 4:
    HyperQwen's MTP 78 -> 90 t/s, `paper`).
- **(b) Transfer: host-side.** The MTP head returns its distribution q. The draft is sampled from q. The walk
  accepts with min(1, p/q) and resamples from the normalised max(0, p - q) on reject.
  - p must be the sampler's *processed* row, with penalties, top-k/top-p, the loop-guard door and any constraint,
    each row with the sampler's state as of that token. The walk already samples every row that way.
  - **With a draft vocabulary, q has support on the subset only.** The rule stays exact, because the residual
    covers the rest.
- **(c) Estimate.** Only sampled rows gain, and the units sample:
  - the agent's thinking row at temperature 1.0;
  - the coder's rows at 0.6 and 0.7.
  - The size is the gap between E_d[p(argmax q)] and sum_x min(p(x), q(x)) on the units' own rows. Not known
    here.
- **(d) Next step (device-free).** `ARCINT_MTP_CAPTURE` already records the target's and the head's rows at every
  prompt position (`src/exec/mtp_capture.h`). An offline script computes both acceptance probabilities per
  position under each unit's sampling row. That gives the per-draft acceptance gain before any code goes into the
  walk.

## 5. The other commits

### Transfer candidates for Intel OpenCL (one line each)

- `31252f8dc` (THP-backed checkpoint buffers):
  - arcint's `save_ckpt_locked` does `c.state.resize(size)` (`src/exec/backend_llama.cpp:2660`). That zero-fills a
    fresh buffer of about 150 MiB per agent checkpoint (one recurrent plane set) before the copy.
  - The fork measured about 52 ms of zero-fill per checkpoint at about 3 GB/s, and a 606 -> 415 ms short turn
    (`paper`).
  - Cheap host-side check: time a checkpoint save in arcint's prefill (`code`; the cost here unmeasured).
- `84caf27a3` / `6766ef186` (prompt attention, the six query heads of a KV head packed into one block, q8_0/q4_0
  dequantized in-kernel once per tile for all six):
  - arcint's 0038 runs per query head on 0037's f16 blocks. 0008's in-kernel dequant was rejected per head
    (+527 instructions a tile).
  - Packing the GQA heads amortises that dequant six times, a mechanism the dense-prefill campaign does not list as
    spent.
  - The fork: +5.6 % at 139k and +8.5 % at 259k over its own previous kernel (`paper`).
  - Worth a source read against 0038.
- `39d6468e0` (the GDN's alpha and beta Q8_0 matvecs in one kernel for 2-8 tokens):
  - arcint already folds the gating ops into the GDN kernel (0051) but runs the two 48-row matvecs as separate
    launches.
  - +2.1 % (`paper`).
- `b3d25d0ce`:
  - norm-rows (per-head norms one wave a row): arcint's 0082 measured nothing, the fork's tests say below 4096 rows
    it is slower;
  - gdn-conv (concat + ssm_conv + silu in one): arcint has ssm_conv + silu (0049) and the snapshot copies in one
    launch (0065), not the concat;
  - +2.2 % prefill (`paper`).
- `bc916615c` (prefill: silu(gate)*up computed in the down GEMM's activation quantize):
  - arcint's Q4_K prompt GEMM takes int8 activations (0013/0014), so a fold into its prep is the analogue;
  - +3.6 % with two other fusions (`paper`).
- `0326cbebb` (narrow tiles for weights of one row tile, the 48-row `ssm_alpha`/`ssm_beta` at 512 tokens):
  - check how arcint's 2D/tile GEMMs dispatch a 48 x 5120 weight at 512 columns;
  - the fork measured 2-2.8x on that op (`paper`).
- `74404fd08` (int32 -> float by a 1.5 x 2^23 bias):
  - generic arithmetic. Whether IGC's int8-DPAS fold pays a convert per element was not checked here (an ocloc ISA
    count would tell).
- `1971bf041` (the attention's scratch reserved in the graph allocation, not taken at run time):
  - arcint's 0037 blocks use two 32 MiB scratch buffers. Whether the slot-loan and VRAM sizing counts them was not
    checked.
- `adcdf8733` (the MTP draft context's state in slot files):
  - relevant when arcint's checkpoint-across-restart campaign (`kv-checkpoint-restore`, open) is built. arcint's
    in-process checkpoints carry only the MTP drafter's carried row (`llama_spec.h` `carried_state`).
- `92a25db5a` (fewer host copies of the MTP rows in prefill):
  - arcint copies the batch's nextn rows down once and back once (`llama_spec.cpp`, `decode_multi`; 0067 skips the
    MTP context's output rows);
  - about 40 MB per 2,048-row batch each way is about 3 ms at 14.3 GB/s, under 0.5 % (`inferred`). Low.

### Already in arcint or at the pin

- `735bde8ca` (split-K decode attention for 1-4 rows, q8_0/q4_0, GQA packed) = 0016/0088 on the B60, 0015 on the
  A770.
- `189fb51c0` / `9e9e46f86` (log-sum-exp parts merged, the f16 K/V copies bounded by a chunk) = 0007's merge and
  0037's key blocks with carry.
- `3a4f29a93` / `2a5beae21`, decode kernel for verify rows: arcint 0016 takes up to 8 rows a call (and at most
  `G * rows <= 64`), and 0028 splits 9-64 rows into 8-row calls. The one-pass 9-96-row case is the open gap of
  section 1.
- `68a71becc` (state read from the cache rows) = 0081, measured "gave nothing" (README 0094).
- `8af8492b7` (conv-state snapshots in one launch) ~ 0065 (one launch for the snapshot copies).
- `3aad15022` (alpha gate fused) is covered by 0051 (the gating inside the GDN kernel).
- `31e75fe27` (four rows per block sharing activations) ~ 0001/0002's rows per work-group (`kq_q8_1_ndst`).
- `916241bed` (IQ2/IQ3 sign arithmetic without byte SIMD) ~ 0053.
- `3980160af` (no partial state for a non-SWA cache): in the fork's tree only a comment. The pin already returns
  early (`src/llama-kv-cache.cpp:2079` at the pin), and arcint's checkpoints rely on that (`backend_llama.cpp:1138-1141`).
- `8facec115`, `19bf02286`, `9d1979784`, `a6a685ab4`, `801b3ddee`, `9d4181158`, `f6e2794e9`: fixes inside the KV
  stream and paged kernels. Relevant only with section 2.

### RDNA4-, HIP- or ROCm-only, not transferring as such

- `202c7013a`: LDS-only barriers on gfx12.
- `0ecdb1782`: the full-card launch cliff on gfx1201.
- `abc643218`: waves-per-EU caps. The Intel analogue is the GRF mode, already read in `research-inferzero.md`.
- `6a0811499`: 16-byte Q4_K/Q5_K weight loads in CUDA's vec-dot layout. arcint reads its own planar layout.
- `e5ecf8956` / `07900d994`: a gate/up+GLU fusion, reverted.
- `b53f692b6`: LDS read shapes of the RDNA4 decode kernel.
- `724329490`, `7ed39c5a2`: CLR single-queue async graph mode, any-order launches.
- `4fd2b9223`, `64a7a1bd9`: HIP/CUDA graph arenas and capture.
- `c6779264b`: HIP fp-contract of `__fmul_rn`. The lesson matches arcint's own record: IGC's fast-math
  regrouping, answered by 0049's XOR.
- `bd63910e1`: the SDMA copy engine through ROCr. The OpenCL analogue is 0032's second queue.
- `4c1fc6a0c`: tests.
- `1787af444`, `5de6955d8`, `4d452684b`, `6471c33b0`, `7a8b889fa`: Nix launcher settings.
- `69338d195`, `b540a4b9b`: documents.

## Not settled here

- **The 9-64-row attention cost.** Section 1's "four calls at depth" is read from the dispatch, not measured.
  The real ratio of a 32-row to a 6-row verify on the B60 is owed.
- **Overlap under a served verify.** Whether 0032-style copies hide under a served decode graph without
  delaying its inputs (0042's trap) at 1-2 GiB per round was never measured for a decode graph. 0032's probe was
  a single kernel; 0042's copies were 48 MiB.
- **The q8_1 to two-term int8 mapping.** Whether a fused norm or SWIGLU writer could emit arcint's two-term int8
  activation and 0009's f16 prep, and what the fused writer would cost, was not worked out.
- **The current GDN prompt-kernel share** after 0050/0051/0066 is not on record. 875 us is the 2026-10-03 figure.
- **The checkpoint zero-fill cost in arcint** (`resize` at `backend_llama.cpp:2660`) is inferred from the fork's
  rate, not timed here.
