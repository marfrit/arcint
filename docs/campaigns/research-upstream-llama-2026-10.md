# Upstream llama.cpp against the patch series (2026-10-15)

What ggml-org/llama.cpp did between the pin `bed0a85` (2026-10-02) and master
`3e4dfa8` (2026-10-15, 288 commits), and what is open there, set against
`contrib/llama.cpp/patches` 0001-0056. The survey for charter item C3's move
(`docs/milestone-1.0.0.md`). Evidence: `code` (the diff was read), `paper`
(the title and description only), `measured-here` (on this repository's
hosts).

## Merged since the pin that meets the series

- **An MoE expert cache** (#29887, #30112; `code`): `src/llama-moe-cache.*`,
  `--moe-cache-mib` / `llama_context_params.moe_cache_size`, an LRU of
  host-resident experts in a device bank for ubatches of up to 32 tokens; in
  `build_moe_ffn` a `GET_ROWS(slot_map, ids)` maps the ids to slots and
  `build_lora_mm_id` gains a `slots` argument; misses are uploaded
  synchronously in the scheduler's copy callback. #30112 splits it across
  GPUs. Same goal as 0021/0024, a different mechanism: no adapt policy (0055
  is LFRU, the default Strata's rule), no asynchronous swaps (0042), no bank
  the card reads in place, no CPU computation of misses (0056), no prefill
  bank stream (0032/0044). It needs I32 `GET_ROWS` on the device, which
  OpenCL gets from 0021, so on OpenCL it is off upstream. The port keeps
  ours and passes upstream's `slots` where the experts are the model's own
  tensors.
- **The qwen4exp indexer as one op** (889edf43d, #29825; `code`):
  `ggml_lightning_indexer` replaces the product, ReLU and head sum, with an
  F16 pool mask; the call is unconditional in qwen4exp, and OpenCL has no
  kernel, so it would run on the CPU. Upstream's `resolve_fused_ops` probe
  exists but is off for it (`auto_flid = false`). The port turns the probe
  on and keeps 0052's graph for a backend without the op.
- **The scheduler's input copies** (6753a033f, #29943; `code`): moved into
  `ggml_backend_sched_copy_input()`, host weights copied last. 0047's single
  wait for a split's user inputs moved into that function.
- **The OpenCL compile options** (f2cc7282c, #30266; `code`): split out of
  `load_cl_kernels` into `ggml_opencl_make_compile_opts()`, plus an
  Adreno/non-Intel decode route with a pooled partials buffer (`fa.fd_partial`).
  0007's Intel partials buffer stays beside it.
- **The batch API** (#29622, #30257, #30506; `code`): a token's embedding
  is checked against the model's input width; an MTP context's hidden state
  from the target is a separate state embedding
  (`llama_batch_ext_set_embd_state`). arcint's draft batch
  (`src/exec/llama_spec.cpp`) moved to it.
- **Reserve and realloc** (972b5fcc1 `sched_new`, cbe28bb3c `embd_all`, the
  no-realloc commits; `code`): context for 0052, resolved by hand.
- Smaller: Prism Hadamard loading (a9d27ac69) beside 0021/0024's cache
  setup; the context-state tests consolidated (a17e94d22) beside 0056's
  test; OpenCL `kernel_clamp_nc` and `kernel_cpy_f32_f32_pack` members beside
  0022's; f16 FILL test cases (OpenCL FILL is still f32-only upstream, so
  0022's f16 fill stays).
- **Intel upstream** (`code`): ggml-opencl's authors are Qualcomm's; Intel's
  only commits are older multi-device fixes (the design 0023 replaces).
  Intel works in SYCL (grouped MoE XMX GEMM, IQ3_S multi-column, a GDN gate
  fusion) and in the OpenVINO ggml backend. SYCL is not a route here (it
  aborts under the xe driver).

## Open upstream that meets the series (2026-10-15)

| PR / issue | what | to us |
|---|---|---|
| #29354 (draft) | ggml-opencl.cpp split into `ops/*.cpp` | conflicts with the 44 patches in that file; a move after it is a re-placement per op file, and the natural shape for upstreaming (`code`, the file list) |
| #30180 (draft) | Qualcomm's generic dense low-bit and i-quant matmuls | overlaps the dense parts of 0002, 0022, 0030; MUL_MAT_ID and the expert cache stay ours (`code`) |
| #30359 / #30360 (draft) | GDN snapshots written into the recurrent cache / input state read from it | #30359 is 0050's mechanism for the generic kernel; #30360 is a lever we do not have (`code` / `paper`) |
| #30345 | the MTP masked output rows | supersedes 0010 (`code`) |
| #26631 (approved) | MUL_MAT_ID id -1 skips the pair | the semantics 0021 and 0056 rely on, standardised (`code`) |
| #29510 | `flash_attn_ext_rows` for unified multi-sequence KV | the problem 0026/0029 solve, another way (`code`) |
| #30539 (draft) | OpenCL declines tensors above `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | could push large bank or expert tensors to the CPU on Intel; check before a move past it (`code`) |
| #30471 / #30541 | quantized KV `set_tensor` views; on-device KV copy sizes | bug fixes on paths arcint may use (`paper`) |
| #30181-#30184 (drafts) | Qualcomm dp4a MoE GEMMs, chunked GDN prefill, add+rms_norm fusion | textual conflicts with 0001/0006/0007/0015 likely; #30183 and #30184 are levers not in the series (`paper`) |
| #21067 | prefetching tensor overrides | the bank stream's idea (0032/0044), CUDA only (`paper`) |

Upstream has no open OpenCL issue for Intel Arc; the Intel Arc issues are
SYCL or Vulkan.

## The port (`measured-here`, 2026-10-15)

All 56 patches carried onto `3e4dfa8` one commit at a time: 46 applied
unchanged, 10 resolved by hand (0001, 0002, 0005, 0007, 0021, 0022, 0024,
0047, 0052, 0056; the resolutions are named above and in the commit
messages of the llama tree). The record of the move, its suites, KL and
served numbers is in `contrib/llama.cpp/README.md`, "Moving the pin".
