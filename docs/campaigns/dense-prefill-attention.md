# dense-prefill-attention — the dense agent's prompt attention at depth (B60)

**Open** (2026-10-13; operator: "Proceed with open topics from small to large",
then "review all kernels if we can pack sub 32 bit ops").

## Charter

The dense agent (Qwen3.8-27B Q4_K_M, libllama engine, q8_0:q4_0 KV, the B60)
prefills 787-837 t/s at 4k and 157 t/s at 237k (`agent-context.md`). Stock
vLLM on a B70 keeps ~1,170 t/s at 110k (`paper`, `research-b70-field.md`): the
gap grows with depth, so it is attention. Make the prompt attention at depth
faster, the answers unchanged.

## References to follow

- **llama.cpp's CUDA backend** (`code`, `ggml/src/ggml-cuda/fattn-common.cuh`,
  `launch_fattn` and `ggml_cuda_flash_attn_ext_get_f16_extra_data`): a
  quantized K/V is converted to f16 once a call (`need_f16_K/V`) and the f16
  MMA kernel runs on the copy. The copy is the whole K/V view, reserved in the
  compute buffer.
- **sycl-tla's FMHA** (`code`, `~/src/sycl-tla-ref` at `7b56b93`,
  `applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp`,
  `examples/06_bmg_flash_attention/06_xe_fmha_fwd.cpp`), what vLLM's XPU
  attention runs on Xe2:
  - K and V are loaded per sub-group straight into registers by 2D block
    loads, prefetched `Stages` tiles ahead, with no local memory and no
    transpose of V;
  - there is only a split barrier (arrive/wait) to keep the sub-groups
    together;
  - the causal mask is computed from row and column indices, and only on the
    diagonal tile;
  - each sub-group takes 16 query rows; head size 128 runs 256 rows a
    work-group with 32-key tiles.
- **llama.cpp's Vulkan backend** (`code`,
  `vulkan-shaders/flash_attn_mask_opt.comp`): a pre-pass classifies each mask
  block as all -inf (skip the tile), all zero (skip the mask reads) or mixed.
  It is ggml's own answer to the explicit mask.

## Where the time goes (`measured-here`, 2026-10-13, B60)

**Served profile** (the profiling build of the dev arcint, the agent's model
and flags, a 58,269-token prompt, one request, device time):

| KV, n_ctx | prefill | `flash_attn_f32_f16_dpas` | its share of device time |
|---|---|---|---|
| q8_0:q4_0, 131,072 (deployed) | 425.9 t/s | 75.2 s (38.8 ms a call) | 56 % |
| f16, 65,536 | 498.8 t/s | 56.3 s (29.1 ms a call) | 50 % |

The GEMMs take the same time in both arms (`kernel_mul_mm_kq2d_q4_K_i8` 26.3 s,
`q6_K` 7.5 s). An f16 arm at 131,072 does not fit: its 8 GiB of KV put 9.3 GB
in GTT, and that arm was discarded.

**Ablations** (test-backend-ops perf, head size 256, 24 query heads on 4, 512
rows, the kernel with one part removed at a time; results are wrong, only the
timing counts):

| variant | f16 at 65k keys | q8_0:q4_0 at 65k |
|---|---|---|
| the kernel | 13.1 TFLOPS | 7.4 |
| no mask reads | 10.1 (slower: not a clean ablation) | 6.9 |
| no V loads / transposed stores | 15.9 | 8.9 |
| no exp | 13.2 | 7.4 |
| no next-K-tile load | 14.7 | 9.1 |
| no inner barriers | 13.7 | 7.5 |
| all of these removed | 21.3 | 13.3 |

The card's fp16 DPAS peak is 98.3 TFLOPS (`llama-engine-kernel-gap.md`). The
kernel's core, with everything else removed, reaches about 22 % of it.

**ISA** (offline: `ocloc -device bmg`, the backend's build options, IGC dumps;
the main loop over one 32-key tile):

| build | instructions | DPAS | mov | byte-typed | word-typed |
|---|---|---|---|---|---|
| f16 | 1,366 | 64 | 223 | 4 | 144 |
| q8_0:q4_0 | 1,893 | 64 | 579 | 228 | 443 |

- **The kernel issues ~20 other instructions per DPAS.** That issue rate, not
  the matrix units, bounds it.
- **The quantized build adds 527 instructions a tile.** They are the
  byte → word → float → half unpacking of the dequantization. Every
  work-group repeats it, so each K/V value is dequantized once per query head
  and query block: 24 times for a 512-row chunk.
- **Not the cause: register prefetch.** A variant that stages the quantized
  tiles straight into local memory, with no register prefetch, measured 7.2
  against 7.4 TFLOPS.

## Stage 1: patch 0037, quantized K/V as f16 in key blocks

The CUDA mechanism, with one deviation, named: CUDA converts the whole K/V
view. At the agent's 245,760 tokens that is 960 MiB, against the unit's
238 MiB margin (`agent-context.md`). Here the conversion runs in blocks of
16,384 keys (two 32 MiB scratch buffers). The f16 kernel runs a block at a
time, its online-softmax state (O, m, l per lane, f32) carried between
blocks.

**Development, on the record:**
- **A slow conversion first:** reusing ggml's `kernel_dequant_*_f16_view_aos`
  for the conversion measured 2.8 TFLOPS. That kernel was not the cause:
  unitrace put the conversion at 0.17 ms a call.
- **Then a spill:** the first carry build spilled 1,408 bytes. Loading O
  element by element needed 128 temporaries. IGC then recompiled the kernel
  with rolled inner loops, 3 DPAS in the hot loop instead of 64, 4x slower.
  This was found in the offline ISA (dumps with `_1_` in the name).
- **The fix:** the carried state is stored in the registers' own layout,
  vector loads straight into the accumulators. That build has no spill, and
  its main loop is 1,387 instructions.

**test-backend-ops** (both cards):
- the FA cases at head size 256 pass 260/261, with the same three failures
  as before the patch;
- the same with 64-key blocks, so the carry runs: 260/261;
- the red case (`-DFA_CARRY_RED`, the carried sums dropped) fails 18
  quantized prompt cases;
- the A770 is unchanged (the carry kernel is Xe2 only).

**Perf** (B60, 512 rows; the test's quantized arms include a per-call
unflattening of the uploaded K/V, ~2.8 ms at 8k, that the served cache does
not have):

| shape | f16 | q8_0:q4_0 before | q8_0:q4_0 with 0037 |
|---|---|---|---|
| 8,192 keys | 8.3 ms | 14.6 ms | 12.3 ms |
| 65,536 keys | 63.0 ms | 111.9 ms | 92.5 ms |

**Served** (the 58,269-token profile above, q8_0:q4_0 at 131,072): prefill
425.9 -> 479.1 t/s (+12.5 %); attention 75.2 s -> 57.7 s plus 2.2 s of
conversion; the f16 arm 498.8.

**Passed** (the dev arcint with 0001-0037 on the B60, q8_0:q4_0, n_ctx
245,760, MTP 5):

| check | result |
|---|---|
| the capital | right |
| needle at 40 % of 231,786 tokens | right; prefill 192.0 t/s (the record: 157.2 at 236,938 with 0.7.3); peak 24,275 MiB, GTT 28 MiB (158 MiB to the idle card's free memory) |
| acceptance task, greedy | 10/10 |
| A/B at 132,874 tokens, one window | 249.0 -> 292.3 t/s prefill, the needle right in both; peak 24,144 -> 24,224 MiB; decode 16.2 in both |
| KL 16 x 512 | 0.005950 / 97.304 % off, on, and with 64-key blocks; the red case (64-key blocks, carried sums dropped) 4.4957 / 36.7 % |

**Review** (before the push, a second model): no defects. It traced the
carry layout against the host's sizing, padded rows, all-masked blocks, the
first block, the mask offset, the reconstructed-SoA buffers, the in-order
queue, and that the gating is no wider than the quantized kernels' (the
A770's sub-group-8 path untouched). Noted: `GGML_OPENCL_FA_DPAS_Q=0` turns
the path off too (in the README now); two comments describe half-open
ranges loosely (fixed with the next patch to the same files).

## Next

1. **The ISA audit** (operator, 2026-10-13), ranked by device time, the hot
   loops offline (B60 build):

   | kernel | prefill share (58k) | instructions | DPAS | mov | finding |
   |---|---|---|---|---|---|
   | FA DPAS, q8_0:q4_0 | 56 % | 1,893 | 64 | 579 | the dequantization's byte/word unpacking: **0037** |
   | FA DPAS, f16 | (after 0037) | 1,366 | 64 | 223 | see 2 |
   | Q4_K, int8 activations | 22 % | 839 | 36 (int8, k32) | 115 | half ops packed already (0013's vISA `mad16`) |
   | Q6_K | 6 % | 1,204 | 128 | 203 | half ops packed already: all 256 run 32-wide |
   | Q5_K | 2 % | 1,228 | 128 | 174 | the same |

   - **The FA f16 build's unpacked 16-bit work** (`code`, its ISA):
     - V's transpose through local memory: 48 word moves and 33
       word-to-dword widenings feed 32 scalar 16-bit stores a tile;
     - the explicit mask: 16 scalar half loads, 91 `cmp`, 62 `sel` a tile.
   - **The Q4_K kernel's 32 activation copies are not worth a window.**
     They serve the four DPAS of the first sub-block of a 256-block, a
     register move across the `kb` loop. A ping-pong buffer left the ISA
     unchanged (offline). The ceiling is 32 of 839 instructions in a kernel
     that is 22 % of prefill, under 1 %. Not pursued.
2. **The f16 core after vLLM's head-256 policy** (`code`, vllm-xpu-kernels
   `5210e3d`, `csrc/xpu/attn/fmha_utils.hpp:101` `chunk_policy_head256` and
   `:194` `_b16`, dispatched by `xe_2/fmha_xe2.cpp`):
   - **Same split as ours:** 8 query rows a sub-group with all 256 output
     columns.
   - **What differs:**
     - 32 sub-groups, so 256 rows a work-group (ours: 16 sub-groups, 128
       rows);
     - 32-key tiles;
     - K and V by 2D block loads straight into registers (sycl-tla's
       mainloop), with no local memory, no V transpose and no barrier but
       the split one.
   - **The mask:** the reference computes the causal mask by index. ggml
     passes an explicit mask, so the ggml-native form is Vulkan's
     `flash_attn_mask_opt` pre-pass: all-zero tiles skip the mask reads,
     all -inf tiles are skipped.
   - **The bound:** the ablations show the core without any K/V traffic at
     21.3 TFLOPS. The limit is instructions per DPAS, and the rewrite
     removes the local-memory traffic and most of the mask work.
   - **Two side effects of 0037:** its f16 copies are laid out by us, so
     they can be laid out for 2D block reads; and the A770's sub-group-8
     build has no carry yet (the coder also serves q8_0:q4_0).

## Where it lives

- `contrib/llama.cpp/patches/0037-opencl-fa-dpas-f16kv.patch`, its README
  section.
- `GGML_OPENCL_FA_DPAS_F16KV`, `GGML_OPENCL_FA_DPAS_F16KV_BLOCK`.
