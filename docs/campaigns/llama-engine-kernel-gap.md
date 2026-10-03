# llama-engine-kernel-gap — the libllama engine to the OpenVINO path's speed on the dense 27B and the coder

**Open.** Built and gated: lever 1 (gated delta-net, patch 0003), the cheap
decode items of lever 6 (0004), lever 5 (few-row F32, 0005), lever 3 (MoE
GEMM, 0006), lever 2's split half (decode attention, 0007), lever 4 (prompt
attention on XMX, 0008).
Results per patch: `contrib/llama.cpp/README.md`.

## Charter

On the same cards the OpenVINO path served the dense Qwen3.8-27B (B60) at
24.0 t/s decode and 1,141 t/s prefill at 4k, the Qwen3.6 coder (A770) at 43.9
and 1,379; the libllama engine with patches 0001/0002 serves them at 18.0 /
418 and 37.7 / 527 (`measured-here`; README, `docs/llama-engine.md`). Close
the gap kernel by kernel, then pass it.

## Where the time goes (`measured-here`, 2026-10-03, final 0002 build)

Device time per decoded token / per 512-token prefill (`GGML_OPENCL_PROFILING`):
- dense decode 54.5 ms: K-quant matvecs 44.2 (Q4_K 28.7, Q6_K 11.6 incl. the
  lm_head, Q5_K 3.9); ~2,150 launches; decode attention 2.0 at 256 keys;
- dense prefill 1,233 ms: our GEMMs 574, gated delta-net 272, flash attention
  186, the F32 alpha/beta projections 92, activation conversion 26;
- coder decode 23.0 ms: decode attention 3.8, matvecs 3.3 + experts 3.0,
  activation quantize 2.8 (271 launches);
- coder prefill 838 ms: gated delta-net 360, MoE GEMM 234, flash attention 97.

## Levers, ranked by expected wall-clock gain

Five reviews (2026-10-03; each read our kernels, ggml's, the profiles, and
the OpenVINO plugin's counterpart kernels and arcint's plugin patches) give
these, all `estimate` unless marked. The decode-side small-op and launch
items of the fifth review are listed separately below.

1. **Gated delta-net, OpenVINO's shape** (prefill −250 ms dense, −345 ms
   coder). ggml's `gated_delta_net.cl` keeps 96 floats per SIMD32 lane
   (state spilled every token), reduces through SLM with 7 barriers, runs
   0.5 threads per EU on the A770 (`code`). OpenVINO's
   `paged_gated_delta_net_opt.cl`: a sub-group per (head, 4 state rows),
   lanes along k, the state in registers (`float8`), q/k by block reads,
   l2-norm fused, shuffles for reductions, the state written at the end
   (`code`); it was 1.5 % of coder prefill at 4k on the A770 (`measured-here`,
   design record). Add Strata's next-token prefetch
   (`~/src/Strata-ref/src/prefill/kernels.cu:380-423`, `code`).
   **Built** (`gated_delta_net_intel.cl`, patch 0003): the state in
   registers, lanes along k; for a prompt a work-group of NSG sub-groups per
   (head, NSG x R rows) with q/k/v/g/beta staged in local memory in chunks
   of TC tokens, the next chunk loaded while the current one is computed;
   for one token one sub-group per R rows with the next token prefetched.
   Kernel time at 512 tokens (`measured-here`, harness against a CPU port
   of ggml-cpu's op, errors ~5e-6 of RMS): A770 (32 heads, SG 8, R 2 x 64
   sub-groups, 256 GRF) 12,042 -> 451 us; B60 (48 heads, SG 16, R 4 x 32,
   TC 8, 128 GRF) 5,692 -> 875 us; one token 82 -> 12.7 us (A770), 51 ->
   5.5 us (B60). The first build, one sub-group per work-group (OpenVINO's
   shape), spilled at 128 GRF and ran 5,225 us on the A770 (1,230 at 256
   GRF); a butterfly reduction of the R rows (R - 1 + log2(SG/R) shuffles
   instead of R log2 SG) measured slower on both cards (B60 875 -> 1,352
   us), so the reductions are not the limit. llama-bench (`measured-here`,
   pp512 / tg128, 3 reps): coder (A770) 600.3 -> 1,005.6 / 37.4 -> 38.8
   t/s; dense (B60) 409.2 -> 493.8 / 17.97 -> 18.20; the build with
   `GGML_OPENCL_GDN_INTEL=0` reproduces the baseline (601.3, 409.9).
2. **Decode attention: GQA-coalesced flash decoding** (dense 2.0 → ~0.3 ms at
   256 keys, ~250 → ~6.5 ms per token at 32k; coder 3.8 → ~0.2). ggml forces
   Intel onto one 32-lane thread per q head that re-reads K/V per head and
   spills (`code`, `flash_attn_f32_f16.cl:667-825`). Reference:
   `paged_attention_opt.cl` (KV partitions, HEADS_PER_WI q heads per K/V
   read, a finalization pass) and arcint patch 0015.
   **Built** (patch 0007): the KV split with lanes along the head dimension
   and ggml's merge; the coder's decode 43.4 / 13.6 / 4.2 -> 47.8 / 47.3 /
   43.5 t/s at 0 / 4,096 / 16,384 tokens of context, the dense model's 19.2 /
   12.8 / 6.4 -> 19.75 / 19.39 / 17.95 (`measured-here`). The query heads of
   a KV head still read K/V each: at 16,384 keys the kernel moves ~12x the
   K/V bytes (270-400 us against a ~67 us bandwidth bound). A form with a KV
   head's query heads in one work-group and 16-key K/V tiles staged in local
   memory measured slower on both cards (A770 395 us against 96 at 4,096
   keys) -- a verdict on that build (a barrier per 16 keys, six sub-groups),
   not on GQA sharing; OpenVINO's `paged_attention_opt.cl` shares the K/V
   reads in registers (HEADS_PER_WI), which is the next form to build.
3. **MoE GEMM with tokens as the SLM operand** (coder −150..−190 ms). Ours
   runs one 8-lane sub-group per work-group, 4 % of peak (`measured-here`);
   arcint's patch 0064 (`code`) gathers the pairs into SLM, builds each lane's
   weight row in registers, fuses gate and up, 8 sub-groups per work-group:
   653.7 → 952.1 t/s on this model and card (`measured-here`, DESIGN 7.0.2co).
   Plus: the route computed once per layer, not per matmul.
   **Built** (patch 0006, `kernel_mul_mm_id2_*`): 0064's layout -- a tile of
   up to XT pairs of one expert, XSG sub-groups of weight rows, the pairs'
   activations staged per super-block in the DPAS a layout, each lane's row
   as the b operand in registers. Measured on the coder's Q4_K gate/up
   (A770, 512-token prefill, 100 calls; `measured-here`): the old kernel
   193 ms; 0064's layout with each lane dequantizing to f16 175 ms (16 x 8),
   143 ms (32 x 8); with the dequantization replaced by a constant 77 ms, so
   it was ALU-bound. The ISA dump (IGC, `ShaderDumpEnable`) showed why a
   first "cheap" form ran slower (175 ms): IGC does not pack `half2`
   arithmetic in SIMD8 (each half op two instructions plus moves), and
   per-super-block scales held in an array indexed by a runtime step went to
   private memory. The form kept: the codes as halves 1024 + q built by
   integer operations on whole words (two halves an instruction; the
   activations staged with the middle two of every four halves swapped to
   match the byte pairs), one DPAS chain per 32-weight sub-block, applied in
   f32 as D sc_j (chain - 1024 S_j) - M m_j S_j with the sub-block sums S_j
   staged with the activations: 117 ms at 32 x 16. The coder's prefill
   (llama-bench, A770): 1,032 -> 1,231 t/s at 512 tokens, 747 -> 847 at
   2,048. Gate and up are not fused yet, and the route is still computed per
   matmul.
4. **Prefill flash attention** (dense −130 ms, coder −50 ms; more at depth).
   First the stock split kernel unlocked for Intel (one gate,
   `ggml-opencl.cpp` `use_split_kernel`), then a DPAS kernel after
   `sdpa_micro.cl` (Q·K and P·V on XMX, f32 softmax).
   **Done (0008)**, the DPAS kernel directly (upstream gates the split
   kernel off where its shuffle reduction depends on the sub-group size,
   `use_split_kernel`, `code`; not tried here): 512 x 4,096 x 24/4 at
   test-backend-ops B60 149.9 -> 8.2 ms, A770 167.0 -> 22.6 ms; prefill
   (0008 before its review, which made the kernel 5-7 % faster) dense 525
   -> 633 t/s at 512, 145 -> 540 at 512 after 4,096; coder 1,285 -> 1,608,
   602 -> 1,322 at 4,096 (`measured-here`). KL coder 0.007059 -> 0.007043,
   dense unchanged.
5. **F32 small projections** (dense −84 ms prefill, −1.3 ms decode; coder
   −33 ms): split-K for ≤ 64 rows, the multi-column F32 gemv for 1-8 rows;
   alpha|beta as one weight.
6. **Decode matvec occupancy and launches** (dense −7..−8 ms per token,
   coder −8 ms): K split inside the work-group for N ≤ 8192 (OpenVINO
   `fully_connected_gpu_gemv.cl:84-96, 222-227`, `code`); the activation
   quantize fused into rms_norm_mul / swiglu; the dedupe key without the
   tensor pointer (the coder quantizes the MoE input twice, `code`); a fused
   gate+up+SwiGLU(+q8_1) kernel for experts and dense FFN (OpenVINO
   `moe_3gemm_swiglu_mlp.cl:449-644`, `code`). The matvecs themselves already
   reach 436 GB/s on the lm_head (`measured-here`): the dense decode gap is
   also bytes (the OpenVINO int4 export ~18 % smaller than Q4_K_M).
7. **Dense GEMM** (dense −100..−200 ms): the activation operand by 2D block
   reads (B60) / a VNNI-tiled layout (A770), the dequant overlapped with the
   DPAS loop, narrow-tile variants. Already within 15-30 % of arcint's best
   XMX K-quant kernel on the B60 (patch 0029, 59 TFLOPS, `measured-here`).

   **Diagnosis, 2026-10-03** (B60, all `measured-here` unless marked).
   The lever is bigger than this entry first estimated.

   *Size of the prize.* On a 4,096-token dense prefill (7.06 s,
   ggml-opencl profiling build) the K-quant GEMMs (`mul_mm_kq_f16.cl`) are
   73 % of device time: Q4_K 53.8 %, Q6_K 15.2 %, Q5_K 4.4 %. The rest is
   prompt attention 8.3 %, gated delta-net 4.9 % and the f32 -> f16
   activation conversion 2.9 %. The FFN down projection alone is 25 %. The
   OpenVINO path takes 3.59 s for the same prefill, and our non-GEMM work is
   already 1.86 s, so matching it needs the GEMM about 3x faster.

   *What the OpenVINO path runs.* `ONEDNN_VERBOSE` on the deployed binary
   (`arcint 0.5.4-1`, `marfrit-openvino +p25`): every dense projection is a
   oneDNN `jit:gemm:any` matmul. The activations are `s8`, quantized per 64
   at run time; the weights are `u4` with f16 scales and u8 zero points per
   64. The inner loop is int8 x u4, on DPAS at twice fp16's rate.

   *Where our GEMM's time goes.* Q4_K, 4,096 x 512 x 14,336:

   | arm | time | TFLOPS |
   |---|---|---|
   | as is | 1,722 us | 34.9 |
   | the B operand a constant | 1,212 us | 49.6 |
   | no dequantization | 1,380 us | 43.6 |
   | both | ~955 us | 63 |

   unitrace `VectorEngineStalls` over the kernel: the vector engines are
   active 29 % of the time with every thread slot occupied. Stall reasons:
   scoreboard waits 52 %, barriers 6 %, instruction fetch 4 %. No spills
   (ISA).

   *Negatives, each with the mechanism it tested:*
   - B tiled so that each B operand is one sub-group block read, in place of
     16-address gathers: 1,722 -> 1,703 us. The gather shape was not the
     cost.
   - Taller tiles, for less B traffic from L2: worse at every shape tried,
     up to 2.2x.
   - B loads pipelined one k-step ahead in registers: 1,692 -> 2,325 us.
   - k-tiles of 64 with A and B in local memory, double-buffered (MM2;
     first written so a thread awaited its own staging loads, then
     register-staged so the loads overlap the DPAS): at best 2,527 us. The
     counters show why. At 80 KB a work-group, occupancy is 25 %. At
     4,2,4,4 the shared functions held requests 41 % of the time: B moved
     through local memory just moves the queue to the local-memory port.

   *What follows.* The kernel's whole data path caps it near 35 % of the
   card's fp16 peak: B from global per sub-group, A dequantized into local
   memory, fp16 DPAS, small register tiles. The references (`code` where
   read: the oneDNN trace above; Intel's XeTLA / sycl-tla Xe2 GEMMs, not
   yet read here) load both operands with the hardware 2D block loads
   (`cl_intel_subgroup_2d_block_io`, present on the B60), prefetch k-tiles
   into L1, keep larger register tiles, and multiply in int8 where the
   format allows. Plan, in order:
   1. A B60 GEMM on 2D block loads with L1 prefetch, against the 63 TFLOPS
      ceiling above.
   2. int8 activations with int8 DPAS. An earlier int8 GEMM moved dense KL
      by 0.0028 nats and top-1 by 0.9 points, inside the bar.
   3. The activation conversion (2.9 %).

   The few-token verify (0009) is unaffected: it is a different kernel.

   **Step 1 done (0011):** patch 0029's 2D-block kernel, ported onto the
   flat planes.
   - Q4_K: 1,722 -> 1,262 us at the shape above, the activation conversion
     included.
   - Q5_K: +14 %. Q6_K: +12 %.
   - Dense prefill 594 -> 702 t/s at 4,096 tokens, 632 -> 758 at 512.
   - KL unchanged.

   The 4,096-token dense prefill after 0011 (profiling build, 701.7 t/s):
   - device time 7.06 -> 5.81 s;
   - the GEMMs 68 % of it: Q4_K 2.90 s, Q6_K 0.81 s, Q5_K 0.25 s;
   - prompt attention 10 %, gated delta-net 6 %, the conversion 3.5 %.

   Matching the OpenVINO path's 3.59 s still needs the GEMMs about 2.5x
   faster.

   unitrace on the 2D kernel: the vector engines are active 59.5 % of the
   time (the old kernel 29 %), scoreboard waits 33 %.

   *int8, first attempt (not shipped):*
   - the activations quantized per (token, 32), Q4_K codes as the u8 b
     operand, one i8 x u8 DPAS a sub-block;
   - correct (MUL_MAT 1,123/1,123) but 2.2x slower: 2,747 us against
     1,247;
   - unitrace: active 37.8 %, scoreboard waits 65 %. With the float decode
     gone, nothing hides the loads, and it issues 133 load messages a
     super-block, 16 of them a sub-block's scales and sums;
   - needs those loads merged and the next sub-block's activations read
     ahead. A measured shape, not a verdict on int8.

   A single-pass f32 -> f16 conversion measured 0.5 % at the kernel and was
   left out.

   *oneDNN's kernel read at ISA.* `ONEDNN_JIT_DUMP` on the OpenVINO path,
   disassembled with IGA built from IGC's `visa/iga`, compared with IGC's
   dump of 0011 by a reviewer (`code`). The three main-kernel variants
   share one loop:
   - 64 tokens x 16 rows a thread, int8 DPAS k32, 128 k a loop body;
   - both operands by 2D block loads, double-buffered one k-step ahead;
   - a cooperative 2D-block L1 prefetch of the activation tile, split
     across the 16 threads, about one super-block ahead;
   - u4 -> s8 decode at 6 ALU per DPAS;
   - DPAS in 8-long chains with independent int32 partials;
   - a per-64 rescale epilogue at 12 ALU per DPAS.

   So oneDNN also pays an O(tokens) epilogue per weight group. The argument
   above that int8 cannot pay on Q4_K's per-32 scales does not hold as
   stated: the int8 attempts lost to latency and to DPAS chains read back at
   once, not to the epilogue count.

   Ours by comparison:
   - 11.9 ALU a DPAS (the float decode, about 5.5 ops per 16 weights);
   - weight reads consumed 21-76 instructions after issue;
   - DPAS chains broken by adjacent dependent pairs.

   **0012**, from that reading: the b operand outer in each sub-block, and
   the token tiles as the fast grid dimension. Dense prefill 703 -> 816 t/s
   at 4,096 tokens, 758 -> 891 at 512, KL unchanged.

   **0013 (2026-10-03)**, all `measured-here` on the B60 unless marked.
   Dense prefill 816 -> 931 t/s at 4,096 tokens and 892 -> 1,030 at 512
   (llama-bench). The steps and their numbers are in
   `contrib/llama.cpp/README.md`, 0013. What the measurements showed:

   - *The card's matrix rate is reachable.* A DPAS microbenchmark with
     operands in registers and 0012's chain shape (8 accumulators, b0 pass
     then b1 pass) runs at 98.3 TFLOPS fp16 and 196.6 TOPS int8, the
     rated peaks, at 128 or 256 GRF and with 4, 8 or 16 accumulators. The
     chain shape was never the limit.
   - *The fp16 kernel at 4,096 x 512 x 14,336 is bound by its activation
     reads.* Ablations of the Q4_K kernel (device time, avg us): real
     1,066; without the decode 1,125; without the A reads 808; without
     either 700. 700 us is the DPAS floor with 3.2 waves rounded to 4.
     The 16 sub-groups of a work-group re-read the same 64 tokens, about
     70 B/clk a core through L1.
   - *More rows a sub-group does not pay.* Two 16-row tiles a sub-group
     halve the A bytes a DPAS. That needs 128 GRF of accumulators: each
     part alone is cheap (no decode 840 us, no A 814) but together 1,285,
     the reads issued late. Slower at the endpoint at every shape tried.
     Prefetch and register double-buffers of A did not help either (README
     0013).
   - *The test shape misled.* At the model's own shapes (K = 5,120) the
     fp16 kernel already ran at ~77 TFLOPS, ~80 % of the floor. The long-K
     down projection (K = 17,408) was the weak one, through a falling L1
     hit rate: a barrier every 16 blocks fixed most of it (KSYNC).
   - *"One instruction, two operations"* (the operator's question): oneDNN
     does SIMD32 byte and half operations on packed registers, two or four
     values a lane an instruction. OpenCL C has no way to ask IGC for that:
     a per-lane `half2` is kept as two planar SIMD16 halves. IGC's inline
     vISA can (`__asm__`, `.decl ... alias=<%n, 0>` to retype a register,
     `(M1_NM, 32)` because a SIMD16 thread's dispatch mask covers only 16
     channels). The packed decode and the int8 kernel's word-source `mad`
     are written that way.
   - *int8 needed a different scale path than oneDNN's.* oneDNN's u4 has
     scales per 64 and rescales in float after each group. Q4_K's are per
     32. The per-32 float rescale (v2-v4 attempts, 1,599-2,140 us) and a
     split-scale form (two DPAS a sub-block) lost or tied. What worked:
     codes as s8 (code - 8), so each sub-block's product fits 16 bits and
     its scale is one integer `mad` on word sources. 9-15 % faster than
     the fp16 kernel at every model shape. KL +0.0005 nats.

   Open:
   - the int8 kernel is bound by load latency, at ~44 % of the int8 rate
     at gate/up. Without its sub-block read-ahead it is 19-30 % slower; L1
     prefetch does not help, and 8 threads an XVE (128 GRF) does not fit;
   - Q6_K (658 of 2,968 ms of K-quant GEMM device time in a 4,096-token
     prefill, before KSYNC) has signed 8-bit scales per 16: no int8 form
     found;
   - Q5_K on int8 (the fifth bit makes codes 0..31: s8 code - 16 fits the
     same 16-bit bound at activations of 7 bits only).
8. **GDN conv chain and decode fusions** (dense −20 ms prefill, −4.5 ms per
   token; coder −11 / −2.5): concat + ssm_conv + silu + state copy as one
   kernel after `paged_causal_conv1d_ref.cl`; l2-norm into the GDN kernel;
   the gate activations into the alpha/beta matvec epilogue.

### Decode: small ops and launches (fifth review)

Per decoded token (`measured-here`, the review's trace of the final 0002
build): dense 2,148 launches, coder 2,034 (the OpenVINO path's coder 1,171).
Device time outside matmuls and attention: dense 8.3 of 54.5 ms, coder 11.0
of 22.9 ms. The per-launch floor: A770 2.2-2.6 us, B60 0.6-0.9 us. In-order
gaps with a deep queue: median 1.0 us, 0.57 ms per dense token. The dense
model is device-bound.

A. **Coder: the GDN gate projections on the card.** The trace shows 60
   gaps per token of 250-490 us, always where `ssm_alpha` / `ssm_beta`
   (Q4_K [2048, 32] in the coder GGUF; F32 in the dense one) would launch.
   No OpenCL kernel runs for them: a scheduler split, drained 60 times a
   token, ~3.5 ms of 26.5 (`measured-here`). Why they land on the CPU after
   the supports_op fix of patch 0001 is not settled (`code` says they
   should pass); verify the placement first. Then one kernel for alpha mm
   -> add -> softplus -> mul and beta mm -> sigmoid (a two-output
   `ggml_can_fuse_subgraph` pattern).
B. **Activation quantize**: a sub-group-cooperative kernel (one work-item
   per 32-block today: 7.4 us on the B60, 10 on the A770), then emitted by
   `rms_norm_mul` / `swiglu` / `add_row`, and the dedupe keyed on (buffer,
   offset, size) rather than the tensor (the coder's MoE input is quantized
   twice, `code` `ggml_cl_kq_cached`). Dense ~1.9 ms, coder ~2.7 ms.
C. **Coder routing chain in two kernels**: a K-split logits matvec and one
   softmax + top-k + normalise kernel (OpenVINO `moe_router_fused.cl`
   `softmax_topk`, `code`) replacing 7 launches (52 us) a layer: ~1.7 ms.
D. **Coder shared-expert tail into `moe_combine`** (sigmoid, mul, two
   add_rows): ~1.2 ms; `kernel_mul` for one row runs one work-group of 64
   (`code`).
E. **swiglu dispatch** (one work-group for 17,408 elements, 15 us, `code`
   `ggml_cl_glu`), then gate/up/GLU fused into the matvec epilogue (the
   Adreno-only arm of the fusion matcher, OpenVINO
   `fully_connected_gpu_bf_tiled.cl` `SWIGLU_LENGTH`): dense ~1.1 ms.
F. **Recurrent state in place** (a llama.cpp graph change): the state read
   and written by the GDN and conv kernels instead of get_rows + copy
   (OpenVINO's paged kernels, `code`): dense ~1.3 ms, coder ~1.1 ms.
G-I. residual add + norm as one kernel; attention-layer gate and q/k
   norm + rope fusions; the q/k l2-norm into the conv kernel: ~0.8 ms
   dense, ~0.8 ms coder together.

Estimated sum: dense ~6.5-7 ms of 55.6 (~20.5 t/s), coder ~9-10 ms of
26.5. Not worth it on NEO 26.27 (no `cl_khr_command_buffer`): out-of-order
queues (event bookkeeping costs about what the ~1 us gaps do) and
multi-threaded enqueue. A hazard: with two OpenCL devices registered every
node runs `sync_with_other_backends`; arcint sets `GGML_OPENCL_DEVICE`, which
keeps one.

### What the outside references add (web and source research, 2026-10-03)

A research pass over Intel's stacks (the agent's reading; `code` where it
quoted source, not yet read here):
- oneDNN converts int4 to half without a convert instruction (u4 as a
  denormal half, two multiplies by 2^12;
  `src/gpu/intel/gemm/jit/generator/pieces/quantization.cxx:543-612`): the
  same idea as 0006's 0x6400 | q codes.
- OpenVINO's `fully_connected_gpu_bf_tiled.cl` dynamic-quantization path and
  oneDNN's `grouped_micro_gemm.cpp:221-249` keep the inner loop in int8
  (activations quantized per group at run time): int8 DPAS on Xe2, f16
  dequantization on Xe-HPG. A card-specific split worth a B60 arm; our int8
  GEMM of 0001's development moved the dense KL by 0.0028 nats and top-1 by
  0.9 points (`measured-here`), inside the answer-level bar, and was dropped
  for speed, not quality.
- 2D block loads with a prefetch three k-tiles ahead (vLLM's Xe2 grouped GEMM
  policy, `gemm_xe2.hpp:157-194`; Intel's SimpleOpenCLSamples
  `20_matrixexperiments-bf16`): for the B60's GEMMs (lever 7).
- `sdpa_micro.cl` (OpenVINO) and sycl-tla's FA2 (`xe_fmha_fwd_mainloop.hpp`)
  are the structure for lever 4.
- Launch cost (`docs`, compute-runtime FAQ): out-of-order queues overlap only
  independent kernels, and immediate command lists are Level Zero only; the
  decode chain is dependent, so fusion (ggml's `ggml_can_fuse_subgraph`
  matcher) is the lever, as the fifth review said.
- Discarded: its top finding, an "upstream" Intel DPAS MoE kernel, was our own
  patch 0001 in the tree it read.

## Gate

Per lever, against the build before it on the same card:
- the answer-level bar (`CLAUDE.md`): KL against the CPU-backend reference
  (16 x 512 tokens) within 0.03 nats of the baseline arm, top-1 down at most
  1 point, prefill and decode (`-ub 1`) modes; the acceptance task 10/10 at
  temperature 0 on the coder, the dense model's sampled pass rate not
  separable from the baseline (20 runs);
- `test-backend-ops` for the op on both cards, a mutant failing it;
- per phase: the target phase faster, the other within the spread.

## Where it lives

`contrib/llama.cpp/patches/` (one patch per lever); reviews and profiles in
the session record of 2026-10-03.
