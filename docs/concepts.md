# arcint concept catalogue

This is a catalogue of the engine concepts arcint implements for inference on Intel Arc GPUs, grouped by mechanism. Each entry says what the idea is in terms that can be applied to another codebase, what it gained on which card, where the code is, which switch controls it, and where the idea came from. A section at the end lists what was tried and not adopted, with the numbers and the reason.

**What arcint is.** arcint is an inference server (OpenAI-compatible HTTP, chat templates, tool calls, speculative decoding, context checkpoints) that runs on llama.cpp's OpenCL backend, with Intel-specific kernels and graph changes carried as a patch series. It serves, on two cards: the Arc A770 (16 GB, Alchemist, Xe-HPG, sub-group size 8, no 2D block I/O, behind the chipset's PCIe 3.0 x4 link) and the Arc Pro B60 (24 GB, Battlemage, Xe2, sub-group size 16, 2D block I/O). The language models are a dense Qwen3.8-27B (the agent, on the B60), the MoE Qwen3.6-35B-A3B coder (on the A770) and Qwen3.8-Flash-Next (a 125B-A6B MoE with 512 experts per layer, on the B60 with an expert cache over host memory). The B60 also runs an image service (stable-diffusion.cpp on the same patched ggml tree), speech and vision. The architecture, the rules and the state of play are in `DESIGN.md`; every environment switch is classified in `docs/switches.md`; the per-release changes are in `CHANGELOG.md`.

**Where the code lives.** arcint's own code (server, sampler, drafter, gates) is in `src/`. The engine changes are a patch series over a pinned llama.cpp commit, `3e4dfa8`, in `contrib/llama.cpp/patches/`; the pinned llama.cpp source is not in this repository. `contrib/llama.cpp/README.md` explains how to apply the series (`git -C <llama.cpp> checkout 3e4dfa8`, then `git -C <llama.cpp> apply <arcint>/contrib/llama.cpp/patches/*.patch`) and has one section per patch with its measurements, red cases and reviews; this catalogue is a reading of it by mechanism. Patch numbers are labels, shown in parentheses after the location, and will change when the series is folded; the file and the kernel or function name are what to search for. Every symbol named below was found in the patches or in `src/`. The branch `flattened` of this repository ([github.com/marfrit/arcint/tree/flattened](https://github.com/marfrit/arcint/tree/flattened)) holds the pin with the released series applied, so a symbol can be read there as a file instead of as a diff; it carries released series only, so the newest patches may be missing from it. References to other projects' files name the file and the symbol and not a line number, because the cited trees move.

**How to read an entry.**
- *Effect* gives the measured number with its card, model, depth and KV configuration as the source states them, with an evidence class: `measured-here` (run on this project's cards), `code` (read in source, not run) or `paper` (a published or third-party figure, not run here). Where the source has no measurement the entry says so. A pair written "A -> B" comes from one comparison in one window (the same binary, card and prompt) unless the entry says otherwise; a figure from another window or another engine is marked as such in the entry, and two figures that look comparable are comparable only if the entry says so. Kernel times are device times from test-backend-ops or unitrace; "prefill" and "decode" are llama-bench `pp` and `tg` or served rates as stated; t/s is tokens per second.
- *Quality* is judged at the answer. The bar the project uses is: the answers stay right (facts, a long-context needle, the acceptance task, which is a Lua CSV parser to RFC 4180 scored by executing the candidate code, 10 cases); the candidate's mean KL against the reference is at most 0.03 nats worse than the baseline arm's, on the same card and window; argmax (top-1) agreement drops by at most 1 point. KL is measured over 16 chunks of 512 tokens against the model's own CPU or Q8_0 reference unless the entry says otherwise. A change that reorders sums can leave KL unchanged and still flip a greedy answer, so some entries record an md5 of the served text as a tripwire.
- *Location* names the file and the kernel or function first, then the patch number. *Switch* is the environment variable or flag, if one exists. *Source* is the reference the idea came from when the record says: Strata, FreeToken, Infernix, IPEX-LLM, vLLM's XPU kernels, llama.cpp's CUDA and Vulkan backends, HyperQwen, a fork of ggml for Arc, or arcint's own.
- "The unit" or "the unit's flags" is one served configuration, a service and its command-line flags and environment as shipped in [`contrib/systemd/arcint-*.service`](../contrib/systemd/) (agent, coder, Flash-Next, writer and the others); a measurement "with the unit's flags" ran those flags. A "real-time factor" (RTF) is seconds of audio produced divided by seconds of process time, so higher is better and 1.0 is real time; "the operator" is the project owner, whose decisions are dated where the source dates them. A "window" is one measurement session on the cards; numbers from a window are only compared with numbers from the same window. Each entry carries two fields beside the effect: *Applies to* (which card generation or sub-group size, which weight types and shape limits) and *Default* (on, opt-in or off, and what sets it).
- A campaign document is cited as "(on the `qfndev` branch)"; each entry is written to be complete without following it.

**Contents.** 1 Integer DPAS and dot-product kernels; 2 Low-bit and codebook weight types; 3 Expert routing and expert GEMM tiles; 4 Expert cache and bank; 5 Prefill: bank streaming, dispatch by token count, memory lending; 6 Attention at depth; 7 GatedDeltaNet and recurrent state; 8 Graph launches, placement and transfers; 9 Speculative decoding; 10 Lanes and windows in one KV pool; 11 Serving and device hygiene; 12 Image and speech; 13 Weight-shared and looped architectures; then the index of references and the section on what was tried and not adopted.

## 1. Integer DPAS and dot-product kernels for quantized weights

Matrix-vector and matrix-matrix products on the XMX (DPAS) units and with 4x8-bit integer dot products, for K-quant, Q4_0, Q8_0 and f16 weights, on both cards.

### Integer-dot matrix-vector products on the quantized weights

Quantize the activation once per 32 values to two int8 terms (the codes, and the codes of what they leave), then multiply the packed weights with 4x8-bit integer dot products, as llama.cpp's CUDA MMVQ does. One to four columns are taken per pass, which is the shape of a speculative verify of a few tokens (the cost of the extra columns is measured below). For `MUL_MAT_ID` one (token, slot) pair is one work-group and the expert is read on the device. The same patch lets K-quant `MUL_MAT_ID` report itself as supported on Intel, so the experts of a K-quant MoE load into VRAM instead of running on the CPU, and exempts K-quant `MUL_MAT` from the stock Adreno rule that declines 512 or more columns (llama.cpp asks `supports_op` with a 512-token op when it places weights, so that rule had put every small K-quant tensor on the CPU).
- **Effect:** `measured-here`, A770 and B60, llama-bench `-p 512 -n 128`, one sequence. Qwen3.8-27B Q4_K_M on the B60: prefill 71.1 -> 417.9 t/s, decode 10.3 -> 18.0 t/s. Qwen3.6-35B-A3B coder Q4_K_M on the A770: prefill 111.2 -> 605.0 t/s, decode 7.7 -> 37.7 t/s, where the stock number has the experts on the CPU. Of the coder's gain, 527 -> 605 t/s prefill and 20.9 -> 37.7 t/s decode is the placement exemption alone, measured with the kernels unchanged. Kernel alone, Q4_K 17,408 x 5,120, one column: 84 -> 351 GB/s on the A770 and 163 -> about 400 GB/s on the B60. The patch's own text later corrects its "2 to 4 columns cost about one": the kernel is compute-bound from 2 columns (Q4_K 4,096 x 14,336 on the B60: 76 / 123 / 169 us at 1 / 2 / 4 columns). KL against the CPU backend, 16 x 512 tokens, ggml's float kernels as the baseline arm: dense 0.003560 -> 0.003560 nats, coder 0.00728 -> 0.00696.
- **Applies to:** Both cards (needs `cl_khr_integer_dot_product`); Q4_K, Q5_K, Q6_K weights, 1 to 4 columns, `MUL_MAT` and `MUL_MAT_ID`; the same machinery later takes the IQ types and Q4_0.
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mv_kq_q8_1.cl`, `kernel_mul_mv_q4_K_q8_1`, `kernel_mul_mv_id_q4_K_q8_1` (Q5_K and Q6_K likewise), activation `kernel_quantize_q8_1`. (0001)
- **Switch:** `GGML_OPENCL_KQ_Q8_1=0` (all of 0001), `GGML_OPENCL_KQ_MV=0` (the `MUL_MAT` matvec), `GGML_OPENCL_KQ_DEDUP=0`.
- **Source:** llama.cpp's CUDA `vec_dot_q4_K_q8_1` and its siblings (`code`).

### fp16 DPAS GEMM with the weights dequantized into a local tile

Dequantize the weight tile to f16 in local memory in a layout where one sub-group block read is one DPAS operand, and run fp16 DPAS on the XMX units with f32 accumulation. The activations are f16 columns with a power-of-two scale per column so that they stay inside f16's range. `MUL_MAT_ID` sorts the (token, slot) pairs by expert on the device and runs one tile of one expert per work-group.
- **Effect:** `measured-here`, GEMM at 512 columns 34 TOPS (A770) and 41-43 TOPS (B60), against about 4 TFLOPS for ggml's kernel (standalone harness). Included in the prefill numbers of the previous entry. An int8-activation DPAS GEMM in the shape of CUDA's MMQ was built first and dropped (see "tried and not adopted").
- **Applies to:** Both cards (needs `cl_intel_subgroup_matrix_multiply_accumulate`; sub-group 8 on Xe-HPG, 16 on Xe2); Q4_K, Q5_K, Q6_K and later the IQ types; `MUL_MAT_ID` from 16 tokens.
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_f16.cl`, `kernel_mul_mm_q4_K_f16`, `kernel_mul_mm_id_q4_K_f16`, route `kernel_moe_route_tiles`. (0001)
- **Switch:** `GGML_OPENCL_KQ_MM=0`, `GGML_OPENCL_KQ_MM_ID_MIN_TOK` (default 16), `GGML_OPENCL_KQ_MM_TILE`, `GGML_OPENCL_KQ_MM_ID_TILE`.
- **Later change on the A770:** the prompt's expert products for IQ4_XS, IQ4_NL and Q4_K (K a multiple of 256, rows a multiple of 8) now take the int8 kernel of 0103/0104 ahead of this tile; the expert cache's slots and bank still decline to the tile.
- **Source:** DPAS operand layouts from the emulation in bashbaug/SimpleOpenCLSamples `20_matrixexperiments` (`code`).

### K-quant GEMM with 2D block loads on both operands (Xe2)

Read the weights with transposed 32-bit 2D block loads (32 bytes of each of the sub-group's sixteen rows), decode them in registers to the fp16 DPAS b operand, and read the tokens (the a operand) as 32 rows x 16 halves straight from the f16 activations. A work-group's 16 sub-groups share the activation tile through L1, with no local memory and no barrier; weights are decoded once per 64 tokens. Two ordering changes found by disassembling oneDNN's JIT GEMM beside IGC's output for this kernel: the b operand outer loop, so that a group's two dependent DPAS are 8 apart and IGC no longer drains the matrix unit between them (single and paired DPAS chains became chains of 4 and 8, DPAS-completion waits per super-block 53 -> 45), and the token tiles as the fast grid dimension, so a 256-row weight panel is read by its token tiles in turn while it is in L2.
- **Effect:** `measured-here`, B60. Dense 27B prefill, llama-bench: 632 -> 758 t/s at 512 tokens and 594 -> 702 at 4,096 for the 2D loads (0011); then 891 / 816 t/s with the order change (0012). Q4_K 4,096 x 512 x 14,336 kernel: 1,722 -> 1,262 us (47.6 TFLOPS) -> 1,229 us. Motivation: `mul_mm_kq_f16.cl` was 73 % of a 4,096-token dense prefill and unitrace showed its threads waiting on loads 52 % of the time. KL 0.003559 unchanged.
- **Applies to:** Xe2 (B60) only: needs `cl_intel_subgroup_2d_block_io` and a base alignment that is a multiple of 64 bytes; Q4_K, Q5_K from K = 512, Q6_K from K = 1,024; within the extension's 2^24 limits.
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_2d.cl`, `kernel_mul_mm_kq2d_q4_K`, `kernel_mul_mm_kq2d_q5_K`, `kernel_mul_mm_kq2d_q6_K`. Needs `cl_intel_subgroup_2d_block_io`, B60 only. (0011, 0012)
- **Switch:** `GGML_OPENCL_KQ_2D=0`, `GGML_OPENCL_KQ_2D_SHAPE="TM,WG,AT"`.
- **Source:** arcint's OpenVINO kernel (marfrit-openvino patch 0029) moved onto ggml's flat planes; oneDNN's JIT output read for the ordering (`code`).

### Packed fp16 decode of the codes, and the barrier that keeps sub-groups in step

Decode two codes at once: one `bfn` masks two codes into a dword as halves 1024 + q, and two 32-wide half instructions subtract 1024 (exact) and scale, written as inline vISA because IGC keeps a per-lane `half2` as two planar halves and moves them apart and back around every operation. The activations are written with k permuted inside every group of four (`kernel_cvt_f32_f16_p4`) to match the code pairs. A work-group barrier every 16 blocks (`KSYNC`) keeps the sub-groups, which share the activation tile through L1, from drifting apart over a long k loop (at K = 17,408 the load-store cache hit 86.5 % of reads against 92 % at K = 5,120).
- **Effect:** `measured-here`, B60, dense 27B Q4_K_M, `-fa 1`. Prefill 512 / 4,096: 892 / 816 t/s (0012) -> 919 / 839 (packed decode, Q4_K) -> 935 / 852 (Q5_K, Q6_K) -> 941 / 857 (conversion) -> 968 / 880 (KSYNC 16). Q4_K main loop 1,640 -> 970 instructions for 128 DPAS. Down projection, 512 tokens: Q4_K 1,496 -> 1,342 us.
- **Applies to:** Xe2 (B60), the 2D-block kernel; Q4_K, Q5_K, Q6_K.
- **Default:** on.
- **Location:** `mul_mm_kq_2d.cl` (`hscale` inline vISA), `kernel_cvt_f32_f16_p4`. (0013)
- **Switch:** the 2D switches above.
- **Source:** arcint's own, from IGC ISA dumps.

### int8 DPAS for Q4_K with the activation quantized per 256 values

One i8 x i8 DPAS per 32-value sub-block, at twice fp16's multiply rate. The activations are int8 per (token, 256), with their integer sums per 32. The codes are s8 (code - 8), so a sub-block's product is at most 32 x 127 x 8 = 32,512 and fits 16 bits. The 6-bit scale is applied by one integer `mad` on word sources (inline vISA `mad16`) into an int32 accumulator that spans the super-block; the constant part (8 D sc - DM m) is one fp16 DPAS per super-block against the sums. The deviation from the references is the per-256 activation scale: llama.cpp's MMQ quantizes per 32 and oneDNN per 64, and the per-32 forms rescale in float after every sub-block and measured slower here (1,599-2,140 us against 1,229 us for fp16).
- **Effect:** `measured-here`, B60. Dense 27B prefill 512 / 4,096: 968 / 880 t/s (KSYNC 16) -> 1,029 / 930 t/s with int8 Q4_K; with `GGML_OPENCL_KQ_2D_I8=0` 973 / 884. The price: KL 0.003562 -> 0.004034 nats, top-1 97.843 -> 97.721 %. The acceptance task, served, dense B60 and coder A770, 10/10 at temperature 0; 20 sampled runs per arm on the dense model: 14 of 20 (mean 8.7) with int8, 12 of 20 (7.7) without, 15 of 20 (8.15) for 0012, which do not separate. Decode is unaffected (its matvec keeps q8_1 per 32).
- **Applies to:** Xe2 (B60), the 2D-block kernel; Q4_K only; prompt GEMM (decode keeps the q8_1 matvec).
- **Default:** on (`GGML_OPENCL_KQ_2D_I8=0` keeps the fp16 kernel).
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_2d.cl`, `kernel_mul_mm_kq2d_q4_K_i8`, activation `kernel_quant_i8`. (0013)
- **Switch:** `GGML_OPENCL_KQ_2D_I8=0`.
- **Source:** the MMQ idea from llama.cpp CUDA and oneDNN; the per-256 variant is arcint's own.

### Genetic search of tile shapes

Search the GEMM tile parameters with a genetic algorithm on the real kernels and take the winners as defaults; the hooks (per-type shape variables, a defines variable, test-backend-ops perf cases for the dense model's prompt shapes) stay in the tree, inert when unset. Findings of the first search: on the A770 the tokens-as-a `MUL_MAT_ID` kernel lost to the older tile kernel once that kernel's tile was searched (2,4,1,1 -> 2,4,2,2; plain GEMM 4,4,1,16 -> 4,4,2,8), and on the B60 the int8 Q4_K kernel's barrier interval is 32 blocks.
- **Effect:** `measured-here`, llama-bench `-fa 1`. Coder A770 prefill 512: 1,613 -> 1,739-1,742 t/s (+7.9 %); at 4,096: 1,345-1,346 -> 1,431-1,432 (+6.4 %); at 2,048 tokens with ubatch 1,024, a shape the search never saw, 1,659 -> 1,882 t/s (+13.4 %). Dense B60: +0.5 % at both depths. KL coder 0.007043 -> 0.007040.
- **Applies to:** Both cards; the K-quant GEMM tiles (the attention tiles were searched separately); the hooks are inert unless set.
- **Default:** on as defaults; hooks unset.
- **Location:** `ggml/src/ggml-opencl/ggml-opencl.cpp` (tile defaults); hooks `GGML_OPENCL_KQ_2D_T<i>`, `GGML_OPENCL_KQ_2D_OPTS`; search tool `tools/kq_tune/`. (0014)
- **Switch:** `GGML_OPENCL_KQ_2D_T<i>`, `GGML_OPENCL_KQ_2D_OPTS`, `GGML_OPENCL_KQ_MM_ID2=1` to turn ID2 back on.
- **Source:** arcint's own (campaign `kernel-autotune-ga`, on the `qfndev` branch).

### Few-column XMX product for speculative verification

A weight times 2 to 16 f32 columns, each weight read once for all columns. A lane is a weight row; its code bytes become halves 1024 + q (the integer-built codes of the previous entry); the tokens are the a operand, 8 a DPAS, read from a tiled f16 copy of the columns that a prep kernel writes once per input (and that matmuls sharing the input reuse). A DPAS chain starts at -1024 S, so it ends at the sum of q x. S must be the sum of exactly the halves the DPAS multiplies: under `-cl-unsafe-math-optimizations` the compiler summed the unrounded floats, which put results 4-9 % off, so the prep decodes each half from its bits with integer operations and sends subnormals to zero first.
- **Effect:** `measured-here`. B60, Q4_K 4,096 x 14,336: 111 us at 2, 4 and 8 columns, 158 us at 16; Q6_K at 8 columns 191 us against 902 us for the GEMM. Dense 27B on the B60, step at n tokens over the 1-token step (50.2 ms): at 8 tokens 6.0x -> 1.21x, at 16 tokens 6.2x -> 1.51x. Served with MTP, acceptance task 10/10: dense B60, 3 drafts 31-35 -> 43.5 t/s, 5 drafts 48.0 t/s; coder A770, 2 drafts 67-69 before and 65.2 after, 3 drafts 69.2, 4 drafts 71.7 t/s. The path starts at 2 columns on Xe2 and at 4 on Xe-HPG; on the A770 the q8_1 matvec stays ahead at 2 and 3 columns (3 tokens: 104.3 against 96.3 t/s).
- **Applies to:** Both cards; Q4_K, Q5_K, Q6_K; 2 to 16 columns on Xe2, 4 to 16 on Xe-HPG (where the q8_1 matvec stays ahead at 2 and 3 columns).
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_few.cl`, `kernel_mul_mm_few_prep`, `kernel_mul_mm_few_q4_K_f16` (Q5_K, Q6_K likewise). (0009)
- **Switch:** `GGML_OPENCL_KQ_FEW=0`, `GGML_OPENCL_KQ_FEW_MIN`.
- **Source:** Strata and NInfer verify with the same kind of kernel (`code`, per the patch text).

### Dense Q4_0 on the integer-dot matvec and the XMX GEMM

Dense Q4_0 weights (the community GPU tables' Llama 2 7B Q4_0) took upstream's float kernels: the tiled GEMM was about 95 % of a 512-token prefill and the 8x matvec two thirds to four fifths of a decode step. Q4_0 has IQ4_NL's flat planes (16 code bytes and one f16 scale per 32 weights) with value n - 8, so the path is IQ4_NL's: up to 4 columns the q8_1 integer-dot matvec, from 16 columns the fp16 XMX GEMM with n - 8 taken in integers. A KV cache's Q4_0 is AoS and declines; 5 to 15 columns stay on upstream's path, unmeasured.
- **Effect:** `measured-here`, windows 73/74, A770 GT pinned at 2,000 MHz, Llama 2 7B Q4_0, n_ctx 4,096, f16 KV, 5 repetitions. A770: pp512 362.73 -> 1,956.15 t/s (5.4x), tg128 39.48 -> 63.56 (+61 %). B60: pp512 425.57 -> 2,332.94 (5.5x), tg128 76.24 -> 75.77 (-0.6 %). Answer-level bar: mean KL 0.000431-0.000432 nats on and off at `-b 512` and `-b 4`; argmax agreement 99.07 % on against 99.09 % off (A770); all 8 fact questions right. The B60 matvec takes 10.4 ms a token over about 3.7 GB (about 360 GB/s of the card's specified about 456 GB/s), against 10.7 ms for the old kernel. The source corrects its own earlier statement that this decode was memory-bound: 0097 later moved it 6.5 % by attention alone.
- **Applies to:** Both cards; dense Q4_0 weights; the matvec up to 4 columns, the GEMM from 16 columns, 5 to 15 columns stay on upstream's path; a KV cache's Q4_0 declines.
- **Default:** on (`GGML_OPENCL_Q40_XMX=0` keeps the float kernels).
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl` (`kernel_mul_mv_q4_0_q8_1`), `kernels/mul_mm_kq_f16.cl` (`kernel_mul_mm_q4_0_f16`). (0096)
- **Switch:** `GGML_OPENCL_Q40_XMX=0`.
- **Source:** arcint's own, after the community GPU tables (a Llama 2 7B Q4_0 file).

### Int8 DPAS without local memory for Q4_0 and Q8_0 (and Q4_K), activations packed per graph

IPEX-LLM's oneDNN f16 GEMM took about 95 ms of a 192 ms 512-token prefill on the B60, its dequantization about 40 ms, and the local-memory tile of the previous entry about 160 ms for the same products. On Xe2 the Q4_0 weights get the 2D-block GEMM the K-quants already had: an fp16 form `(q - 8) d` rounded as a dequantization to f16, and an int8 form with one int8 DPAS per 32 weights. On Xe-HPG (no 2D loads) the activations are quantized once per graph into `[k / 32][token][32 bytes]`, so an `a` operand of 8 tokens is one 256-byte block read and a lane is a weight row; Q8_0 had no fast path on sub-group 8 at all. Q4_K followed on the same shape, with the quantizer also writing the integer sums per 32 for Q4_K's constant term (a k16 fp16 DPAS per block). From 16 columns, K a multiple of 256.
- **Effect:** `measured-here`, Llama 2 7B Q4_0, flash attention on, 5 repetitions, two rounds. B60 pp512: 0096's tile 2,482 -> 2D fp16 3,575 -> 2D int8 4,389 t/s, tg128 81.7 throughout; whole-chunk KL 0.00043 -> 0.00043 -> 0.00105, argmax 99.09 -> 99.09 -> 98.82 %. A770 pp512: 2,073 -> 2,813 t/s at the default shape 32,8, which has no KL row; at shape 32,16 (2,794 t/s) whole-chunk KL 0.00043 -> 0.00104 and argmax 99.07 -> 98.65 %. A one-token ubatch takes the matvec, so its KL does not see these kernels. IPEX-LLM's wheel on the same host: B60 2,826 and A770 2,604 t/s with a whole-chunk KL of 0.016. The coder's own file on the A770 (4,096 tokens, ubatch 1,024, f16 KV): Q8_0's kernel off the profile's top 12 (810 ms of 6.7 s), prefill 1,695 -> 1,866 t/s (+10.1 %), decode 58.4 both; Q4_K int8 (0102) 1,865 -> 1,919 t/s (+2.9 %), the kernel 838 -> 632 ms, KL 0.00572 -> 0.00585, argmax 96.91 -> 97.40 % (ubatch 512) and 96.77 -> 96.62 % (ubatch 32); the served A/B and the acceptance task for 0102 were still owed in the source.
- **Applies to:** Xe2 for Q4_0 (2D-block kernel); Xe-HPG for Q4_0, Q8_0 and Q4_K; from 16 columns, K a multiple of 256, otherwise the old paths take the product.
- **Default:** on (`GGML_OPENCL_Q40_2D=2`, `GGML_OPENCL_Q8_I8`, `GGML_OPENCL_KQ_SG8_I8`; each `0` keeps the kernel below it).
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_2d.cl` (`kernel_mul_mm_kq2d_q4_0`, `kernel_mul_mm_kq2d_q4_0_i8`); `kernels/mul_mm_q40_i8_sg8.cl` (`kernel_mul_mm_q4_0_i8_sg8`, `kernel_mul_mm_q8_0_i8_sg8`, `kernel_mul_mm_q4_K_i8_sg8`, activation `kernel_quant_i8_sg8`). (0099, 0102)
- **Switch:** `GGML_OPENCL_Q40_2D` (0 tile, 1 fp16 Xe2 only, 2 int8 default), `GGML_OPENCL_Q8_I8=0`, `GGML_OPENCL_Q40_SG8_SHAPE` (`TMM,WG`, default `32,8`), `GGML_OPENCL_KQ_SG8_I8=0`.
- **Source:** IPEX-LLM's lead priced in `docs/campaigns/research-concept-mining/ipex-llm/` (on the `qfndev` branch); the kernels are arcint's own.

## 2. Low-bit and codebook weight types, and small-operand kernels

Weight formats that stock ggml-opencl left on the CPU (IQ and Q2 families), how their decode is made cheap on an integer-issue-bound pipe, and the small kernels a MoE decode step also needs.

### Codebook and IQ weight types on the same machinery

Expand IQ3_XXS (through its grid and sign tables held in local memory), IQ4_XS and IQ4_NL (through `kvalues_iq4nl`) to signed int8 per 32-weight sub-block and feed them to the same integer-dot matvec and fp16 GEMM. `supports_op` claims an op only when its kernels were built and declines IQ shapes no kernel takes, so those weights stay on the CPU rather than reaching a missing kernel.
- **Effect:** `measured-here`, A770, standalone harness: IQ4_XS / IQ4_NL matvec about 200 GB/s on 5,120 x 2,560, IQ3_XXS 82 GB/s (98-byte blocks read with 16-bit loads). Before it, every expert of the Flash-Next GGUF ran on the CPU. No served number in this section.
- **Applies to:** Both cards; IQ3_XXS, IQ4_XS, IQ4_NL (IQ4_NL with K a multiple of 32); matvec 1 to 4 columns, the fp16 GEMM, `MUL_MAT_ID`.
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl`; the dequantizers in `kernels/mul_mm_kq_f16.cl`. (0002)
- **Switch:** none beyond 0001's.
- **Source:** arcint's own.

### Low-bit expert types and planar storage on the card

The GSQ-RCO quantizations of Qwen3.8-Flash-Next (the files Strata serves) keep experts in IQ2_S, IQ2_XXS, IQ1_M and Q2_0 (down), with dense IQ3_S; llama.cpp left those layers on the CPU. The patch adds matvec (1 to 4 columns), `MUL_MAT_ID` and the f16-tile GEMM for them, with expanders written in 32-bit arithmetic only (64-bit integer arithmetic is emulated on these cards). IQ2_XXS, IQ2_S, Q2_0 and IQ3_S are stored in planes per weight slice on the card (each field of every block back to back) so a lane reads aligned 32-bit words where the 18-, 66-, 82- and 110-byte GGUF blocks left fields at 2-byte offsets. A slice keeps its size and place, so the expert cache's slot and bank copies stay byte copies; the host conversion happens in `set_tensor`, `get_tensor` and the USM bank. IQ2_XS followed as a further type, with the grid and sign tables in local memory for the matvec (the GEMM reads them from constant memory), which lets the cache take 10 further layers of the IQ3_XXS file.
- **Effect:** `measured-here`, B60, 4,096 x 14,336, one column: Q2_0 184 -> 99 us, IQ2_XXS 211 -> 111, IQ3_S 255 -> 156, IQ2_S 146 -> 142 (the 32-bit expanders took IQ2_S 168 -> 146 and IQ3_S 284 -> 255 before the planes). Q2_0 rows of 20 sub-blocks (Flash-Next's 2,560 x 640 down projections) on 4 lanes a row: 135 -> 89 us at 3 tokens. IQ2_XS: Flash-Next IQ3_XXS served with all 48 layers cached instead of 38; 300-token answers 3.7 -> 4.2-5.0 t/s, bound by storage reads from ZFS (4,000-4,700 major faults and about 1 GB per answer); IQ2_XS on NVMe decodes about 33 t/s. A claimed "27.6-28.0 t/s" for that setup was retracted in the source: the answers repeated one prompt whose pages were in the page cache.
- **Applies to:** Both cards; IQ2_S, IQ2_XXS, IQ1_M, Q2_0, IQ3_S and IQ2_XS; planar storage per weight slice, an odd block count leaves a slice in GGUF blocks and its ops on the CPU.
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl`, `kernels/mul_mm_kq_f16.cl`; conversion `ggml_cl_lb_convert` in `ggml-opencl.cpp`. (0022, 0030)
- **Switch:** none new.
- **Source:** Strata's file format (ISTA-DASLab GSQ-RCO); the kernel shapes are arcint's.

### Packed-byte (SWAR) arithmetic for codebook weights

Do the per-byte weight arithmetic of the low-bit kernels on 32-bit words rather than on `char4` vectors. The sign step becomes `(g ^ m) + (m & 0x01010101)` instead of `char4(g ^ m) - char4(m)`, exact because no grid byte is 0 so that `~g + 1` never carries out of a byte; IQ1_M's plus/minus delta and Q2_0's -1 offset use the same trick with a 0x80808080 bias. Reason, from the ISA (`ocloc compile -device bmg`): IGC lowers `char4` arithmetic one byte at a time, moving each byte into its own register, widening, computing and packing back, and those moves were more than half of the matvecs' instructions.
- **Effect:** `measured-here`, 2026-10-11. ISA on the B60, instructions / movs: IQ2_XXS 1,405 / 526 -> 1,052 / 206; IQ2_S 1,585 / 541 -> 1,230 / 221; IQ1_M 1,355 / 408 -> 1,138 / 222; Q2_0 1,291 / 384 -> 1,069 / 194 (dp4a count unchanged at 32). Kernel at Flash-Next's shapes, one token, B60: IQ2_XXS 40.7 -> 25.0 us, IQ2_S 50.6 -> 31.1, Q2_0 gate/up 33.4 -> 21.7. A770 kernels within 7 %. Served Flash-Next IQ2_XS on the B60: decode 30.7-31.9 -> 31.7-33.5 t/s, prefill equal; the coder's decode 42.1 -> 42.1. KL unchanged at 0.294050. The word forms were checked exhaustively against the `char4` forms.
- **Applies to:** Both cards (the gain was measured on the B60; within 7 % on the A770); the sign step of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_M and Q2_0.
- **Default:** on (`-DLB_CHAR4_ARITH` keeps the old form).
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl`, expand helpers shared with `kernels/mul_mm_kq_f16.cl`. (0034)
- **Switch:** `-DLB_CHAR4_ARITH` through `GGML_OPENCL_IQ_OPTS` keeps the old form.
- **Source:** arcint's own.

### Local-memory codebook with the values pre-shifted into their byte

The IQ4 lookup (`kv4`, IQ4_XS and IQ4_NL) reads four 16-entry tables from local memory, one per byte position with each value already shifted into its byte (64 words), and ORs the results: `t[n0] | t[16 + n1] | t[32 + n2] | t[48 + n3]`. It replaces constant-memory byte lookups assembled as a `char4` (B60) and register selects (A770). The dense IQ4_XS matvec was bound by the integer pipe (XVE active 68 %, ALU1 62 %, memory stall 35 %; Q4_K 52 / 41 / 53 %), and each constant lookup widened its index to a 64-bit address.
- **Effect:** `measured-here`, 2026-10-11, kernels in test-backend-ops perf. B60 dense IQ4_XS 4,096 x 14,336 one column 131.1 -> 92.3 us; 6,144 x 2,560 43.4 -> 24.9 us; MUL_MAT_ID IQ4_XS 36.7 -> 22.1 us. A770 4,096 x 14,336 115.2 -> 91.2 us. Served Flash-Next, device time: dense matvecs 11.13 -> 10.35 ms per token, all kernels 26.89 -> 26.09 ms; wall time inside the run-to-run spread (32.3-33.4 against 33.0-34.4 t/s). Coder on the A770 (IQ4_XS experts, MTP 4): 41.0-42.1 -> 42.0-43.0 t/s. Outputs bit-identical to the old forms. A previous local-memory codebook measured slower (59.5 against 44.2 us) because it was a per-lane 16-byte copy; the difference is one table per work-group with the values pre-shifted.
- **Applies to:** Both cards; IQ4_XS and IQ4_NL matvecs.
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl` (`LOCAL_TABLES`, `kv4`). (0035)
- **Switch:** `GGML_OPENCL_IQ_OPTS` with `-DIQ4_KV_CHAR4`, `-DIQ4_KV_SHIFTED`, `-DIQ4_KV_SELECT`, `-DIQ4_LT_RED` (red case).
- **Source:** arcint's own, from unitrace stall metrics and ISA counts.

### Multiply-spread decode of sign bits and 2-bit fields

Two decodes did per-byte work with shifts and masks in Flash-Next's low-bit expert kernels, which are bound by integer issue and not by memory. IQ2_S and IQ3_S signs: four sign bits spread to the bytes' low bits by one multiply, `((b & 0xF) * 0x00204081) & 0x01010101`, with the signed value `(g ^ (s * 0xFF)) + s`. Q2_0's four 2-bit fields: fields 0, 2 and 1, 3 spread by one multiply each after masking. Both identities hold for every input byte (checked exhaustively), so results are bit-identical. The prompt GEMM's own copy of the decode got the same change in 0054.
- **Effect:** `measured-here`, 2026-10-15, B60. Instructions (offline ISA): IQ2_S 487 -> 425, Q2_0 372 -> 340, IQ3_S 553 -> 489. Kernel, MUL_MAT_ID 512 experts top 10: IQ2_S 640 x 2,560 at 1 / 3 tokens 31.1 / 70.9 -> 29.1 / 63.6 us. Served Flash-Next decode +0.9 % on the mean (38.6 -> 38.7-39.2 and so on), prefill 664 t/s in both arms. The GEMM copy (0054): the 20k needle's prefill 664.0 / 664.2 -> 665.8 / 665.3 t/s (+0.2 %).
- **Applies to:** Both cards (measured on the B60); IQ2_S, IQ3_S and Q2_0, the matvec and, since 0054, the prompt GEMM.
- **Default:** on (`-DLB_SIGN_MASK4`, `-DQ2_SPREAD_SHIFT` keep the old forms).
- **Location:** `lb_signmask4` replacement in `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl` and `kernels/mul_mm_kq_f16.cl`. (0053, 0054)
- **Switch:** `-DLB_SIGN_MASK4`, `-DQ2_SPREAD_SHIFT` (old forms), `-DLB_SIGN_RED`, `-DQ2_SPREAD_RED` through `GGML_OPENCL_IQ_OPTS`.
- **Source:** arcint's own.

### Decode-path kernels for F16 weights, fused router, hyper-connection ops

Collected in one patch because Flash-Next's decode needed them. F16 / BF16 weights times 2 to 8 columns in one read of each weight row (4,096 x 14,336: 2.80 ms for the tiled GEMM at 2, 4 or 8 columns against 0.29 ms at one, now 0.70-0.89 ms); F16 / BF16 weights times 16 or more columns on the XMX tiles of the K-quant GEMM; the MoE router chain SOFT_MAX .. ARGSORT .. GET_ROWS .. SUM_ROWS, CLAMP, DIV fused into one kernel; `GGML_OP_DSV4_HC_PRE / _POST` so that llama.cpp's fusion probe builds the hyper-connection mixers fused; SCALE -> SILU and SCALE -> SIGMOID fused; QSA indexer ops on the card (TOP_K radix select a row, CPY F32 <-> I32, FILL, REPEAT, SET_ROWS for F16), without which the scheduler ran four CPU islands per attention layer (98 splits per Flash-Next verify graph), each draining the queue.
- **Effect:** `measured-here`, B60, Flash-Next. XMX f16 weights: 320 x 10,240 at 512 tokens 1,907 -> 468 us, the needle's prefill 342.8 -> 383.4 t/s. Indexer ops on the card: decode 32.0 -> 33.0-34.2 t/s, the needle 381 -> 422-425 t/s, the long text identical. Fused SCALE activations: no decode change measurable above the run-to-run spread. For the IQ, low-bit, F16 and BF16 types MUL_MAT 557/557 and MUL_MAT_ID 181/181 pass in test-backend-ops (DSV4_HC_PRE 18/18, DSV4_HC_POST 4/4, TOPK_MOE 416/416); the 74 MXFP4 MUL_MAT_ID failures on the B60 are in the pin, not in the series.
- **Applies to:** Intel devices; F16 and BF16 weights (stored as F16) at 2 to 8 columns and at 16 or more on the XMX tile; the fused router and the DSV4 hyper-connection ops are model-specific (Flash-Next).
- **Default:** on (`GGML_OPENCL_F16_XMX`, `GGML_OPENCL_FUSE_TOPK_MOE`, `GGML_OPENCL_FUSE_SCALE_ACT`).
- **Location:** `kernel_mul_mv_f16_f32_nc`, `kernel_mul_mv_f16_f32_ksplit`, `kernel_scale_act_f32` in `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl`; `kernel_mul_mm_f16w_f16` in `kernels/mul_mm_kq_f16.cl`; the fused router is the OpenCL counterpart of ggml-cuda's topk_moe. (0022)
- **Switch:** `GGML_OPENCL_F16_XMX=0`, `GGML_OPENCL_FUSE_TOPK_MOE=0`, `GGML_OPENCL_FUSE_SCALE_ACT=0`.
- **Source:** ggml-sycl for the DSV4 ops; ggml-cuda's topk_moe for the router (`code`).

### Skinny F32 weights: K-sliced GEMM

F32 weights of at most 256 rows (the delta-net's `ssm_alpha` / `ssm_beta`, a MoE router) got one 64 x 64 output tile per 64 columns over the whole K, so 8 work-groups for 48 rows and 512 tokens. Up to 8 columns now take ggml's `kernel_gemv_f32_f32_mc` (K split over a work-group); more take the tiled GEMM with K cut into slices run as its batch dimension, followed by a slice-summing kernel.
- **Effect:** `measured-here`. The coder's router in a 512-token prefill: 24.3 -> 9.0 ms (A770). The ssm_alpha/ssm_beta cost was 92 ms of the dense model's 512-token prefill before.
- **Applies to:** Intel (both cards); F32 weights of at most 256 rows.
- **Default:** on (`GGML_OPENCL_F32_SKINNY=0` keeps ggml's paths).
- **Location:** dispatch in `ggml/src/ggml-opencl/ggml-opencl.cpp` (the pin's `kernel_gemv_f32_f32_mc` for up to 8 columns); the slice sum `kernel_sum_slices_f32` in `ggml/src/ggml-opencl/kernels/mul_mm_f32_f32_l4_lm.cl`. (0005)
- **Switch:** `GGML_OPENCL_F32_SKINNY=0`.
- **Source:** arcint's own.

### F32 weights multiplied through an f16 copy made at load

Some GGUFs keep 2-D weights F32 (Gemma 4 E4B's per-layer-input projections `blk.N.proj` and `blk.N.inp_gate`, 220 MB). The backend had no fast F32 product: a prompt took ggml's F32 tiled GEMM or the skinny path, and decode read F32. When such a weight is written, the backend now keeps an f16 copy beside it and `MUL_MAT` runs the F16 kernels on the copy: the XMX GEMM from 16 columns, the f16 matvecs at 1 and 2-8. Gemma's values are bf16, which f16 holds exactly from 2^-17 up.
- **Effect:** `measured-here`, 2026-11-07, Gemma 4 E4B: the op suites pass on both cards (full `MUL_MAT` 2,299 pass, 0 fail), the red switch fails the cases it should (B60 14, A770 8), and the key `type=f32 k=256` vanishes. KL against the CPU reference, off -> on: f16 KV ubatch 512 0.0079 -> 0.0095, ubatch 1 0.0044 -> 0.0044, q8_0:q4_0 0.0192 -> 0.0190. Served on the B60 the prefill is within the run-to-run spread (f16 KV 2,108-2,429 t/s across the arms) and decode is equal, so the `inferred` 250-300 ms off a 6,656-token prefill (+7-10 %) is not confirmed; a repeated benchmark is pending. The coder's peak VRAM on the A770 is unchanged at 16,276 MiB.
- **Applies to:** Intel; F32 2-D weights of at least 256 rows and a K of at least 256, not routers (`ffn_gate_inp`), Hadamard rotation matrices or vocabulary-size products (`output.weight`, `token_embd`); a 3-D activation whose columns lie back to back is flattened and takes the copy; 9-15 columns keep F32; from 16 columns sub-group 16 only (the A770 has no f16 XMX GEMM). The copies are VRAM outside the weight buffers (110 MB on Gemma 4 E4B), logged per device; a weight that should have a copy and has none reports `reason=no-f16-copy`.
- **Default:** prompts only (`GGML_OPENCL_F32W_F16=2`; `=1` every column count, `=0` off). Gemma 4 E4B on the B60, llama-bench, 5 repetitions, two interleaved rounds (`measured-here`): pp4096 4,244 -> 4,408 t/s (+3.9 %) at f16 KV, 4,026 -> 4,173 (+3.7 %) at q8_0:q4_0, decode unchanged (83.70 / 83.77, 77.46 / 77.40); `=1` gains the same prefill and loses 0.7 % decode (83.07, 76.92).
- **Location:** `ggml_cl_f16_copy_build` and `ggml_cl_mul_mat_f16_copy` in `ggml/src/ggml-opencl/ggml-opencl.cpp`; test `test_mul_mat_weight`. (0110)
- **Switch:** `GGML_OPENCL_F32W_F16`.
- **Source:** llama.cpp's Vulkan backend rounds F32 weights to f16 inside `matmul_f32_f32` on fp16 devices (`code`); here the copy is made once at load.

## 3. Expert routing and expert GEMM tiles

How a MoE layer's (token, expert) pairs are turned into matrix products.

### Experts as DPAS "a" operand: the tokens are the matrix, the weight rows are lanes

For `MUL_MAT_ID` prefill, a work-group takes a tile of up to XT (token, slot) pairs of one expert and XSG sub-groups of weight rows, one row per lane. Per 256-weight super-block the pairs' activations are staged once in local memory in the DPAS a layout and each lane builds its row's b operand in registers, so weights never pass through local memory and are read once a tile. For Q4_K, Q5_K and Q6_K the b operand is the quantization codes as halves 1024 + q, built with integer operations on whole words; the DPAS chain per sub-block is corrected in f32 with the sub-block's activation sum (D sc (chain - 1024 S) - M m S). The route (pairs sorted by expert, cut into tiles) is computed once for a layer's gate, up and down, which share their ids.
- **Effect:** `measured-here`, 2026-10-03, A770 GT clock pinned at 2,000 MHz. The way there, the coder's Q4_K gate/up, 100 calls of a 512-token prefill: 193 ms (0001's kernel) -> 143 ms (f16 dequantization per lane, 32 x 8) -> 117 ms (integer-built codes, 32 x 16); with the dequantization replaced by a constant, 77 ms. Together with 0004 and 0005: coder A770 prefill 512 1,005.6 -> 1,292.2 t/s, 2,048 tokens 747 (0004 and 0005 only) -> 874.7, decode 38.8 -> 43.4; dense B60 prefill 521.1 from 491.6. KL, coder 0.00700 -> 0.00706 batched. Later, 0014 turned this kernel off by default on the A770 (the 0001 tile kernel with a searched tile was faster), and kept it on the B60; on the A770 the tile kernel is in turn replaced for IQ4_XS, IQ4_NL and Q4_K expert products by the int8 kernel of 0103/0104 (K a multiple of 256, not for the cache's slots and bank).
- **Applies to:** Both cards; Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_XS, IQ4_NL expert weights; prompts of 16 or more tokens.
- **Default:** on for sub-group 16 (B60), off for sub-group 8 (A770), where the tile kernel replaced it (`GGML_OPENCL_KQ_MM_ID2=1` turns it on).
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_f16.cl`, `kernel_mul_mm_id2_q4_K_f16`, `kernel_mul_mm_id2_q5_K_f16`, `kernel_mul_mm_id2_q6_K_f16`, `kernel_mul_mm_id2_iq3_xxs_f16`, `kernel_mul_mm_id2_iq4_nl_f16`. (0006)
- **Switch:** `GGML_OPENCL_KQ_MM_ID2=0` or `=1`, `GGML_OPENCL_KQ_MM_ID2_SHAPE=XT,XSG`, `GGML_OPENCL_KQ_ROUTE_CACHE=0`.
- **Source:** arcint's retired OpenVINO kernel `kernel_mul_mm_id2_*` (arcint's own).

### Expert-GEMM tile sized to the tokens an expert gets

The sub-group-16 default tile of `MUL_MAT_ID` was 16 rows x 32 tokens, which dequantizes a weight block once per token tile. At about 40 tokens per expert in a 2,048-token ubatch of Flash-Next, an expert paid one full and one nearly empty tile. A tile of 64 rows x 256 tokens with 8 sub-groups shares each dequantized block across more tokens and rows.
- **Effect:** `measured-here`, 2026-10-11, B60. Served needle (Flash-Next IQ2_XS, 13,000 MiB, 32,768 context, bank stream on), by tile: 494.7 t/s with the old 2,2,1,1, 634.1 with 4,4,2,4 set by the override (the 0032 table), 647.7 with the new default and no override. Deployed configuration (9,500 MiB, 131,072 context, MTP 2), two processes each: prefill 408.3 / 408.2 -> 639.8 / 644.5 t/s with the stream and this tile; decode equal (26.6-29.1 against 26.7-28.6). Control, the coder on the B60 (its IQ3_XXS / IQ4_XS experts take the tokens-as-a kernel, which does not read this tile): pp4096 1,905.5 -> 1,906.0. The A770 tile is unchanged.
- **Applies to:** Xe2 (sub-group 16, B60) `MUL_MAT_ID` tile kernel; the A770 tile is unchanged.
- **Default:** on (`GGML_OPENCL_KQ_MM_ID_TILE` overrides).
- **Location:** `ggml/src/ggml-opencl/ggml-opencl.cpp` (the SG 16 `MUL_MAT_ID` tile default `4,4,2,4`). (0033)
- **Switch:** `GGML_OPENCL_KQ_MM_ID_TILE`.
- **Source:** arcint's own.

### MoE router on one sub-group per token

The fused top-k router (softmax, the k largest, their sum clamped, the k weights divided by it) ran one work-group of 256 per token with k rounds of local-memory scans and barriers. Now each token gets one 16-lane sub-group with its probabilities in registers (four tokens a work-group) and the k rounds use sub-group reductions only, ties to the lower expert index. The private array is sized by compile-time buckets of the expert count, because one 64-float array for every count spilled on both cards (A770 about 239 registers, B60 about 87).
- **Effect:** `measured-here`. Router on the A770 coder 34.1 -> 13.5 us a call (1.364 -> 0.542 ms a token, of a 19.25 ms token). KL against the coder reference 0.023656 -> 0.023594. The sum over the probabilities runs in another order, so greedy texts and draft acceptance changed, and the coder's A/B harness failed two of its three decode cells on that while the device time per verify step fell 1.4-2.5 %. Flash-Next (B60, 512 experts): A/B PASS, prefill +0.7 %, decode +0.4 %.
- **Applies to:** Both cards; the fused top-k MoE router; expert counts bucketed by compile-time array size (184 and 512 experts measured).
- **Default:** on (`GGML_OPENCL_TOPK_MOE_LEGACY=1` keeps the old kernel).
- **Location:** `kernel_topk_moe_softmax_norm` in `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl`; dispatch `ggml_cl_topk_moe_fused` in `ggml-opencl.cpp`. (0064)
- **Switch:** `GGML_OPENCL_TOPK_MOE_LEGACY=1`.
- **Source:** arcint's own (CUDA's topk_moe fusion for the chain itself, see 0022).

### int8 DPAS for the prompt's routed experts (opt-in)

`MUL_MAT_ID` on int8 DPAS on a sub-group-16 device: the IQ3_S grid and IQ4_NL values decoded in-kernel to s8, q8_1-style int8 activations per 32, one s8 8x16x32 DPAS per 32-weight block per 8-pair group, scaled in float by (weight d x (1 + 2 sc)) x (activation scale). The activation buffers hold [column][K] codes and [column][K/32] scales, not [pair][K]: 21 MB of codes and 2.6 MB of scales at an 8k chunk and K 2,560 against 210 MB + 26 MB, which at Flash-Next's VRAM margin is real. Auto mode picks int8 while the mean routed pairs per expert are at or under a crossover (128 by default, Strata's measured 130-250 rows per expert on a B70). A first card run found the block-loop index counted in elements of K where the kernels read it as a 32-block index (weight reads out of bounds; forced int8 passed 2 of 12 cases); fixed to 12 of 12.
- **Effect:** `measured-here`, B60. Kernel: IQ3_S m=640 k=2,560 n=16 2,259 -> 1,143 us, n=512 8,482 -> 4,155 us; IQ4_NL k=640 n=16 1,199 -> 930 us, n=512 5,035 -> 3,826 us. Served Flash-Next IQ3_S: prompt 8k +1.2 % with the slot loan, 19k +2.1 %, 8k +3.2 % without the loan; with the IQ3_S gate/up arm alone no gain beyond the spread; decode equal. The GEMMs it covers are measured at 7.1 % (2,048 tokens) and 10.8 % (8,192 tokens) of Flash-Next's prefill span (OpenCL intercept trace, `measured-here`), a ceiling of about +3.3 % at 8k, which is what the served A/B delivered: the prefill's lever on this model is moving the expert weights, not the GEMM. Quality (one 8,192-token window, `-ub 2048`, expert cache, fp16 twins as floor): both formats +0.0054 to +0.0098 nats over the twins; per format +0.0039 to +0.0117; argmax -2.47 to -4.27 points in the window where the twins were bit-identical to the reference (the earlier window's -1.07 came from twins at a noisy 0.9692). The KL bar (0.03) passes and the argmax bar (1 point) fails in every arm. Verdict in the source and CHANGELOG: opt-in, off by default, not adopted on any unit. Strata's own figure, `paper`: KL 0.0002, top-1 99.7 % on teacher-forced positions, a different method; the same-method comparison was still owed in the source.
- **Applies to:** Xe2 (sub-group 16, B60) only; IQ3_S gate/up (K a multiple of 256, rows a multiple of 16) and IQ4_NL down (K a multiple of 32); other types and sub-group 8 keep the fp16 tile.
- **Default:** off (opt-in; `GGML_OPENCL_KQ_MM_ID_I8=1` or `auto`); not adopted on any unit.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_i8.cl` (`kernel_mul_mm_id_iq3_s_i8`, `kernel_mul_mm_id_iq4_nl_i8`, activation `kernel_kq_mm_id_quant_i8`), shared text `kernels/kq_i8_ops.inc`. (0071)
- **Switch:** `GGML_OPENCL_KQ_MM_ID_I8` (unset = off, `1`, `auto`), `GGML_OPENCL_KQ_MM_ID_I8_CROSS`, `GGML_OPENCL_KQ_MM_ID_I8_FMT`.
- **Source:** Strata's fused ESIMD form (`code`, `sycl/src/prefill/esimd_moe.dp.cpp`, `moe_fused_iq.dp.cpp`; its +21 % at 1,024 tokens and +12 % at 4,095 on a B70 and KL 2e-4 are `paper`).

### int8 DPAS for the prompt's expert products on Xe-HPG, a sub-group owning several row sets

On the A770 the prompt's expert products (IQ4_XS, IQ4_NL, Q4_K) take int8 DPAS with no local memory and no barrier. A pack kernel copies the quantized activation columns once per (activation, route) into the route's tile order `[tile][k/32][TMM][32]`, so gate and up share the copy. The IQ4 weight codes (`kvalues_iq4nl`, -127..113) are never rounded; the lookup `id_kv4` is a mux over the code's bit planes with no memory read. Alone this was a loss (see "tried and not adopted"); the fix is 0104: a sub-group owns NR sets of 8 weight rows and reads each activation tile once, into registers, for all of them, which divides the activation reads that were half of the kernel. Alchemist's DPAS takes 8 rows, so Strata's 16-row thread becomes two DPAS on one A operand. The register budget binds: the accumulator and the activation-scale vector `af` are 16 GRF per set and pair group, and NR 3 needs about 296 of 256, so IGC spills and retries with rolled loops. The default therefore drops `af` by folding the activation's scale into each 32-block's fma (8 GRF, one more multiply per DPAS, the same value to one more f32 rounding), and Q4_K's constant term moves out of the K loop. The expert cache's slots and the bank decline to the tile; a decline at dispatch is reported through the downgrade lines.
- **Effect:** `measured-here`, the coder's file on the A770, 4,096 tokens, ubatch 1,024, f16 KV, 3 repetitions, one window. Prefill: off 1,916 / 1,918 t/s; NR 1 2,043; NR 2 with `af` 1,938; **NR 2 without `af` 2,151 (+12.2 %)**; with the table 2,149; NR 3 without `af` 2,042, with the table 2,092; NR 4 without `af` 1,430 (spilled). Decode 58.6 in every arm. On the exact tree, two rounds: 1,915 -> 2,150 t/s. Served on the coder's deployed flags (q8_0:q4_0 KV) against 1.5.4: prefill at 41k 843.3 -> 958.5 t/s (+13.7 %); a 90k prompt 166.8 -> 159.2 s. Answers: the acceptance task 10/10 x3; KL against the file's CPU reference, 8 chunks, off -> on: ubatch 512 0.00585 -> 0.00803 with top-1 agreement 97.40 -> 96.67 %; ubatch 32 0.00600 -> 0.00909, 96.62 -> 96.81 %; inside the bar. The A/B's decode cells went red (answer b 58.8 -> 54.5 t/s, c 52.25 -> 51.4, a 56.35 -> 56.55) with different texts; a verify costs the same in both arms (30.6-31.1 ms) and the new texts took 389 verifies against 376; measured apart over 24 generated prompts, each with its own 2k-token prefix, tokens per verify 2.45, 2.40 -> 2.43, 2.47 and acceptance 75, 76 % -> 75, 76 %: no loss, the red is text variation over three answers. Memory: +43 MiB, peak 16,208 -> 16,251 MiB of the A770's 16,288, GTT unchanged at 32 MiB, so nothing is evicted but the margin is 25 MiB. Suites on the A770: iq4_xs 22/22, q4_K 105/105, iq4_nl 25/25; red (`-DID_RED`, a negated weight scale) 3/22, 40/105, 7/25. The first form made Q4_K spill at every NR, seen in the IGC dump and not in the suite.
- **Applies to:** Xe-HPG (A770) only: both programs build at sub-group 8 only, so the B60 does not change. IQ4_XS, IQ4_NL and Q4_K expert products, K a multiple of 256, rows a multiple of 8; not the expert cache's slots and bank.
- **Default:** on (`GGML_OPENCL_KQ_ID_SG8_I8=0` keeps the tile and does not build the program); NR 2.
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_id_i8_sg8.cl`, `kernel_id_pack_i8_sg8`, `kernel_mul_mm_id_iq4_xs_i8_sg8`, `kernel_mul_mm_id_iq4_nl_i8_sg8`, `kernel_mul_mm_id_q4_K_i8_sg8`, `id_kv4`; the activation quantizer `kernel_quant_i8_sg8` is shared with the dense kernels. (0103, 0104)
- **Switch:** `GGML_OPENCL_KQ_ID_SG8_I8=0`, `GGML_OPENCL_KQ_ID_NR` (2), `-DID_KEEP_AF`, `-DID_KV_TABLE` (Strata's byte-pair table; the two tie once 0104 is in).
- **Source:** Strata's fused int8 expert form (`code`, `sycl/src/prefill/esimd_moe.dp.cpp`); the row-set organisation is arcint's own.

## 4. Expert cache and bank

A MoE whose experts do not fit in VRAM: slots on the card, a bank in host memory, and who computes what.

### Per-expert cache: VRAM slots for hot experts, a USM host bank for the rest

For a MoE whose experts do not fit in VRAM: per layer, slot tensors on the card hold the hot experts (sized in bytes), a bank of all other experts sits in host memory allocated as USM host memory, and residency tables map the router's ids to one or the other. The graph looks the router's ids up in the tables and sums the slot branch and the bank branch. Decode's ids are read back once a step for usage counts, and every 6 decode tokens (`LLAMA_EXPERT_CACHE_ADAPT_EVERY`) hot bank experts exchange with cold slots on the card's queue (the adapt step). A cached model needs `expert_cache_bytes` (or `LLAMA_EXPERT_CACHE_MIB`) and its experts on the CPU. Host USM memory is passed to the kernels as pointers: a `cl_mem` over host memory is migrated to the card by the driver (`measured-here`, B60), so pointers are the zero-copy route.
- **Effect:** `measured-here`, 2026-10-06, B60, Qwen3.8-Flash-Next UD-Q3_K_XL from NVMe, 12,800 MiB of slots, 36.5 GiB of bank: the 20,045-token needle prefilled at 149.6 t/s against 81.9 for the previous stage (16 expert layers on the card), 235.0 t/s with 2,048-token ubatches; decode 12.2-12.3 t/s against 10.5-12.4; 75 % of decode's routed experts on the card after adaptation. KL against the CPU reference 0.029634 (top-1 95.784 %) against 0.029206 (95.931 %) for the previous stage.
- **Applies to:** Both cards in principle (measured on the B60 and, for the layer split, the A770); a MoE whose experts are placed on the CPU; IQ and low-bit types, Q8_0 down projections.
- **Default:** off until a budget is set (`--llama-expert-cache MIB`).
- **Location:** `src/llama-expert-cache.cpp`, `src/llama-expert-cache.h`, `llama_expert_cache_lookup`, `llama_expert_cache_owner`, `struct llama_expert_cache_layer`; `build_moe_ffn` in `src/llama-graph.cpp`; USM entry point `ggml_backend_opencl_set_alloc_host` in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0021)
- **Switch:** `--llama-expert-cache <MiB>` / `LLAMA_EXPERT_CACHE_MIB`; adaptation constants `LLAMA_EXPERT_CACHE_{ADAPT_EVERY,MAX_SWAPS,MIN_GAIN,MIN_USAGE,DECAY}` (the hit rate stayed at 89.9-90.7 % across them).
- **Source:** Strata's expert cache and adapt step, and FreeToken's gather of missing experts (`code`, both).

### Decode reads a missed expert in place, and a verify's pairs are ordered by where they live

At decode the expert cache's matvec reads a missed expert directly from the USM bank by pointer argument, with no gather into a card-side mirror first. Separately, a permutation computed once per layer (the three projections share the ids) puts the (token, expert) pairs that are read over the link first and the pairs in VRAM last, so the long link reads start early.
- **Effect:** `measured-here`, B60, Flash-Next, MTP 2: direct read 26.0 -> 28.2 t/s; pair order 29.8 -> 30.7 t/s (bank pairs last: 29.3). Served, IQ2_XS from the cache with the gate's requests (2026-10-06): decode 21.3 t/s plain, 27.0 t/s with MTP 2, prefill 371.1 t/s at 20k; before the patch's decode work 16.0 and 13.8. With the rest of the patch (2026-10-07): MTP 2 30.7-31.6 t/s, needle prefill 383 t/s.
- **Applies to:** Both cards; decode with the expert cache's bank in USM host memory.
- **Default:** on when the cache is on (`GGML_OPENCL_BANK_DIRECT`, `GGML_OPENCL_PAIR_ORDER` = 1).
- **Location:** `kernel_moe_pair_perm` in `ggml/src/ggml-opencl/kernels/mul_mv_iq_q8_1.cl`, dispatch in `ggml-opencl.cpp`. (0022)
- **Switch:** `GGML_OPENCL_BANK_DIRECT=0` (gather first), `GGML_OPENCL_PAIR_ORDER` (1 by default).
- **Source:** Strata's expert kernels read its pinned host mirror the same way (`code`, per the patch text).

### Expert swaps leave the compute queue: two-step non-blocking exchange

An adapt step exchanged experts between slots and bank entries on the card's in-order queue ahead of the next graph. A swap moves about 1.4 MB each way over the link, and 1,800-3,100 swaps happen in a 300-token answer, so the copies held up the next verify's input writes (2.5-6 ms against 0.93-1.0 ms with no swaps; `LLAMA_EXPERT_CACHE_MAX_SWAPS=0` as the discriminator). Each layer's bank gets spare entries (4 by default, about 263 MB more pinned host memory for Flash-Next). A swap runs in two steps on a second in-order queue, polled between ubatches without a wait: the victim's bytes are copied from its slot into a spare entry; once that landed, the victim is served from the bank and the newcomer is copied into the freed slot; once that landed, the newcomer is served from the slot. No adapt is planned while a swap is in flight. The tables change only between graphs, and a fence placed after each table upload keeps a copy from writing a slice an earlier graph still reads. The copies are enqueued right after the owner's graph, not before the next one: the first build enqueued them at step start, left the input wait at 2.4-5.9 ms and lost about 1.5 %.
- **Effect:** `measured-here`, 2026-10-13, B60, Flash-Next with the unit's flags (q8_0 KV, 11,300 MiB of slots, MTP 2, the bank stream), A B A B, one process an arm: decode on three 300-token answers 32.3 / 36.2 / 34.5 t/s -> 33.1 / 37.0 / 35.2 (+2.4 % on the mean), the verify's input wait 2.6-6.0 -> 0.94-0.95 ms, experts on the card 89.9 / 93.8 / 92.2 % -> 88.5 / 93.3 / 91.3 % (the newcomers land two verifies later, about one point of hits). Prefill unchanged (673 t/s). Answers right, draft acceptance counts identical. A probe showed a 48 MiB host-to-card copy on a second queue finishing in 3.5 ms while a 37.5 ms kernel ran on the first.
- **Applies to:** Any cached layer; needs the adapt entry points.
- **Default:** on with the cache (`LLAMA_EXPERT_CACHE_NB=0` restores the exchange).
- **Location:** `src/llama-expert-cache.cpp` (`LLAMA_EXPERT_CACHE_SPARES`, the ticket list, the `after_graph` hook). (0042)
- **Switch:** `LLAMA_EXPERT_CACHE_NB=0` (the old exchange), `LLAMA_EXPERT_CACHE_SPARES`, `LLAMA_EXPERT_CACHE_MUTATE=1` (the red case; the first landed swap is checked against the file).
- **Source:** Strata's adapt stream (`code`, the adapt planning and exchange in `src/program/generate.cpp`). Strata reserves 96 exchange buffers over all layers; here the spares sit in each layer's bank, so a swap beyond its layer's spares waits for a later adapt.

### Residency tables as rows of one tensor, one write per adapt step

Each layer's residency tables were their own tensor and their own non-blocking write; an adapt step touches up to 48 layers in two phases, so up to about 290 writes of 2 KiB. The tables are now rows of one I32 tensor per card (`tab_all`), each layer's tables views of its rows, and the changed range is written once.
- **Effect:** `measured-here`, 2026-10-14, B60, Flash-Next with the unit's flags. A probe: 240 small writes 1.64 ms against 54 us as one write. Served: time writing swaps and tables per 300-token answer 0.09 / 0.06 / 0.07 s -> 0.05 / 0.03 / 0.04 s; decode 31.4 / 36.6 / 35.1 t/s -> 31.6 / 36.7 / 35.3 (inside noise); texts and hit rates equal. Table check against the host maps: 0 of 144 rows differ; red case 1 of 144.
- **Applies to:** Any expert cache.
- **Default:** on with the cache.
- **Location:** `src/llama-expert-cache.cpp`, `tab_all`, `upload_tables`, `flush_tables`. (0046)
- **Switch:** `LLAMA_EXPERT_CACHE_MUTATE=1|2` (red cases).
- **Source:** Strata's one residency table and one upload (`code`, `res_upload` and the residency table in `src/program/generate.cpp`).

### CPU computes the first missed experts while the card computes the hits (doorbell tier)

At a decode layer's first expert projection a kernel hands the first N missed (token, expert) pairs to the CPU: it writes -1 into the combined ids, so the card's three projections skip them, and publishes the pairs and the activation to USM host memory behind a release store. A coordinator thread and a spinning pool compute the pairs' gate, up, SwiGLU and down from the bank in host RAM while the card computes the hits. After the down projection a kernel polls a done flag with L1/L3-uncached loads and copies the rows in. The card reads the other misses over the link as before. The CPU rows are multi-token AVX2 kernels: a 32-value chunk is decoded once and every token of the window applies it, and pairs are grouped by expert, so an expert missed by several tokens of a verify is decoded once. Two departures from Strata: weights are read from the bank's planes, and the activation quantizers were written here. A prototype with ggml-cpu's dot products did not pay, because ggml-cpu's Q2_0 dot product has no SIMD form.
- **Effect:** `measured-here`, 2026-10-15, B60, Flash-Next IQ2_XS with the unit's flags, six answers, mean decode: card only 36.3 t/s; CPU takes up to 2 / 4 / 8 / 16 pairs a layer: 37.8 / 39.4 / 40.4 / 41.8 t/s (+4.2 / +8.5 / +11.3 / +15.2 %), with 8 threads for 16; 24 or 32 pairs fall back below 16. The gain is largest where the cache misses most: the code answer (78 % of its experts on the card) +32 %. Kernel, one thread: Q2_0 640 x 2,560 at 1 / 4 tokens 661 / 2,637 us in ggml-cpu against 72 / 153 us. The texts differ from the card-only arm (the CPU's sums are not the card's) and repeat exactly within an arm. KL as served (8 x 512, ubatch 4): card only 0.367287 / 84.363 %, CPU up to 16 0.370044 / 84.412 % (+0.0028 nats); red case (CPU rows left zero) 0.436437 / 81.912 %. The coder decoding on the A770 at the same time lost 1 % (42.0 against 42.4 t/s) while Flash-Next kept +14 %. The CPU rows cost 8 threads spinning at 100 % while Flash-Next decodes. A follow-up fixed a layout trap: an IQ4_NL tensor in USM host memory is stored as two arrays (scales, quants), and the slow path read them as GGUF blocks, giving 13,230 non-finite values in layer 0; after the fix IQ3_XXS at 12,000 MiB of slots decoded 33.9 / 37.5 / 37.3 / 33.5 / 30.6 / 30.9 t/s on six answers against 20.7 / 26.2 / 28.9 / 18.7 / 21.2 / 18.5 without the CPU misses (+52 % on the mean).
- **Applies to:** Both cards in principle (measured on the B60); needs a CPU with AVX2 for Strata's rows; layers whose three projections are IQ2_S, IQ2_XXS, Q2_0 (0056), later IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL; other types use ggml-cpu's dot products.
- **Default:** off in the code (`LLAMA_EXPERT_CACHE_CPU_MISSES=1`); the Flash-Next unit sets it.
- **Location:** `ggml/src/ggml-opencl/ggml-opencl-miss-cpu.cpp`, `ggml-opencl-miss-cpu.h`, `ggml_cl_miss_cpu_join_iq4nl`; test `tests/test-opencl-miss-cpu.cpp`. (0056)
- **Switch:** `LLAMA_EXPERT_CACHE_CPU_MISSES=1`, `GGML_OPENCL_CPU_MISS_MAX`, `LLAMA_EXPERT_CACHE_CPU_MISS_THREADS` (8), `GGML_OPENCL_CPU_MISS_MT=0`; red `GGML_OPENCL_CPU_MISS_RED=1`.
- **Source:** Strata's doorbell tier with Strata's CPU kernels (`code`, `src/kernels/cpu/iq_avx2.cpp`, `q2_avx2.cpp`, MIT).

### Multi-token AVX2 rows for more quantization types, and a larger miss cap

The fast path is taken only when all three projections of a layer have multi-token rows, so every layer with an IQ2_XS, IQ3_XXS, IQ3_S or IQ4_NL projection ran ggml-cpu's single-token dot products. The rows for those four types were ported, with a per-type block constant and with single-token loops shaped like ggml's where the multi-token form lost at one token. The miss cap went from 16 to 64 pairs: with 10 experts a token and at most 4 tokens in a verify window, that is every miss on the CPU.
- **Effect:** `measured-here`, B60, Flash-Next with the unit's flags, six answers, mean decode: IQ3_XXS 33.85 / 33.87 -> 36.13 / 36.08 t/s at the old cap 16, 37.73 at 32, 37.93 at 64; the new rows forced off 32.83. UD-Q3_K_XL 12.48 -> 12.55 at cap 16, 16.68 at 64 and at 128. At four tokens the new rows are 2.48x (IQ2_XS), 2.86x (IQ3_XXS), 2.71x (IQ3_S) and 1.77x (IQ4_NL) as fast as ggml-cpu's dot products (the source writes "x ggml's time"; its code comment says "faster for two and more", and 0056's table for IQ2_S has 168 us against 394 us at four tokens), and 1.00-1.04x at one token. KL as served at ubatch 4: 0.2178 / 89.66 % against 0.2191 / 89.56 %. Three of eight IQ3_XXS answers changed wording, and the forced-old-path control differs from the base build in four; the source says this is not explained.
- **Applies to:** AVX2 CPUs; IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL added to the rows; cap of missed pairs per decode layer 64.
- **Default:** follows the doorbell tier (off unless `LLAMA_EXPERT_CACHE_CPU_MISSES=1`); cap 64 by default.
- **Location:** `ggml/src/ggml-opencl/ggml-opencl-miss-cpu.cpp`. (0063)
- **Switch:** `GGML_OPENCL_CPU_MISS_MAX` (64).
- **Source:** Strata's `src/kernels/cpu/iq_avx2.cpp` (`code`).

### One cache per GPU under a layer split

Each card keeps slots, tables and a USM bank for the layers it runs, in its own OpenCL context, with its own budget and counters. One swap plan covers every cache with one cap: the largest gains win wherever they are, and each swap runs on the queue of the card that owns its layer. Under the pin, Intel's OpenCL runtime lists two different GPUs as two platforms with one device each, and ggml-opencl made one context per platform; `GGML_OPENCL_DEVICES="P:D,..."` selects devices across platforms, programs are built for the device that uses them, and buffer types and backends are named `OpenCL`, `OpenCL1`, and so on.
- **Effect:** `measured-here`, 2026-10-07, B60 + A770, Flash-Next IQ2_XS, MTP 2, 131,072 context. Split at layer 40: the A770 holds every expert of its 8 layers, 90.4 % of the routed experts are on a card, prefill 378 t/s, decode 18.6 t/s; the B60 alone 84.4 %, 408 t/s, 19.8 t/s. The campaign gate (decode above the one-card arm) was not met. A model that fits on one card loses decode when split: the coder Q4_K_M on the B60 alone 77.2 t/s tg64, split 3/2 57.5, 1/1 55.0 (pp4096 improves 2,470 -> 2,945-2,976 t/s).
- **Applies to:** Two cards in one process; each card's layers.
- **Default:** off (needs `--device` with both cards and a layer split); not a production configuration.
- **Location:** `src/llama-expert-cache.cpp` (the shared plan); device selection in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0023, 0024)
- **Switch:** `GGML_OPENCL_DEVICES`, `llama_model_params.expert_cache_bytes_dev`, `LLAMA_EXPERT_CACHE_MIB=A,B`.
- **Source:** Strata's adapt across its stages (`code`, the adapt plan in `src/program/generate.cpp`).

### Expert routing trace for offline predictor pricing

An off-by-default recorder of the expert cache's routing into binary trace files (a file per process, rotated at a size cap, each file replayable alone): each graph's kind (prompt, decode, verify, MTP draft), first position, token ids, every layer's routed ids in the router's rank order, and every residency change the cache applies, so a replay rebuilds the run's resident set. The draft head's own routing is recorded through the same read-back hook as the cache's usage counts. The point is to price a predictor offline before any serving change. A trace with token ids is the prompt and the answer in another form; it is therefore a privacy decision of whoever enables it.
- **Effect:** no throughput measurement in the source; the cost statement is `code`: unset, one `getenv` at init; on, the decode ids are the ones the cache already reads back for its usage counts, so no new device wait is added; a traced prompt adds one host read and one wait at the next ubatch boundary, and a draft record is a read behind the draft graph written after that context's synchronize. A prompt record is about 1.9 KB per prompt token (a 100k prompt about 190 MB). A subtlety recorded: the MTP block must be expanded after the FFN's weights expand, otherwise the draft head's router stops matching the fused-router pattern and runs unfused with a different softmax sum order.
- **Applies to:** Any cached model.
- **Default:** off (`LLAMA_EXPERT_ROUTE_TRACE` unset costs one `getenv`).
- **Location:** `src/llama-expert-route-trace.h`, `src/llama-expert-route-trace.cpp` (`llama_expert_route_graph_kind`), `llama_expert_route_trace_flush` in `src/llama-context.cpp` and `src/llama-ext.h`; reader `tools/expert_cache_replay.py`. (0069)
- **Switch:** `LLAMA_EXPERT_ROUTE_TRACE=<dir>` (arcint `--llama-expert-route-trace`), `LLAMA_EXPERT_ROUTE_TRACE_MAX_MB`.
- **Source:** arcint's own (campaign `expert-foresight`, on the `qfndev` branch).

### Save the learned expert ranking as the next start's profile

At teardown the cache writes its learned routing as Strata's STRP profile, so a warm start can load it with `--llama-expert-profile`. The default ranking is Strata's xe save rule: the raw routing counts mixed with the start's ranking as pseudo-counts (lambda 2, 800,000 routed entries of prior by default), so a short conversation does not wipe the prior. The counting sits in `count()` and not in the adapt step, so it keeps accumulating with adaptation off, and it is held in double because a month at about 33 t/s passes float's 2^24 on the hottest pairs.
- **Effect:** no served measurement in the source; device-free tests only (C++ and Python writers byte-equal).
- **Applies to:** A single-card expert cache (a layer split writes nothing).
- **Default:** off (`LLAMA_EXPERT_CACHE_SAVE` unset).
- **Location:** `llama_expert_profile_add_heat` in `src/llama-expert-profile.cpp` (called from `src/llama-expert-cache.cpp`). (0070)
- **Switch:** `LLAMA_EXPERT_CACHE_SAVE=<file>` (arcint `--llama-expert-cache-save`, `--llama-expert-cache-save-rule xe|cu`).
- **Source:** Strata's `save_profile` (`code`, `sycl/src/program/generate.cpp`) and `rank_learned_profile` (`src/core/expert_cache.cpp`), commit `3d6f1718`.

### Lazy reads of a huge embedding table

Flash-Next's 27 GiB n-gram embedding table is mapped and read lazily with `MADV_RANDOM`, a graph's rows prefetched together with `MADV_WILLNEED`, instead of being touched whole. The mechanism is llama.cpp's own (`TENSOR_READ_LAZY` and `llama_prefetch_rows`); its AUTO mode resolves to off for a device without mmap support, the OpenCL device among them, so the table was an ordinary mapped weight with a synchronous fault per row. arcint's change is to set `mp.lazy_mode = LLAMA_LAZY_MODE_ON`.
- **Effect:** `measured-here` (0.7.5): 2.0-7.7 -> 0.3-0.6 ms per verify.
- **Applies to:** Models with a host-row-read tensor (Flash-Next's n-gram table); OpenCL, where llama.cpp's AUTO resolves to off.
- **Default:** on when experts are placed on the CPU (arcint sets the mode).
- **Location:** `src/exec/backend_llama.cpp` (`mp.lazy_mode = LLAMA_LAZY_MODE_ON` near the model load).
- **Switch:** none.
- **Source:** llama.cpp's lazy tensor reads (`code`, the lazy-mode resolution in `src/llama-model.cpp`, `src/llama-mmap.cpp`); arcint only forces the mode on.

## 5. Prefill: bank streaming, dispatch by token count, memory lending

Making a long prompt cheap on a card whose VRAM is also the decode working set.

### Prefill streams each bank tensor on a second queue

A prefill ubatch read each layer's bank experts from USM host memory through a gather kernel that copied the routed ones into a card-side mirror on the compute queue, ahead of each projection's GEMM: about 0.83 s of a 2,048-token chunk's 4 s of kernels, with the copy engine 2 % busy. The stream copies each bank tensor whole on a second in-order queue into a ring of VRAM slots, `ring - 1` tensors ahead of the GEMM that reads it. A GEMM waits for its copy event and a copy waits for the GEMM that last read its slot. The first prefill graph learns the order of its bank tensors; lookahead stays inside a graph because decode's swaps change the banks between graphs. IQ4_NL banks (scales and quants stored split) are not streamed and still gather.
- **Effect:** `measured-here`, B60, Flash-Next IQ2_XS, served needle prefill (20,045 tokens, MTP 2): 416.1 -> 497.6 t/s, compute engine 96 -> 95 %, copy engine 2 -> 88 %; deployed unit configuration (9,500 MiB of slots, 131,072 context): 406.6 -> 494.9 t/s (+21.7 %); ring 2 gave 392.9 t/s, so one tensor of lookahead is not enough. A ring of 4 costs 4 x 187.7 MiB of VRAM; VRAM peak 24,243 MiB against 23,448 MiB off. A standalone probe showed a 2 GiB USM-to-card copy on a second queue (150 ms, 14.3 GB/s) running entirely under a kernel on the first queue. KL as served 0.294050 / 87.083 %, same as off. The source marks the decode figures of the window void (a backup window). With a larger expert-GEMM tile (4,4,2,4) the same needle prefill reached 634.1 t/s against 494.7 with the default tile, above Strata's own engine on this card (620.5 t/s, a different engine and window, from the strata-sycl-b60 campaign document on the `qfndev` branch).
- **Applies to:** Intel OpenCL with the expert cache's USM bank; IQ4_NL banks still gather; from 256 tokens per ubatch; the B60 was measured.
- **Default:** off in the code (`GGML_OPENCL_BANK_STREAM=1`); the Flash-Next unit sets it.
- **Location:** `ggml/src/ggml-opencl/ggml-opencl.cpp`, `ggml_cl_bank_stream`, `struct bank_stream_state`, `ggml_cl_bank_stream_reset`, `ggml_cl_bank_stream_forget`. (0032)
- **Switch:** `GGML_OPENCL_BANK_STREAM=1`, `GGML_OPENCL_BANK_STREAM_RING` (default 4), `GGML_OPENCL_KQ_MM_ID_TILE`; red switch `GGML_OPENCL_BANK_STREAM_RED=1`.
- **Source:** Strata's prefill ring (`code`, the ring in `src/prefill/prefill.cpp`) and Infernix's expert stream.

### Choose the streamed GEMM, the gather or the matvec by token count

Three ways to run a cached MoE layer exist, and the cheapest depends on the tokens in the ubatch. Streaming the whole bank tensor through the ring costs the same link time whatever the token count (the whole bank, 22.37 GiB at Flash-Next's sizes, about 1.84 s a graph up to roughly 1,000 tokens); gathering only the routed experts into a card-side mirror is cheaper for short prompts; the matvec (the q8_1 arithmetic that decode uses) is cheapest for a handful of tokens. The expert cache's tensors therefore take the GEMM from 64 tokens (the other MoE tensors from 16), and the bank stream runs from 256 tokens a ubatch.
- **Effect:** `measured-here`, B60, Flash-Next with the unit's flags, A B A B, one process an arm. A 24-token follow-up prompt 1.86-1.87 s -> 0.64-0.65 s; 28 tokens 1.82-1.83 -> 0.57 s; 77 tokens 1.84 -> 1.11 s; 173 tokens 1.83 -> 1.54 s; 269 tokens and above unchanged; the 20,045-token needle 29.7 s either way. The crossover measured: 173 tokens 1.84 s streamed against 1.56 s gathered; 317: 1.84 / 1.96; 461: 1.85 / 2.26; 1,470: 2.36 / 3.51. The matvec path changes the arithmetic for the prompt, so the texts differ: the hash-table answer after a 28-token prompt accepted 166 of 267 drafts against 170 of 259; KL 0.294000 against 0.293562 (+0.0004 nats, -0.10 points of top-1).
- **Applies to:** The expert cache's tensors only (`.slots` and `.bank`); the coder and the dense models do not take this branch.
- **Default:** on (thresholds 64 and 256 tokens).
- **Location:** `ggml/src/ggml-opencl/ggml-opencl.cpp`, `ggml_cl_bank_stream` and the `MUL_MAT_ID` dispatch. (0043, 0044)
- **Switch:** `GGML_OPENCL_KQ_MM_ID_CACHE_MIN_TOK` (64), `GGML_OPENCL_KQ_MM_ID_MIN_TOK` (16), `GGML_OPENCL_BANK_STREAM_MIN_TOK` (256; 0 streams every GEMM ubatch).
- **Source:** arcint's own, priced first with the existing switches.

### Prefill slot loan: lend the expert cache's tail slots to a long prompt

Decode wants VRAM as expert slots, a prompt wants it as a larger ubatch (a bigger compute buffer, reserved for the session at load). A static split makes a larger chunk cost decode its slots for good. The loan lends the cache's tail layers' slots to the prompt: the tail slot tensors live in their own buffer, a loan frees it and re-reserves the compute buffers at the prompt's chunk, the lent layers' experts stay reachable in a lend bank in host memory through a third `MUL_MAT_ID` branch (so tensors stay under the grouped kernel's 512-expert local table), and on return the compute buffers are re-reserved at the resident ubatch, the tail buffer is allocated again, every lent slot is refilled from the lend bank by region copies and verified. A failed return or an unverifiable refill exits non-zero so a supervisor restarts the unit clean. The lend is sized against the kernel's own device-wide memory state: before a loan the engine asks the xe driver for the card's used and total memory and lends layer by layer until `used - lend + need + first_touch + allowance <= total` holds at zero margin (`need` is what the prompt's compute buffers require; the allowance, `LLAMA_SLOT_LOAN_PREFILL_GROWTH_MIB`, is 0 by default), and it falls back to the resident path when the tail layers cannot reach it. After the loan an eviction guard compares the process's resident VRAM and GTT growth with the expectation, logs it, and feeds a true shortfall (at or above a 64 MiB noise floor) into the next loan's sizing; `LLAMA_SLOT_LOAN_EVICTION_GUARD=0` turns it off. `LLAMA_SLOT_LOAN_CARD_MARGIN_MIB` only moves the fallback used when the kernel gives no figure. What a static trade would cost, measured before: a chunk of 8,192 at 8,000 MiB of slots prefilled a 19,196-token prompt at 569.9 / 572.1 t/s against 442.1 at the unit's 2,048 chunk and 11,600 MiB (+29 %), while decode at 8,000 MiB of slots was 31.08 against 34.30 t/s (-9.4 %). The cause: the grouped GEMM's 32-token tiles hold +396.5 % (gate/up, bank experts) and +192.5 % (slot experts) more rows than were routed at 2,048, +115.8 % and +48.6 % at 8,192.
- **Effect:** `measured-here`, B60, Flash-Next IQ3_S with the unit's flags plus the loan (`--llama-slot-loan 6400 --llama-loan-chunk 8192 --prefill-chunk 8192`). An 18,802-token prompt: 436.9 -> 485.2 t/s (+11.1 %) in the first window, +18.0 % (515.5 against 436.7 t/s) later; decode on six answers 34.62 / 34.58 against 34.67 / 34.65 (inside the spread); answers byte-identical. The CHANGELOG states prefill at 103,638 tokens 217.7 -> 347.9 t/s (+59.8 %), +18 to +19.5 % at 18.8k, KL inside the twin floor. The lend frees 6.20 GiB in 26 layers (3,094 slots); the refill took 2.66-2.86 s (about 2.3 GB/s against the link's 14.3 GB/s) before the region-copy rewrite and 0.514-0.539 s after it, with the lend bank compared byte for byte (0 of 12,250 rows differ). A card-side fault in the exit path was traced to the runtime's live submission rings: the poison exit waits an idle interval (default 2,000 ms) after draining the queues, because with idle 0 it faulted. The sizing rule above replaced an earlier fixed-margin rule (rounds 17-20 of the campaign, replaced in rounds 21-22); the campaign document (on the `qfndev` branch) was still open when this was written.
- **Applies to:** Flash-Next with the expert cache; lendable are the planar types and IQ4_NL (Q8_0 layers are not); measured on the B60.
- **Default:** off (needs `--llama-slot-loan MIB`); the Flash-Next unit sets it.
- **Location:** `src/llama-expert-cache.cpp` (`extend_lend`, `lend_undo`), `src/llama-loan-book.h` (`llama_loan_book`), `ggml_backend_opencl_lend_write`, `ggml_backend_opencl_drain` in `ggml/src/ggml-opencl/ggml-opencl.cpp`; the lent branch in `build_moe_ffn` (`src/llama-graph.cpp`); arcint side `src/exec/slot_loan.h`. (0068)
- **Switch:** `--llama-slot-loan MIB`, `--llama-loan-chunk N` (`LLAMA_SLOT_LOAN_MIB`), `LLAMA_SLOT_LOAN_VERIFY`, `LLAMA_SLOT_LOAN_EXIT_IDLE_MS`; red `LLAMA_SLOT_LOAN_MUTATE=1..5`.
- **Source:** Strata's lend region (`code`, the lend region and the refill in `src/core/expert_source.cpp`); Strata's B70 figures (16K 1,222 -> 1,347 t/s) are `paper`.

### Reserve compute buffers for the cells a prompt needs, not for a full cache

The scheduler reserves for a prompt ubatch at a completely full cache: 2,048 tokens against 131,072 cells for Flash-Next, 3,048 MiB on the B60 plus 770 MiB for the MTP draft context, for the whole session. The largest tensors were the QSA indexer's head scores (pools x 4 heads x ubatch tokens, f32, 1,026 MiB), a ReLU copy of the same size, and the selection mask. Two changes: the indexer's ReLU in place (3,048 -> 2,278 MiB and 770 -> 577), and a reservation parameter (`n_kv_reserve`) so the reservation sees min(cache, N) cells while `decode` cuts a batch's ubatches so that tokens x cells stays within the budget.
- **Effect:** `measured-here`, 2026-10-14, B60, Flash-Next at 131,072 tokens, 3 drafts with the 0.5 stop. Before: 8,191 slots (11,300 MiB), decode 34.8 / 38.0 t/s, 72k prefill 560 t/s (128.7 s), peak 23.82 GiB. ReLU in place: peak 22.85 GiB. `--llama-reserve-cells 20480` with 13,400 MiB of slots (9,724 slots): decode 38.4 / 43.0 t/s, 20k prefill 662 against 657 t/s, 72k prefill 416 t/s (173.0 s, deeper chunks are cut), peak 23.35 GiB. The cost is stated: a deep prompt streams the bank once per smaller ubatch. A first build capped the cells but not the indexer's pool count (3,048 -> 2,613 MiB only), and an arm that spent the expected saving on slots ran 1.8 GiB over the card.
- **Applies to:** Flash-Next (QSA indexer tensors dominate the reservation); the MTP draft context gets the same budget.
- **Default:** off (needs `--llama-reserve-cells N`); the Flash-Next unit sets it.
- **Location:** `llama_context_params::n_kv_reserve` (`include/llama.h`), `llama_context::decode` in `src/llama-context.cpp`; the in-place ReLU in `src/models/qwen4exp.cpp`. (0052)
- **Switch:** `--llama-reserve-cells N`.
- **Source:** arcint's own (the listing of the reservation's largest tensors).

## 6. Attention at depth

Decode, verify and prompt attention on both cards, with f16 and quantized KV.

### Decode attention with lanes along the head dimension, split over keys

For decode and short verifies (up to 8 query rows, f16 K/V, head size a multiple of 128), one sub-group of 16 per (head, split of 128 keys): the scaled query in registers, K and V rows by sub-group block reads, one sub-group reduction per key, online softmax over four keys a step, partial records merged by ggml's merge kernel. ggml's own split kernel keeps all DV accumulators in each lane and is limited to head size 128; its basic q1 kernel walks every key with one sub-group per head, so decode slowed with context. The logit softcap is applied before the mask (as ggml-cpu does) and scores are clamped at -3e38 so that no `-inf` reaches `exp()` under `-cl-finite-math-only`.
- **Effect:** `measured-here`. Before: llama-bench decode at 0 / 4,096 / 16,384 tokens, coder A770 43.4 / 13.6 / 4.2 t/s, dense 27B B60 19.2 / 12.8 / 6.4. After: coder 47.8 / 47.3 / 43.5, dense 19.75 / 19.39 / 17.95. Kernel, head size 256, 24 heads on 2 KV heads: 4,096 keys 4,159 -> 96 us (A770), 1,560 -> 57 us (B60); 16,384 keys 21,315 -> 396 us and 6,514 -> 270 us. A 2-row MTP verify had gone to ggml's prefill kernel at 3.5 ms a call, 61 % of device time of a speculative run on the coder; with up to 8 rows on this kernel, llama.cpp's draft-mtp loop gave the coder 56.9 / 68.1 / 67.6 t/s with 1 / 2 / 3 drafts (24.1 / 32.0 / 35.4 before). KL, one token per ubatch, coder 0.007034 -> 0.006986.
- **Applies to:** Both cards; up to 8 query rows, f16 K/V (quantized K/V in the next entry), head size a multiple of 128 (256 measured).
- **Default:** on (`GGML_OPENCL_FA_INTEL_SPLIT=0` keeps the basic kernel).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl`, `flash_attn_f32_f16_q1_split_intel`, merged by `flash_attn_f32_merge`. (0007)
- **Switch:** `GGML_OPENCL_FA_INTEL_SPLIT=0`, `GGML_OPENCL_FA_INTEL_KV_PER_SPLIT` (128, by depth since 0097), `GGML_OPENCL_FA_INTEL_OPTS`.
- **Source:** the split-KV flash-decoding idea from ggml's own split kernel; the lane layout is arcint's.

### Split size of decode attention by depth and by card

The per-head Intel decode kernel took 128 keys per split at any depth, and the switch had never been searched. A grid of 16 to 512 keys per split at 0 to 32,768 keys of depth (llama-bench decode, both cards, Llama 2 at f16 and q8_0:q4_0, the coder on the A770 and the dense 27B on the B60) found: short KVs want short splits (at depth 0, 16 keys beat 128 by 11 and 7 % on Llama 2 f16, 21 and 14 % with q8_0:q4_0, 14 % on the coder, 2 % on the 27B), deep f16 wants long ones (512 keys beat 128 by 8 and 10 % at 16,384), quantized K/V stays near 128 from 4,096 on. The rule: f16 takes n_kv/16 in 16-512 keys; quantized takes n_kv/32 in 16-128 (0097), refined by card in 0098: the B60 n_kv/32 up to 512, the A770 n_kv/64 in 16-64 and 128 from 16,384 keys. The GQA kernel uses 64 keys below 2,048 keys of context and 512 from there. 0098 also fixes the plain K-split kernel's work-group on Intel (it ran 64 items with FA_SG 32, so keys past the first 32 were visited twice); no served model has a head size that reaches it.
- **Effect:** `measured-here`, 0097 against the then-deployed build (arcint commit 31e021e4), one window. Llama 2 7B served: A770 f16 63.57 -> 70.22 t/s, q8_0:q4_0 54.89 -> 66.06; B60 f16 75.77 -> 80.70, q8_0:q4_0 68.63 -> 78.87; prefill level. The coder on its deployed flags 74.65 -> 82.78 t/s, a verify step 26.2 -> 24.1 ms, draft acceptance 72.95 -> 71.9 %. Agent equal (47.71 -> 47.76 t/s). 0098's llama-bench rule grid (q8_0:q4_0, not the release gate): Llama 2 on the B60 at 16,384 keys 36.52 -> 41.26 t/s, at 8,192 keys 50.52 -> 53.92. 0098's gate against the 1.5.1 package, served with the deployed flags: coder at ~3,300 keys 75.91 -> 78.13 t/s; agent at ~300 keys 48.56 -> 48.04 t/s (inside the spread). KL 0.000431 in all arms (Llama 2, one token a ubatch). Flash-Next's A/B was red on its variance rule only, accepted by the operator.
- **Applies to:** Both cards; the per-head decode kernel (all head sizes it builds) and the GQA kernel (Xe2).
- **Default:** on (a value in `GGML_OPENCL_FA_INTEL_KV_PER_SPLIT` fixes it).
- **Location:** `flash_attn_f32_f16_q1_split_intel` and `flash_attn_gqa_dpas.cl`; split rule in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0097, 0098)
- **Switch:** `GGML_OPENCL_FA_INTEL_KV_PER_SPLIT`, `GGML_OPENCL_FA_GQA_KV_PER_SPLIT` (a value fixes it).
- **Source:** arcint's own.

### Attention over a quantized KV cache: dequantize while staging

K and V as q8_0 or q4_0 (pairs 8:8, 8:4, 4:4). The prompt kernel is the same DPAS kernel built three more times with `-DKT -DVT`; only the staging load changes, the tiles go to local memory as f16 either way. The decode and verify kernels are the K-split decode with a loader in place of the f16 row read (one byte a lane for q8_0, a nibble for q4_0, the block scale broadcast). The f16 decode kernel keeps its own body: built from the shared one it decoded 4 % slower on the A770. Upstream has no kernel for an asymmetric pair; every call dequantizes all of K and V to f32 on the GPU first. arcint offers 8:8 and 8:4; 4:4 misses the answer-level bar.
- **Effect:** `measured-here`, llama-bench `-fa 1 -r 2`. Stock kernels with q8_0 on the dense 27B (B60): prefill 201 t/s at 4,096 tokens, 27 t/s at 16k depth, decode 4.65 t/s at 16k. With this patch, dense B60 8:8: prefill 896 at 4,096 and 435 at 16k, decode 18.9 and 17.1 t/s (f16: 934, 474, 19.7, 17.95); coder A770 8:4: prefill 1,361 / 446, decode 42.0 / 38.0 (f16: 1,435 / 493, 47.85 / 43.65). Cache size: q8_0 is 53 % of f16, 8:4 41 %. KL against the CPU reference, f16 KV as the baseline: dense 0.004034 (f16), 0.003966 (8:8), 0.005756 (8:4), 0.007800 (4:4); coder 0.007040, 0.006996, 0.009245, 0.011329. 4:4 loses 1.2 points of top-1 on the dense model and 1.05 on the coder, so it fails the bar. The dense 27B at 131,072 context with an 8:8 cache: peak VRAM 23.06 GB, a 128,133-token prompt at 166 t/s, decode 7.5 t/s at that depth, acceptance task 10/10.
- **Applies to:** Both cards; K/V pairs q8_0:q8_0 and q8_0:q4_0 (4:4 is refused); prompt kernel head size 256 (128 after 0017), decode kernel head size a multiple of 32.
- **Default:** on when the KV is quantized; the KV default is f16 (`--llama-kv q8_0` or `q8_0:q4_0`).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas.cl` (built with `-DKT -DVT`), `kernels/flash_attn_f32_f16.cl` (the quantized variants of `flash_attn_f32_f16_q1_split_intel`). (0015)
- **Switch:** `--llama-kv q8_0` or `q8_0:q4_0`; `GGML_OPENCL_FA_DPAS_Q=0`, `GGML_OPENCL_FA_INTEL_SPLIT=0`.
- **Source:** arcint's own; KV-quantization survey in `docs/campaigns/research-kv-quantisation.md` (on the `qfndev` branch).

### One pass over K and V per KV head for verify rows (GQA decode)

For decode and MTP verify on Xe2: a work-group per (KV head, split) takes the KV head's query heads and the rows together (row m of the tile is head m mod G, token m div G), so K and V are read once instead of once per row. 8 rows a DPAS M, two sub-groups each, one per half of the output columns; V staged transposed as f16 per 32 keys; K read directly when f16 and staged when quantized. The per-head split kernel gives every (query head, row, split) its own sub-group, so 6 verify rows cost 5.4x one row at 131k keys. Routed from 4 rows; 1-3 rows stay on the split kernel. Not built on the A770 (sub-group 8).
- **Effect:** `measured-here`, B60, dense 27B geometry (24 query heads on 4, head size 256). One attention layer, f16, 6 rows: 8k keys 0.67 -> 0.48 ms, 32k 2.75 -> 1.79 ms, 131k 10.3 -> 4.83 ms. Served, agent flags (131,072 context, q8_0, MTP 5), 62,597-token prompt: decode at that depth 11.0 -> 15.6 t/s (+42 %), verify time 9.78 -> 6.65 s, prefill unchanged at 286.9 t/s. A 3-row forward got slower (29.2 -> 27.7 t/s at 32k), hence the 4-row threshold. The kernel's own split is 512 keys, 64 keys below 2,048 keys of context (0098). Forms tried at 131k, 6 rows: no local memory 8.86 ms, K and V staged 5.27, K direct and V staged 4.83. Open in the source: a 6-row call at 131k is still 2.5x a 1-row call. KL through the kernel 0.003593 -> 0.003589.
- **Applies to:** Xe2 (B60) only; sub-group 16; head size 256 (128 after 0017), no ALiBi or softcap; f16 and the quantized pairs; from 4 query rows.
- **Default:** on (`GGML_OPENCL_FA_GQA=0` turns it off).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl`, merge via `flash_attn_f32_merge`. (0016)
- **Switch:** `GGML_OPENCL_FA_GQA=0`, `GGML_OPENCL_FA_GQA_MIN_ROWS`, `GGML_OPENCL_FA_GQA_SIDES`, `GGML_OPENCL_FA_GQA_OPTS`.
- **Source:** NInfer's small-T attention kernel (`code`, `small_t_bf16.cuh` under `src/ops/attention/causal_softmax/` as read when the idea was taken; the tree has been reorganised since and the path no longer exists in it).

### Structural search of the verify kernel: S-split and the split size

The 8-row tile's two sides split Q K^T by 16-key group and exchange S through local memory (one barrier more per tile), where before both sides computed all of S; the kernel gets its own split size of 512 keys because a work-group covers a whole KV head. Prefetching the next V tile into registers was found and left off: with S-split IGC falls back to a rolled form at 2-2.5x the time.
- **Effect:** `measured-here`, B60, q8_0 KV, verify rows 4 / 6 / 8 at 32k depth: dense 27B 36.7 / 52.6 / 62.1 -> 39.4 / 64.2 / 78.5 t/s; Cydonia 24B 29.2 / 42.3 / 58.6 -> 36.1 / 58.8 / 76.8 t/s. Served, a 62,597-token prompt: decode at that depth 15.6 -> 16.4 t/s, acceptance task 10/10 in both. The search: exhaustive over 90 genomes best 0.863 (against the earlier kernel as 1.0); GA adding S-split best 0.783.
- **Applies to:** Xe2 (B60) only; the GQA verify kernel.
- **Default:** on (`GQA_NO_S_SPLIT` turns the S-split off).
- **Location:** `flash_attn_gqa_dpas.cl` (`GQA_S_SPLIT`; the prefetch gene `GQA_PF` was removed again by 0019). (0018)
- **Switch:** `GQA_NO_S_SPLIT` (via `GGML_OPENCL_FA_GQA_OPTS`), `GGML_OPENCL_FA_GQA_KV_PER_SPLIT`.
- **Source:** arcint's own search, `tools/kq_tune/fagqa.py`.

### Head size 128 on the XMX attention kernels

The prompt and verify kernels built a second time with `-DDK=128`, for f16 and the quantized pairs, routed by head size. For Mistral-Small-3.2-class 24B models (32 query heads on 8 KV heads, all layers full attention).
- **Effect:** `measured-here`, B60, Cydonia 24B Q4_K_M, llama-bench against the build where head size 128 took the upstream kernels: f16 prefill 512 1,099 -> 1,407 t/s, 4,096 tokens 472 -> 1,225 t/s, decode 25.2 unchanged; q8_0 prefill 512 at 16k depth 168 -> 451 t/s. Served at 98,304 context with q8_0 KV: an 89,265-token prompt at 221 t/s, decode 8.4 t/s at that depth.
- **Applies to:** Both cards for the prompt kernel; Xe2 for the verify kernel; f16 and quantized pairs; 32 query heads on 8 KV heads measured.
- **Default:** on.
- **Location:** `flash_attn_dpas.cl`, `flash_attn_gqa_dpas.cl` (the `DK=128` builds). (0017)
- **Switch:** `GGML_OPENCL_FA_GQA`, `GGML_OPENCL_FA_GQA_MIN_ROWS`, `GGML_OPENCL_FA_GQA_OPTS`, `GGML_OPENCL_FA_DPAS`.
- **Source:** arcint's own.

### Prompt attention on the XMX units

Attention for more than 8 query rows (f32 Q, f16 K/V, head size 256, f16 mask) with S = Q K^T and O += P V on DPAS and an f32 online softmax. On Xe2 (sub-group 16) a work-group of 16 sub-groups takes 128 query rows and a 32-key tile staged in local memory (V transposed, key fastest so that neighbouring work-items store neighbouring halves); S's result layout (lane = key, component = row) is already P's a layout. On Xe-HPG (sub-group 8) a pair of sub-groups shares 8 query rows, each owning half of the 256 output columns, and P is re-laid from S by shuffles (which must run in every lane: under a branch they read inactive lanes and produce NaN). The work-group stops after the last key any of its rows sees, each row's masked tail scanned backwards. The row sums add the f16-rounded probabilities that P V multiplies.
- **Effect:** `measured-here`, test-backend-ops, 512 rows x 4,096 keys, 24 heads on 4, f16 mask: B60 149.9 -> 8.2 ms (6.3 TFLOPS), A770 167.0 -> 22.6 ms (2.3 TFLOPS), against ggml's `flash_attn_f32_f16.cl` (344 and 309 GFLOPS; at 4,096 tokens of context it was 66 % of the coder's prefill). llama-bench prefill on the version before the review: dense B60 525 -> 633 t/s at 512 tokens, 145 -> 540 at 512 after 4,096; coder A770 1,285 -> 1,608, 350 -> 1,032 and 602 -> 1,322 at 4,096 tokens. KL coder 0.007059 -> 0.007043, dense 0.003559 unchanged.
- **Applies to:** Both cards (sub-group 16 and 8 variants); more than 8 query rows, f32 Q, f16 K/V, head size 256 (128 after 0017), f16 mask, no sinks, ALiBi or softcap.
- **Default:** on (`GGML_OPENCL_FA_DPAS=0` keeps ggml's kernel).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas.cl`. (0008)
- **Switch:** `GGML_OPENCL_FA_DPAS=0`, `GGML_OPENCL_FA_DPAS_OPTS`.
- **Source:** arcint's own.

### Quantized prompt K/V converted to f16 block by block, flash state carried across blocks

The quantized builds of the DPAS prompt kernel dequantize K/V in every work-group, so each value is converted once per query head and query block (24 times in a 512-row chunk of the dense 27B), adding 527 instructions to a tile's main loop (1,893 against the f16 build's 1,366, offline ISA). Instead, convert K/V to f16 in blocks of keys and run the f16 kernel on each block, carrying the online-softmax state (O, m, l per lane, f32, in the registers' own layout) between blocks. Deviation from the reference: llama.cpp's CUDA backend converts the whole K/V view once per call; at the dense agent's 245,760 tokens that is 960 MiB, more than the unit's margin, so the conversion here runs 16,384 keys at a time (two 32 MiB scratch buffers plus 14 MB of carried state at 512 rows). Loading O row by row spilled 1,408 bytes and IGC then recompiled the kernel with rolled loops, 4x slower; vector loads straight into the accumulators do not spill.
- **Effect:** `measured-here`, B60. Kernel, 512 rows, q8_0:q4_0: 8,192 keys 14.6 -> 12.3 ms, 65,536 keys 111.9 -> 92.5 ms (f16: 8.3 and 63.0). Served, dense agent's flags, q8_0:q4_0, 58,269-token prompt: prefill 425.9 -> 479.1 t/s, attention 75.2 -> 57.7 s plus 2.2 s of conversion (the f16-KV arm: 498.8). At 132,874 tokens and n_ctx 245,760, 249.0 -> 292.3 t/s (+17 %). KL identical to six digits with the path, with 64-key blocks, and without (0.005950); the red case that drops the carried sums gives 4.4957.
- **Applies to:** Xe2 (B60) only (sub-group 16); quantized K/V prompt attention.
- **Default:** on (`GGML_OPENCL_FA_DPAS_F16KV=0` keeps the quantized kernels; `GGML_OPENCL_FA_DPAS_Q=0` turns it off too).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas.cl` (`FA_CARRY`), `flash_attn_kv_to_f16` (a work-item per 8 values, using the kernel's own `load8`). (0037)
- **Switch:** `GGML_OPENCL_FA_DPAS_F16KV=0`, `GGML_OPENCL_FA_DPAS_F16KV_BLOCK`; red `-DFA_CARRY_RED`.
- **Source:** llama.cpp CUDA `launch_fattn` (`need_f16_K/V`); the blockwise carry is arcint's own.

### Prompt attention without local memory: 2D block reads, prefetch, mask classes (Xe2)

A second prompt kernel for the B60. K arrives as the b operand of Q K^T by a transposed 2D block read and V as the b operand of P V by a VNNI-transforming one, straight from global memory into registers, the next tile prefetched, no local memory, no barriers. 8 query rows a sub-group with all output columns, 32 sub-groups a work-group, 32-key tiles. Q is converted to f16 once per call (scaled) and re-read by 2D reads, since holding it in registers leaves no room at head size 256. A mask pre-pass writes one byte per 8 rows x 32 keys (mixed, all zero, nothing visible): all-zero blocks skip the mask reads and blocks with nothing visible are skipped by that sub-group, which is possible because the loop has no barrier. IGC's value-returning builtins (`__builtin_IB_subgroup_block_read_flat_*`) do the reads: the extension's functions write a private array, which in this kernel put the accumulators in scratch memory with no spill reported.
- **Effect:** `measured-here`, B60, 512 rows, 24 query heads on 4, causal mask. f16 8,192 keys: 8.21 ms (0008) -> 5.43 ms (without classes) -> 4.02 ms; 65,536 keys: 63.0 ms (13.1 TFLOPS) -> 43.1 -> 30.6 ms (26.9 TFLOPS); q8_0:q4_0 65,536 keys 92.2 -> 56.5 ms. Main loop 972 instructions per 64 DPAS against 1,366. Served, dense agent's flags (q8_0:q4_0), against 0037 in the same window: prefill at 58,269 tokens 479.1 -> 641.2 t/s, at 132,874 tokens 292.3 -> 451.6, at 231,786 tokens 192.0 -> 324.2. Against arcint 0.7.3 (0001-0035), which includes the gain of 0036 and 0037: 425.9, 249.0 and 157.2 t/s (the last is the 0.7.3 record, taken at 236,938 tokens). Decode unchanged (16.2 t/s at 128k). KL unchanged at 0.005950; reds 0.6815 (keys off) and 0.4797 (mixed blocks as all zero). Later, 0041 read Q in pairs of chunks (8 reads a tile instead of 32; ablation had put the Q reads at about 9 of 31 ms), f16 65,536 keys 30.46 -> 28.41 ms, bit-identical because each DPAS chain keeps its accumulation order.
- **Applies to:** Xe2 (B60) only; head sizes 256 and 128; f16 K/V and 0037's f16 blocks; 64-byte aligned heads and a row pitch of a multiple of 16 bytes, otherwise the old kernel runs.
- **Default:** on (`GGML_OPENCL_FA_DPAS2=0`, `GGML_OPENCL_FA_DPAS2_MCLS=0`).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas2.cl`, `flash_attn_q_to_f16`, `flash_attn_mask_cls`. (0038, 0041)
- **Switch:** `GGML_OPENCL_FA_DPAS2=0`, `GGML_OPENCL_FA_DPAS2_MCLS=0`, `GGML_OPENCL_FA_DPAS2_NSG`; red `-DFA2_RED`, `-DFA2_MCLS_RED`; `-DFA2_Q1` keeps one-chunk reads.
- **Source:** vllm-xpu-kernels `csrc/xpu/attn/fmha_utils.hpp` `chunk_policy_head256` and sycl-tla `xe_fmha_fwd_mainloop.hpp` (`code`); the mask classes after llama.cpp's Vulkan `flash_attn_mask_opt.comp`.

### Split the keys across work-groups when the row tiles leave a wave part empty

A work-group takes 256 query rows of one head and a Xe core holds one work-group (32 sub-groups at 256 GRF), so the B60 runs 20 at a time and time is the number of waves times 9.4 ms (rows 256 / 512 / 768 / 1,280 / 2,048 -> 2 / 3 / 4 / 6 / 10 waves; 18.9 / 28.2 / 37.6 / 56.9 / 93.9 ms at 65,536 keys). The agent's 512-token ubatch gives 48 work-groups, so the third wave is 40 % full. The kernel therefore splits a call's keys across work-groups by the reference's rule (the fewest splits whose waves are at least 95 % full, else the fullest, at most 16), with the wave size taken from `CL_DEVICE_MAX_COMPUTE_UNITS`, each split keeping its own online-softmax state in the carry layout and a combine kernel merging them.
- **Effect:** `measured-here`, 2026-10-14, B60. Kernel, f16 65,536 keys: 512 rows 28.17 -> 23.89 ms (34.5 TFLOPS), 256 rows 18.9 -> 12.34 ms; 768 rows and more unchanged. Served, dense agent's flags at 245,760, q8_0:q4_0: prefill at 54,895 tokens 668.0 -> 690.6 t/s (+3.4 %), at 138,028 tokens 461.6 -> 490.1 (+6.2 %), at 228,120 tokens 342.9 -> 370.1 (+7.9 %). Peak VRAM 24,271 -> 24,291 MiB. The merge reorders softmax sums: the acceptance task stayed 10/10 but is a different program (md5 changed). A first KL run with the default batch (2,048 rows, one split chosen) measured nothing, since the red case read the same as the baseline.
- **Applies to:** Xe2 (B60) only, the no-local-memory prompt kernel; mostly for 256 to 512 rows per call.
- **Default:** on (`GGML_OPENCL_FA_DPAS2_SPLIT=1` turns it off).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas2.cl`, `flash_attn_split_combine`. (0045)
- **Switch:** `GGML_OPENCL_FA_DPAS2_SPLIT` (1 = off), `GGML_OPENCL_FA_DPAS2_WAVE`; red `-DFA_SPLIT_RED`.
- **Source:** llama.cpp CUDA `parallel_blocks` in `launch_fattn` and `flash_attn_combine_results` (`code`, `ggml-cuda/fattn-common.cuh` at `bed0a85`); vLLM uses the same 256-row tile without a split but with thousands-of-tokens chunks.

### Occupancy on Xe-HPG: take Q out of local memory and double the sub-groups

The sub-group-8 prompt kernel (A770) staged Q in local memory next to the K and V tiles: 58 KB of the core's 64, so a core ran one 8-thread work-group, half a thread per EU, with nothing to hide latency. The A770 has no 2D block I/O and head size 256 does not leave enough registers to reuse K/V reads, so the local-memory K/V sharing stays. Q is converted to f16 once per call in the a-operand blocks (`flash_attn_q_to_f16_blk`) and read with `intel_sub_group_block_read8`; local memory falls to 45 KB and the work-group takes 16 sub-groups. Moving Q out of local memory alone changed nothing; the work-group size did.
- **Effect:** `measured-here`, 2026-10-13, A770 at its pinned 2,000 MHz, 512 rows, causal: q8_0:q4_0 32,768 keys 142.5 ms -> 70.8 ms (16 sub-groups; 144.2 with 8 and 73.2 with 32); f16 32,768 keys 108.5 -> 83.0 ms (16), 66.7 (32). Served, coder (q8_0:q4_0, n_ctx 98,304, MTP 4): prefill at 44,329 tokens 476.6 -> 685.5 t/s (attention 55.6 -> 27.1 s), at 56,800 tokens 410.1 -> 616.2 t/s; decode unchanged. KL 0.026908 in both arms; red 1.1127. The acceptance task was 9/10 greedy in both arms with the identical answer. A follow-up (0040) turns q8_0 / q4_0 codes into halves with word operations (byte-typed operations in the main loop 282 -> 10): about 1 % at 16 sub-groups, because staging is load-bound there.
- **Applies to:** Xe-HPG (A770) only; the sub-group-8 prompt kernel, f16 and quantized K/V.
- **Default:** on (`GGML_OPENCL_FA_SG8_QG=0` restores the old build).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas.cl` (`FA_QG`, `load8`), `flash_attn_q_to_f16_blk`. (0039, 0040)
- **Switch:** `GGML_OPENCL_FA_SG8_QG=0`, `GGML_OPENCL_FA_SG8_NSG`; red `-DFA_QG_RED`; `-DFA_LOAD8_BYTES`.
- **Source:** arcint's own; the code-to-half trick is 0006's.

### V staged as key pairs, no shuffles (Xe-HPG prompt attention)

For f16 V the A770 prompt kernel stages V a key pair per work-item: two keys' 8 values packed into 8 uints (the P V b operand's pairs) and stored as 8 dwords, one per dim, instead of 16 scalar halves. The pair is keys j and j + 8 of a 16-key group, so P's a operand comes straight from S's two 8-key halves without shuffles. Quantized V keeps the old staging: the pair form exposes the dequantization that the old loop hid under its loads, and neither packed decoding nor batched loads won it back.
- **Effect:** `measured-here`, 2026-10-14, A770 at 2,000 MHz, f16, 512 rows, 65,536 keys: 166.1 -> 128.6 ms (forms tried on the way: V in the b operand's blocked layout 155.6 ms with 1,760 B of spill; key pairs (2i, 2i+1) 137.8 ms; pairs (j, j+8) 128.6-129.0 ms). Served, coder with f16 KV at 49,152 context, MTP 4: the 40,841-token needle 661.6-665.8 -> 734.4-741.2 t/s (+11 %). The deployed coder runs q8_0:q4_0 at 98,304 and does not take this path; the source notes the f16-KV arm is no better than that today (725.8 t/s on the same needle) and retracts an earlier comparison against a stale number.
- **Applies to:** Xe-HPG (A770) only; f16 V only (quantized V keeps the old staging).
- **Default:** on (`GGML_OPENCL_FA_SG8_VBLK=0`).
- **Location:** `ggml/src/ggml-opencl/kernels/flash_attn_dpas.cl` (`FA_VBLK`). (0048)
- **Switch:** `GGML_OPENCL_FA_SG8_VBLK=0`; red `-DFA_VBLK_RED`.
- **Source:** arcint's own.

### Kernel rewrites by LLM mutation agents, accepted by test and timing

Two agents rewrote one kernel each overnight; every candidate was compiled, tested and timed on the B60 before the next. Kept changes, prompt kernel: Q in registers instead of local memory, K and V sharing one local buffer (96 KB of local memory per work-group became 16 KB), early loads, two DPAS chains per key group, and one sub-group vote that skips the row reductions and the O/l rescale when no row's running maximum can rise (exact: the skipped factor is 1). Verify kernel: 64-value staging units, S exchange laid out lane-major, `fmax(s + mask, M_INIT)`. For the A770 (sub-group 8) a second pass: bank-conflict padding of K and V, the reciprocal of l precomputed (which moved IGC's register allocation as a whole, 12,093 -> 9,744 assembly lines, pp512 at 16k +14 % in one step), a double-buffered weight tile in the K-quant GEMM.
- **Effect:** `measured-here`. B60, dense 27B Q4_K_M, q8_0 KV, 32k keys deep, llama-bench: pp512 297.5 -> 428.7 t/s (+44.1 %), pp6 64.2 -> 71.2 (+10.9 %), pp8 78.5 -> 87.3 (+11.2 %). Served, 62,597-token prompt: prefill 287 -> 408 t/s (+42 %), decode after it 18.7 -> 21.6 t/s (+15.5 %, MTP 5). KL prompt path 0.003966 -> 0.004099. The acceptance task at T=0 scored 8/10 for both builds in that window; the source states a greedy score is one trajectory. A770, coder, q8_0: pp512 after 16,384 496.2 -> 692.3 t/s (+39.5 %), after 4,096 1,116.2 -> 1,346.8 (+20.7 %), after 0 +3.1 %; served 15,004-token prompt 762 -> 1,008 t/s (+32 %), decode 41.5 -> 39.5 t/s at a lower draft acceptance (55.6 -> 51.3 %). One review fix (a guard around `native_exp`) was dropped again after it put 896 bytes of spill in the q8_0 prompt build and cost 15 % of pp512.
- **Applies to:** B60 sub-group-16 prompt and verify kernels; A770 sub-group-8 prompt kernel and K-quant GEMM tiles (every A770 change is under `#if SG == 8`).
- **Default:** on (kernel source).
- **Location:** `flash_attn_dpas.cl`, `flash_attn_gqa_dpas.cl`, `mul_mm_kq_f16.cl` (the `#if SG == 8` blocks). (0019, 0020)
- **Switch:** none new.
- **Source:** arcint's own (campaign `kernel-autotune-ga`, "LLM mutation search", on the `qfndev` branch).

### KV rotations as a fast Walsh-Hadamard transform

The KV cache rotations are `MUL_MAT` nodes carrying a hint that src0 is the orthonormal Sylvester Hadamard matrix. The CPU and CUDA backends compute them as a fast Walsh-Hadamard transform of each src1 row; the OpenCL backend ignored the hint and ran dense F32 products (the skinny split-K path, 24,615 products and 750 ms of a 22.3 s B60 decode window, about 2.35 ms per verify step). The kernel takes one row per 16-wide sub-group, the stages below 16 by sub-group shuffles and the rest in registers, for widths 64 to 256; 512 and above keep the dense product (the 512 kernel spilled at SIMD16 on the A770).
- **Effect:** `measured-here`, same binary with the switch off against on, one window per card. Agent (B60, q8_0:q4_0 KV, 262,144 tokens, replay, MTP 5): verify step 59.5 -> 57.5 ms, decode 47.7 -> 50.0 t/s. Coder (A770): verify step 35.0 -> 33.1 ms; in the larger window-76 set decode +4.0 % (72.26 / 71.83 -> 74.47 / 75.33 t/s) with draft acceptance unchanged (72.0 -> 72.1 %), agent +2.7 % (46.23 / 46.38 -> 47.59 / 47.51). KL of the agent 0.004946 -> 0.004095 at `-ub 512`, 0.006860 -> 0.006872 at `-ub 1`. A first window saw lower draft acceptance on the coder on three prompts (e.g. 39.2 -> 33.7 %) and the release A/B went red on one answer's decode (53.75 -> 51.50 t/s); the source attributes that to one answer's text moving with the rounding, and the operator accepted the cell.
- **Applies to:** Intel devices that can require a 16-wide sub-group (`cl_intel_required_subgroup_size`); F32 products with a Hadamard hint, widths 64 to 256; 512 and above and F16 src1 keep the dense product.
- **Default:** on (`GGML_OPENCL_FWHT=0` keeps the dense product).
- **Location:** `ggml/src/ggml-opencl/kernels/fwht_f32.cl`, `kernel_fwht_f32_64`, `kernel_fwht_f32_128`, `kernel_fwht_f32_256`; test `MUL_MAT_HADAMARD`. (0095)
- **Switch:** `GGML_OPENCL_FWHT=0`.
- **Source:** the CPU backend's `ggml_compute_forward_fwht` and CUDA's `ggml-cuda/fwht.cu` (`code`).

### Flash attention at head size 512 on Intel, as two V halves

ggml declines `FLASH_ATTN_EXT` at head size 512 on every Intel device, so Gemma 4's seven full-attention layers went to the CPU (q8_0:q4_0 KV, with their Q, K, V and mask copied) or, with an f16 cache, switched flash attention off for the whole model. The prompt kernels now take a V column split: two work-groups a head, each computing S and the softmax over all 512 dimensions and P V for 256 of V's columns, so a sub-group's accumulators stay as at head size 256. Decode and verify rows take the existing split kernel at 512.
- **Effect:** `measured-here`, 2026-11-07, B60, Gemma 4 E4B, ctx 16,384, a 6.5k-token prompt, against 1.6.0. q8_0:q4_0 KV: prefill 176-246 -> 1,905-2,053 t/s, decode about 65 -> 75.5 t/s. f16 KV: prefill 1,276-1,294 -> 2,113-2,381 t/s, decode 79 -> 81 t/s, thinking answers broken in 2 of 3 runs before and right in 3 of 3 after. KL against the CPU reference: f16 ubatch 512 0.0079 (A770 0.0095), q8_0:q4_0 0.019 (1.6.0-era 0.0205), q8_0:q4_0 ubatch 1 0.011. The head-512 filtered op run passes 82 cases on both cards; `-DFA_DVH_RED` fails 36 head-512 prompt cases on each; the full `FLASH_ATTN_EXT` suite passes 3,078 with 0 fail on both. Granite 4.2 8B is unchanged.
- **Applies to:** Intel; head size 512, f32 Q, f16 K/V or q8_0:q8_0, q8_0:q4_0, q4_0:q4_0, an f16 mask or none, no sinks, ALiBi or softcap.
- **Default:** on (`GGML_OPENCL_FA_DK512=0` keeps ggml's decline).
- **Location:** `flash_attn_dpas2.cl`, `flash_attn_dpas.cl` (the `DVH` build option), `supports_op` in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0106)
- **Switch:** `GGML_OPENCL_FA_DK512`.
- **Source:** the CUDA backend's head-512 configurations (`fattn-mma-f16.cuh`, `code`) compute all 512 columns in one block; here S is computed once a half, 1.5x the minimal DPAS work.

### The f16 matvec's local staging, for 32-lane sub-groups

`kernel_mul_mat_f16_f32_l4_x8`, the f16 matvec for one column a batch at K 128-256, staged its column with one float4 a sub-group lane, written for Adreno's 64 lanes; on Intel's 32 the second half of a K 256 column was never written. Every work-item of the work-group now stages its share. This is a defect fix in upstream's kernel, and it only bites attention without flash attention at decode.
- **Effect:** `measured-here`, 2026-11-07. Gemma 4 E4B Q4_K_M, f16 KV, flash attention off, c 2,048, CPU reference, B60: ubatch 1 KL 3.37, top-1 35.5 %, PPL 129 against 14.1 before; 0.0044 and top-1 98.5 % after, on both cards. `test-backend-ops` on Gemma's own graph ops failed at `kq-0` with ERR 1.04 (B60) and 0.997 (A770). The tree without the patch fails the K 256 one-column cases (3 FAIL a card); with it 41/41.
- **Applies to:** Intel; batched f16 x f32 products of one column, K 128-256.
- **Default:** on.
- **Location:** `mul_mv_f16_f32_l4.cl`, `kernel_mul_mat_f16_f32_l4_x8`. (0107)
- **Switch:** none; the opt-in `GGML_OPENCL_MM_KQ_PAIR` and `GGML_OPENCL_MM_KQ_GQA` forms are refused off Adreno.
- **Source:** arcint's own, a fix in upstream's kernel (`code`).

## 7. GatedDeltaNet and recurrent state

The linear-attention layers of the Qwen3.5/3.6/3.8 hybrids: the recurrence kernel, the rollback a rejected draft needs, and the copies around it.

### GatedDeltaNet recurrence with the state in registers

`GGML_OP_GATED_DELTA_NET` (the linear-attention layers of Qwen3.5/3.6/3.8) at head size 128: lanes along the key dimension, each sub-group holding R state rows in registers, reductions by sub-group shuffles. ggml's `gated_delta_net.cl` keeps 96 floats per SIMD32 lane (the state spills every token) and reduces through local memory. For a prompt, q/k/v/g/beta come in chunks of TC tokens staged in local memory with the next chunk loaded into registers while the current one is computed, so q/k are read once per work-group; for one token, one sub-group per R rows with the next token's inputs prefetched.
- **Effect:** `measured-here`, harness against a CPU port of ggml-cpu's op, 512 tokens: A770 (32 heads) 451 us against 12,042 us for ggml's kernel, B60 (48 heads) 875 against 5,692; one token 82 -> 12.7 us (A770), 51 -> 5.5 us (B60). llama-bench pp512: coder 600.3 -> 1,005.6 t/s (decode 37.4 -> 38.8), dense 409.2 -> 493.8 (17.97 -> 18.20). OpenVINO's own shape (one sub-group a work-group, R 4) ran 1,230 us on the A770 and spilled at 128 GRF; a butterfly reduction of the R rows was slower on both cards. KL: dense 0.003560 -> 0.003562, coder 0.00696 -> 0.00700.
- **Applies to:** Both cards; head size 128, scalar gate.
- **Default:** on (`GGML_OPENCL_GDN_INTEL=0` keeps ggml's kernel).
- **Location:** `ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl`. (0003)
- **Switch:** `GGML_OPENCL_GDN_INTEL=0`, or `GGML_OPENCL_GDN_INTEL=Rtg,Rpp,NSG,TC`.
- **Source:** OpenVINO's `paged_gated_delta_net_opt.cl` (`code`); the chunk pipelining from Strata's `gdn_rec_cols_pipe_kernel` in `src/prefill/kernels.cu` (`code`).

### Verify batches use the one-token GatedDeltaNet kernel

The Intel GDN dispatch sent every batch over one token to the prompt kernel, which stages chunks of 16 tokens in local memory, so a 5-token verify paid the staging for one partial chunk: 35 us a launch against 11.7 us for the one-token generation kernel, which already loops over the tokens with the state in registers and writes the K snapshots a token. Batches up to 8 tokens now go to it.
- **Effect:** `measured-here`. Coder GDN time a verify step 1.044 -> 0.674 ms, decode +1.1 to +1.2 %, same answers; with the copy-run entry on the dense agent (B60, MTP 5) decode +2.3 to +2.4 %, prefill equal; Flash-Next with the router and copy-run entries: prefill +0.7 %, decode +1.2 %.
- **Applies to:** Both cards; batches up to 8 tokens.
- **Default:** on (`GGML_OPENCL_GDN_TG_MAX`, 8).
- **Location:** `kernel_gated_delta_net_tg` dispatch in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0066)
- **Switch:** `GGML_OPENCL_GDN_TG_MAX=N` (1 = before).
- **Source:** arcint's own.

### GatedDeltaNet writes its rollback snapshots straight into the recurrent cache

In the rollback path (K = n_rs_seq + 1 > 1) the op wrote the attention output and K state snapshots into its own output, and a `ggml_cpy` then moved the snapshots into the recurrent cache: one more kernel and a copy of K x 128 x 128 x heads floats per GDN layer every verify. The Intel GDN kernels now take the snapshot destination (buffer, offset, slot stride); when the node's next real node is that `CPY`, with the shapes and strides the builder gives it, they write into the cache rows and the `CPY` is skipped.
- **Effect:** `measured-here`, 2026-10-14, A B A B. Dense agent (B60, MTP 5, K = 6): 37.3 / 39.9 / 37.7 -> 39.2-39.3 / 42.0 / 39.6 t/s (+5.1-5.3 %); coder (A770, MTP 4, K = 5): 49.7 / 46.2 / 42.9 -> 50.7 / 47.1 / 43.6 (+1.8-2.0 %); Flash-Next (B60, MTP 2, K = 3): 32.2 / 37.5 / 35.9 -> 32.6-32.7 / 38.2 / 36.6 (+1.2-1.9 %). Every md5 and draft-acceptance figure equal in both arms. The scheduler's host time does not move: the saving is the copy on the card.
- **Applies to:** Both cards; the rollback path (K = n_rs_seq + 1 > 1) when the node's next real node is the copy.
- **Default:** on (`GGML_OPENCL_FUSE_0050=0` keeps the copy).
- **Location:** `ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl` (the `_tg` and `_pp` kernels); the fold in `ggml-opencl.cpp`. (0050)
- **Switch:** `GGML_OPENCL_FUSE_0050=0`; red `GGML_OPENCL_FUSE_0050_RED=1`.
- **Source:** vllm-xpu-kernels, whose delta stage updates the state in place (`code`, `csrc/xpu/gdn_attn/gated_delta_rule.hpp`).

### GatedDeltaNet gating computed inside the kernel

The callers form the gate as `softplus(alpha + dt_bias) * A` and beta as `sigmoid(b)`: four small ops ahead of each GDN node. The backend finds the chain in a window before each GDN node (every intermediate used once), skips the ADD, SOFTPLUS, MUL and SIGMOID, and hands the kernel the raw projections with dt and A per head. The kernel repeats the separate kernels' expressions with an always-0 XOR between the steps so the roundings match. The graph builder attaches alpha and b as extra sources of the GDN node, because the allocator would otherwise reuse alpha's memory between the ADD and the GDN; an overlap check had refused 36 of 36 folds before that.
- **Effect:** `measured-here`, 2026-10-14, A B A B. Dense agent 39.2 / 42.0 / 39.6 -> 39.4 / 42.1 / 39.7 t/s (+0.3 %); coder 50.7 / 47.0 / 43.6 -> 51.2 / 47.6 / 44.1 (+0.9-1.2 %); Flash-Next neutral. Flash-Next's verify graph spends 0.3-0.4 ms less enqueueing, but its roughly 63 ms round is not host-bound. Texts bit-identical. The first build's helpers sat inside `#ifdef GDN_PP`, the generation program did not compile and the backend silently fell back to the generic kernel, so its "42/42" and md5s measured the generic kernel. The source notes upstream's `graph_optimize` hook with `add_alloc_dep` (used by CUDA) as the cleaner way to keep alpha and b alive at a rebase.
- **Applies to:** Both cards; the softplus-add-mul and sigmoid chain before a GDN node; a model whose beta is `reshape(sigmoid(b))` (qwen3next) folds nothing.
- **Default:** on (bit 2 of `GGML_OPENCL_FUSE_0050`, default 3).
- **Location:** `gated_delta_net_intel.cl`, `ggml-opencl.cpp`; `gdn_keep_gating_inputs` in `src/models/delta-net-base.cpp` (later `src/models/gdn-gate-inputs.h`). (0051)
- **Switch:** `GGML_OPENCL_FUSE_0050` bit 2 (default 3), `GGML_OPENCL_FUSE_0051_PART`; red `GGML_OPENCL_FUSE_0051_RED=1`.
- **Source:** vllm-xpu-kernels, which takes a and b raw with A_log and dt_bias (`code`, `csrc/xpu/gdn_attn/gated_delta_rule.hpp`).

### Rollback by replay: one base state and small per-token records

The gated delta-net rolls a recurrent sequence back by keeping whole state snapshots (`n_rs_seq + 1` planes per layer). Replay keeps one base state and, for each token after it, a record of the step's own small vectors (the value delta, the key, and the gate: `R = H * (2 * S_v + 1)` floats per token a layer instead of a plane `S_v * S_v * H`) and re-applies them onto the base on rollback. The new op `GGML_OP_GATED_DELTA_NET_REPLAY` appends to the enum so no existing op number moves; its CPU reference is bit-identical to the snapshot path's per-token step; the Intel kernel has a `GDN_RPL` build with a replay prologue and a record-writing epilogue. Rollback reduces the pending record count; a state save writes the records with a distinct header. A hybrid model needs the recurrent-input helper in all three hybrid input classes (0091): a first version filled the controls only for the plain recurrent input and the first draft rollback was refused (HTTP 500).
- **Effect:** `measured-here`, agent on the B60. Resident VRAM at a 115,395-token prompt 23,725 against 24,301 MiB (-576 MiB); decode 20.96 against 21.03 t/s; at that prompt prefill took 214.9 s against 211.5 s. In an earlier comparison (replay off against on, same window) prefill at 55k went 715.4 -> 690.5 t/s (-3.5 %) and at 138k 501.1 -> 494.1 (-1.4 %). The saved memory buys the model's trained length: at `--n-ctx 262144` with replay the unit loads at 23,256 MiB (245,760 without replay: 23,390 MiB), a needle at 40 % of a 254,970-token prompt is right (731.5 s, 348.6 t/s), peak 24,199 of 24,480 MiB, GTT flat, a verify costs the same 124.4 ms. The acceptance task is 10/10 greedy both ways but a different program. The expected saving from the shapes (`code`): agent 588 MiB, Flash-Next 216 MiB, coder 186 MiB. Replay is a per-unit choice and not a default.
- **Applies to:** Intel OpenCL; F32 state, head size 128, one sequence, MTP rollback, qwen35, qwen35moe and qwen4exp models with every recurrent layer on OpenCL; otherwise the snapshot path stays.
- **Default:** off (`LLAMA_GDN_REPLAY=1`, a per-unit choice; the sample agent unit sets it).
- **Location:** `ggml_gated_delta_net_replay` in `ggml/src/ggml.c`; `GDN_RPL` in `ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl`; `llama_memory_recurrent::rpl_reset` in `src/llama-memory-recurrent.cpp`; `rs_set_input_rpl` in `src/llama-graph.cpp`. (0086, 0087, 0091)
- **Switch:** `LLAMA_GDN_REPLAY=1` (arcint's per-unit choice).
- **Source:** the arc-b580 fork's replay (`code`, the replay op in the fork's `ggml/include/ggml.h` and its CPU reference in `ggml-cpu/ops.cpp`); its replay == snapshot test is `paper`.

### Lesson: one source slot, three meanings

The GDN node's `src[6]` was claimed three times (the raw alpha of the gating fold, the state-rows index, the replay records). With the fold on by default, `src[6]` held the F32 alpha, OpenCL `supports_op` declined the node, it fell to the CPU, and the CPU op asserted `src_rows->type == GGML_TYPE_I32`: every unit crashed with every new switch off. The fix gives the op a mode word (`op_params[1]`: gathered, rows, replay) that selects the state path in the CPU op, `supports_op` and the dispatch, and moves the gating inputs to `src[8]` and `src[9]`.
- **Effect:** `measured-here`, the crash on the units; no throughput claim. The check that follows from it (a served binary with every new switch off must answer a short prompt on each unit) is covered by the later release gates, which start each unit that way.
- **Applies to:** All devices; the GDN node's source slots.
- **Default:** fixed (the mode word is always set).
- **Location:** `GGML_GDN_MODE_*` in `ggml/include/ggml.h`; `src/models/gdn-gate-inputs.h`. (0081)
- **Switch:** none.
- **Source:** arcint's own.

### Row-parallel strided f32 copy for the recurrent-state snapshot

A same-shape f32 copy with contiguous rows of at least 256 floats and a non-contiguous destination; each work item moves one float4 of a row. The case that mattered is the MTP snapshot write of the hybrid models into a view whose planes are `mem_size` rows apart, contiguous only when every configured sequence is in the ubatch. Before it, ggml-opencl ran such copies through its generic copy, one 64-item work-group a row.
- **Effect:** `measured-here`, B60. Snapshot-sized case (786,432 floats x 6 planes into a strided view) 4,595.6 -> 95.2 us. With two sequences configured and one busy each GDN layer's write took about 4.4 ms; a verify of the dense 27B 260 ms instead of 55 ms; same 300-token answer, two lanes with one busy 10.9 -> 40.9 t/s (one lane 41.0), text byte-identical.
- **Applies to:** Both cards; same-shape f32 copies with contiguous rows of at least 256 floats and a non-contiguous destination.
- **Default:** on (`GGML_OPENCL_CPY_FLAT=0` also turns it off).
- **Location:** `kernel_cpy_f32_f32_rows` (`ggml/src/ggml-opencl/kernels/cpy.cl`). (0025)
- **Switch:** `GGML_OPENCL_CPY_FLAT=0` (the pin's own switch for flat copies) also turns this path off.
- **Source:** arcint's own.

### Run of consecutive small copies as one launch

A speculative verify writes the conv-state rollback snapshots as K = n_draft + 1 separate copies per gated-delta-net layer, each moving three floats a row and launch-bound. The backend matches a run of up to 8 consecutive f32 `CPY` nodes (VIEW no-ops skipped) whose sources and destinations are views of one tensor each with equal shapes, narrow rows and disjoint destinations, and runs them as one launch in which a work-item per row writes all K windows. The copy is exact.
- **Effect:** `measured-here`, A770 coder: the snapshots 150 launches / 0.900 ms -> 30 / 0.175 ms a step (of a 39.1 ms verify step); decode +1.8 to +1.9 %, same answers.
- **Applies to:** Both cards; up to 8 consecutive f32 copy nodes with equal shapes, narrow rows (ne0 < 32) and disjoint destinations.
- **Default:** on (`GGML_OPENCL_FUSE_CPY_RUN=0`).
- **Location:** `kernel_cpy_f32_f32_run` in `ggml/src/ggml-opencl/kernels/cpy.cl`. (0065)
- **Switch:** `GGML_OPENCL_FUSE_CPY_RUN=0`.
- **Source:** arcint's own.

## 8. Graph launches, placement and host-to-device transfers

Fewer and cheaper launches per decode step, keeping ops on the card, and the cost of small host-to-device transfers.

### Decode-step small ops resized for Intel

Three launches a decode step makes per layer were sized for Adreno. The q8_1 activation quantization ran one work-item per 32-value block (64 work-items for a 2,048-wide row); it now uses a sub-group per block with coalesced reads and sub-group reductions. The GLU kernels ran one work-group per row (a 17,408-wide decode row in one group, 15 us); each row is now split over the grid's second dimension (up to 64 work-groups per row). `kernel_mul` on a non-broadcast row used 64 work-items per row (the coder's gated shared expert at decode, 14.5 us), now up to 256.
- **Effect:** `measured-here`. Quantization 7.4 us on the B60 and 10 us on the A770 before (a trace of the decode step); no per-kernel after-figure in the source. Gated with 0005 and 0006, see the experts entry above for the combined prefill and decode numbers.
- **Applies to:** Intel devices (both cards); other devices dispatch as before.
- **Default:** on (`GGML_OPENCL_KQ_QUANT_SG=0` keeps the old quantizer).
- **Location:** `kernel_quantize_q8_1_sg` in `ggml/src/ggml-opencl/kernels/mul_mv_kq_q8_1.cl`; the GLU kernels in `kernels/glu.cl`. (0004)
- **Switch:** `GGML_OPENCL_KQ_QUANT_SG=0`.
- **Source:** arcint's own.

### Fewer launches in a decode graph, with the same bits

A Flash-Next verify graph launched about 4,150 kernels (58.4 ms of kernel time), which at about 3 us an enqueue is about 12 ms of host time per graph. Four changes, three of them llama.cpp CUDA's fusions: the rollback snapshots of the conv state written by `cpy` without a preceding `cont`; rms_norm and scale in one kernel; ssm_conv with silu; sigmoid with mul (the GDN z gate). The traps: under `-cl-fast-relaxed-math` IGC regrouped `(x * scale) * post` to `x * (scale * post)`, one rounding fewer. KL stayed inside the bar, but the texts changed and MTP draft acceptance fell (coder 40.7 -> 36.4 % on one answer). The intermediate now passes through an XOR with a kernel argument that is always 0, which keeps the two roundings. The model for such a test is the per-fusion md5 of served answers, not KL.
- **Effect:** `measured-here`, 2026-10-14, units' flags, A B A B, three 300-token answers, every answer and every draft-acceptance figure bit-identical to the unfused arm. Flash-Next (B60) verify graph scheduler time 12,346-12,407 -> 11,666-11,678 us, decode 32.0-32.1 / 37.3-37.4 / 35.7-35.8 -> 32.2-32.3 / 37.5-37.6 / 36.0-36.1 t/s; dense agent verify graph scheduler time 8,640-8,662 -> 8,139-8,143 us, decode 37.2 / 39.8 / 37.5 -> 37.3 / 39.9 / 37.7; coder (A770) verify graph 6,695-6,699 -> 6,380-6,422 us, decode 49.4 / 45.9 / 42.5 -> 49.8 / 46.4 / 43.0.
- **Applies to:** Both cards; hybrid (gated delta-net) models for the GDN-side fusions; rms_norm plus scale, ssm_conv plus silu, sigmoid plus mul.
- **Default:** on (`GGML_OPENCL_FUSE_0049=<mask>`).
- **Location:** `ggml/src/ggml-opencl/kernels/rms_norm.cl`, `kernels/ssm_conv.cl`, `kernels/sigmoid.cl`; `build_conv_state_at` in `src/models/qwen4exp.cpp`; the fusion matcher in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0049)
- **Switch:** `GGML_OPENCL_FUSE_0049=<mask>` (1 rms_norm + scale, 2 ssm_conv + silu, 4 sigmoid + mul; 0 off); red `GGML_OPENCL_FUSE_0049_RED=1`.
- **Source:** llama.cpp CUDA `ggml_cuda_can_fuse` in `ggml-cuda.cu` (`code`).

### Gather from a quantized cache on the card (GET_ROWS for q8_0)

OpenCL `GET_ROWS` took f32, f16, i32 and plain Q4_0 only, so every gather from a q8_0 tensor ran on the CPU. With `--llama-kv q8_0`, the KV type also types the QSA indexer's key cache, and the indexer gathers rows from it: the scheduler copied each layer's 8 MB indexer cache to the CPU, gathered there and copied back, in the target graph and in every MTP draft. Two layouts need two kernels: a cache written by SET_ROWS keeps ggml's plain 34-byte blocks, and a q8_0 tensor uploaded through `set_tensor` is flattened into quants and scales. The host tells them apart by the tensor's extra, and for a view the parent's decides.
- **Effect:** `measured-here`, 2026-10-13, B60, Flash-Next IQ2_XS, 131,072 context, MTP 2, bank stream, each arm's freed VRAM given to expert slots. f16 KV (9,500 MiB): needle prefill 649.8 t/s, decode 30.0 / 31.3; q8_0 before the patch (11,300 MiB): 521.5 t/s, decode 13.3 / 13.4; q8_0 with it: 617.2 t/s, decode 30.9 / 31.0. Scheduler splits on the CPU for an 80-token answer with q8_0 KV: 222 of 444 before. KL f16 KV 0.294050, q8_0 KV 0.294037. Neither the KL runs nor the served needle caught the red kernel (the indexer's top-k survives a perturbed scale), which is why the patch has a dedicated test. A later fix (0.8.1) in the source: the support check read the q8_0 kernel before it was loaded.
- **Applies to:** Both cards; q8_0 sources (plain and flattened layouts); matters when `--llama-kv q8_0` types the QSA indexer's cache (Flash-Next).
- **Default:** on.
- **Location:** `ggml/src/ggml-opencl/kernels/get_rows.cl`, `kernel_get_rows_q8_0`, `kernel_get_rows_q8_0_flat`; test `test_get_rows_cache` in `tests/test-backend-ops.cpp`. (0036)
- **Switch:** none.
- **Source:** arcint's own.

### Graph inputs without a blocking write and a wait each

With one GPU, the scheduler creates no events (`n_copies == 1`), so for each user input of a split it synchronizes the backend and then copies synchronously. On the B60 a sync costs about 10.6 us and a blocking 1 KiB write about 10.4 us; Flash-Next's verify graph has 34 inputs and a draft graph 17, and the GPU idles through them. The fix has two parts: the scheduler waits once before a split's first user input instead of before each, and the OpenCL backend copies a write of at most 4 MiB into a buffer that holds no weights into a 32 MiB USM host ring and enqueues it without a wait, so the caller may reuse its bytes at once.
- **Effect:** `measured-here`, 2026-10-14, A B A B, three 300-token answers. Flash-Next (B60): a verify's input copies 926-949 -> 582-593 us, a draft's 395-404 -> 134-141 us; decode 31.6 / 36.8 / 35.3 -> 32.0 / 37.3 / 35.8 t/s; prefill unchanged. Dense agent (B60): input copies 203 / 245-249 -> 65-67 / 114-118 us, decode 36.8 / 39.4 / 37.1 -> 37.1-37.2 / 39.7-39.8 / 37.5-37.6 t/s. Coder (A770, behind the chipset's PCIe 3.0 x4, where a sync and a blocking write cost about 60 us an input): input copies 1,930-2,020 -> 133-255 us, decode 42.9-43.0 / 39.9 / 36.9-37.0 -> 49.3-49.4 / 45.9-46.0 / 42.6 t/s (+14-15 %). Same text in every arm. test-backend-ops cannot catch a staging fault, because its CPU reference reads its inputs back from the GPU buffer; the served red case (stale ring bytes reach the card) answered "!!!!!!!!" and lost the needle.
- **Applies to:** Both cards (largest on the A770); non-weight buffers, writes of at most 4 MiB.
- **Default:** on (`GGML_OPENCL_STAGE_INPUTS=0`, `GGML_SCHED_INPUT_SYNC_EACH=1`).
- **Location:** `ggml_backend_sched_copy_input` in `ggml/src/ggml-backend.cpp` (the single wait); the staging ring in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0047)
- **Switch:** `GGML_SCHED_INPUT_SYNC_EACH=1`, `GGML_OPENCL_STAGE_INPUTS=0`; red `GGML_OPENCL_STAGE_RED=1`.
- **Source:** arcint's own (campaign `flash-next-llama-engine`, stage 8, on the `qfndev` branch); CUDA on one GPU runs the same sequence with cheaper calls.

### Keep constant matrices on the device; read sampled ids once

Round-trip discipline for a decode or MTP step, in three bits. Bit 2 is the one that paid: the KV-cache rotation matrices (a 256x256 f32 Hadamard for K and a 64x64 for V) were uploaded as graph inputs on every graph, two split inputs and two blocking copies of about 272 KB. The cache now allocates them once in its own buffer type and `build_input_k_rot` / `build_input_v_rot` return those tensors. Bit 1 (a getter after the first on a clean context skips the device wait) and bit 4 (sampled ids concatenated into one tensor and read once, not on arcint's served path since arcint samples on the host) gave nothing and stay off.
- **Effect:** `measured-here`, window 58, 2 rounds an arm, together with the MTP K/V-only pass: coder (A770) decode +1.8 / +2.0 / +1.8 %, prefill at 41k +6.6 %; window 57 put the decode gain on bit 2. Agent and Flash-Next equal. On by default since 0094. The round-trip counts are `code`: a verify of 5 rows made 6 context synchronizes, 1 with bit 1.
- **Applies to:** All backends with one non-host buffer type; models with KV rotations; bits 1 and 4 apply to other paths.
- **Default:** bit 2 on since 0094; bits 1 and 4 off (`LLAMA_SYNC_TRIM`).
- **Location:** `attn_rot_dev` in `src/llama-kv-cache.cpp`, `build_input_k_rot`, `build_input_v_rot`, `set_input_*_rot`; `synchronize()` in `src/llama-context.cpp`. (0083, default in 0094)
- **Switch:** `LLAMA_SYNC_TRIM` (bit mask, 2 by default; `0` restores upstream).
- **Source:** the arc-b580 fork's commits (`code`, 60ef575f3b, 4ec75c3416, b0376aec39); its per-call costs are the fork's `measured-here` on its SYCL stack, `paper` for us.

### A cache keyed by a stack temporary serves stale data (defect class)

Intel K-quant products convert an f32 activation once per graph and reuse it for every product sharing the input, keyed by the tensor pointer, buffer, offset, size and graph count. A flat-columns path multiplied a 3D `src1` through a stack copy of the tensor (`ggml_tensor s1 = *src1`), which has the same address in every call, and ggml-alloc puts each layer's input at the same offset, so layer N+1 matched layer N's conversion and multiplied stale data. Fix: clear the slot keyed by the temporary after the product.
- **Effect:** `measured-here`, B60, four sequences a batch in mixed ubatches: KL 0.2438 (top-1 81.96 %) before, 0.004099 (97.745 %) after, the same as one sequence per call. arcint's served path decodes one sequence per call and never took the stale hit. The note is a transferable warning for any per-graph conversion cache.
- **Applies to:** Intel K-quant products on the flat-columns path; hit only by batches that mix sequences.
- **Default:** fixed (always).
- **Location:** `ggml_cl_kq_forget` in `ggml/src/ggml-opencl/ggml-opencl.cpp`. (0027)
- **Switch:** `GGML_OPENCL_KQ_DEDUP=0` also masked it.
- **Source:** arcint's own, found by a four-sequence KL arm.

## 9. Speculative decoding

The model's own MTP head, the draft vocabulary and the width policy.

### The model's own MTP head as a multi-draft speculator, verified row by row

The checkpoint's MTP head drafts up to N tokens in a chain (draft 0 comes from the catch-up batch, as in Strata); the target verifies them in one decode of N + 1 rows; the sampler walks the verified rows in order and keeps the draft while it agrees, so every emitted token is the target's own pick at its row, greedy or seeded sampled. A draft stops at a token the head gives less than min-p (Strata's `--spec-min-p`). A recurrent model rejects a draft by choosing the snapshot (or the replay record) for the accepted length. The verify's numbers depend on how many rows it carries, so the text is not byte for byte the plain loop's; the gate is the answer-level bar. Flash-Next's checkpoint carries the head (31 `mtp.*` tensors, about 5 GB), which an earlier "no head" verdict had missed.
- **Effect:** `measured-here`. llama.cpp's own draft-mtp loop with the decode attention split (0007): coder 47.8 -> 67.6 t/s with 3 drafts, dense 19.75 -> 31.9 t/s. Served on the libllama engine (the campaigns index, `docs/campaigns/README.md`, on the `qfndev` branch): dense 18-19.5 -> 31-35 t/s, coder 47.5 -> 69 t/s; with the few-column verify (0009) dense 3 / 5 drafts 43.5 / 48.0 t/s and coder 4 drafts 71.7 t/s. Flash-Next IQ2_XS from the expert cache, 2026-10-06: 21.3 t/s plain, 27.0 t/s with MTP 2, 30.7-31.6 with the rest of 0022; 3 drafts with the 0.5 stop 37.9 t/s on Strata's request against Strata's own 37.8 t/s on the same card (2026-10-14). The width that is best on Flash-Next is 3: width 2 -1.2 %, 4 -5.5 %, 5 -8.4 % (acceptance 73.5 / 66.6 / 60.3 / 54.8 % for 2 / 3 / 4 / 5).
- **Applies to:** Models that carry an MTP head (the dense Qwen3.8-27B, the coder, Flash-Next via `--llama-mtp-gguf`); recurrent models need a snapshot or replay rollback.
- **Default:** off until `--llama-mtp N` is set; the sample units set it (agent 5, coder 4, Flash-Next 3).
- **Location:** `src/exec/llama_spec.cpp` (the drafter), `src/exec/verify_walk.h` (`walk_verify`), `src/exec/spec_state.h`. Engine side: patches 0010, 0067, 0080.
- **Switch:** `--llama-mtp N`, `--llama-mtp-min-p P`, `--llama-mtp-gguf FILE`.
- **Source:** Strata `src/core/mtp.cpp` (the draft chain with `min_p`) and `tools/mtp_fetch.py` (`code`); llama.cpp `draft-mtp`.

### Select the output rows before capturing the MTP hidden state

In the Qwen3.5/3.6 MTP graphs the nextn row was captured before the output rows were selected, so a batch of several tokens with one output returned token 0's row in place of the output token's. The patch selects first (as the trunk graph does), only for masked contexts.
- **Effect:** `measured-here`. Without it, drafts after the first chained from the wrong row: served acceptance fell from 78 to 57 % on the dense 27B at 3 drafts and from 91 to 75 % on the coder at 2 drafts. Its red case (0010 rows in the old order) left the coder's texts the same bytes and cut decode to 43.5 / 39.6 / 38.6 from 50.7 / 47.1 / 43.6 t/s.
- **Applies to:** qwen35 and qwen35moe graphs with masked nextn embeddings.
- **Default:** on.
- **Location:** `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp` (`t_h_nextn`). (0010)
- **Switch:** none.
- **Source:** Strata's drafter chains the same way (`code`, per the patch text).

### MTP head fills only its K/V for the prompt

The MTP draft context's catch-up decode has no output rows: it exists only so later draft steps can read this layer's K/V. The graph nevertheless built the whole MTP block for every prompt row (query and gate, attention, gated output projection, residual, FFN, shared head) and cut only the output rows. With no outputs and masked nextn embeddings, the graph now returns right after the cache store. The retained K/V rows are the full graph's, so the quality class is exact. Scoped to the dense and the coder architecture; Flash-Next's MTP graph (hyper-connections, K/V inside the layer builder) is not covered.
- **Effect:** `measured-here`, window 58, 2 rounds an arm, bundled with one round-trip trim: coder (A770) decode +1.8 / +2.0 / +1.8 %, prefill at 41k 729.75 -> 777.80 t/s (+6.6 %), texts md5 identical; window 57 put the prefill gain on this patch and the decode gain on the trim. Agent prefill equal (55k 715.85 -> 714.95); Flash-Next equal. On by default since 0094. The Metal-class price quoted by the audit (about 5 % on long prompts) is `paper`.
- **Applies to:** qwen35 (dense) and qwen35moe (coder) only; not Flash-Next's qwen4exp graph; layers without a fused qkv or K/V bias.
- **Default:** on since 0094 (`LLAMA_MTP_KV_ONLY=0` keeps the full graph).
- **Location:** `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` (`graph_mtp` constructors). (0080, default in 0094)
- **Switch:** `LLAMA_MTP_KV_ONLY` (on unless `=0`).
- **Source:** the arc-b580 fork's `LLAMA_MTP_KV_ONLY` (`code`, `arc-b580/src/models/qwen35.cpp`).

### Leave the draft context's catch-up rows on the device

With a draft vocabulary, the MTP context runs with unmasked nextn embeddings (a row for every token, no logits), so the prompt catch-up copied one row per prompt token to the host that nothing reads: 41k rows x 2,048 floats on the coder over the A770's PCIe 3.0 x4 link. A context flag leaves the rows on the device for the following decodes without changing the graph (toggling the mask would force a re-reserve).
- **Effect:** `measured-here`. The copy cost the coder 0.8 % of prefill and Flash-Next (4 hyper-connection streams per row) 0.9 %. Coder with the vocabulary, A/B with vs without the patch: prefill at 41k 723.25 -> 729.20 t/s, decode equal; Flash-Next prefill with the vocabulary equals prefill without (622.65 / 622.88 t/s).
- **Applies to:** Draft contexts with unmasked nextn embeddings, i.e. with a draft vocabulary.
- **Default:** on, set by the engine around the catch-up decodes.
- **Location:** `llama_set_embeddings_nextn_skip` in `src/llama-context.cpp`, `src/llama-ext.h`. (0067)
- **Switch:** set by the engine around the draft context's catch-up decodes.
- **Source:** arcint's own.

### Draft vocabulary: score a slice of the output head for the draft

The drafter scores only a 40,960-row slice of the output head (ids counted over 5.4M of the model's own tokens, 97.5 % coverage, 96 % on code), so a draft step reads a fraction of the head. The target still verifies every token and every emitted token is the target's own pick at its row, but the text can move with the verify's shape; it is judged at the answer-level bar. The id file transfers between models with the same tokenizer. With a draft vocabulary the draft context's unread catch-up rows stay on the device (0067).
- **Effect:** `measured-here`. Dense 27B with 4 drafts: 47.1 -> 50.8 t/s with draft acceptance 72.1 -> 68.0 % (the HyperQwen figures, `paper`, for the reference: a draft about 3 ms with the full bf16 head, 0.5-1 ms after). Served by arcint's production units: the coder +10 to +13 % decode (task 10/10), Flash-Next +2.9 %. Stacked with the 1.2.7 changes and the coder's prefill ubatch 1,024, the coder against 1.2.6: decode +13.6 / +21.9 / +14.7 %, prefill at 41k +15.7 %.
- **Applies to:** Any MTP draft with a tokenizer-matched id file; measured on the dense 27B, the coder and Flash-Next.
- **Default:** off unless `--llama-mtp-vocab FILE` is set; the sample agent unit sets it, and the CHANGELOG (1.2.7) records it on the coder and Flash-Next units.
- **Location:** `src/exec/llama_spec.cpp` (`vocab` argument of the drafter constructor, `src/exec/llama_spec.h`); `llama_set_embeddings_nextn_skip` (0067).
- **Switch:** `--llama-mtp-vocab FILE`.
- **Source:** HyperQwen's draft-vocab patch (`code`, `patches/qwen3_5-mtp-draft-vocab.patch`); id tool from Strata.

### Cost-aware draft width

Choose the number of draft rows from a learned cost per width: the expected tokens of a window of j + 2 rows, E_j = 1 + q_0 + .. + q_j with q_i the product of the draft probabilities' calibrated acceptance, divided by the window's measured cost; the chain goes on while one or two more drafts could beat the best width by a margin. The cost table is seeded with the card's measured round times and learned online. The min-p floor stays arcint's (the reference's cost arm has no floor).
- **Effect:** `measured-here`, coder on the A770: decode on the six-answer set 56.27 -> 64.92 / 65.42 t/s (+15.8 %); the acceptance task 10/10 x3 at 90.1 / 90.0 / 90.1 -> 92.8 / 90.9 / 93.5 t/s; a sampled battery 10/10 x9 and 9/10 x1. Not adopted on Flash-Next (-2.6 %) or the agent (the greedy task flipped). An oracle that picks the best fixed width per answer on Flash-Next would reach 35.33 t/s, +2.5 % over width 3; Strata's cost-aware width measured +4.7 % on one GB10 in its first run but not in its own later check (0.996x and 1.02x), and Infernix's adaptive MTP +2.0-2.1 % geomean on two models and a loss on a third (all `paper`).
- **Applies to:** MTP with one lane; measured on all three units, adopted on the coder.
- **Default:** off (`--llama-mtp-policy cost`); the CHANGELOG (1.3.0) records it on the coder unit.
- **Location:** `src/exec/draft_policy.h` (`DraftPolicy`, `PolicyChain`), `tools/mtp_width_replay.py` (offline replay of the same rule), `src/exec/mtp_trace.h`.
- **Switch:** `--llama-mtp-policy cost`, `--llama-mtp-cost-seed`, `ARCINT_MTP_TRACE`.
- **Source:** Strata `src/spec/draft_policy.cpp` (`keep_draft`, `code`).

## 10. Lanes and windows in one KV pool

Serving an agent and its subagent from one process.

### Cell windows: a sequence attends over its own range of a unified KV pool

In a unified KV cache a sequence's attention spans the pool up to the highest used cell, so it grows with another sequence's cells (masked, but read). `llama_memory_seq_windows` gives sequence i a range of the pool; `find_slot` searches that window, `get_n_kv` counts to the window's last used cell, and the K/V views start at the window's first cell. In a ubatch with several windowed sequences, the attention builder runs one op per sequence on that sequence's K/V window and its slice of the mask, and the outputs are concatenated. Graph reuse compares the window parts, because a graph's views have the offsets built in. A search-head reset on `seq_rm` makes a sequence refill from its end rather than above its old high-water mark, and the slot search starts no higher than the window's last used cell. Windows are refused (the call returns false and changes nothing) for several streams, a shared source cache, the indexed hybrid (Flash-Next's QSA k-pool selects cells by absolute index) and any memory other than the attention cache of `llama_kv_cache` or the hybrid memory.
- **Effect:** `measured-here`, B60, dense Qwen3.8-27B Q4_K_M, q8_0 KV, MTP 5, named lanes 131,072 + 32,768. The subagent lane's 300-token answer with the agent lane at about 20k, 64k and 118k: 32.1 / 32.0 / 32.0 t/s, the same text each time (before 0025 and these patches: 7.7 / 7.3 / 6.5 t/s; with 0025's snapshot-write fix alone, a 4k subagent cell next to a 130k agent: 14.2 t/s). Two lanes in shared steps, 21.6-21.7 t/s a lane, 43.3 together, against 15.9 + 15.8 taking turns. The search-head red case: a 4k request after a 29k one, 34.1 t/s and 22.7 s without the reset, 37.8 t/s and 15.6 s with it. Reusing a graph with stale part views had cut draft acceptance with both lanes busy to 23-26 % against 28.2 % alone; fixed. KL four sequences a batch 0.004099 / 97.745 %, equal to one sequence. Upstream's `--kv-unified-per-slot` is one cap for all slots; there is no per-sequence range.
- **Applies to:** Unified KV caches of the attention-only or hybrid-attention kind; refused for several streams, a shared source cache and Flash-Next's indexed hybrid.
- **Default:** off unless named lanes are configured; the production agent runs one lane.
- **Location:** `src/llama-kv-cache.cpp`, `llama_memory_seq_windows`, `llama_kv_cache::get_win_parts`, `get_k_part`, `get_v_part`; `include/llama.h`. (0026, 0029)
- **Switch:** `LLAMA_PPL_SEQ_WINDOWS=1`, `LLAMA_PPL_PER_SEQ=1`, `LLAMA_PPL_SEQ_FLIP=1` (perplexity test hooks); arcint's lanes configuration.
- **Source:** arcint's own (campaign `lanes-agent-subagent`, on the `qfndev` branch; the lanes arrangement was later ended, see "tried and not adopted").

### Row blocks for 9 to 64 query rows of attention

An attention call with 9 to 64 query rows runs in blocks of up to 8, each a recursive call on views of q, the mask and the output, so the small-row kernels (the q1 split kernels and the GQA kernel, at most 8 rows) take it. Without blocks a verify of two lanes (12 rows) fell to the prefill kernel, which pays a K/V conversion every call.
- **Effect:** `measured-here`, B60. Two lanes of the dense 27B in shared verifies: the prefill kernel had taken 2,034 calls at 4.3 ms; a shared step took 282 ms instead of 1,030, and one such step at 131,072 + 32,768 had outrun xe's job timeout (an engine reset). After 0029 the lanes' path no longer fires it; it still matters for a prompt tail of 9-64 tokens over a deep context: a 17-25-token follow-up over 122,383 tokens prefilled in 0.65-0.88 s with blocks and 1.35-1.59 s without. Red case: mask offset dropped, 426 of 2,806 attention cases fail.
- **Applies to:** Intel (both cards); attention calls with a mask and 9 to 64 query rows.
- **Default:** on (`GGML_OPENCL_FA_ROW_BLOCKS=0`).
- **Location:** `ggml/src/ggml-opencl/ggml-opencl.cpp` (the FLASH_ATTN_EXT dispatch). (0028)
- **Switch:** `GGML_OPENCL_FA_ROW_BLOCKS=0`.
- **Source:** arcint's own.

## 11. Serving and device hygiene

What the server does around the engine, and the defensive mechanisms that keep a card from silently going wrong.

### Downgrade reports: a path other than the promised one says so

A weight's type promises a kernel path; the backend can silently take another (an optional program not built, a product sent to the CPU, ggml's generic tiled GEMM, the per-column matvec at 16 or more columns, an expert product on the generic `MUL_MAT_ID`, a prompt's flash attention off the XMX kernel). The first time a key occurs in a process the backend logs one `DOWNGRADE src=opencl <key>` line at warn level and counts it per key; arcint publishes the counts on `/props` as `downgrades`, and the release gate fails a unit on a key outside an expected list. The case that motivated it: the coder's Q8_0 ran ggml's float kernel at prefill for months and was seen only in a profile (0099 fixed it).
- **Effect:** `measured-here`, served on all three units with prompts up to 19,521 tokens: the coder and the agent report nothing; Flash-Next reports 8 keys (BF16 at 5-15 columns at K 10,240, 2,560 and 320, and F32), at most 0.67 % of its kernel time. Red cases: `GGML_OPENCL_Q8_I8=0` on the coder reports 4 keys; `GGML_OPENCL_FA_DPAS=0` on Llama 2 reports `op=FLASH_ATTN head=128 kv=f16 n_q=9+ path=f32_f16 reason=no-dpas-kernel` (prefill 2,814 -> 1,461 t/s). The first version missed the matvec, and short prompts never reached the generic path, so a check of this kind needs long prompts. `llama-bench` installs a null log callback and never shows the lines.
- **Applies to:** The Intel OpenCL backend; keys listed in 0100 and 0101; not yet covered: expert GEMM selection and flash attention's shape limits beyond the two keys of 0101.
- **Default:** always on; counts on `/props` `downgrades`.
- **Location:** `ggml_backend_opencl_downgrades` in `ggml/src/ggml-opencl/ggml-opencl.cpp`; arcint side `src/api/handlers.cpp` (`downgrades` on `/props`); gate tools `tools/downgrade_check.py`, `tests/acceptance/expected-downgrades.json`. (0100, 0101)
- **Switch:** none (always on); not covered yet: expert GEMM selection and flash attention's shape limits beyond the two keys of 0101.
- **Source:** arcint's own (operator decision).

### A weight the card declines at load says so

At load, libllama probes each weight's op on the buffer types its layer asks for and takes the first one the device accepts; when the device declines, the weight lands in host memory and the scheduler splits every forward around it, with no backend path taken that the downgrade keys above could count. The loader now records that decision where it is made, with the reason: one `DOWNGRADE src=llama op=<probe op> type=<type> weight=<name, layer numbers as N> buft=<buffer type> reason=device-declines` line the first time a key occurs and a count of weights per key, served on `/props` `downgrades` with the OpenCL keys, so the release gate's expected list sees it. Placements somebody asked for are not reported: buffer-type overrides (`--llama-cpu-moe`, the expert cache), lazily read tensors, the input layer (`token_embd`, on the host by design), layers left on the CPU. The case that motivated it: Dory on a build whose OpenCL `CPY` refused BF16 put its 217 BF16 scale vectors in CPU buffers, decoded at 14.7 against 23.7 t/s, and said nothing.
- **Effect:** `code` so far: from the model files' headers and the series' `supports_op`, Dory without 0109 reports 6 keys (217 weights) and with it none; Gemma 4 E4B, Granite 4.2 8B, the coder and the 27B none; Flash-Next one, its f16 PLE conv kernel probed as `SSM_CONV` (`blk.N.ple_conv1d.weight`), in host memory until 0112 probed it as the cast the graph applies to it (80 KiB on the card). The card runs are owed.
- **Applies to:** Every weight libllama's loader places, on any backend; not the multimodal projector (libmtmd's own loader).
- **Default:** always on; counts on `/props` `downgrades`.
- **Location:** `src/llama-placement-report.h` and the call in `llama_model_loader::create_tensor` (`src/llama-model-loader.cpp`), `llama_placement_downgrades` in `src/llama-ext.h`; arcint side `collect_downgrades` in `src/exec/backend.h`, `tests/test_placement.cpp`; the first key it found, fixed: `LLM_TENSOR_PLE_CONV1D` in `src/llama-arch.cpp`. (0111, 0112)
- **Switch:** none.
- **Source:** arcint's own; the gap was found by 0109's red arm.

### Context checkpoints for a model whose state cannot be cut back

A hybrid (delta-net) model's recurrent state cannot be trimmed to an earlier token, so each lane keeps checkpoints of that state in host memory: at the start of the batch that begins the last user message, at a user message more than 8,192 tokens past the newest checkpoint, and a few tokens before the prompt's end; at most 32 per lane; a request restores from the newest at or below the shared prefix and the attention KV is trimmed there. A thinking turn's follow-up or an edited message resumes instead of re-prefilling. Checkpoints load together with the adaptive expert cache.
- **Effect:** `measured-here`, dense 27B, B60, MTP 5: a 24.3k-token conversation 38.75 -> 0.61 s. Coder on the A770, a 40,365-token prompt sent twice with different final questions: time to first byte 48.04 / 48.22 s cold, 2.17 / 2.37 s on restore. A resumed turn gives the same text as a full re-prefill with the same batch breaks (193/193 and 209/209 characters); with other breaks the text moves and the answers stay right.
- **Applies to:** Hybrid models with a recurrent state; per lane.
- **Default:** on (32 checkpoints a lane, step 8,192 tokens).
- **Location:** `src/exec/backend_llama.cpp`, `src/config.h` (`llama_checkpoints`, `llama_checkpoint_step`); state save with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` for the recurrent part.
- **Switch:** `--llama-checkpoints N`, `--llama-checkpoint-step T`.
- **Source:** llama.cpp server's context checkpoints (`code`, `tools/server/server-context.cpp` at the pin); Strata's conversation snapshots as a further reference (`docs/campaigns/kv-checkpoint-restore.md`, on the `qfndev` branch).

### Cancellation at the next token or prefill batch

A client that leaves, and a stop signal, end a request at its next token or prefill batch, also while nothing is written (a long prompt's prefill, a buffered tool call): a zero-timeout socket peek on the request's own connection.
- **Effect:** `measured-here`, A770 coder, a 37,723-token prompt, against the parent binary: the slot is free 1.8 s after a streamed client leaves (42.6 s before), 1.6 s non-streamed (42.3 s), SIGTERM exits after 1.8 s (42.7 s); the same needle prompt sent right after the abort is answered correctly. Speech and transcription do not cancel.
- **Applies to:** The llama engine's HTTP server; speech and transcription do not cancel.
- **Default:** on.
- **Location:** `src/http/` (cpp-httplib with poll; the request's `socket` field), `tools/cancel_check.py`.
- **Switch:** none.
- **Source:** arcint's own.

### Token mode: token ids out, a renderer hash in

A gateway that renders the chat template itself can ask the engine for token-level output: the unit serves its renderer material (decode pieces, special tokens, the chat template and its source, the think-tag geometry) with a canonical hash; a request carries the hash and a hash of its rendered prompt; a stale hash or a prompt the unit would render differently is a 409 before a slot is taken. The stream carries one event per accepted output token before the text event, and span events for thinking, tool calls and the guards. The point is that a gateway and an engine agree on exactly the same prompt tokens, so the prefix cache hits. Thinking replay (keeping every assistant turn's thinking, `preserve_thinking`) matters because an agent's steering message otherwise drops the loop's reasoning from the prompt: on the coder one steering turn hit the prefix cache for 10.6 % of the prompt and re-prefilled for 53 s (`measured-here`, `docs/feyntalk.md`, which gives no after-figure there); the 1.4.0 release gate on the coder measured 13.4 % without and 99.6 % with it (prefill 1,476 -> 229 ms).
- **Effect:** `measured-here` on the stub: the chat route's bytes unchanged without the new surface (golden captures 12/12 byte-identical). A silent fallback is the failure mode the project names: a gateway's token mode lost every turn carrying a reasoning effort on a template that reads the level, and only the path label, not the answer, was wrong. The release gate therefore runs real token-mode turns at every effort.
- **Applies to:** Llama units on `/v1/responses` with `stream: true`, text only; off on speech units.
- **Default:** per request opt-in (`arcint.token_mode`); `/props` `token_mode: ready` is only a pre-flight.
- **Location:** `src/api/renderer.cpp`, `src/core/renderer_material.cpp` (`renderer_hash`), `tools/ft_gate.py`.
- **Switch:** per request `arcint.token_mode`, `arcint.renderer_hash`, `arcint.render.prompt_sha256`; `/props` `token_mode`.
- **Source:** arcint's own (Open Responses over HTTP+SSE).

### Hygiene: uniform work-groups, an image limit, a sampled alias tripwire

Three small defensive mechanisms. (a) Every launch rounds its global size up to a whole local size and the kernels exit on the tail (kernels that use sub-group or barrier collectives, such as the gemv 32b-trans kernels, must not lose a lane before the collective: they clamp the row index for the loads and gate only the final write), so that the runtime's Level-Zero-backed OpenCL mode (LEO), which rejects non-uniform work-groups with `CL_INVALID_WORK_GROUP_SIZE`, accepts every launch. (b) On a DG2 device under LEO, an `IMAGE1D_BUFFER` above 16,384 texels aborts the runtime in a 14-bit width check at creation; the patch refuses such an image so the kernel takes its buffer path. (c) A sampled VRAM page-table aliasing tripwire writes then reads back at most 16 blocks per bank-ring allocation and one whole slice per slot tensor; Strata scans every 64 KiB block and retries, so this is a named deviation with the full-scan price stated (2 x allocation bytes of traffic).
- **Effect:** `measured-here` for (a): the full test-backend-ops (12,881 cases) gives identical failure sets under the default driver and LEO on the B60 (12,724 passed both); on the A770 LEO aborted inside the runtime's DG2 command generation after 4,615 cases, at MXFP4 m=2,880, which is the image-width abort that (b) works around. With (b) the A770 runs under LEO at 12,867 of 12,881 cases (the default driver's pass set), the coder decodes +6 % and prefills -5.6 % against the default driver (CHANGELOG 1.4.0, `measured-here`; the 0072 text had +5.9 % and -5.3 % before the workaround). (b) is `measured-here` for the defect (16,384 texels fit, 16,385 abort). (c) has device-free cells; its load cost is inside noise, with 60.5 MB written and read back on Flash-Next (alias check on 61.6 / 56.6 s, off 52.1 / 64.2 s to health, CHANGELOG 1.4.0). See "tried and not adopted" for the LEO route.
- **Applies to:** (a) every Intel launch; (b) DG2 under LEO only; (c) the expert cache's slot tensors (planar types and IQ3_XXS) and the bank-stream ring.
- **Default:** (a) on; (b) on, a no-op elsewhere; (c) slot half `LLAMA_EXPERT_CACHE_ALIAS=0` off, ring scan unconditional.
- **Location:** `ggml/src/ggml-opencl/ggml-opencl-uniform.h`; `ggml/src/ggml-opencl/ggml-opencl-image-limit.h` (the rule), `ggml_cl_image_texels_ok` in `ggml/src/ggml-opencl/ggml-opencl.cpp`; `ggml/src/ggml-opencl/ggml-opencl-alias.h`, `src/llama-expert-cache-alias.h`. (0072, 0079, 0077)
- **Switch:** `LLAMA_EXPERT_CACHE_ALIAS=0` turns off the slot-buffer half of (c) only; the bank-stream ring's scan is unconditional.
- **Source:** (a) arcint's own; (c) Strata's `sycl_queue.hpp` (`code`, the 64 KiB block tagging and re-allocation).

## 12. Image and speech

Engine concepts that the diffusion and speech services carry (the services themselves are in docs/imggen.md and docs/speech-service.md).

### Q8_0 weights on the XMX tile

Upstream's generic q8_0 GEMM was 68-81 % of an image-diffusion step. The patch dequantizes the int8 code and the f16 per-32 scale of the SoA split into an f16 local tile and runs fp16 DPAS with f32 accumulation, the K-quant pattern. Dispatch: Q8_0 x F32, 16 or more columns, K a multiple of 32, flattened weights only. Built only with sub-group 16 (B60).
- **Effect:** `measured-here`, B60, Z-Image Turbo Q8_0 with Qwen3-4B Q8_0, 8 steps, seed 42: 1024^2 step 10.95 s (off) -> 2.14 s (5.1x); 512^2 3.03 s -> 0.476 s (6.4x); the gate against the CPU reference passes in both arms (cos worst 0.999926, PSNR 42.59 dB at 1024^2), and the red case fails it (0.987219, 19.56 dB). At 1024^2 the kernel is 69 % of the step at about 33.5 TFLOP/s, 34 % of the B60's XMX f16 peak. It is faster than the f16 weights (2.21 s) in 5.5 GiB less memory. In the language services only the MTP draft heads have Q8_0 weights on the B60. The first build was wrong (a global K coordinate written into the per-super-block tile; cos worst 0.83) and the gate caught it.
- **Applies to:** Xe2 (B60) only (the kernel needs sub-group 16); Q8_0 x F32, 16 or more columns, K a multiple of 32, 2D contiguous flattened weights; on the A770 Q8_0 takes the int8 kernel of 0099.
- **Default:** on (`GGML_OPENCL_Q8_XMX=0` falls back to upstream's kernels).
- **Location:** `ggml/src/ggml-opencl/kernels/mul_mm_kq_f16.cl`, `deq_q8_0`, `kernel_mul_mm_q8_0_f16`. (0058)
- **Switch:** `GGML_OPENCL_Q8_XMX=0`; red `GGML_OPENCL_Q8_XMX_RED=1`.
- **Source:** arcint's own, after its K-quant tiles.

### Fused rotary-embedding apply for diffusion transformers

`GGML_OP_ROPE_APPLY` takes a precomputed `pe` (`[2,2,d/2,tokens]`, `[[cos,-sin],[sin,cos]]`) and rotates `a` in one pass, instead of the generic two repeats and four muls per layer. The op definition and CPU forward are identical to the fork it comes from; the OpenCL kernel is original.
- **Effect:** `measured-here`, B60, Z-Image Turbo, 8 steps, seed 42: a 1024^2 step 4.77 s with stable-diffusion.cpp's generic RoPE -> 2.14 s (2.2x). At 512^2 the generic RoPE was 37 % of the f16 run's device time (`repeat_f32` 3,054 ms + `mul` 2,188 ms of 14,053 ms) and the fused op is 70 ms a run. The port's first version added element offsets to a `global char *` destination without `sizeof(float)` and left most cells unwritten (an all-white image); the image gate caught it. The program is built with `-cl-mad-enable -cl-fast-relaxed-math`, so the CUDA arithmetic's separate roundings are not reproduced and the gate decides. Nothing in libllama calls the op.
- **Applies to:** OpenCL (Intel); F32, even `d`, contiguous destination, at most `INT_MAX` pairs; used by stable-diffusion.cpp through a patch, not by libllama.
- **Default:** on when the caller emits the op.
- **Location:** `ggml/src/ggml-opencl/kernels/rope_apply_f32.cl`, `kernel_rope_apply_f32`; `test_rope_apply` in `tests/test-backend-ops.cpp`. (0057)
- **Switch:** none (the op is used when the caller emits it).
- **Source:** the `leejet/ggml` fork's `GGML_OP_ROPE_APPLY` (`code`, commit `e05d18a1`, `ggml.h`); the OpenCL kernel is original.

### 3D convolution as implicit GEMM; leaky ReLU as an op

`GGML_OP_CONV_3D` on the implicit-GEMM form (no im2col matrix in global memory): depth folded into the reduction axis (`K = OC`, `CRS = IC*KD*KH*KW`, `NPQ = N*OD*OH*OW`) on the existing conv2d tile. Without it a backend with neither `IM2COL_3D` nor `CONV_3D` runs the Wan VAE's 3D convolutions on the CPU. `GGML_OP_LEAKY_RELU` was the one unary the pin's OpenCL lacked, and a mixed CPU/OpenCL upscaler graph sent it to the CPU and segfaulted (the source's likelier mechanism, not measured: the in-place op wrote through a view of an OpenCL buffer).
- **Effect:** `measured-here`, B60. CONV_3D 276/276 op cases pass (284/284 with the Wan VAE's shapes added in 1.1.1); the red case fails 270 of 276. Served, Qwen-Image's VAE decodes on the card in 2.66 s at 512^2. LEAKY_RELU 7/7 pass; ESRGAN 4x of a 512^2 image to 2048^2 takes 27 s on OpenCL where the same command segfaulted before. The strided LEAKY_RELU cases measured wrong in the release window (ERR 1.4-1.7, not attributed), so only contiguous input is admitted. The source notes that for one frame only the kernel's last time tap meets data, so two thirds of the VAE's conv FLOPs multiply padding (arithmetic, not measured).
- **Applies to:** OpenCL (measured on the B60); CONV_3D with f16 or f32 weights and F32 input, at most 2^32 elements per tensor; LEAKY_RELU contiguous f32 or f16.
- **Default:** on (`GGML_OPENCL_CONV_3D=0` makes the op unsupported, which crashes Qwen-Image's graph).
- **Location:** `ggml/src/ggml-opencl/kernels/conv3d.cl`, `kernels/conv3d_f16_f32.cl` (`kernel_conv_3d`); `kernels/leaky_relu.cl`. (0059, 0060)
- **Switch:** `GGML_OPENCL_CONV_3D=0` (op unsupported), `GGML_OPENCL_CONV3D_RED=1`, `GGML_OPENCL_LEAKY_RELU_RED=1`.
- **Source:** CUDA's `im2col_3d` and Vulkan's `conv3d_mm` for the form (`code`).

### Speech decoder computes only the valid frames

Qwen3-TTS's code2wav decoder built a fixed 72-frame graph from the decoder's attention window and padded the codes to it, so a 16-frame flush cost as much as a 72-frame one (1.766-1.79 s per flush on a pure CPU backend, 99.6 % compute). The graph now takes the caller's valid frame count.
- **Effect:** `measured-here`, 2026-10-19. CPU: flush becomes frame-proportional (44 frames 1.11 s, 64 frames 1.59 s against the fixed 1.77-1.79 s), the 8-sentence total 14.25 -> 10.60 s. B60 (Q8_0 backbone, bf16 mmproj, decoder on the CPU, 8-sentence gate, seed 42, window 72, through `/v1/audio/speech`): real-time factor 0.729 (padded) -> 0.802 (valid frames) in the first run, 0.7285 -> 0.8006 in the final build; worst first audio 11.0 -> 9.52 s; word error rate 0.0455 on both (round trip through a whisper server). Sample difference at most 191/32768 of full scale. Still sub-real-time.
- **Applies to:** Qwen3-TTS `GEN_WAV` code2wav in libmtmd; any backend.
- **Default:** on (`MTMD_TTS_WAV_PAD=1` restores the padded window).
- **Location:** `tools/mtmd/clip.cpp` (`clip_graph_qwen3tts_gen`, the `MTMD_TTS_WAV_PAD` switch), `tools/mtmd/models/models.h` (field `gen_n_frames`). (0062)
- **Switch:** `MTMD_TTS_WAV_PAD=1` restores the padded window (the red case); no flag in arcint.
- **Source:** arcint's own.

### Small autoregressive graphs belong on the CPU: place a speech graph by measured per-frame cost

Qwen3-TTS's code predictor runs a small graph per frame. On the B60 the predictor cost 272.4 ms per frame against 59.9 ms per frame on a single CPU backend (4.5x), with device execution only 12 % of wall and host enqueue negligible: roughly 6,600 tiny kernels per frame were the cost, not the graph splits. Cutting the splits with an OpenCL `ARANGE` op (1,007 -> 299 splits) bought 4.8 % and was dropped (see "tried and not adopted"). The adopted arrangement is a hybrid: the decoder-side mmproj graphs on the CPU, the backbone on the device (`--tts-gen-device cpu|gpu`), with the code predictor's weights quantized except `gen.code.head` and `gen.code.embd`, which stay f16.
- **Effect:** `measured-here`, B60, the fixed 8-sentence gate through `/v1/audio/speech`. Real-time factor: 0.267 (card only), 0.828 (hybrid), 1.1291 and 1.1597 (the enlarged D8 set in two windows, with the valid-frames decoder and the quantized predictor), word error rate 0.0264 (worst 0.333) on that set. On the installed package the same set measured RTF 1.1167 (182.960 s of audio in 163.837 s). The campaign closed on that result.
- **Applies to:** Qwen3-TTS code predictor and decoder graphs; measured on the B60.
- **Default:** per flag (`--tts-gen-device cpu|gpu`); the production set runs `cpu`.
- **Location:** `tools/mtmd` in the pinned tree (`tools/mtmd/clip.cpp`); arcint side `--tts-gen-device`.
- **Switch:** `--tts-gen-device cpu|gpu`.
- **Source:** arcint's own (campaign `qwen3tts-card-real-time`, on the `qfndev` branch).

## 13. Weight-shared and looped architectures

Models that run the same weights more than once per token. Admitted 2026-11-07; neither entry has run on a card yet.

### Physical blocks and executed layers kept apart (a looped nGPT)

Dory1.0-4B stores 72 blocks (16 input, 40 recurrent, 16 output) and runs the 40 recurrent ones `num_loops` times (4 in the file), so the graph has 192 layers over one set of weights. The model keeps two counts. Weight placement, the offload fit and the quantizer count the physical blocks; the graph, the KV cache and the layer count the API reports count the executed layers, and a virtual-to-physical table maps one to the other. The tensors of a recurrent block are created once and every executed layer of it points at the same ones. The recurrent KV is either one cache a loop and layer (`per_loop`) or one a layer that each loop overwrites (`last_loop`, where a token's state then depends on the chunk boundaries of its prefill). In an engine that admits models by geometry, admit on the physical count: the executed count changes with the operator's loop setting.
- **Effect:** `measured-here`, 2026-11-07, Dory1.0-4B Q5_K_M, build 6fa32c1e, decode over 400 tokens including the prefill. A770: 2 loops q8_0:q4_0 23.7 t/s, prefill 512-665 t/s at 5.9k tokens, peak 5.2 GiB; 2 loops f16 24.9 t/s, 727 t/s, 7.1 GiB; 4 loops 14.3 t/s, 396 t/s, 6.2 GiB; 4 loops `last_loop` 14.3 t/s, 392 t/s, 4.7 GiB. B60: 36.4 t/s / 1,287 t/s, 37.9 / 1,592 (f16), 22.0 / 742 (4 loops). The KV layers counted at start were 56, 96 and 36; the answers are right in every cell; KL against the CPU reference on the A770 is 0.0091 at 2 loops, 0.0107 at 4, 0.016 with q8_0:q4_0; 0 DOWNGRADE lines. The graph also agrees with vLLM's implementation on the normalisation, the slerp residual, the gates and the loop seed (`code`).
- **Applies to:** Any architecture that repeats a block range; here Dory1.0-4B, 1 to 4 loops. Refused with MTP, DFlash2 and the layer split.
- **Default:** The file's own loop count and KV mode; `--llama-dory-loops` and `--llama-dory-kv` override, and `/props` `dory` names the value that serves and its source. A start whose reported layer count disagrees with the loop count fails.
- **Location:** `src/models/dory.cpp`, `llama_model_dory`, `n_layer_physical`, `layer_physical`; `src/llama-model.cpp` (`load_tensors`, `create_memory`'s layer filter); `llama_model_n_layer_physical` in `src/llama-ext.h`. (0108) arcint: `src/core/dory.cpp`, `dory_resolve`, `dory_check_loaded`, `dory_option_refusal`; `src/core/admission.cpp`, `is_dory_4b`; `src/exec/backend_llama.cpp` (the metadata overrides `dory.num_loops` and `dory.recurrent_kv_cache_mode`).
- **Switch:** `--llama-dory-loops N`, `--llama-dory-kv per_loop|last_loop`.
- **Source:** llama.cpp pull request 30306 (open, head `1c3cdd23`), cross-read against vLLM's `models/dory.py`.

### A BF16 source for the device copy, so the casts of per-channel scales stay on the card

A model whose small scale vectors are stored in BF16 casts each one to F32 in the graph. If the backend refuses a BF16 source, the loader places that tensor in a CPU buffer and the scheduler splits the graph around it, once per scale (217 in Dory). On OpenCL a BF16 tensor is already held as F16 on the device (the upload converts it), so the copy kernels read it as F16 and `supports_op` lists the pair: no new kernel.
- **Effect:** `measured-here`, 2026-11-07, A770, Dory1.0-4B, 2 loops, q8_0:q4_0, same window: without the patch decode 14.7 t/s, prefill 540 t/s; with it 23.7 and 665. CUDA's same failure cost 3.1x in the upstream discussion (`paper`). The arm without the patch printed no DOWNGRADE line: an op that `supports_op` declines goes to the CPU without a key, an open gap in the downgrade accounting.
- **Applies to:** `CPY`, `DUP`, `CONT` from BF16 to F32 or F16 on the OpenCL backend; both cards.
- **Default:** On.
- **Location:** `ggml/src/ggml-opencl/ggml-opencl.cpp`, `supports_op` (`GGML_OP_CPY`) and `ggml_cl_cpy`; test `tests/test-backend-ops.cpp`, `test_cpy` with BF16 sources at 2560, 4096, 7168 and 2560 x 256. (0109)
- **Switch:** none.
- **Source:** arcint's own, after the same fix on CUDA in the pull request above.

## Where the ideas came from

Two lists. The first names, per reference, the entries whose idea was taken from it (the entry's Source line); an entry may appear under more than one. Entries not listed there are arcint's own. The second names references that were read or measured against in an entry without being its origin.

**Taken from:**

- **Strata:** Per-expert cache; Decode reads a missed expert in place; Expert swaps leave the compute queue; Residency tables as rows of one tensor; CPU computes the first missed experts (doorbell tier); Multi-token AVX2 rows; One cache per GPU under a layer split; Save the learned expert ranking; Prefill streams each bank tensor on a second queue (with Infernix's expert stream); Prefill slot loan; GatedDeltaNet recurrence (the chunk pipelining); The model's own MTP head as a multi-draft speculator; Cost-aware draft width; int8 DPAS for the prompt's routed experts (opt-in) and its Xe-HPG row-set form; Few-column XMX product (Strata and NInfer verify with the same kind of kernel); the sampled alias tripwire in Hygiene (a deviation from Strata's full scan); the id tool of the draft vocabulary.
- **FreeToken:** Per-expert cache (the gather of missing experts).
- **Infernix:** Prefill streams each bank tensor on a second queue (the expert stream). Its LFRU policy is a negative result.
- **IPEX-LLM:** Int8 DPAS without local memory for Q4_0 and Q8_0 (its lead was priced and led to 0099).
- **NInfer:** One pass over K and V per KV head for verify rows; Few-column XMX product.
- **vLLM's XPU kernels and sycl-tla:** Prompt attention without local memory (head-256 policy, 2D-read mainloop); GatedDeltaNet writes its rollback snapshots into the recurrent cache; GatedDeltaNet gating computed inside the kernel.
- **llama.cpp's CUDA and Vulkan backends and server:** Integer-dot matrix-vector products (CUDA MMVQ); Quantized prompt K/V converted to f16 (CUDA `launch_fattn`); Split the keys across work-groups (CUDA `parallel_blocks`); Fewer launches in a decode graph (CUDA fusion checks); the fused MoE router and KV rotations as a Walsh-Hadamard transform (CUDA `topk_moe`, `fwht.cu`); the mask classes (Vulkan `flash_attn_mask_opt`); 3D convolution (CUDA `im2col_3d`, Vulkan `conv3d_mm`); Lazy reads of a huge embedding table (llama.cpp's lazy tensor reads, switched on by arcint); Context checkpoints (the server's); the MTP drafter (llama.cpp's `draft-mtp`).
- **OpenVINO and oneDNN (the retired engine's kernels):** K-quant GEMM with 2D block loads; Experts as DPAS "a" operand; GatedDeltaNet recurrence.
- **The arc-b580 fork of ggml:** Rollback by replay; MTP head fills only its K/V; Keep constant matrices on the device.
- **The `leejet/ggml` fork:** Fused rotary-embedding apply.
- **HyperQwen:** Draft vocabulary.

**Read or measured against, not the origin:** Strata's own engine on the B60 (620.5 t/s prefill, in the bank-stream entry); the IPEX-LLM wheel measured on the same cards (Q4_0 entry); HyperQwen's draft timings (draft vocabulary); Strata's B70 and paper figures quoted for the lend region and the int8 expert GEMM; MMQ's per-32 and oneDNN's per-64 activation scales (the int8 Q4_K entry, which departs from both).

## Tried and not adopted

Each row gives the concept, how it was tried, the result as the source states it, and the reason. Every row ends with "Same mechanism?", which says whether the build matched a reference's mechanism or whether the idea is arcint's own; a row that rests on a build that differs from the reference's is a verdict on that build and not on the technique. Where a row names a window, it is one measurement session, and the numbers it quotes come from that session. Patch numbers are labels from the series as it was when the row was written.

### Kernel shapes

- **int8-activation DPAS GEMM in the shape of CUDA's MMQ (first development of 0001).** Built before the fp16 GEMM; reached 20-30 TOPS but moved the dense 27B's prefill KL by 0.0028 nats and top-1 agreement by 0.9 points (`measured-here`). The patch text says the f16 GEMM was faster and left the KL where ggml's float kernels have it; the later campaign text says it was dropped for speed, not quality, because the change was inside the answer-level bar. A per-256-activation int8 Q4_K kernel was later adopted (0013). **Same mechanism?** The README calls it the CUDA MMQ shape; the tile details of this first build are not recorded, so it is a verdict on that build, not on MMQ.
- **A shared decoded weight tile, as CUDA's MMQ builds it (`load_tiles_q4_K`).** Built for Q4_K int8 on the B60: four token groups x four row groups of sub-groups, each decoding a code chunk, scales and constant part into a double-buffered local tile of about 36 KB, one barrier, every sub-group reading its b operands by block read. Correct (1,221/1,221 cases), instruction count 977 -> 838, and 46-59 % slower at every dense shape (`measured-here`, B60, 512 and 2,048 tokens: gate/up 1,098 -> 1,654 us, down 1,195 -> 1,751 us). Ablations: without the barrier 1,429 / 1,707; without the local reads 1,476 / 1,591; one buffer (18 KB) 1,223 / 1,380, still 11-16 % behind. Reason: the per-super-block barrier, the local tile's traffic and its occupancy cost more than the 14 % of instructions the shared decode removes, and the per-(token, column) scale application binds, which the references pay too. Same mechanism as the reference's, in OpenCL sub-group form on this card; the source calls the ceiling "for this design". **Same mechanism?** In structure yes (codes and scales unpacked once into shared memory, read by every sub-group), in OpenCL sub-group form: DPAS has no per-warp MMA and the 6-bit scale is applied by an integer `mad` per sub-block. The source calls the ceiling "for this design" on this card; a different tiling would be a different test.
- **Serving a KV head's query heads from one work-group with 16-key tiles in local memory (decode attention, 0007).** A770 395 us, B60 79 us against 96 us and 57 us for the lanes-along-the-head-dimension kernel (`measured-here`, head size 256, 24 heads on 2 KV heads, 4,096 keys). Not in the patch. **Same mechanism?** No reference; arcint's own alternative to the lanes-along-the-head-dimension layout.
- **Left-outs in the 2D-block GEMM (0012, 0013).** Each measured on the B60 and slower or equal: the next super-block's weights read a super-block ahead (Q4_K 1,229 -> 1,770 us); 96-token tiles; L1 prefetch of the next activation tile (also for 0013); two 16-row weight tiles per sub-group (dense prefill 742-780 t/s against 839); an 8-row activation read (1,066 -> 1,439 us); a banded raster (down -4 %, gate/up +6 %); a Q4_K int8 form with the 6-bit scale split as 8 s_hi + s_lo (1,251 against 1,180 us); the int8 kernel at 16 tokens per sub-group in 128-GRF mode (spills).  **Same mechanism?** Partly: reading ahead and prefetching 2D tiles is what vLLM's Xe2 grouped GEMM does (a lead recorded in the kernel-gap campaign document, on the `qfndev` branch), but each variant was tried inside this one kernel's register budget, so each is a verdict on this kernel.
- **Prefetching the next V tile into registers in the GQA verify kernel (`GQA_PF`, 0018).** 8 alone gained 7 %, but combined with the S-split or at 4, IGC fell back to a rolled form (3,181-3,383 assembly lines against 10,176-12,453) at 2-2.5x the time. Removed in 0019. **Same mechanism?** No reference; a gene of arcint's own structural search.
- **Reordering a DPAS chain to interleave (0041).** Even chunks into one chain, odd into the other: KL held (0.005748 against 0.005950) but the dense agent's greedy acceptance answer flipped to a different, wrong program, 2/10 three times out of three against 10/10. A sum reordered inside the bar can still move a greedy answer; the task gate caught it. The ordered form (same accumulation order) was adopted. Holding part of Q in registers across tiles spilled from four chunk pairs on. **Same mechanism?** No reference; arcint's own change of accumulation order, rejected by the task gate and not by KL.
- **Quantized K/V converted to f16 once per call on the A770 (0048 addendum).** For a quantized prompt call the sub-group-8 build converted both views to f16 scratch and ran the f16 kernel. Correct, +0.9 % (40,841-token needle 725.8 -> 732.4 t/s) and +1.3 % at 81,974 tokens, but peak VRAM 15,977 -> 16,139 MiB of the card's 16,288 (the scratch grows with depth, about 201 MB at 98,304), leaving 149 MiB; a thin margin is the known way into xe eviction. The premise ("f16 is 8 % faster than quantized") was mostly the test's per-call unflattening of uploaded K/V; served, the quantized kernel is within about 1 % of the f16 one. The same mechanism (0037, converted block by block) is adopted on the B60. Same mechanism as CUDA's `need_f16_K/V` with the whole view converted. **Same mechanism?** Same as CUDA's `need_f16_K/V` with the whole view converted (the form that fails the VRAM margin here); the block-wise form of the same idea (0037) is adopted on the B60, where the margin allows it.
- **Staging V as dword pairs, mask classes and f16 conversion on the A770 prompt kernel (campaign `dense-prefill-attention`, on the `qfndev` branch).** The A770's attention is load-bound with local-memory staging and barriers; mask classes at sub-group level ran 74.6 ms against 70.8; f16 conversion of K/V (0037's mechanism) ran 73.7 ms against the quantized kernel's 65.7 because f16 staging reads twice the bytes; V's transposing stores as dword pairs (a `shuffle_xor` between lanes 2i and 2i+1) ran 82.2 ms against 70.3 on the quantized case, not bank conflicts (every V row stride tried lost 10-12 ms), cause unmeasured, reverted; 0048's later pair forms lost 6-11 % on quantized V because the old loop hides V's dequantization under its loads and the pair form exposes it. Each a verdict on that build, as the campaign says; the f16 V pair form is adopted for f16 V only (0048). The 2D-read payload built once per tensor, as sycl-tla does (`FA2_AP`), compiled to the same 1,733 instructions as the flat builtins and was 1-9 % slower on the B60 (23.9 -> 26.0 ms), cause unmeasured. **Same mechanism?** For the 2D-read payload yes (sycl-tla builds the payload once per tensor; the flat builtins here compile to the same code); for the staging variants no reference, arcint's own on a load-bound kernel. The campaign calls each a verdict on the build named.
- **Low-bit expert matvec at 8 rows per sub-group where 16 lanes divide a row (0053 addendum); IQ2_S grid as 16-bit codes.** Faster in isolation (IQ2_S 71.3 -> 67.5 us at three tokens) but about 1 % slower served (verify 11.44 -> 11.57-11.70 s); the 16-bit-code grid slower in every form tried. **Same mechanism?** No reference; arcint's own variants of its kernels.
- **The q/k RMS norm, weight and interleaved M-RoPE as one kernel (0074).** In the first card window, 45 of 227 cases failed on the A770 and a device fault on the B60 because the kernel stepped the positions' offset in elements instead of bytes (so every row was unrotated). Fixed and re-run: 227/227 on both cards; served coder 55.10 / 55.75 off against 53.92 / 55.08 on, agent 41.97 / 42.47 against 42.05 / 42.40, Flash-Next inconclusive; no decode gain, so the switch stays off. The source's explanation of the slowness is `inferred` (one work-group per row with half the lanes idle in the rotation, behind the norm's two barriers), and the fusion saves 2 launches per side, about 0.4 % at the wall by the audit's pricing. **Same mechanism?** No: Infernix fuses this with CUDA templates at T <= 16 (+22-52 % through 256 tokens and a win at every width to 8,192, `paper`, audit); this is a verdict on the OpenCL sub-group form, which idles half its lanes in the rotation behind the norm's two barriers and saves 2 launches per side.
- **Block-ahead prefetch and a page touch for the CPU miss rows (0075).** Strata's cold reads are file-backed rows; here the bank is USM host memory written at load, so its pages are resident before the first miss (a different build, named in the source). In one card window, Flash-Next with the cap at 64: level 0 34.12 / 34.40 t/s with 0 and 10 major faults, level 1 34.28 t/s (inside the spread), level 2 30.12 and 33.73 with one leg at +479,730 major faults. The level-2 spike came from addressing an IQ4_NL split bank by the tensor's `nb[2]` stride; fixed, but the touch still buys nothing by the premise. Off. **Same mechanism?** No: the premise differs. Strata's cold reads are file-backed rows; here the bank is USM host memory written at load, so faults are about zero by construction.
- **int8 Q Kᵀ from the q8_0 cache, and its 1-3-row staged form (0088, 0089).** Q rows quantized to int8 once per call, the q8_0 key block's raw code bytes as the DPAS B operand, the integer result scaled by `dq * dk`. Opt-in. The first card run failed 135 of 135 attention cases at ERR 0.10 because the Q staging loop covered only the row-tile count (a layout error, not rounding); fixed. Served on the agent at a 115,395-token prompt, decode off 21.03 t/s, int8 alone 15.83 (-25 %), the 1-3-row staged form alone 19.05 (-9 %), both 13.27 (-37 %), same answer (`measured-here`). Not a verdict on the fork's technique: the fork's build is SYCL s8 x s4 over q4_0 keys with Q already in registers, ours OpenCL s8 x s8 over q8_0 keys with Q quantized per call into local memory. The A770 cannot take the form (no s8 x s4 k32 in the public extension; the fork's vISA is Xe2 SIMD16 only). **Same mechanism?** No: the fork's build is SYCL s8 x s4 over q4_0 keys with Q already in registers; ours is OpenCL s8 x s8 over q8_0 keys with Q quantized per call into local memory. A verdict on this build, not on the fork's technique.
- **GDN state read straight from the cache rows (0081, fixed in 0090), joint q/k l2 norm (0082), round-trip trims 1 and 4 (0083), fast top-k (0084).** All opt-in; 0094 and the CHANGELOG record them as measured without a gain and kept for further work. The runs behind that, per patch: 0081 (after 0090's fix) one coder run with the switch on, texts md5 identical, decode 51.5 -> 51.2 t/s (no gain), and one Flash-Next run with the exact bundle, answers right, prefill-20k 536.9 and decode 32.8 t/s against 534.2 and 33.1 in an earlier off arm (README 0090); 0082, 0083 bits 1 and 4 and 0084 have no card run in their README sections (the arms were listed as owed), and I found the runs behind "gave nothing" in none of the sources, so for those three the verdict rests on the CHANGELOG's statement. 0081's OpenCL kernels used the cache-row base for the writes as well as the read, so the state landed at the wrong offset (a served answer gate failed, decode 51 -> 21 t/s; the B60's first rows case faulted the compute engine); 0090 splits the base. The test cases that should have caught it were registered only in the perf list, which times and never compares, and a chain case aborted on a 128-tensor context pool, so none had completed on a card (0092, 0093). The reference figures (+3.8 % decode for state rows; +7.7 % 2B and +2.6 % 27B for the hybrid bundle) are `paper` from the fork's Metal and SYCL stacks. **Same mechanism?** Yes for 0081 (the fork's rows form, with its CPU reference and SYCL reads, ported to OpenCL), 0082 and 0084 (the fork's joint norm and its partial-sort-equivalent selection); the evidence is thin, one run for 0081 and none recorded for the rest.
- **A fixed-size 72-frame speech decoder window, an `ARANGE` op for the card-only predictor (0061).** The op suite passed (2/2, red 0/2), and the gain was weak: RTF 0.267 -> 0.280 (-4.8 % wall), GEN_CODE splits 1,007 -> 299. Its only beneficiary is the card-only predictor path, which the hybrid placement does not use, so the patch was dropped from the numbered series; the diffs are regenerable. **Same mechanism?** No reference; arcint's own op, dropped because its only beneficiary path is not adopted.
- **Int8 DPAS expert products on the A770, first form (0103).** The prompt's IQ4_XS, IQ4_NL and Q4_K expert products on sub-group 8 with int8 DPAS, one sub-group per 8 weight rows, activations packed per route, no local memory. Alone it was a loss (`measured-here`, the coder's file on the A770, 4,096 tokens, ubatch 1,024, f16 KV): prefill 1,918 -> 1,636 t/s, 1,680 with Strata's byte-pair codebook table; tile sizes 16, 32, 64, 128 gave 1,490, 1,636, 1,578, 1,082. Taking the kernel apart with timing-only switches (IQ4_XS, 2,797 ms over the profile against 2,042 for the fp16 tile): activation reads 50 %, weight reads 3 %, decode 20 %, stores 1 %, DPAS and the rest 20 %. Reason: the activation reads, which 0104 then divides by letting a sub-group own several row sets (adopted, see the entry on int8 expert products on Xe-HPG). **Same mechanism?** No, and the source lists the differences: Strata runs int8 only on Battlemage by default and keeps Alchemist on FP16, so an Alchemist int8 path has no reference result; its thread owns 16 rows at sub-group 16, here 8; it picks 2, 4 or 8 row tiles per decode from the experts' mean rows, here fixed; and it scales the activations per 32 of K, here per 256 (8x coarser, the likely part of the KL cost). The first form is a verdict on that build.

### Expert cache, bank and prefill

- **Decode-once grouped expert matvec in the verify window (0031 and the grouped kernels, removed in 0.8.1).** Strata groups the window's (token, expert) pairs by expert, the bank's first, and decodes each expert's weight once. Built here (`GGML_OPENCL_KQ_GROUPED=1`): correct and slower, 30.5 -> 29.5 t/s, verify 15.01 -> 15.56 s, IQ2_S in isolation 120.6 -> 146.9 us. The audit counts this as the reference's mechanism, so the negative counts, while stating that a different kernel shape would be a new candidate; the reference's own note is that the grouped form pays only where an expert has 4 or more entries. Related removals from 0022, each measured: the hyper-connection read fused into three kernels (30.8 t/s off, 30.4 on); a projection's matvec copying the next projection's misses into a mirror (31.4 -> 31.6); a matvec grouped by expert for 2-15 tokens (the coder's verify 7.85 -> 9.18 s on the A770); and the activation as one int8 term, as llama.cpp and Strata do instead of codes plus residual codes (IQ4_XS 49.6 -> 43.9 us at 3 columns and Flash-Next 31.0 -> 31.6-31.8 t/s, but the coder 43.3 -> 40.7 t/s with its drafts accepted 42.7 -> 38.2 %). **Same mechanism?** Yes in form (Strata's `row_dot_multi` with up to 4 entries per group, groups from a per-layer kernel, the bank's experts first), measured at MTP 2, where a verify has 3 tokens and the measured overlap between tokens' experts is 28 %. The reference's own note is that the grouped form pays only where an expert has 4 or more entries and the per-entry cost is low, which a window this narrow cannot give; the verdict holds at this window width, and a different kernel shape is a new candidate.
- **Infernix's LFRU expert-cache policy (0055, dropped in 0.8.1).** Score = uses / (now - last + 1), the round's routed experts protected, a miss admitted only over a lower-scoring victim, within a budget per adapt. On the gate set +1.6 % on the means (40.4 against 39.8 t/s over four and five processes; the first record said +1.3-1.8 %, an arithmetic slip corrected in the 0.7.10 release review); on six other prompts -2.1 % (32.6 against 33.3), per prompt -6.5 to +1.2 %, with 1.5-2x as many swaps (3,840-5,376 per request against 2,265-3,359). Differences from Infernix, named in the source: not built were its single pool over all layers (the slots here are per-layer tensors) and its warm start from saved state; the clock and the candidate set had a faithful variant (`LLAMA_EXPERT_CACHE_LFRU_EXACT=1`), measured +1.4 %. A later history survey (`research-strata-history`, on the `qfndev` branch) reads it as: every past-based policy sits near 90 % of decode's routed experts under 96 swaps a plan, and LFRU pulls ahead only with more swap throughput, so the price of a swap decides. **Same mechanism?** Partly: the policy was built per layer inside the existing slots and swap queue; Infernix's single pool over all layers and its warm start from saved state were not built, so the verdict is on the per-layer build, not on Infernix's global cache.
- **Staging every graph input, including above 4 MiB and CPU-island copies (0078, dropped in 1.4.1).** The 1.4.0 release gate pooled nine runs of each build in three passes: Flash-Next decode 32.97 against 33.70 t/s (-2.2 %); the bisect (four rounds each) named this patch (removing it: 33.75 against 32.85, +2.7 %). The mechanism is not measured; the likely path is the CPU-island copies, which went through the same scheduler function as the graph inputs. The coder's checkpoint-restore gain (time to first token 2.07 against 2.17 and 2.37 s) went with the patch. A narrowed form (graph inputs only) is owed. **Same mechanism?** No reference; arcint's own bound on 0047.
- **A JIT option for buffers above 4 GiB (retired, 0076).** On dg2 the option moves every kernel from stateful to stateless addressing; the A770's `CL_DEVICE_MAX_MEM_ALLOC_SIZE` is 4,294,959,104 bytes so no `cl_mem` exceeds 4 GiB, and on the B60 all builds are already stateless. Read from offline disassembly (`measured-here`) and the OpenCL type split (`code`). **Same mechanism?** Not a technique comparison: a compiler option, read from offline disassembly.
- **Layer split across two cards (0023, 0024).** Flash-Next across the B60 and the A770, a layer range and an expert cache per card (campaign `layer-split-two-cards`, on the `qfndev` branch, still open). In the first window, split at layer 40 with the default MTP placement: decode 18.6 t/s against 19.8 on the B60 alone, prefill 378 against 408 t/s. With the draft model on the B60 (`--llama-mtp-card 0`): K=40 20.4 t/s, K=36 21.7 t/s, which is +9.6 % over the B60 alone at the same context, one run each, prefill -4 %, against a run-to-run spread of about 3-5 % on that request (`measured-here`, 2026-10-08). DESIGN §3.10 calls it "not a production configuration"; the sources I found record no measured reason against it beyond that the gain is within a few spreads and that a model which fits on one card loses decode when split (the coder: 77.2 t/s on the B60 alone, 57.5 and 55.0 split). The first window's "gate not met" was for the default placement. **Same mechanism?** Yes: Strata's layer split (contiguous ranges, the last card runs the head, an expert cache per stage), with the host hand-off through ggml's scheduler copy.
- **Named lanes: an agent lane and a subagent lane in one process.** Ended after a test drive: a lane slowed with the other lane's pool fill (subagent 14.2 t/s with the agent 130k deep). Patches 0026 and 0029 fixed that (the subagent lane 32.1 t/s at any agent fill), and the lanes configuration stays available; the production agent runs one lane. **Same mechanism?** No: llama.cpp's server has one cap for all unified slots and no per-sequence range, so the window mechanism (0026, 0029) is arcint's own; the lane arrangement itself was ended for cost, then fixed.
- **Earlier negatives that did not test the reference mechanism (OpenVINO-era record, `docs/campaigns/research-reference-audit.md`, on the `qfndev` branch).** A per-layer LRU tier that also chose which device computed (rejected for the rule, not speed; later Strata's adapt was built as 0021 and 0042); a hybrid expert fetch from pageable memory with a fixed share (the reference uses pinned memory and a link-probed share); prefill expert streaming v1 with CPU-thread staging and one queue (later built as the bank stream, 0032, which gives 408 -> 645 t/s); "MTP on Flash-Next: no head" (the checkpoint carries the head). These are why the rule "an arcint negative is settled only if it tested the reference's mechanism" exists.

### Speculative decoding

- **A DFlash2 block drafter in place of the MTP head.** Built on libllama (`--llama-dflash-gguf`), opt-in. Against MTP 5 on the agent at 212,992 tokens, the Q4_K drafter, draft ubatch 64 (`measured-here`, B60): +27 % on a code answer, -5 % prose, -8 % at 4k, -2 % at 55k, -33 % at 130k, -36 % at 200k. The reference's own server loses past about 60k too (+30 % at 4k, +15 % at 58k, -26 % at 135k; its drafts accepted 37 -> 16 %), but below that the reference gains and the port does not (-8 %, -2 %): the gap is open and not attributed. The acceptance task scored 0/10 at 212,992 (MTP 5 and 7: 10/10, plain 8/10), twice the same bytes, not attributed. The unit keeps MTP 5 (operator). **Same mechanism?** Yes for the drafter (llama.cpp's `draft-dflash`, rebuilt on libllama); the reference gains below about 60k (+30 % at 4k, +15 % at 58k) where the port does not (-8 %, -2 %), so the port's gap there is a build difference not yet attributed.
- **Cost-aware width on Flash-Next and the agent.** -2.6 % on Flash-Next and a flipped greedy task on the agent; adopted on the coder only. **Same mechanism?** Partly: Strata's rule with the 0.5 min-p floor retained, where the reference's cost arm has no floor (`research-mtp-width-policy.md`, on the `qfndev` branch, which says a faithful price needs a min-p 0 arm).
- **A larger MTP window on Flash-Next without a policy.** Widths 4 and 5 lose 5.5 and 8.4 % against 3 (acceptance 60.3 and 54.8 %). **Same mechanism?** arcint's own measurement of fixed widths with the min-p stop, no reference build involved.
- **Lowering the CPU-miss cap on Flash-Next.** Caps 32, 64 and 128 lie within the window's spread; 16 binds, at -7.4 %. **Same mechanism?** arcint's own tuning of Strata's cap.
- **Four drafts on the dense model before the few-column product.** Fell to 10.7 t/s from 19.75 plain (a 5-row verify on a matvec that took at most 4 columns), the reason for 0009. **Same mechanism?** Not a reference comparison; a kernel gap found by measuring.

### Runtime modes

- **Level-Zero-backed OpenCL (LEO) as the runtime route (0072, 0079).** After the launches were made uniform, the B60 under LEO passes exactly the default driver's cases (12,724) and costs prefill -0.28 % at 55k and -0.08 % at 138k. The A770 first aborted inside the runtime's DG2 command generation after 4,615 cases, at MXFP4 m=2,880 ("in this release LEO is supported only on the later products"); that is the DG2 image-width abort (16,384 texels fit, 16,385 abort) that 0079 works around. With 0079 the A770 runs under LEO at 12,867 of 12,881 cases, the coder decodes +6 % and prefills -5.6 % against the default driver (CHANGELOG 1.4.0; the 0072 text had -5.3 % at 41k, 730.15 -> 691.15 t/s, and +5.9 %); the prefill loss, unexplained and owed a profile under both drivers, is the reason the units stay on the default driver. **Same mechanism?** Not a technique comparison: LEO is a runtime mode, and the measured loss is for this runtime release.
- **Moving the pin to upstream master.** The move costs a port of the host side, not a re-apply: 43 of 55 patches touch `ggml-opencl.cpp`, where upstream changed 445 lines in 8 commits, among them a split of `load_cl_kernels`. The pin moved once (2026-10-15, `bed0a85` to `3e4dfa8`: 46 of 56 patches applied unchanged, 10 resolved by hand; two semantic changes found by running it). A patch that does not apply cleanly is a bug in the series and not a reason to move the pin. **Same mechanism?** Not a technique comparison.

