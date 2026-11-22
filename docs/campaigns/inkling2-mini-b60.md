# inkling2-mini-b60 — serve Inkling 2 Mini on the 24 GB card, at the speed its design allows

**Open** (operator, 2026-11-22: "B60 first").

## Charter

Serve Thinking Machines' Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, Apache-2.0,
revision `b85c3cfa`) on the B60 through arcint's libllama engine, with:

1. the loop mechanism as the model defines it (core layers 6-17 run three times; every pass keeps its own
   KV and short-convolution state; passes 2 and 3 mix the previous pass's messages through rank-64 gates);
2. flash attention that carries the model's learned relative-position bias and per-head sinks inside the
   kernel (decode and prefill), not a dense per-head bias tensor;
3. the optimisations the references use (fused small ops, experts cached once across loop passes, the
   bundled DSpark drafter), not the naive graph;
4. a weight format chosen on measurement (see "Weights").

## The model (`paper`: config.json, the model card, the merged llama.cpp PRs)

26.4B parameters, 3.4B active; 128 routed experts, top 8, no shared expert; 24 physical layers, hidden
2304; 24 query heads, 8 KV heads, head dim 128. Five sliding-window layers (window 1,024) and one global
layer per block of six (global: 5, 11, 17, 23). No RoPE: a learned relative-position bias table
(`d_rel` 16, extent 1,024) added to the attention scores; global layers also scale q by
`1 + 0.1 log(max(1, (pos+1)/8192))`. Per-head sinks from a projection, a sigmoid output gate, a QK L2 clip,
V RMS-norm without gamma. Causal short convolutions (kernel 4) on K, V, the attention output and the FFN
output, carried as recurrent state per layer visit. Loop: core layers [6, 18), 3 passes, fixed; 48 layer
visits per token; KV is not shared between passes (8 global KV streams). Vocabulary 201,024; context
1,048,576. A bundled 3-layer drafter ("DSpark", `mtp.safetensors`, blocks of 8). Released weights: experts
NVFP4, the rest MXFP8 (19.5 GB); ggml-org's GGUF conversion: BF16 52.2 GB, Q8_0 27.8 GB, IQ4_NL 17.6 GB.

## Reference to follow

- **The model graph and memory:** llama.cpp mainline at `690b87d8` (`code`), the commits
  `5c008e00` (#32217, Inkling 2 text and vision), `dd099a6b` (#32294), `4ddf2b94` (#32381, the chat parser),
  `d998c676` (#32430), `8af68d48` (#32429, the attention mask), `a1f42226` (#32657), `29b52d39` (#32694, the
  Mini and its loop: `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, the hybrid-iSWA memory
  input with relative positions `build_inp_mem_hybrid_iswa_rel`) and `690b87d8` (#32704, the loop slots
  after the scale pass). Every ggml op the graph uses exists at our pin `3e4dfa8` (`code`, checked
  2026-11-22); the new parts are in libllama's graph, memory and hparams.
- **Flash attention with the bias in the kernel:** llama.cpp #32740 (open; CUDA, ROCm, Vulkan, declined
  on OpenCL and CPU): the bias stays a table `[n_head, n_bins, n_tokens]` (per query token, since the bias
  is low-rank and token-dependent: `attn_rel_bias · Rcur`, minus the sink; 1,025 bins, 98.4 KB a token)
  indexed by `[n_kv, n_tokens]` bins inside the kernel, `mask + bias` rounded to f16 once (`code`). Its
  op slots `src[6]`/`src[7]`/`op_params[5]` collide with arcint's paged attention, so the port takes its
  own slots. The plan for our kernels (`flash_attn_dpas`, `flash_attn_gqa_dpas`, the split decode and its
  merge, compile-time variants, six red cases): `inkling2-fa-bias-design.md`. The alternative to price:
  GodGMN's in-kernel bias from the static per-layer weight (16 MACs per row and key, no per-token table).
- **Fusions:** the GodGMN `inkling-rdna3` branch (`docs/inkling-rdna3.md`): L2 clip, the loop controller,
  depth mixing, short conv and concat + state store fused, ~7,650 -> ~3,000 kernels a token, decode 31.9 ->
  64.0 t/s on an RX 7900 XTX (`paper`); vLLM #69028's stacked controller GEMMs and the FP32 merge fused
  with RMSNorm (`paper`).
- **Experts once across passes:** llama.cpp #32744 (the MoE cache held loop-layer experts three times:
  19.2 vs 28.4 t/s) (`paper`).
- **The drafter:** llama.cpp #32218 (DSpark, open), vLLM's 7-token probabilistic drafting (acceptance
  length 3.58 on GSM8K) (`paper`).

## Gate

- **Port fidelity (CPU):** the ported libllama on our pin, CPU backend, against mainline `690b87d8` CPU on
  the same GGUF and prompts: mean KL <= 1e-4 nats, argmax agreement >= 99.9 % (it is the same code; a
  larger gap is a port defect).
- **Answers on the card (answer-level bar, `CLAUDE.md`):** the B60 arm against the BF16 GGUF's CPU logits,
  compared with the baseline arm = mainline llama.cpp's Vulkan backend on the same B60 and the same GGUF:
  mean KL at most 0.03 nats worse than the baseline's, argmax agreement at most 1 point lower; Paris; the
  needle at 100k; the acceptance task.
- **Speed, per phase:** decode and prefill at 4k / 32k / 128k against the Vulkan baseline on the same card
  and power cap; the published RDNA3 / Ampere numbers are context, not the gate.
- **Weights:** native NVFP4 experts (the released bytes) against a requantized GGUF, by the answer-level
  bar and decode speed; an iwolucion search only if the budget demands a smaller file.

## Current state

2026-11-22 03:45 (`measured-here` unless marked):
- **Baseline arm** (mainline `690b87d8`, Vulkan, B60 at 165 W, ggml-org IQ4_NL, llama-bench, 2 repetitions):
  pp512 33.3 / 32.3 / 30.1 t/s and tg128 8.3 ± 4.1 / 10.1 / 10.2 t/s at depth 0 / 4,096 / 32,768. A Paris check
  through mainline's server spent 600 tokens reasoning at the template's default effort (0.9) without an answer:
  unchecked, not wrong.
- **Port to the pin, CPU:** against mainline's CPU build on the IQ4_NL GGUF (English and code, 3 chunks of 2,048):
  mean KL below print precision (< 5e-7 nats), max 0.000055, same top token 99.967 %, PPL 3.485480 against
  3.485478; the red case (2 loop passes instead of 3) reads mean KL 0.499, same top 76.3 %. The q log-scaling on
  the global layers is exactly 1 below position 8,191, so a red case on it needs a longer context. Experts are held
  once per physical layer; the loop's cache slots alias them. ggml SCALE on row-strided views took mainline's loop
  (red first: the pin's loop failed the view cases at NMSE 0.65-1.32).
- **The port on the card** (OpenCL, B60 at 165 W, patches 0133-0139 + 0145, no FA patches yet, 2026-11-22 05:00):
  0 of the model's ops unsupported (the graph's 150 prefill / 144 decode shapes); 0145's SCALE on strided views,
  LOG, ARANGE and the 2-5-wide SOFT_MAX pass on the card; a raw completion "The capital of France is Paris." at
  18.2 t/s (llama-simple, 24 tokens; not comparable to llama-bench); KL against the CPU port's logits with FA on:
  mean 0.00197 nats, max 0.070, same top 98.50 %, PPL 3.4847 against 3.4855 (FA off: 0.00205, same top 98.27 %).
  Slow paths named by DOWNGRADE lines: MUL_MAT_ID on Q8_0 (the coda layers) and several f32 MUL_MATs on the
  generic kernel; the prompt FA with sinks on the generic f32_f16 kernel.
- **arcint serves it** (2026-11-22 06:15, B60 at 165 W, arcint c0527425 = the port 0133-0139/0145, the FA bias 0140-0144,
  the chat dialect, the admission entry and the null-BOS fix; IQ4_NL, q8_0 KV, 32,768 context): Paris at reasoning
  effort low, unset and sampled (the reasoning split into its own field, 169-187 characters), a correct tool call;
  decode 18.8-19.8 t/s (19.0 over a 4,000-token answer), prefill 53-128 t/s on short prompts. The acceptance task
  (the harness's own request at reasoning_effort, greedy, 12,000-token cap): **10/10 four times**, at low (2,111
  tokens, 19.4 / 20.9 t/s) and at medium (4,270 tokens, 18.7 / 18.8 t/s); at the template's default effort 0.9 with
  a 4,000-token cap it reasoned to the cap (the harness's enable_thinking=false means nothing to this template).
  A 4,697-token prompt: prefill 127.7 t/s, decode 17.8 t/s; against the Vulkan arm at depth 4,096 (pp512 32.3,
  tg128 10.1 t/s) about 4x and 1.8x, but from different tools (arcint's server lines against llama-bench):
  indicative until the same tool measures both.
- **Same tool against the baseline arm** (llama-bench, pp512 / tg128, -r 2, B60 at 165 W, IQ4_NL; the integration
  series = the pin + 112 patches, OpenCL; 2026-11-22 07:00): FA on pp512 172.7 / 139.1 / 106.5 t/s and tg128 23.0 /
  20.1 / 18.4 t/s at depth 0 / 4,096 / 32,768 (FA off 174.0 / 144.4 / 97.1 and 23.6 / 21.4 / 19.5), against Vulkan's
  33.3 / 32.3 / 30.1 and 8.3 / 10.1 / 10.2: prompt 5.2x / 4.3x / 3.5x, decode 2.8x / 2.0x / 1.8x. The graph still
  folds the bias into the dense mask (the wiring onto the indexed op, 0146, is open).
- **The bias through the indexed FA op** (0146, the full stack aa9eb32d = the pin + 113 patches, 2026-11-22 07:35):
  CPU gate both arms (indexed, and the dense fold with the op forced declined) mean KL 0 against mainline, same top
  99.967 %; red (the table read dropped in the CPU reference) 4.01 nats on the indexed arm only. On the B60: KL
  against the CPU base 0.00199, same top 98.53 % (the dense fold: 0.00197 / 98.50 %); llama-bench pp512 183.9 /
  177.5 / 164.1 and tg128 24.0 / 22.6 / 20.8 t/s at depth 0 / 4,096 / 32,768 (+54 % prompt and +13 % decode at
  32k over the dense fold; against the Vulkan arm 5.5x prompt, 2.0-2.9x decode). The acceptance task, sampled
  (temperature 1.0, effort low), interleaved: dense fold 50/60 (one 0: the function declared `local`, the task
  asks for a global), indexed 60/60; a greedy run had flipped to the same `local` slip, which both builds show.
- **Speed work** (Markus: performance first). Decode profile of one token (B60, the profiling build of the full stack):
  33.2 ms of device time in 6,977 kernels (4.8 us each; wall ~42 ms): Q8_0 matvecs 6.6 ms, IQ4_NL experts 4.8 ms,
  CONCAT 5.7 ms in 356 launches and REPEAT 2.4 ms in 120 on 2304x1..5 tensors, small f32 products 2.7 ms, the
  per-pass elementwise ops ~4.7 ms, flash attention 0.48 ms. 0147 (one element a work item for small CONCAT /
  REPEAT; the generic kernels walked a 2,304-element row serially): tg128 23.40 -> 28.48 t/s (+21.7 %) at depth 0,
  22.60 -> 26.75 (+18.4 %) at 4,096, prefill level. Measured and dropped (CPU): a loop self-draft (1 pass: same top
  70.4 %, 2 passes 76.3 %: no gain at a draft cost of 0.5-0.75 forward a token); expert reuse across passes
  (consecutive passes share 58 % of the experts; the model is resident, nothing to prefetch).
  The output head (Q8_0, 0.49 GB a token) requantized from BF16: Q6_K mean KL 0.000115, same top 99.19 %, -23 %
  bytes; Q4_K KL 0.00116, same top 98.37 %, -47 % (CPU, against the Q8_0 head's logits): inside the bar, worth
  ~0.25-0.5 ms a token; it goes with the weight-format decision.
  0148 (the depth-mix states as one stack, a row per block, read by a view): CONCAT 548 -> 288 and REPEAT
  120 -> 0 per decode graph; card tg128 29.09 -> 29.50 (+1.4 %), KL identical. Profile after it: 25.9 ms device
  in 6,730 kernels, wall 33.9 ms. The gap was the host (`measured-here`, unitrace: ~6,730 enqueues a token at
  ~3.6 us host each, 49 blocking reads, ~154 waits), and gdb plus `GGML_SCHED_DEBUG=2` named the cause: SITU's
  `sigmoid` of the gate half of gate_up (a strided view) was not admitted by OpenCL and ran on the CPU, a graph
  split per MoE visit (48 a token). Graph rebuild was not it (CPU: 14 of 15 decode graphs reused). 0149 (a
  strided f32 sigmoid kernel): CPU splits 49 -> 1 a graph; pp512 185.0 -> 417.9 t/s (+126 %) at depth 0 and
  178.4 -> 387.8 at 4,096; tg128 29.50 -> 34.86 (+18 %) and 27.62 -> 32.25; KL 0.001931 (0148: 0.001986).
  Inkling decode on the B60 since the start of the speed work: 24.0 -> 34.9 t/s at depth 0, prefill 184 -> 418.
  The same class of gap elsewhere (`measured-here`, `GGML_SCHED_DEBUG=2`, the decode graph of each production model on
  the full series tree, flash attention on, q8_0:q4_0 KV): the dense 27B, the iwo4 coder file and Dory on the B60
  have one CPU split, the token embedding; Flash-Next on the B70 with its experts put on the CPU on purpose
  (`-ot exps=CPU`) has only the expert blocks and the embedding there (the CPU-only lightning indexer is switched
  off by `resolve_fused_ops`, as intended). arcint's MTP draft graphs were not in this check.
- **The B70** (ib1, `measured-here`, one Arc Pro B70 at 220 W, `ZE_AFFINITY_MASK=1`, the same 0149 build, IQ4_NL):
  pp512 658.2 / 627.2 / 550.6 t/s and tg128 40.12 / 39.22 / 37.47 at depth 0 / 4,096 / 32,768 (B60: 417.9 / 387.8 /
  331.3 and 34.86 / 32.25 / 28.79); mean KL 0.001931, same top 98.501 %, as on the B60; one CPU split (the
  embedding); arcint 42e8525b served: Paris, the acceptance task 10/10 at effort low (served decode 40.2 t/s). The
  same kernels run on both cards (Xe2). Decode gains 15-30 % on a card with +33 % dense decode bandwidth: Inkling's
  decode is bound by its ~6,800 launches a token, not by bytes.
- **Flash attention on the card** (fa3, fbfcece): all 17 bias cases run and pass (decode 1-8 rows through the GQA
  and split kernels, the 512-row prompt through dpas2), the full FLASH_ATTN_EXT suite 3,171 / 3,171, the decode and
  the Xe2 prompt mutations red. The 9-64-row single-batch shape is declined on purpose (reason=row-blocks).
- **Chat:** an Inkling dialect in arcint (minja renders the shipped template; the golden render equals mainline's
  byte for byte).
- **Flash attention, step 1:** the indexed-bias op in arcint's own slots and its CPU reference; the FA suite
  5,815 / 5,815 on CPU. The OpenCL kernels are next.

## Where it lives

`contrib/llama.cpp/patches/` (the port, numbers from 0133), arcint's chat/reasoning code for the template
and parser (`src/core/`), this document.
