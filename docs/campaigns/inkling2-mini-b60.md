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

2026-11-22 02:30: the port to the pin in progress (four of seven commits: arch and hparams, the
relative-position input and conv-state store, the text model and loader, the expert scales); arcint's
chat side in progress; the FA design note written (`inkling2-fa-bias-design.md`); mainline's Vulkan build
for the baseline arm built; weights downloaded (IQ4_NL complete). Nothing measured.

## Where it lives

`contrib/llama.cpp/patches/` (the port, numbers from 0133), arcint's chat/reasoning code for the template
and parser (`src/core/`), this document.
