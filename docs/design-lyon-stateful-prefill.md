# design-lyon-stateful-prefill — serving-length prefill (0.5.4 LYON)

Acceptance rows: `docs/window-054.md`.

## What exists

- **Compile once, replay at every length** (`code`): the served serving-shape
  graph is T-dynamic (`tools/q4e/serving_shape.py`, `stateful_gdn_core`,
  `stateful_short_conv`, `emit_stateful_attention`); a served GDN block is 164
  nodes whatever the prompt length, the full-depth 35B graph 20,658 nodes
  (`measured-here`, `tests/python/test_gdn_block.py`). The GDN core is a
  token-sequential `Loop` that the plugin fuses into its `GatedDeltaNet`
  primitive.
- **A chunked stateful core** (`stateful_gdn_core_chunked`, selected by
  `Q4E_GDN_CORE=chunked` and `Q4E_GDN_CHUNK`, beam-free): CHUNK tokens per
  iteration, byte-exact against the chunked algebra (`measured-here`, CPU). It
  stays in-graph, outside the fused primitive.
- **The 35B prefill** on the A770, full depth, all-resident: 952.1 t/s at 4096
  tokens after patches 0059–0064 (`measured-here`;
  `docs/design-native-dpas-expert-kernel.md`). The GDN core was 1.5 % of its
  device time in the patch-0059 profile.
- **Flash-Next prefill** on the B60 (`d48q8`, CPU expert tier): ~61–65 t/s at
  20–27k tokens (`measured-here`). The GDN core is 0.3 % of the prefill wall;
  the CPU tier is ~152 ms per MoE layer call, ~51 s of the ~80 s a
  4,096-token prompt adds (`measured-here`, 2026-09-28).

## Open: experts streamed to the card during prefill (the binding term)

Mechanism to build: at prefill chunks of 1,024 tokens and above, every
non-resident expert of a layer is copied to the card from pinned host memory
through a ring of device slots borrowed from the expert cache, on a copy queue
of its own, the next layer's copies overlapping the current layer's compute
(ready/release events per slot), and the card computes every routed expert
with quantised grouped kernels. References (`code`):

- Strata `src/prefill/prefill.cpp`: `stream_all_min()` (1,024), `ring_slots()`
  (384 slots when nearly every streamed expert is DMA'd from pinned RAM, else
  96), a pinned stager ring for the experts the arena could not pin; grouped
  quantised kernels in `src/prefill/moe_mmq.cu`; chunks up to 8,192.
- FreeToken `python/freetoken/moe/offload_cache.py`
  (`begin_prefill`, `prefetch_prefill_layer`): two layer buffers, the next
  layer copied on `prefill_copy_stream` with ready/release events while the
  current one computes; `max_extend_tokens` 8,192
  (`python/freetoken/scheduler/config.py`).
- Effect: Strata reads a 32K prompt at 1,750 t/s with IQ3_XXS experts on an
  RTX 5070 12 GB + 64 GB DDR5 (`paper`, Strata `README.md`).

Campaign: `docs/campaigns/prefill-expert-streaming.md`. It needs the pinned
host bank (`ttm.pages_limit`; `docs/campaigns/research-reference-audit.md` §4
item 4) and prefill chunks above 2,048 on the B60.

## Open: a fused chunked GDN primitive

The fused `GatedDeltaNet` primitive advances one token per step; the chunked
core runs only in-graph. A chunked GDN kernel follows FLA's chunked
gated-delta rule as FreeToken carries it (`code`,
`python/freetoken/kernel/fla/chunk.py`, `chunk_delta_h.py`, `chunk_o.py`,
`wy_fast.py`, `solve_tril.py`). Strata's prefill keeps the recurrence per
token and pipelines the next token's inputs into registers (`code`,
`src/prefill/kernels.cu`, `gdn_rec_cols_pipe_kernel`). Priority follows its
share of prefill time (above).

Full history: `git show b0447b8:docs/design-lyon-stateful-prefill.md`.
