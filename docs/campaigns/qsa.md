# qsa — Qwen Sparse Attention served in Flash-Next's 12 full-attention layers

**Closed 2026-09-30 (served, non-default).** `--qsa` exports the indexer
(`tools/q4e/attention.py`, `tools/q4e/serving_shape.py`) with a compressed
block-key cache: one pooled, normed, roped f32 row per complete 4-token block
in a fixed `[8192, dh]` Variable per QSA layer plus the raw tail, so
`n_ctx ≤ 32,768`. The selection marker travels in model rt_info and is
re-applied at load; the paged attention takes the selection as input 29 and
decode reads only the chunks that hold a selected key. `d48q8qsa` answers the
needle at 20,085 tokens; B60 decode 5.1 t/s against dense 5.9, prefill 1.27x
the dense time (`measured-here`, LYON-001 in `docs/window-054.md`). Dense
`d48q8` stays the served artifact. Owed: the f32 reference re-capture with the
fixed BF16 indexer feed (T8) and the KL above 2,051; one lane, no prefix
cache and no paged speculation with QSA today.

Reopen against the references (`research-qsa.md`): vLLM's paged
compressed-key cache under the KV block table; llama.cpp #28213's gather of
the selected K/V; Strata's QSA kernels
(`~/src/Strata-ref/src/kernels/cuda/qsa_select.cu`, `qsa_decode_attn.cu`,
`native_qsa_indexer.cu`), which append to the indexer at commit time so
speculation keeps working (`code`: `src/core/verify.cpp:990`). Gains start at
≥ 64K context in the paper's own curve (`paper`).

Full history: `git show b0447b8:docs/campaigns/qsa.md`.
