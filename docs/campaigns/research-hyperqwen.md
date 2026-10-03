# research-hyperqwen — HyperQwen's Qwen3.8-27B serving stack, read for the libllama engine

HyperQwen (syv-ai, `github.com/syv-ai/HyperQwen`, read at e1459c7 on
2026-10-03): vLLM 0.30.0 plus 45 patches serving Qwen3.8-27B on one RTX 3090
(250 W). Read from its source (patches, scripts); where its README and code
disagree, the code is quoted. Evidence classes: `code` (file:line in that
tree), `doc-measured` (its own tables, no raw output in the tree),
`bench-raw` (the one raw capture, `bench/demo/data.js`, recomputed here).

## What it does, ranked for the dense 27B on the B60

1. **A cheap MTP step: a draft vocabulary and an int4 head.** The MTP drafter
   scores a 40,960-row slice of lm_head (`patches/qwen3_5-mtp-draft-vocab.patch:30-43`,
   the rest -inf at `:65-76`; ids in `prepare/draft_vocab_ids.json`, counted
   over 5.4M of the model's own tokens, 97.5 % coverage, 96 % on code); MTP
   module and head in GPTQ int4 (`drafter/README.md:14-28`). Exact. Their
   draft: ~3 ms with the full bf16 head, 0.5-1 ms after (`docs/optimizations.md:75-88`);
   98.0 -> 108.6 -> 118.8 t/s (`drafter/README.md:14-28`); an MTP step 24.8 ms
   (`optimizations.md:234`), against 21.7 ms a token in their batch-mode
   table at one request (`batch/README.md:17`, not a like-for-like plain
   step; `doc-measured`). Same tokenizer: the id file transfers. The
   libllama engine takes it with `--llama-mtp-vocab` (2026-10-03). On the
   dense 27B with 4 drafts it ran 47.1 -> 50.8 t/s, draft acceptance
   72.1 -> 68.0 % (`measured-here`, `mtp-cycle-wall.md`).
   `tools/draft_vocab.py` is Strata's (`~/src/Strata-ref`, `code`), not
   arcint's.
2. **A verify that reads the weights once for up to 16 rows.** Not int8 (the
   single-user path is W4A16, `single-user/start_qwen.sh:139`): Marlin tiles
   16 token rows, so a decode step already pays for 16 (`docs/gotchas.md:252-262`);
   step time rises in stairs (39.5 ms at 16 query tokens, 47.8 at 19). From
   `bench-raw`: an 8-row step with the drafter 1.19x a plain step at short
   context, a 16-row step 1.6x at 25k (attention grows ~1 ms per position).
   MTP's knee at 4 drafts (`optimizations.md:399`).
3. **Prompt lookup fused with the drafter** (`patches/dflash2-lookup-drafting.patch`):
   the longest suffix of the history matched against all of it (`:317-385`;
   4 to 32 tokens, `:515-525`), its continuation replacing the drafter's
   tokens on a match of 8 or more, of 6-7 when it agrees with the drafter on
   2 tokens, of 4 at the tail (`:568-580`), and filling a 16-row block beyond
   them (`:449-459`), point-mass
   draft probabilities so sampling stays exact (`:476-512`), the long block
   requested after two full-accept-with-match steps (`:858-906`). 25k
   context, greedy: verbatim reproduction 159 -> 260 -> 381 t/s, chat 126 ->
   133 (`optimizations.md:306-316`, `doc-measured`; `bench-raw` 359).
4. **Rejection sampling with the draft's probabilities** at temperature > 0,
   drafts truncated like the target (`start_qwen.sh:263-274`,
   `sampler-small-topk-fast-softmax.patch:15-19`): MTP 78 -> 90 t/s, 2.2 ->
   2.6 tokens a step (`optimizations.md:383-384`, `doc-measured`).
5. **DFlash2** (incoai's 5-layer drafter over target layers 5/19/33/47/61,
   7 drafts in one pass, a 2,048-token window; requantised to W4A16):
   117.8-125.7 t/s against a tuned MTP's ~110, MTP ahead beyond ~12k
   (`optimizations.md:202-238`, `drafter/README.md:64-101`, `doc-measured`).
6. **The gated delta-net under speculation**: 1 + k state slots per request,
   the next step starting from slot `num_accepted - 1` (upstream vLLM,
   `vllm-pr50021-gdn-spec-bounds.patch:9-30, 83-111`, `code`); the state in
   fp16 (`--mamba-ssm-cache-dtype float16`, perplexity unchanged to three
   places, `optimizations.md:54-62`). llama.cpp's `n_rs_seq` snapshots are
   the same mechanism.
7. **Split-KV attention for the multi-row verify** (`spec-decode-attn.patch`):
   q_len x GQA rows packed into one tile, KV split with a combine; 250 /
   583 / 1,132 us per layer at 8 / 16 / 32 query tokens and 25k context
   against FA2's ~2,000.
8. Lower for us: int8-activation GEMMs for prefill (+27-30 %, perplexity
   +4.1 %), int8 QK prefill attention (+0.3-3.3 % end to end), KVarN 4/2-bit
   KV (context only; decode 2.13x slower at 112k), 2,048-token prefill chunks.
   Tried there and lost: skipping the drafter while copying (-6 %), n-gram
   chains at temperature (-8 %), MTP head fine-tuning, 5 MTP drafts.

## What it means here (`measured-here` where marked)

- Our MTP on the B60: 3 drafts, 31-35 t/s served against 19.75 plain
  (`docs/llama-engine.md`, `measured-here`); a cycle 102 ms (verify 90,
  drafting 11.5) against a 51 ms step (`mtp-cycle-wall.md`, `measured-here`).
  The verify, not the draft, is our expensive part: a 4-row verify costs
  ~1.8 plain steps where HyperQwen's 8-row one costs 1.19. First lever: the
  K-quant product for 2-16 rows reading each weight once (item 2), then the
  draft vocabulary (item 1) and prompt lookup (item 3), which both need it
  to pay.
- Our sampler accepts by sampling the target and comparing with a greedy
  draft (`src/exec/verify_walk.h`): exact, but item 4's acceptance gain at
  temperature is open.
