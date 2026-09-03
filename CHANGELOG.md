# Changelog

The human record of what each release changed and what it depends on. The
package record lives in `contrib/packaging/arcint/debian/changelog` (one
entry per built package) and the plugin patch record in
`contrib/packaging/marfrit-openvino/patches/README.md`; this file names the
same releases by tag and adds what a reader upgrading needs first: the
runtime dependency. Every number here was measured; the measurement
protocol and the retractions are in `DESIGN.md` §7.

Runtime dependency throughout: `marfrit-openvino`, a source build of
OpenVINO at the pinned upstream commit `71640275` (the 2026.4.0 nightly of
2026-08-21) with the patch series in `patches/` applied. The patch level
is part of the package version (`+p1`, `+p2`) and a release names the level
it was built and measured against; the dependency is strict because a
different nightly is a different ABI.

## Unreleased

### Fixed since 0.2.13 (unreleased)
- The DFlash2 drafter past 2,048 tokens: plugin patch 0014 lets an Assign
  adopt a same-type, same-rank output layout instead of asserting (the
  served head now drafts at 17k-token prompts and through 3,000-token
  decodes); the export's state trim takes a runtime slice start and the
  export tool gained `--compress int4` with the recipe verified
  byte-identical against the served head; the engine's drafter disable is
  per lane and re-armed on the next request, with a feed cap one row below
  the window for plugins without 0014.

### Known defects in 0.2.13, found by the long-context window (2026-09-03)
- `--dflash` at depth: on the same card and model the DFlash2 drafter
  reaches 1.69 / 1.43 / 1.00 tokens per cycle at 8.9k / 37.7k / 76k tokens
  and decodes below plain at each (19.8 vs 22.3, 11.0 vs 19.9, 5.3 vs 16.3
  t/s); the zero acceptance at 76k is not yet explained. Serve deep
  contexts without a drafter.
- `--paged-kv u8:i4` prefill price (the M8 item owed since 0.2.13): +7% /
  +25% / +72% of prefill time at 8.9k / 37.7k / 71.7k tokens on the coder,
  decode at parity; by code reading it is the opt attention path u8:i4 is
  forced onto, not yet measured as such.
- `--mtp on` at depth: on the 24 GB card the dense 27B agent's MTP verify
  accepts 79% of drafts at 37.7k tokens (12.6 t/s) but 0% at 76k and 143k
  tokens, where decode falls to 1.0 and 0.3 t/s against plain decoding's
  16.3 at 76k, and its prefill runs 13% slower than plain there and at
  half plain's rate at 143k; the cause
  is not yet measured (by code reading the reconstructed MTP layer keeps an
  unpaged state and a dense mask over the whole context). Serve deep
  contexts with `--mtp off` until it is.
- `--paged-kv u8:i4`: a 141,902-token prefill on the coder (16 GiB card,
  auto-fit 171,312) failed with a GPU out-of-resources error and every
  later request in that process failed with it; prompts up to 8,909
  tokens succeed at every depth setting tried. The failing depth lies
  between those two numbers; bracketed one process per depth: 35,227 and
  71,689 tokens prefill and decode, and the same 119,074-token prompt that
  u8 prefills on this card fails at u8:i4, so the fault is specific to the
  u8:i4 prefill path. An earlier
  version of this entry said "every prompt over about 2,048 tokens", which
  was an artifact of running the depths deepest-first in one process and
  is retracted. Until the cause is found, do not deploy u8:i4 for prompts
  beyond about 70k tokens on the 16 GiB card.
- `--dflash`: the exported draft head carries a state variable fixed at
  2,048 rows; the first draft after a prompt longer than that fails and the
  drafter disables itself for the process (decode continues without
  drafts). Every DFlash number on the record was taken with prompts under
  2,048 tokens.

- Drafting II measured (M11, `DESIGN.md` §7.0.2z): four host-side levers on
  the DFlash2 chain, all free of training. Viterbi over the selector's
  lattice accepts fewer drafts than the greedy commit on every probe;
  blocks of 12 and 16 gain a few percent of tokens per verify cycle and
  lose 6 to 21% of throughput; the ngram drafter falls below plain decoding
  on the dense 27B. The offline oracle puts a floor of +0.74 accepted
  drafts per cycle under any re-ranker restricted to the same candidates.
- New flags: `--dflash-block N`, `--dflash-select greedy|viterbi` (greedy
  stays the default and byte-for-byte the previous selector),
  `--dflash-lambda X`, `--dflash-topk K`; the `ARCINT_DFLASH_DUMP` cycle
  dump and `tools/dflash_oracle.py`.

## 0.2.13 — 2026-09-03

Requires `marfrit-openvino 2026.4.0~dev20260821+p2` (patches 0003–0013).

### Added
- Measured reservation (M7): the fit pass adopts the maximum admissible
  `--n-ctx` when the flag is omitted and prints the ledger it used; an
  explicit `--n-ctx` is verify-only and never lowered. On the 24 GB card
  the dense 27B agent at u8 with an 8 GiB prefix cache admits 155,568
  tokens; a unit hand-set above that is refused by design.
- `--paged-kv KEY[:VALUE]` (M8): asymmetric KV, u8 keys with i4 values, at
  8.8 KiB per token against u8's 11.3. Auto-fit context on the coder
  (16 GiB card) 133,456 → 171,312, on the dense agent (24 GB card) 155,376
  → 199,424; acceptance task 10/10 on the coder at u8:i4. Still owed:
  the u8:i4 prefill price and prefix byte-exactness.
- `--offload-ratio` device-tier expert slot pool with asynchronous uploads
  (M9, plugin patches 0004–0007): the 35B on the 16 GiB card from 0.4 t/s
  (ratio 25, unpatched) to 9.1 (ratio 50, 8 GiB pool, 16-token probe) and
  10.4 (64-token probe).
- `--moe-cpu-tier` and `--moe-cpu-tier-threads` (M14, plugin patches
  0011–0012): experts that would evict a device slot are computed on the
  host instead. 35B on the 16 GiB card: 15.0 / 15.5 t/s against 10.4 / 10.6
  at ratio 50 with an 8 GiB pool, 14.1 / 14.8 against 7.4 / 7.5 at ratio
  75 with 5 GiB; greedy text byte-identical over 64 tokens; 10/10.
- `--prefix-cache-reserve PCT`: under auto-fit, hold that share of the
  affordable KV pages spare for cached prefixes (default 0, the previous
  behaviour: all pages live). The dense agent at 25% adopts 116,528 with
  2,428 pages spare.
- `--pin-dispatch` (M12, measured null on a quiet host, opt-in) and the
  exporter's `--moe-lowering tiled` form.
- `--vision`: reserved and refused (M13). The vision IRs a VLM export ships
  are reported at load and never read (coder: 6 files, 428.3 MiB on disk).
- Per-expert routing histogram in the plugin (patch 0013, env
  `MOE_OTD_ROUTING_HIST=<path>`), the input for any expert placement or
  bit-width policy.
- `llm.txt` carries the complete `--help`.

### Fixed
- An explicit `--n-ctx` below the admissible maximum was refused whenever a
  prefix cache was configured: the cache's reserve pages were sized from
  the whole budget and the overshoot was blamed on the request. The
  reserve is trimmed first now, the request never; the refusal is itemized
  in pages.
- The auto-fit correction could not converge when the pool held spare
  pages: a sub-page overshoot moved live pages into spare and the pool
  total never changed. It trims the pool total with a 4/16/64/256-page
  floor and refuses loudly, with the attempt history, above an allocation
  granule of 84 pages.
- Two help-text defects (a literal `%%`, a paragraph under the wrong flag).

### Dependencies
- `marfrit-openvino +p2`: ten new plugin patches (0004–0013) on the same
  upstream commit; the runtime reports `...-marfrit-p2`. Patches 0001 and
  0002 remain deliberately unapplied.
- Unit files ship `--n-ctx` as a multiple of 4096 below the admissible
  number the fit pass prints: coder 131072 (u8), agent 262144 (the 35B's
  trained maximum, verified), the dense 27B with MTP 131072.

## 0.2.12 — 2026-09-01
- DFlash2 block-diffusion drafter: `--dflash DIR` serves the public
  Qwen3.8-27B draft head, seven drafts per verify pass; 44.8 t/s on the
  24 GB card against 24.0 plain and 33.0 with the MTP head;
  `--dflash-device` parks the draft on the other card with byte-identical
  output. `tools/export_dflash.py` and the pairing-probe instruments.

## 0.2.11 — 2026-09-01
- Operator serving defaults: `--temp`, `--top-p`, `--top-k`,
  `--repetition-penalty`, `--presence-penalty`, `--chat-template-kwarg`;
  precedence request > flags > artifact > family card.
- `usage.completion_tokens_details` reports accepted and rejected
  prediction tokens per response.

## 0.2.10 — 2026-09-01
- Review fixes on the paged path (index-tensor map pruning, embeddings
  byte check, MTP argmax guard, a 16-token page tripwire);
  `ARCINT_PA_HOST_INPUTS` experiment. Requires `marfrit-openvino +p1`
  (patch 0003: the MoE subbuffer churn at token_num > 1). The -3 package
  corrected the self-reported version.

## 0.2.9 — 2026-08-30
- `/props` publishes `chat_template_caps`; `reasoning_effort` accepted as
  the template's on/off switch.

## 0.2.8 — 2026-08-30
- `reasoning_content` split from `content` for templates that open the
  think block in the prompt.

## 0.2.7 — 2026-08-30
- Tool-call arguments reach the template as the type its capability flag
  asks for (the 500 on tool-using turns).

## 0.2.6 — 2026-08-30
- `--gate-pad N` (shared-expert gate widened for a JIT GEMM: −13% prefill
  wall, −5..6% decode, off by default); profiler tables name their capture.

## 0.2.5 — 2026-08-30
- The logits slice fix: +27% prefill, six device-to-host copies per
  prefill gone, 2.09 GiB of activation reservation returned.

## 0.2.4 — 2026-08-29
- The public repository becomes the upstream (no sanitised export);
  u8-vs-f16 KV corrected at depth (f16 7.8% faster at 53.5k tokens).

## 0.2.3 — 2026-08-29
- `--paged-kv u8|f16`; u8 default with its prefill price on the record;
  two retractions kept in `DESIGN.md`.

## 0.2.2 — 2026-08-29
- `--served-model-name`; `/props` carries the context length; the unit
  template installs from CMake.

## 0.2.1 — 2026-08-29
- `/v1/models` carries `n_ctx`, `n_ctx_train`, quant, lanes; the shipped
  unit starts on a package-only host. The -2 package was a rebuild of a
  stale archive.

## 0.2.0 — 2026-08-29
- First packaged release: amd64, trixie, strict dependency on
  `marfrit-openvino` at the pinned nightly.
