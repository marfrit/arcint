# clean-turns step 2 — handover (branch `clean-turns-wip`, 2026-10-21)

This note is the handover for step 2 of the [clean-turns](clean-turns.md)
campaign. It is written from the state of this branch, which carries the
unfinished work; `clean-turns` itself stays at `ae2bc93`.

> **Superseded (2026-10-21):** the next steps below were carried out on branch
> `clean-turns-step2`; the result, its measurements and the deviations from the
> reference are in [clean-turns.md](clean-turns.md#step-2-built--grammar-constrained-tool-calls-measured-here-2026-10-21).
> This note is kept as the record of the state it describes.

## What step 2 is

Grammar-constrained tool calls for the Qwen3-coder artifact. The pinned
llama.cpp's PEG parser for exactly this model
(`common/parsers/qwen3-coder.cpp` in the pinned tree) builds a GBNF grammar
from the declared tool schemas; arcint drives that grammar around its own
sampler so that a tool call the model starts is one it finishes — an
unterminated call block becomes unreachable, not merely detectable.

- **Behind a switch, default off.** `--tool-grammar` (operator default),
  per-request `tool_grammar` (non-standard field, named as such), reported by
  `/props`. Off keeps today's behaviour exactly; a request that declared no
  tools is never constrained (the generator returns an empty grammar).
- **The gate** (from the campaign charter): a forced-malformed call must be
  impossible with the switch on; the CSV sanity glance stays green; decode
  t/s moves by no more than the run-to-run spread (one number each, on/off).
  None of that is proven yet.

## What is wired (on this branch)

| Piece | State |
|---|---|
| `third_party/llama-peg/` | the pinned tree's PEG grammar closure — `peg-parser`, `json`, `json-schema`, `json-schema-to-grammar`, `trie`, `unicode` — copied verbatim (MIT, see `THIRD_PARTY.md`), made self-contained by `llama-peg-shim.h`. Builds as the `arcint_peg` static library, linked into `arcint_core`. |
| `src/core/toolgrammar.{h,cpp}` | `generate_tool_grammar(tools)` and `tool_grammar_trigger()`. Mirrors the reference parser's tool-call block: lazy trigger rule on the call opener, function name drawn from the declared set, required parameters in any order (`permute`, copied from the reference's `chat-peg-parser.cpp`), optional parameters, each value typed to its declared schema. |
| `tool_grammar` flag | plumbed end to end: `Config` (`--tool-grammar`), `SamplerDefaults`, `SamplerParams`, `SamplerOverrides`, the chat request field (`src/core/chat.cpp`), `/props` (`src/api/handlers.cpp`). |
| `src/core/toolcall.{h,cpp}` | `ToolCallTags` accessor so the parser and the grammar share the call tags and cannot drift. |

## What is NOT wired (the actual work)

1. **The switch does nothing.** Nothing in the sampling path reads
   `SamplerParams::tool_grammar`. No `llama_sampler` for the grammar is
   created, and `llama_sampler_apply` / `llama_sampler_accept` are never
   called.
2. **The generator is not in the build's test path.** `toolgrammar.cpp`
   compiles into `arcint_core`, but `tests/tool_grammar_test.cpp` is a
   standalone `main()` that CMake does **not** register, and no production
   code calls `generate_tool_grammar`. It was verified once by hand
   (device-free, emits the expected GBNF); it is not a regression test.
3. **The backend does not receive the schemas.** `GenerationInput` carries
   `tool_names` (names only, for the stub). The grammar needs name + JSON
   Schema, i.e. the full `std::vector<ToolSpec>` (or equivalent) must reach
   the llama backend.
4. **The decode loops are untouched.** All three sample arcint's own
   `lgc::Sampler` over a raw logits row — `src/exec/backend_llama.cpp:975,
   :1009` (plain), `:1114, :1148` (multimodal), `:1252, :1289` (spec/verify).
   The grammar must be driven by hand around `sampler.sample()` (apply before
   the sample) and fed every accepted token via `sampler.observe()` (accept
   after). The spec/verify loop is the awkward one: speculative tokens that
   are verified must be accepted into the grammar too.
5. **The gate is unproven.** No forced-malformed case, no CSV glance, no
   decode t/s number with the switch on/off.

## What was found about libcommon (and the pinned tree)

- The pinned tree **does** ship `common/parsers/qwen3-coder.cpp` — the
  complete upstream reference for this model's tool-call form. The campaign
  doc's earlier claim ("the pinned tree ships no grammar for the coder's
  call form") was written before that file was found and is now wrong; the
  reference is the starting point, not a file to write.
- The schema→grammar half does **not** drag libcommon in: the closure is six
  translation units plus three string helpers (`string_repeat/join/split`,
  inlined in `llama-peg-shim.h`), compiled against arcint's own vendored
  nlohmann/json. `llama-peg-shim.h` provides exactly what the closure takes
  from `common.h`/`log.h`/`ggml.h`: `GGML_ASSERT`, `llama_token`, and the
  string helpers. `LOG_DBG` is a no-op there.
- arcint's `ToolSpec.parameters` is `nlohmann::json`; the closure speaks
  `common_json` (a pimpl over `nlohmann/ordered_json`). The conversion in
  `toolgrammar.cpp` is a `dump()`/`parse()` round trip — one schema per
  declared function, cheap.
- The reference parser's full root also covers the generation prefix,
  reasoning, and free content. arcint does **not** need that part: it applies
  its own chat template and `think_eog_mask`. Only the tool-call sub-grammar
  transfers, as a lazy trigger on the call opener.

## Exact next steps

1. **Register the test.** Add `tests/tool_grammar_test.cpp` to the build as a
   real test (or fold its assertions into `arcint-test`), so the grammar
   output is a regression check.
2. **Pass the schemas to the backend.** Extend `GenerationInput` (or add a
   field) to carry the full tool specs; populate it where `tool_names` is
   populated today.
3. **Drive the grammar in the plain loop first.** In
   `src/exec/backend_llama.cpp`, when `sampler.tool_grammar` is set and the
   request declared tools: build the grammar once per request
   (`generate_tool_grammar`), create a `llama_sampler` with
   `llama_sampler_init_grammar_lazy_patterns` (trigger:
   `tool_grammar_trigger()`), `llama_sampler_apply` the row before
   `sampler.sample()`, and `llama_sampler_accept` in `sampler.observe()`'s
   path for every committed token.
4. **Then the multimodal loop, then spec/verify** (speculative accepts must
   feed the grammar).
5. **Prove the gate.** Forced-malformed call unreachable with the switch on
   (reachable with it off, as a control); CSV glance green; one decode t/s
   number on and off.
6. **Measure the placement price.** `apply` decodes every candidate token to
   a piece before testing it. arcint hands `sample()` the full ~152k-vocab
   row, so a naive placement costs a per-candidate decode per step. Placing
   it after candidate collection bounds the cost but changes which tokens the
   grammar ever sees — that trade is a measurement, not a guess, and it
   belongs with the t/s number.

## Warnings for the next person

- **Do not trust the campaign doc's step-2 feasibility section as-is.** It
  predates finding `qwen3-coder.cpp` in the pinned tree; its "the call
  grammar would be arcint's file" is superseded by the reference parser.
  (The rest of that section — no sampler chain, drive by hand, apply/accept
  entry points — still holds.)
- **The switch is a no-op today.** Anyone who turns on `--tool-grammar` gets
  no error and no effect. If you land partial work, say so in the commit
  message; do not let the flag read as live.
- **`tests/tool_grammar_test.cpp` is not run by CTest.** It has a `main()`;
  it will not be built or executed by `ctest` until registered.
- **The call opener is a single special token** on the coder vocabulary
  (measured). The lazy trigger pattern is therefore the opener itself,
  unescaped. If the artifact or template changes, re-measure before assuming
  the trigger still fires.
- **Spec/verify is the riskiest loop.** A grammar that sees the sampled
  token but not the verified speculative tokens will desync. Cover it with
  the forced-malformed gate, not by inspection.
- **Card discipline applies** (`docs/sop-card-window.md`): sampler before the
  leg, card identity by PCI id, zombie sweep by pid. The t/s measurement
  needs a card window; nothing here has been run on a card.
- **Vendored code is verbatim from the pin.** If the pin moves, the closure
  in `third_party/llama-peg/` must be re-copied and re-checked against the
  new `common/parsers/qwen3-coder.cpp`, not patched in place.
