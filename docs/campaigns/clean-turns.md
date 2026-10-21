# clean-turns — fewer unwanted stops and clean tool calls in agent sessions

**Open.** Mission (operator Markus, 2026-10-20): the CSV acceptance task is
only a concentration indicator here; the goal is agent sessions that do not
stop empty and whose tool calls arrive parseable. Per change: the CSV sanity
glance (T=0 ×3 plus 5 sampled at 0.7 on the served unit) and one decode t/s
number. No 30-run batteries, no statistics campaigns.

## Charter

Two mechanisms, one defect family:

1. **Unwanted stop inside a think block.** A Qwen-family model that *mentions*
   a chat-template control token emits the real special token instead of the
   text. If the end-of-turn marker is written where the model means it as
   text, the turn ends there: finish `stop`, no content, no tool call. If the
   closing think-tag is written as a mention, reasoning and content split at
   the wrong place. Both reproduced against the served endpoint with
   `ignore_eos` (the generated text shows the marker where the model meant to
   write it).
2. **Tool calls parsed after the fact.** `parse_qwen_tool_calls` runs on the
   finished text; an unterminated call block is a warning plus raw content. No
   grammar constrains the call.

## Gate

- Step 1 must be able to fail: the probe prompt that provokes the mention ends
  **empty without the switch** and **continues with it** (`measured-here`).
- Step 2: if the pinned llama.cpp's grammar machinery can be wired in, a
  forced-malformed call must become impossible with it on; if it cannot, the
  answer is a `file:line` reason, not a shrug.
- Both: the CSV sanity glance stays green, and decode t/s moves by no more
  than the run-to-run spread (one number each).
- Default stays the current behaviour; every new behaviour is a switch until
  the operator decides.

## Baseline (step 0) — `measured-here`, existing data only, no card window

Tool: `tools/turn_defect_audit.py` (counts only; session content never leaves
the machine, and the script builds every control token from parts so the file
contains none).

Session side — the pi session logs on this host, 27 files, 2653 assistant
turns; every file containing the operator's exclusion word skipped whole (2
files, never opened beyond the grep):

| class | count | of turns |
|---|---|---|
| (a) ends with neither text nor tool call | 6 | 0.23 % |
| — of those, finish `stop` (the model's own end-of-generation token won) | 4 | |
| — of those, finish `length` (the token cap) | 2 | |
| (b) ends inside an open think block | 6 | 0.23 % |
| (c) aborted or errored turns | 85 | 3.2 % |
| (d) tool calls that failed at the tool with an argument/JSON/shell-quoting error | 8 | of 2561 calls |
| (d) tool calls whose `arguments` field is not parseable JSON | 0 | |
| control token found in client-visible text (the mention signature) | 4 | |

Classes (a) and (b) are the same six turns: every empty turn left its output
inside the think block. By model the four `stop`-finished ones are
`qwen3.8-flash-next` ×2, `qwen38-flash-next-e` ×1, `deepseek-v4-flash` ×1;
the two `length` ones are `ornith-evoquant` and an older `qwen-agent`.
The mention signature sits in client-visible text of three models (three
closing think-tags, one call opener).

Card-host journals (whole journal per served unit, `--journal` mode; unit names
are operator-local and stay in the git-ignored local file, per the campaigns
README):

| unit | completions logged | unterminated call-block warnings |
|---|---|---|
| the coder unit on the A770 | 861 | 0 |
| the Flash-Next unit on the B60 | 648 | 1 |
| the dense agent unit on the B60 | 886 | 1 |

Reading of the baseline:

- The empty-stop defect is **rare per turn and total per occurrence**: 4 in
  2653 turns, but each one costs the agent a wasted round and the operator a
  confused session. It is a tail event, so a mean over 30 runs will not see
  it; only a probe that provokes it can.
- The session counts **understate** it: this session's own file contains the
  exclusion word (the mission text carries it) and is therefore skipped, and
  in it the same shape was observed twice on 2026-10-20 while working this
  campaign — finish `stop`, content only in reasoning. Three more instances
  were observed live in the operator's sessions the same day. So: 4 counted +
  3 observed-and-excluded.
- Tool-call syntax failure is **not** the dominant cost on the served units
  (2 warnings in 2395 completions). Class (d)'s 8 argument errors are mostly
  the client's own shell quoting, not the model's JSON. This lowers the
  priority of step 2 relative to step 1 and says what step 2's real benefit
  would be: not fixing observed breakage, but making the malformed call
  unreachable.

## Where it lives

- `src/core/sampler.{h,cpp}` — arcint's own sampler, deliberately outside the
  backend so it is testable without a GPU (`src/core/sampler.h` header
  comment). `sample()` gets one row of logits and returns a token id.
- `src/exec/backend_llama.cpp` — the decode loops (plain `generate` at 908-989,
  `generate_spec` at 1059-1123, the MTP path at 1203-1264, the combined-lane
  path at 1429-1513) and `is_stop` at 1711, which is where an end-of-generation
  token currently ends the turn.
- `src/api/handlers.cpp` — `prep.think_open` at 674 (the rendered prompt ends
  inside a think block), the streaming path's `kToolOpen` at 937, the
  unterminated-block warning at 834 and 1030.
- `src/core/reasoning.cpp` — the think-block openers/closers table
  (`kTags`), the same tag family the mask has to know.

## Step 1 — mask the end-of-generation tokens while a think block is open

Design (written before the code, `code`):

While a think block is open — opened by the template (`prep.think_open`) or by
the model's own opener — an end-of-generation token is not a legal next token
in this template family: the turn's content has not started. So the sampler
sets those logits to `-inf` until the closing think-tag has been emitted. The
second opener of the same family is masked too (a reopened block is the same
defect one step later). `max_tokens` and the lane's context limit stay the
only exits; both already return `FinishReason::Length`.

Why the sampler and not `is_stop`: masking the logits keeps the invariant that
the token the loop sees is the token that was legal. A filter in `is_stop`
would let the end-of-generation token be *sampled*, consume a step, and then
be ignored — and with top-p/top-k applied before it, the model's probability
mass would still be spent on an illegal token.

Scope: per-request switch first (`think_eog_mask` on the request, plus the
operator flag), current behaviour as default. The mask is a no-op once the
block closes, so a normal answer pays one `-inf` write per masked token id per
step.

Red-first check that it can fail: the probe prompt that provokes the mention
must end with empty content on the unmasked unit and continue on the masked
one.

## Step 2 — grammar-constrained tool calls: feasibility

(recon below, `code`)

## Step 1 vocabulary facts on the served artifact (`measured-here`, 2026-10-21, no card)

The mask's design has an assumption under it: that a think-tag is ONE token in
the served vocabulary, so there is one logit to block. That is a fact about the
artifact's file, and it was never checked. It is now, two ways, neither needing
a GPU:

| how | what it reads |
|---|---|
| `tools/think_vocab_probe.py` | the GGUF's metadata only (added-token list, token-type table, eos/eot ids) — pure Python, no llama, no device |
| `tools/vocab_eog_probe.cpp` | the vocabulary as **the pinned llama.cpp loads it**: `vocab_only=true`, so no tensors, no device, no card seat; asks `llama_vocab_is_eog` over the whole vocabulary and `llama_tokenize(parse_special=true)` per tag |

On the served coder artifact (`qwen36-coder-iwo3-c5f495ac-shq8.gguf`, 248320
tokens, 753 tensors, Q4_K - Medium), by the second probe, run against the
branch's own `libllama.a` on the GPU dev host:

| fact | value | consequence for the mask |
|---|---|---|
| ids `llama_vocab_is_eog` answers true for | **5**: 248044, 248046, 248063, 248064, 248065 | the mask blocks five logits per step while the block is open, not one — and it is the same predicate `is_stop` asks, so the two cannot disagree about which ids those are |
| the think opener | **one id**, 248068, attr `user_defined` | blockable by id, as designed |
| the think closer | **one id**, 248069, attr `user_defined` | never blocked (emitting it is what re-opens the turn), matched as text |
| the longer reasoning-tag variants (`thinking`, `reason`) | 3-4 ids each | no id to block; the text-match path is what covers them |

The code already takes exactly that set: `prepare_think_mask` copies
`eog_ids_`, and `eog_ids_` is built by asking `llama_vocab_is_eog` over the
whole vocabulary (`src/exec/backend_llama.cpp:507-509`) — the same predicate
`is_stop` asks. Nothing hard-codes one end-of-turn id, so the four ids nobody
named did not get missed. The measurement confirmed the code rather than
changing it, which is the order it should happen in.

The `user_defined` type is the part worth keeping: llama.cpp's special-token
cache is built over `CONTROL | USER_DEFINED | UNKNOWN`
(`src/llama-vocab.cpp:3177`), and `tokenize(..., parse_special=true)` partitions
on that cache (`src/llama-vocab.cpp:3399-3410`), so a tag typed `user_defined`
still tokenizes to one id. A probe that only looked for the `CONTROL` type
would have called these tags "text" and been wrong. The Python probe reports
the type rather than deciding on it, for that reason.

Two bugs found while writing the probes, both worth naming: GGUF array elements
carry no type byte of their own (reading one per element desyncs the stream at
the token-type array, which is the array the probe needs); and in
`meta[string()] = value()` CPython evaluates the **right-hand side first**, so
the value was decoded before its key. Both produced confident nonsense, not
errors.

## Step 1 endpoint pair — what the leg needs, and what it costs

The gate is a pair on one served unit: the same arm empty without
`think_eog_mask`, continuing with it. Running it needs a serving process built
from this branch, and that is where the cost sits:

- The **served coder unit is the installed package** (`arcint 1.2.1`, build
  `83d7609`) on the 16 GiB card. It cannot serve the `on` arm: the field does
  not exist in it. So the pair cannot be run against the unit as it stands.
- The branch **does build** on the GPU dev host against the pin with all 60
  patches (`~/wp/clean-turns/build/arcint`, configured
  `-DARCINT_LLAMA=ON -DARCINT_LLAMA_DIR=~/lcpp-clean -DGGML_OPENCL=ON`, the
  recipe `~/wp/wt-0019/build-0020` used). Device-free suite in that build:
  **194 cases, 0 failed**. `--think-eog-mask` is in its `--help`.
- Running it means loading a 13.58 GiB artifact, and **both cards are held by
  the two serving units**. Stopping a serving unit is the operator's call, not
  a step an agent takes: the fleet rule is one seat per card, announced, and
  mneme records a past accident where an engineer's own rollout at the unit he
  was building on killed his own agent loop.
- The leg is otherwise prepared: the gate script and the probe are on the host
  (`~/ct-tools/`), and the gate's verdict logic is already proven fail-capable
  against a stand-in server. Against the current package the gate prints
  **NOT SHOWN** (exit 1), which is the correct reading of that run: the arms
  came back `EMPTY-LENGTH` at a 256-token cap, a different defect.

So step 1 is one seat away, and the seat is the operator's decision.

## Step 2 feasibility — grammar machinery: it can be wired in (`code` + `measured-here`)

The gate asked for a `file:line` reason if it cannot be done. It can, and the
reasons it is not free are these:


**What the pinned tree offers** (`code`, llama.cpp `3e4dfa8` + the 60
`contrib/llama.cpp/patches`, all 60 applying clean to the pin):
| driven through the chain entry points | `include/llama.h:1493` (`llama_sampler_accept`), `:1494` (`llama_sampler_apply`) |
| the mask itself sets rejected logits to `-INFINITY` on a `llama_token_data_array` | `src/llama-grammar.cpp:1359` (`llama_grammar_apply_impl`), rejects applied at `:1386`, `:1389`, `:1398` |
| it also blocks the turn-ending token while no stack is complete | `src/llama-grammar.cpp:1384-1387` |
| lazy mode: grammar engages only after a trigger pattern or trigger token | `src/llama-grammar.cpp:1313`, `:1362`, `:1407-1409` |
| JSON-Schema to GBNF converter | `common/json-schema-to-grammar.cpp:993` |

**The integration cost, which is the part a paper would not tell us** (`code`):
arcint has **no llama.cpp sampler chain to add it to**. Every llama decode loop
samples arcint's own `lgc::Sampler` over a raw logits row —
`src/exec/backend_llama.cpp:975`, `:1009` (plain), `:1114`, `:1148`
(multimodal), `:1252`, `:1289` (spec/verify). So "wire in the grammar sampler"
means driving `llama_sampler_apply` / `llama_sampler_accept` by hand around
arcint's `sample()` on the same row, and feeding it every token the loop
accepts, including speculative ones. The mechanism itself is compatible —
`apply` mutates logits in place, which is exactly what `Sampler::sample` does.

**The converter does not drag libcommon in** (`measured-here`, 2026-10-21, on
this machine, no GPU): a test binary that turns a two-property schema into
GBNF links and runs against exactly five upstream translation units —
`json-schema-to-grammar.cpp`, `common/json.cpp`, `common/json-schema.cpp`,
`common/trie.cpp`, `common/unicode.cpp` — plus three string helpers
(`string_repeat`, `string_join`, `string_split`; declared in `common/common.h`,
defined in `common/common.cpp`, so a handful of functions, not the library) and
`ggml_abort`, which arcint already links. Compiled with arcint's own vendored
`nlohmann/json` 3.11.3 on the include path (`THIRD_PARTY.md`), not upstream's,
and it produced a 360-byte grammar. So the schema→grammar half is a small,
contained dependency, not a fork of upstream's `common`.

**What is ours to write**: the pinned tree ships no grammar for the Qwen
coder's call form (`grammars/` holds `json.gbnf`, `json_arr.gbnf`, arithmetic,
chess — nothing with the function/parameter block form). The call grammar
would be arcint's file, generated per request from the declared schemas.

**The price to measure, not yet measured**: `apply` decodes every candidate
token to a piece (`src/llama-grammar.cpp:1382`) before testing it. arcint
hands `sample()` the full vocabulary row, so a naive placement costs a
per-candidate decode over ~152k ids per step. Placing it after
`collect_candidates` (top-k/top-p first) bounds that to the candidate set but
changes which tokens the grammar ever sees — that trade is a measurement, not
a guess, and it belongs with the decode t/s number step 3 asks for.

**Priority stays where the baseline put it**: 2 unterminated call blocks in
2395 served completions. Step 2's benefit is unreachability of the malformed
call, not repair of observed breakage.

## Status

- Step 0 done (this note).
- Step 1 in progress: landed at unit level, gate written and proven able to
  fail; the endpoint pair needs the branch's llama build (running on the GPU
  dev host as of this writing).
- Step 2 feasibility answered above: wireable, at a named cost.
- Step 3 not started.

## Step 1 measured at the endpoint (before any code change)

Probe: `tools/think_eog_probe.py` (arms in words only; every control token
assembled from parts, none in the file). Run against the **already running**
coder unit on the A770 — no card window, no build, one process already on the
card. `measured-here`:

- The mention mechanism does **not** reproduce on the coder artifact: asked to
  produce a control token as text, it *spells it as text* (48 arms: the marker
  appears in the output text, the turn does not end there). The substitution
  was observed on the dense/Flash-Next family, which is the B60 and out of
  scope for this branch.
- The **empty stop itself reproduces reliably on the coder artifact**, and its
  mechanism is simpler than the mention story: with `max_tokens` 4096 and the
  prompt "Reply with an empty answer.", the turn ends at the model's
  end-of-generation token **while the think block is still open** —
  `finish=stop`, content 0 characters, ~600 characters of reasoning, no tool
  call, 2/2 arms at T=0 and T=0.7. With `max_tokens` 64 the same shape comes
  back as `length`, which is why the earlier arms hid it.
- With tools declared (the shape of a real agent turn): 10/16 turns produced a
  tool call, 3 answered, 3 ended empty at the token cap. No empty stop there.

So the mask's target is not only "the model mentions the marker" but the
general case: **the end-of-generation token wins a step in which the think
block has not been closed**. That is the same six turns in the baseline and
the same shape this session suffered three times while working on it.

## Step 1 design (rewritten after the measurement)

Mask, in the sampler, the ids that end generation, plus a second opening
think-tag, while a think block is open. The block is open when the template
opened it, or when the model wrote an opener of the family; it closes on that
family's closer. `max_tokens` and the lane's context stay the only exits.

Default off (current behaviour), on by request field and by operator flag.

## Step 1 landed (unit level) — `code` + `measured-here` (device-free suite)

What the switch does, in one sentence: while a think block is open, the ids
that end generation are not offered, and neither is a second opener.

Where each part lives:

| part | file |
|---|---|
| the tag family, as one accessor (`think_tag_pairs`) | `src/core/reasoning.{h,cpp}` |
| the mask itself (`mask_illegal_while_thinking`, applied before the penalties and before greedy) | `src/core/sampler.{h,cpp}` |
| the params, the request field `think_eog_mask`, the operator flag `--think-eog-mask`, `/props` | `src/core/sampling.{h,cpp}`, `src/core/chat.cpp`, `src/config.{h,cpp}`, `src/api/handlers.cpp` |
| which ids to mask, read from the vocabulary at load | `src/exec/backend_llama.cpp` (`eog_ids_`, `prepare_think_mask`) |
| the block's state fed to the sampler in all three llama decode loops (plain, multimodal, spec/verify) | same file, at each `on_piece` |

Three decisions worth keeping:

- **The mask is in the sampler, not `is_stop`.** Masking the logits keeps the
  invariant that the token the loop sees is a token that was legal; a filter in
  the stop check would let the turn-ender be sampled, spend a step and its
  probability mass, then be ignored.
- **Nothing outside `reasoning.cpp` writes a tag out.** The ids come from
  `llama_tokenize` over the table's own strings (`Tokenizer::special_ids`),
  the end-of-generation set from `llama_vocab_is_eog` over the whole
  vocabulary — the same predicate `is_stop` asks, so the two cannot drift.
  During this work a version that composed tag strings from characters in a
  test and in the probe looped on itself; that is the defect this campaign
  tracks, reproduced in the tooling, and the accessor is the fix for it.
- **The lead rule is one-sided.** Before the block's opening is settled the
  mask engages as soon as the generation's lead matches an opener *or could
  still grow into one*, and lets go only when the lead has provably become
  something else. Masking the turn-ender for the two or three tokens of a
  partial opener costs nothing; releasing a turn the client still reads as
  reasoning is the defect.

Evidence at unit level (`measured-here`, device-free build, 194 cases green):
the red case fails without the switch and passes with it — with the mask off a
row whose argmax is the turn-ending id ends the turn inside an open block; with
it on the same row samples the next token. Nine further cases cover the closer
clearing the mask, a closer split across two emitted pieces, a closer inside
one long piece, an opener at the lead engaging it, a quoted opener later *not*
engaging it, and the mask-off row being untouched. `tests/roundtrip.sh` checks
the flag reaches `/props`.

Not yet shown, and it is the gate: the same red case **at the endpoint on the
served coder unit**, which needs the llama build of this branch (a card window,
`docs/sop-card-window.md`). Until that pair exists the step is not closed.
