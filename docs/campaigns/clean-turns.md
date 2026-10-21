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

- **Step 2 built and its gate shown (2026-10-21, branch `clean-turns-step2`):**
  the switch drives the grammar in every llama decode loop; forced-malformed
  calls are unreachable with it on (vocab-level probe and endpoint pair), the
  CSV glance stays 10/10, decode t/s does not move. A **proposal**: the default
  stays off, no unit or package changed. See
  [Step 2 built](#step-2-built--grammar-constrained-tool-calls-measured-here-2026-10-21)
  at the end of this note. The [handover](clean-turns-handover.md) that preceded
  it is history now.

- Step 0 done (this note).
- Step 1 in progress: landed at unit level, gate written and proven able to
  fail; the endpoint pair needs the branch's llama build (running on the GPU
  dev host as of this writing).
- Step 2 done as a proposal (see above).
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

## Hardware note (2026-10-21, operator)

The B60 now serves `qwen3.8-agent` (dense Qwen3.8-27B) on port 8087; Flash-Next
(port 8089) is **stopped** and must not be started. The A770 rules are
unchanged. This changes which endpoints are live for probes; the coder unit on
the A770 is the one this campaign's leg targets.

## Step 1 endpoint leg — `measured-here` (2026-10-21, one seat, 6.4 min)

Seat: the operator's night shift. The leg stopped the coder unit, loaded this
branch's llama build on the A770 (n_ctx 98304, mtp 4, kv q8_0:q4_0, port 18080),
ran the gate, and restored the unit (verified `:8080/v1/models`, build
`83d7609`). Card identity by PCI id, sampler before the leg, zombie sweep by
pid, per `docs/sop-card-window.md`.

Gate, `empty-answer` shape (the only shape that provokes the defect on this
artifact), 4096 cap:

| arm | finish | content | reasoning | verdict |
|---|---|---|---|---|
| T=0.0 mask=off | stop | 0 | 673 | EMPTY-STOP |
| T=0.0 mask=on  | stop | 0 | 632 | EMPTY-STOP |
| T=0.7 mask=off | stop | 0 | 721 | EMPTY-STOP |
| T=0.7 mask=on  | stop | 7 | 624 | answered |

The T=0.7 arm is a clean red→fixed: without the mask the turn ends empty, with
it the model answers. The T=0.0 arm is the problem: the mask demonstrably
engaged (at greedy, off=673 vs on=632 reasoning chars — different streams, so
the mask changed a sampled token) yet the on-arm still ended `finish=stop` with
empty content. The gate read that as "the mask does not prevent the empty
stop" and printed FAILED.

**The gate was conflating two different outcomes.** The defect is a
turn-ending token winning a step *while the think block is still open*. But an
empty stop *after* the block closed is a legitimate empty answer — the model
thought, closed the block, and chose to say nothing — which the mask is allowed
to permit. The two are byte-identical to a client that cannot see the block's
state: both give `reasoning_content` = the block body and `content` = "". The
API never exposed whether the block closed, so the gate could not tell them
apart.

**The mask is complete across all three decode loops** (`code`, read against
the pin): every sampled token in the plain, multimodal, and spec/verify loops
passes through `sampler.sample()`, which applies the mask before greedy or
draws. In the verify walk (`src/exec/verify_walk.h`) a draft token is kept only
if it equals what the masked sampler drew, so a drafted turn-ender is kept only
if the sampler drew it — which it cannot while the block is open. `is_stop`
(`backend_llama.cpp:1739`) is the eog-predicate or the stop-token ids, the same
set the mask kills. Therefore, with the mask on and engaged, `finish=stop`
implies the block had already closed. The T=0.0 on-arm is a legitimate empty
answer (block closed), not the defect — the gate misread it.

**The fix is observability, not the mask.** `split_reasoning` already computes
`closed` (the closer was seen); the handler discarded it. It is now exposed as
`reasoning_closed` in the message, alongside the existing non-standard
`reasoning_content` (`src/api/handlers.cpp`). The gate now classifies on
mid-block vs closed-block: an empty stop that closed the block is not a
failure; an empty stop that did not is. When a unit does not report the field
(an older binary) the gate reads an empty stop as the defect — the conservative
call — and says so. `tests/python/test_clean_turns_gate.py` covers the new
classifier and the leg's exact T=0.0 scenario (off: mid-block stop; on:
closed-block empty answer → fixed). 9 cases green, device-free.

## Step 1 full battery — `measured-here` (2026-10-21, second seat, ~21 min)

**Scope: this result holds for the A770 coder artifact, and only for it.** On
2026-10-20 the operator session measured the mid-block stop *directly* on
Flash-Next (B60, IQ3_S): a raw /v1/completions probe with `ignore_eos=true`
showed the generation continuing past an end-of-turn token the model emitted
*inside* its open reasoning (the text after it was more reasoning), and the
chat probe ended `finish=stop`, empty content, reasoning cut mid-sentence. So
the mid-block EOG is **demonstrated on Flash-Next, not on the coder; it is
model-specific.** The mask is the fix for the Flash-Next artifact, and its
benefit test belongs there — a B60 window, the operator's call (the B60
currently serves the operator's own model; nothing is run on it). The coder
battery below shows the mask is a no-op safety net on *this* artifact, which
is the correct outcome for an artifact that does not exhibit the defect.

The first leg's `reasoning_closed` field made the block's state visible. The
full battery — all five probe shapes, both temperatures, both masks, 20 arms,
4096 cap, on the A770 (n_ctx 98304, mtp 4, kv q8_0:q4_0, port 18080) — then
answered the question the first leg could not:

**Every one of the 20 arms reports `closed=true`.** Zero arms ended the turn
while the think block was still open. The `empty-answer` shape's empty stops
(`finish=stop`, content 0) are all *closed-block* empty answers: the model
closes the block, then ends the turn with no content. The other four shapes
all answered (content non-empty), also with `closed=true`.

This overturns the campaign's premise. The defect step 1 targets — a
turn-ending token winning a step *while the think block is still open* — does
**not** reproduce on this artifact, across any of the five shapes. The
"empty stop" that does reproduce is a legitimate empty answer (the model
closed the block and chose to say nothing), which the mask is allowed to
permit and cannot fix. Note the first leg's T=0.7 off-arm (EMPTY-STOP) did not
reproduce here (answered, content 16): the empty stop is not deterministic at
T=0.7, so the "reliably reproduces" claim was itself unstable.

**The 2026-10-20 "think block open" was an inference, never a measurement.**
It was read from client-visible fields — content 0 + reasoning non-empty +
finish=stop — which cannot distinguish a mid-block stop from a closed-block
empty answer. The `reasoning_closed` field is the first direct measurement of
the block's state, and it says the block closed. Rule 5 holds: a label is not
evidence, and the label "block open" had no basis.

**The baseline audit's class (b) is the same blind heuristic.**
`tools/turn_defect_audit.py` counts `b_open_think_block` as `not has_text and
not has_call and has_think`, where `has_think` only checks that a thinking
block *exists*, never that it *closed*. So the "4 empty-stop turns in 2653"
(class b) were measured with the same client-visible proxy. They are, on this
evidence, closed-block empty answers (class a), not mid-block stops. The
count is real (the model does give empty answers) but the class label is
wrong, and the mask does not address it.

**What this means for step 1.** The mask is complete and correct across all
three decode loops (the proof above stands), and it is a sound safety net: if
any artifact or edge case did produce a mid-block stop, the mask would prevent
it. But on *this* artifact it is not fixing a demonstrated defect, because the
defect does not occur. The empty answers the baseline counted are a
model-quality issue (the model chose to say nothing after closing the block),
which the mask — by design — does not and should not fix. Whether to keep the
mask as a safety net, and whether to pursue the empty-answer defect as a
separate (model-quality) lever, is the operator's call.

## Step 2 — grammar-constrained tool calls (feasibility answered; built below)

The pinned llama.cpp exposes the grammar sampler publicly
(`include/llama.h:1581,1600`), driven through `llama_sampler_apply`/`accept`,
and its apply sets rejected logits to -INFINITY — the same shape arcint's
`Sampler` already uses. The cost: arcint has no llama.cpp sampler chain to add
it to; every llama loop samples `lgc::Sampler` over a raw logits row, so the
grammar must be driven by hand around `sampler.sample()`. The schema converter
does not drag in libcommon (a test binary links and runs on five upstream
translation units plus three string helpers, against arcint's vendored
nlohmann/json, and emits a 360-byte grammar). The pinned tree ships no grammar
for the coder's call form — that file would be ours. The price to measure:
`apply` decodes every candidate token, so its placement relative to
`collect_candidates` is a measurement, not an assumption. Not started.

## Step 2 built — grammar-constrained tool calls (`measured-here`, 2026-10-21)

Branch `clean-turns-step2` (from `clean-turns-wip`). Off by default; on by
`--tool-grammar` or the request field `tool_grammar`; `/props` reports it.

### What it does, and where (`code`)

| part | file | follows |
|---|---|---|
| the GBNF for the declared tools: opener, a declared function, required parameters in any order, optional ones, each non-string value typed by its schema, closer; parallel calls when the request allows them (OpenAI default), min 0 under the lazy trigger | `src/core/toolgrammar.cpp` (tags from `tool_call_tags()`, the parser's table) | `common/parsers/qwen3-coder.cpp` at the pin, the think-block template branch |
| the constraint hook: draw, test the drawn token alone, only on a refusal mask the row and draw again; `observe()` accepts every generated token | `src/core/sampler.{h,cpp}` (`TokenConstraint`) | `common_sampler_sample` (grammar-based rejection sampling) |
| the constraint on libllama's lazy grammar sampler; the opener is a trigger TOKEN (one token in this vocabulary), else a regex-escaped pattern | `src/exec/llama_toolgrammar.{h,cpp}` | `common_sampling_add_grammar_triggers` |
| the schemas reach the backend only when the calls are parsed (tools declared, `tool_choice` not `none`), with `parallel_tool_calls` | `src/exec/backend.h`, `src/api/handlers.cpp`, `src/core/chat.cpp` | |
| built per request next to the think mask in the plain, multimodal and speculative loops (MTP, DFlash2, combined lanes) | `src/exec/backend_llama.cpp` (`prepare_tool_grammar`) | |

**Spec/verify is correct by construction, and tested to be.** The verify walk
draws row *i* from the history the sampler itself accepted and `observe()`s only
the tokens it keeps, so the grammar's state is the emitted stream and a
rejected draft never reaches it — the reference's
`common_sampler_sample_and_accept_n` does the same. No force-disable of
speculation was needed. Device-free case `verify_walk_constraint_sees_only_kept_tokens`
covers it on synthetic rows; the vocab probe below covers it on the real
grammar, with a red mutant.

**The placement price** (the handover's step 6), answered by following the
reference: the grammar never sees the full row unless the drawn token is
refused. A token that fits costs one single-token `apply`; a refusal costs one
full-row mask. Measured at the endpoint below: sample time over the CSV call's
681 tokens 0.06 s off, 0.07 s on (no refusal); a request with one refusal
adds ~0.02 s of sample time (one full 248k-row mask).

### Deviations from the reference, named

- **Dormant while a think block is open.** The reference ends reasoning at the
  call opener (`thinking_end_tags` includes it) and triggers the grammar there.
  arcint parses calls from the content only (`split_reasoning`), so a call
  opener inside an open block is reasoning text to the API, and constraining
  it would build a call nobody parses. The grammar therefore waits for the
  closer (the sampler's think-block tracker, which now runs when either
  `think_eog_mask` or `tool_grammar` is set; the mask itself still engages
  only under `think_eog_mask`). Rests on the API contract of
  `split_reasoning`, not on an invariant; the operator may prefer the
  reference's behaviour.
- **`tool_choice: required` and a named function stay lazy.** The reference
  builds a non-lazy grammar with min one call for `required`; that grammar
  covers the generation prefix and the reasoning, which arcint renders itself.
  A named function is not narrowed to that one function. Not built: no agent
  in the baseline sends either.
- **Kept from the reference, so worth knowing:** a string-typed value is raw
  text up to the parameter closer, so a string `enum` is not enforced
  (recorded in `toolgrammar_llama_refuses_the_malformed_calls`); a call opener
  the model spells out of several text tokens instead of the special token
  does not trigger the grammar (the reference uses the token trigger alone for
  a one-token word); once a call is complete only another call or the
  turn-ender may follow.
- When the grammar would make the model call something, it makes it call a
  **declared** function — not necessarily the one the user meant (endpoint:
  asked for an undeclared `delete_everything`, the on arm called `read_file`).
  Malformed becomes unreachable; "what the user wanted" is not the grammar's
  to know.

### The gate

**1. Device-free suite** — `tests/test_toolgrammar.cpp` in `arcint-test`
(replacing the unregistered standalone file): libllama build 204 cases, stub
build with `-Werror` 201, 0 failed; `ctest -L unit` green in both. Red controls
inside: without the constraint the refused token wins; libllama's own grammar
engine (text, no vocabulary) refuses an undeclared function, a missing
required parameter, an undeclared parameter, a word for an integer, and
reports an unterminated block as incomplete.

**2. Vocabulary probe, no card** — `tools/tool_grammar_probe.cpp` on the served
artifact's vocabulary (248320 tokens; the opener is one token, so the trigger
is a token): an adversary writes logits that prefer a malformed call over a
well-formed one, five shapes, plain and speculative, T=0 and two seeds at 0.7.

| arms | grammar off | grammar on |
|---|---|---|
| 30 + 30 | 30/30 the adversary's malformed text, byte for byte | 30/30 schema-valid |
| red mutant `ARCINT_PROBE_RED_DESYNC=1` (feeds the first rejected draft) | — | 12 of the 15 speculative arms fail (the other 3 reject only at the turn-ender, which is never fed); the 15 plain arms stay valid |

**3. Endpoint pair, A770** — one seat of 8 min (16:49–16:57 UTC): the coder unit
stopped, renderD128 confirmed the A770 by PCI id (`8086:56a0`) and held by
nothing but the GPU.0 unit's process, this branch's build (sha256 `0ff818f0…`,
tree chain hash `9ed3e8df…` both ends) loaded with the unit's own flags
(n_ctx 98304, `--llama-mtp 4`, KV q8_0:q4_0, checkpoints 0) on port 18080,
`tools/tool_grammar_gate.py` run, the unit restarted and `/v1/models`
answering. The installed package cannot serve the field, so the branch build
with the unit's flags is the served configuration's stand-in.

Forced-malformed prompts (synthetic; T=0, T=0.7 seed 1, and T=0 with thinking on):

| shape | off | on |
|---|---|---|
| call an undeclared `delete_everything` | 3/3 undeclared function | 3/3 valid (`read_file`), 1 masked draw each |
| `read_file` with `filename` instead of `path` | 3/3 required `path` missing | 3/3 valid, 1–2 masked draws |
| `run_query` with `limit` as the word "ten" | 3/3 `limit` not an integer | 3/3 valid, 1 masked draw |
| `run_query` without `limit` | 2/2 required `limit` missing, + 1 no call (thinking on, 2048 cap) | 2/2 valid, + the same no-call at the cap |

11 of 12 off arms malformed, 0 of 12 on arms invalid. The thinking-on arms go
through the dormant-then-engaged path (the block closes, then the call is
constrained).

CSV task through a `write_file` call (the grammar spans the whole answer,
thinking off, 4000 cap, scored by executing the answer):

| arm | PUNKTE | decode | sample time |
|---|---|---|---|
| off, T=0 ×2 | 10, 10 | 681 tok, 79.4 / 79.5 t/s | 0.06 s |
| on, T=0 ×3 | 10, 10, 10 — byte-identical to the off answer | 681 tok, 79.5 / 79.5 / 79.4 t/s | 0.07 s |
| on, T=0.7 ×5 | 10, 10, 10, 10, 10 | 75.8–80.7 t/s | 0.11–0.15 s |

No draw was masked in any CSV run: the model's own calls were always inside
the grammar. Decode t/s on against off: 79.5 against 79.5 — no movement beyond
the ±0.1 run-to-run spread. The standard glance (no tools, so no grammar is
built) with the switch on: 8/8 at 10/10, 72.7–77.2 t/s.

### Open

- The **default stays off** — turning it on for a unit is the operator's
  decision; nothing here changes a unit or a package.
- The grammar is built per request (`generate_tool_grammar` + libllama's
  parse); not measured on its own. Upper bound from the probe: 31 builds, 60
  arms, the vocabulary load and its piece table in 7.6 s wall. A cache keyed on the declared tools is the obvious
  next step if a measurement says it is worth it.
- The deviations above (dormant in the think block, `required` lazy) are the
  operator's to accept or send back.
- Only the A770 coder artifact is measured. The B60's models were not touched.

