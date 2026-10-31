# feyntalk — Open Responses for agent harnesses in 1.3.0

**Open.** Design note (2026-10-31), nothing built. arcint serves OpenAI Chat
Completions; agent harnesses need server-side conversation state, cancel by id,
typed events. Decision taken (operator): arcint implements **Open Responses**
(`/v1/responses`, the Linux-Foundation specification) over HTTP + SSE in 1.3.0,
WebSocket and the rest in 1.3.x, plus a versioned `arcint:` extension
namespace and an optional token mode.

## Charter

1. A second, native protocol surface — not a translation layer over
   `/v1/chat/completions`. One request path, one generation path, one event
   model; the chat and completions routes keep their current behaviour.
2. The spec is the authority. Every deviation is named with its reason and its
   evidence class. No home-grown protocol dialect in the standard namespace.
3. No persistence to disk; conversation state is in-memory and bounded.
4. No touch of the engine step loop. Cancellation, token events and span events
   ride the existing callbacks (`GenerationInput::peer_alive`,
   `TokenCallback`) and the sampler; the concurrent `mtpw` branch's trace writes
   in the step loop stay untouched.
5. Text is the default. Token mode is an option, proven by a shared renderer
   before an operator turns it on for a unit.

## Reference to follow

- **Open Responses 2026-04-24** (the specification repository, checked out
  read-only): the normative text `src/specifications/2026-04-24.mdx`, the
  OpenAPI document `public/openapi/openapi.json`, the earlier release
  `2026-01-15.mdx` (no removal of anything used here), the compliance CLI
  `bin/compliance-test.ts` and its `src/lib/compliance-tests` library. Evidence
  class for every claim below taken from it: `paper`.
- **2026-04-24 delta** over 2026-01-15: WebSocket transport, connection-local
  `store=false` state, `/responses/compact`, assistant-message `phase`,
  optional `logprobs`, and the `response.create` envelope. `paper`.
- Prior art for token mode, from the operator's brief, not re-read at source in
  this note: **SGLang's model gateway gRPC mode** (workers speak token ids; the
  tokenizer, template and reasoning/tool parsers live in the router) and
  **vLLM tokens-only** serving; **llama.cpp and vLLM take token ids in and
  out**. `paper`. They establish that token-authoritative serving is a used
  architecture, not an experiment.
- arcint's own record, read at source for this note: `src/http/server.cpp`
  (routes 112/116/155/216, `peer_probe` 41-46), `src/api/handlers.{h,cpp}`
  (`SlotPool` 301-360, `prepare_chat` 746, `drive` 250, `stream_chat` 1119,
  `sse_send` 182, `run_chat` 978, `/props` 514), `src/exec/backend.h`
  (`GenerationInput`, `peer_alive`, `FinishReason`, `AbortCause`),
  `src/exec/backend_llama.cpp` (`abort_cause_of` 150-152, `peer_gone` 160-172,
  the prefill boundary 1951, the prefix reuse 1110-1129),
  `src/exec/backend_stub.cpp` (120-200), `src/core/{chat,toolcall,reasoning,
  sampler,sampling}.{h,cpp}`, `src/api/error.h`. `code`.

## Gate

**Acceptance gate (1.3.0), all must pass:**

1. **Device-free suite**: `ctest -L unit` green on the stub build with
   `-Werror`, 0 failed; every new case shown failing before its change (red
   first). The suite baseline is 204 cases in the libllama build and 201 in the
   stub build (`measured-here`, clean-turns, 2026-10-21); the feyntalk cases
   add to it.
2. **Round-trip**: `tests/roundtrip.sh` gains `/v1/responses` non-stream and
   SSE checks against `--stub`; 0 failed.
3. **Compliance CLI** against a stub (or CPU llama) build on a test port:
   `failed == 0` among the runnable tests, exit 0, and the skipped set equals
   the named list below. The CLI's per-request deadline is 30 s (`paper`, its
   README); a stub build meets it.
4. **Live leg**: a pi session with provider type `openai-responses` (SSE) runs
   a short multi-turn tool-using task against a test port on the card host in
   one card window (`docs/sop-card-window.md`). Gates: the task's expected
   answer is right; ≥ 2 tool round-trips; the continuation's
   `cache_hit_tokens >= prior prompt_tokens - 1` (the guaranteed shared prefix,
   `code`); 0 malformed tool calls; decode within the run-to-run spread of the
   chat-route arm on the same window; the stop classes unchanged.
   **Outcome (`measured-here`, 2026-10-31, A770, the coder model, serving
   build 83d4e26, one card window):** PASS — 4 tool calls (read, read, write,
   read), 0 malformed, the expected answer right. The continuation shared
   93–97 % of the prior prompt in cache, consistently 4 tokens below
   `prior prompt_tokens - 1`; the chat route shows the same property: the
   prior turn's generation prefix is re-rendered differently in history, so
   the gate's `- 1` is not met by 4 tokens and the shared-prefix guarantee is
   `code`-bounded, not exact. Recorded rather than waived: the answer and the
   tool loop are right, the cache share is high.
5. **Cancel measurement**: at a 100,000-token prefill, the cancel POST to the
   aborted request's `response.incomplete` is bounded by **one ubatch**: the
   server's `prompt_tokens_prefilled` is within one `--prefill-chunk` of the
   point where cancel was issued, and the control arm (connection close, today's
   path) shows the same bound. Recorded with card, depth, KV precision and
   configuration.
   **Outcome (`measured-here`, 2026-10-31, A770, the coder model, build
   83d4e26, ~14–15k-token prefill, cancel issued 10 s in):** the bound is
   **~1.6 ubatches for both arms, not one** — cancel-by-id aborted at 14,336
   prefilled after 1.28 s, the socket-close control at 14,336 after 1.25 s;
   `response.created` reached the client after 0.01 s. The extra 0.6 ubatch is
   the existing boundary check's property (the ubatch in flight at the check),
   identical for the control arm; recorded honestly, not narrowed to one.

**Knowingly skipped in 1.3.0** (each is refused loudly or absent from the
served surface, never silently accepted): every WebSocket test; `/responses/
compact`; `background: true`; hosted tools (`implementor_slug:` tool types);
`allowed_tools`; `text.format` `json_object`/`json_schema`; logprobs /
encrypted reasoning / obfuscation; assistant `phase`; prior reasoning replay
fidelity (see the open question). `include` is not skipped: it is accepted and
ignored, because clients send it on every reasoning request. See the Scope
section for the reasons and the status codes.

## Scope — 1.3.0 vs 1.3.x

| spec part (2026-04-24) | 1.3.0 | 1.3.x | reason / evidence |
|---|---|---|---|
| `POST /v1/responses`, JSON bodies, non-stream `ResponseResource` | **yes** | — | the base surface; reuses `prepare_chat` + `drive` `code` |
| HTTP framing: JSON bodies, `application/json` responses, `text/event-stream` + `Cache-Control: no-cache` for SSE | **yes** | — | `paper`; the chat SSE route already sets that header (`server.cpp:196`) `code` |
| SSE (`text/event-stream`, `event:` line equals `type`, terminal event, `[DONE]`) | **yes** | — | `paper` spec §HTTP Responses; the writer is new, `sse_send` 182 writes `data:` only `code` |
| item lifecycle + status machines (message, reasoning, function_call) | **yes** | — | `paper` §Items; the states come from `FinishReason` `code` |
| semantic event set (`created`, `in_progress`, `output_item.added/done`, `content_part.added/done`, text/reasoning/function-call deltas, `completed`/`failed`/`incomplete`) | **yes** | — | `paper` OpenAPI event schemas |
| `previous_response_id`, `store`, eviction on a failed continuation | **yes** (HTTP, in-memory) | — | `paper` §`previous_response_id`; `store=false` over HTTP decided below |
| `tool_choice` auto/required/none/function object; flat function tools; `parallel_tool_calls` | **yes** | — | maps to `ChatRequest` `code` |
| `instructions`, temperature, top_p, penalties, `max_output_tokens`, `metadata`, `service_tier`, `safety_identifier`, `prompt_cache_key`, `truncation` | **yes** (accepted/echoed) | — | `paper` CreateResponseBody |
| error envelope with spec types | **yes** | — | `paper` §Errors; `error_body` already has the shape, different type spellings `code` |
| `arcint:` extension namespace (cancel, token mode, spans, loop guard) | **yes** | — | `paper` §Extending; operator decision |
| WebSocket transport (`response.create`, sequential turns, connection-local `store=false`, 60-min limit, WS error events) | — | **yes** | the spec's transport mode; cpp-httplib 0.18.3 has no WebSocket support `code`; separate work package |
| `POST /v1/responses/compact` + `CompactionBody` | — | **yes** | no compaction pass exists; a summariser is a model job and a card/money decision `code` |
| `background: true` + a background cancel lifecycle | — | **yes** | the spec defines the request flag but no lifecycle route `paper`; needs a job layer |
| `allowed_tools` hard constraint | — | **yes** | needs grammar narrowing or suppression; 400 `unsupported_parameter` in 1.3.0 |
| `item_reference` input items, `compaction` input items | — | **yes** | need an item index in the store and a compaction pass respectively; 400 in 1.3.0 |
| `text.format` json_object/json_schema | — | **yes** | the schema→GBNF machinery exists (`src/core/toolgrammar.cpp`) but is opt-in and not wired for answers `code`; 400 in 1.3.0 |
| reasoning `summary`, `encrypted_content`, logprobs, obfuscation, `phase` | — | **yes** | arcint surfaces raw reasoning only; summaries/encryption are model-provider features `code` |
| `include` (2026-04-24 enum) | **accepted and ignored** | — | clients send `reasoning.encrypted_content` every turn; a 400 would lock them out, and arcint serves neither encrypted reasoning nor output logprobs. The enum is accepted even where a future value is unknown; only a non-string entry is a 400 `invalid_request`, param `include` (`measured-here`, live client) |
| hosted tools (`implementor_slug:`) | — | **later** | arcint executes no tools; the item type would be a receipt for a tool it never ran `code` |
| `max_tool_calls` | — | **later** | no cross-call counting exists; 400 in 1.3.0 |
| `response.queued` | never | — | admission is a status code before `response.created` (`SlotPool`/503 `code`); no queue state exists |
| Authorization header enforcement | ignored | — | server-side auth is not part of this surface; the header is a client obligation `paper`; arcint has no auth today `code` |

`store=false` over HTTP: the spec's connection-local state is a WebSocket
concept (`paper`); over HTTP there is no connection-local cache, so a response
created with `store=false` is not addressable after its stream ends. A later
`previous_response_id` naming it answers **400 `previous_response_not_found`**
(`paper`); a client holding `store=false` must resend the full input, which is
what Open Responses clients do on that error anyway. The alternative — keeping
`store=false` entries in RAM for a TTL — is refused because it would contradict
the zero-retention meaning the flag carries.

## Request mapping (point 2)

Conversion sits in a new `src/api/responses.{h,cpp}`, in two functions:
`parse_response_request` (body → an internal `ResponseRequest`: `ChatRequest`
fields + the spec fields the resource must echo + the `arcint` extension
object) and `prepare_response` (that → the existing `PreparedChat`). The chat
path's `prepare_chat` (`src/api/handlers.cpp:746`) is split into
`parse_chat_request` (unchanged) and `prepare_from_chat` (the body from line 756
on: lane resolution, template default, `render_chat`, `finish_prepare`, think
state, media). Both routes then call **the same** `prepare_from_chat`; the
Responses route never gets a second prompt builder. `drive`
(`src/api/handlers.cpp:250`), `log_stats` and `log_slot_line` move to
`src/api/generate.{h,cpp}` so the Responses stream uses the identical stop
scan, reasoning split, UTF-8 hold-back and tool parsing.

| Responses input | arcint | note / evidence |
|---|---|---|
| `input` string (max 10,485,760 chars) | one `user` `ChatMessage` | `paper` CreateResponseBody |
| `message` role `user`/`system`/`developer`/`assistant`, content parts `input_text`, `output_text`, `text` | `ChatMessage` role, joined text | `developer` is a known role (`src/core/chat.cpp:13`); part types are concatenated with `\n` between parts `code` |
| `message` content `input_image`/`input_file`/`input_video` | `ImagePart` for `input_image` only, with the existing mmproj refusal when unloaded; the other two are 400 `unsupported_parameter` | same refusal as chat `code` |
| `function_call` item | `assistant` `ChatMessage` with `tool_calls` (`id`=`call_id`, `name`=`name`, `arguments`) | replay shape `code` |
| `function_call_output` item, `output` string | `tool` `ChatMessage` (`tool_call_id`=`call_id`, content=`output`) | `src/core/chat.h` requires `tool_call_id` `code`; array output: `input_text` parts joined, images refused |
| `reasoning` item | accepted, kept in the stored state, **not rendered** into the prompt | see Open questions; parity with `/v1/chat/completions`, which has no reasoning replay `code` |
| `item_reference` item | 400 `unsupported_parameter`, param `input` | resolving item ids from a stored response is 1.3.x; the store has no item index in 1.3.0 `code` |
| `compaction` summary item (input) | 400 `unsupported_parameter`, param `input` | no compaction pass exists `code`; `paper` ItemParam allows it |
| `tools[]` `strict` field | accepted and ignored | arcint's tool grammar is a separate opt-in (`--tool-grammar`); `strict` is a request-time enforcement hint `paper`, `code` |
| `instructions` | one `system` message at the head of `messages`; echoed in the resource | `paper` field; arcint templates take a system turn `code` |
| `tools` `{"type":"function","name","description","parameters"}` | `ToolSpec` (flat, no `function` wrapper) | `code` `ToolSpec` |
| `tool_choice` `"auto"`/`"required"`/`"none"` | same string | `code` |
| `tool_choice` `{"type":"function","name":"f"}` | `ChatRequest::tool_choice="required"` + `tool_choice_function="f"` | arcint's internal form is the chat shape `code`; a named function stays lazy in the grammar (clean-turns deviation, unchanged) |
| `tool_choice` `{"type":"allowed_tools", ...}` | 400 `unsupported_parameter`, param `tool_choice` | 1.3.x (Scope) |
| `text.format` absent or `{"type":"text"}` | nothing | model default `paper` |
| `text.verbosity` | accepted and ignored | no verbosity control exists `code` |
| `text.format` json_object/json_schema | 400 `unsupported_parameter`, param `text.format` | 1.3.x |
| `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` | `SamplerOverrides` | same fields as chat `code` |
| `max_output_tokens` (≥ 16) | `sampler.max_tokens` | `code` |
| `parallel_tool_calls` | `ChatRequest::parallel_tool_calls` | `code` |
| `reasoning.effort` `"none"` | `enable_thinking=false` | the fleet's switch `code` |
| `reasoning.effort` low/medium/high/xhigh/minimal | `enable_thinking=true` | Qwen templates know on/off only (`src/core/chat.cpp:354` comment) `code`; echoed unchanged |
| `reasoning.summary` | accepted, echoed `null`; no summary is produced | 1.3.x |
| `truncation` `"disabled"` or `"auto"` | both: no truncation, `§3.8` 400 on overflow with `prompt_tokens`/`n_ctx` | `paper`: `auto` MAY truncate, not MUST; `disabled` MUST fail — same behaviour `code` |
| `stream`, `stream_options.include_obfuscation` | `stream` honoured; obfuscation accepted and ignored (arcint emits no `obfuscation`) | `paper` optional field |
| `store`, `background`, `metadata`, `service_tier`, `safety_identifier`, `prompt_cache_key` | echoed; `background:true` → 400 `unsupported_parameter`, `service_tier` other than `auto`/`default` → 400 | see Scope `code` |
| `max_tool_calls` | 400 `unsupported_parameter`, param `max_tool_calls` | 1.3.x |
| `include` (array of strings, any enum value) | accepted and ignored: nothing is added to the output | clients send `reasoning.encrypted_content` on every reasoning turn; the strings are the spec's enum, a future value is accepted too; a non-string entry is 400 `invalid_request`, param `include` `measured-here` |
| `arcint` object (extension) | parsed by `src/api/responses.cpp` (see Extension namespace and Token mode) | `paper` §Extending existing schemas |

## Event mapping (point 3)

Text mode is the default stream. `sequence_number` starts at 1 and increments by
1 per event, no gaps; the `event:` line carries the same string as the body's
`type` (`paper` spec §Streaming HTTP Responses). `[DONE]` follows the terminal
event (MAY, `paper`).

| arcint source (`code`) | spec event (`paper`) | payload notes |
|---|---|---|
| request admitted | `response.created` | full `ResponseResource`, `status:"in_progress"`, `output:[]`, every required key present, zero values where the spec says so |
| immediately after | `response.in_progress` | same snapshot |
| first reasoning piece (`ReasoningStreamer::Step::reasoning`, `src/core/reasoning.h:49`) | `response.output_item.added` (`reasoning`) then `response.content_part.added` (`reasoning_text`) | item `status:"in_progress"`, `summary:[]`; part `{"type":"reasoning_text","text":""}` |
| reasoning piece | `response.reasoning.delta` | the UTF-8-safe text after `utf8::Streamer` (`code`: `stream_chat` reasoning stream 1141-1160) |
| think closer in a step (`step.content` non-empty / `closed`) | `response.reasoning.done`, `response.content_part.done`, `response.output_item.done` | item `status:"completed"`, summary `[]`, content = the reasoning text |
| first content piece | if no reasoning item: `response.output_item.added` (`message`) then `response.content_part.added` (`output_text`) | part `{"type":"output_text","annotations":[],"text":""}` |
| content piece | `response.output_text.delta` | same UTF-8 hold-back as today |
| content end | `response.output_text.done`, `response.content_part.done`, `response.output_item.done` | item `status:"completed"`, content the full text |
| parsed call (`parse_qwen_tool_calls`, at turn end) | `response.output_item.added` (`function_call`) | `name`, `call_id`, `arguments:""`, `status:"in_progress"`; the spec needs `name` at added time `paper` |
| parsed call arguments | `response.function_call_arguments.delta` (one delta: the whole JSON string) then `.done` | arcint parses only the finished call, so the argument stream has no token granularity in text mode `code`; token mode gives the span boundaries |
| finish `Stop` (with or without calls) | `response.completed` | `usage` filled from `usage_json` |
| finish `Length` | `response.incomplete`, `incomplete_details.reason:"max_output_tokens"` | the open item closes `incomplete` and is the **last** item `paper` §Items |
| finish `Loop` (loop-guard abort) | `response.incomplete`, `incomplete_details.reason:"loop_guard_abort"` | same item rule; `arcint:loop_guard` event carries the counters |
| finish `Abort` / `ClientGone` | nothing more | nobody is there to receive a terminal event; the write failed `code` |
| finish `Abort` / `StopSignal` | `response.failed` (`server_error`, code `server_stopping`) | the process is stopping; delivery is best-effort |
| finish `Abort` / `Cancelled` (extension) | `arcint:response.cancelled` then `response.incomplete`, reason `"cancelled"` | spec has no cancelled state `paper`; `incomplete` is the nearest terminal state |
| generation exception (`guarded`, `src/api/handlers.cpp:962`) | `error` streaming event then `response.failed` | `paper` §Errors: an in-stream error is followed by `failed`; the open item closes `incomplete` (the two MUSTs meet at a tension, see Open questions) |
| loop-guard counters (`report_loop_guard`) | `arcint:loop_guard` (end of stream) | the seven keys of `loop_guard_json` (`src/api/handlers.cpp:119`), `paper` §Extending streaming events |
| `stats` | `response.completed`'s `usage` | `input_tokens`=prompt, `input_tokens_details.cached_tokens`=cache hit, `output_tokens`=completion, `output_tokens_details.reasoning_tokens`=0 in 1.3.0 (no separate count `code`) |
| rotation (expert cache) | `arcint:rotation` | type defined, emitted in 1.3.x when the cache exposes a hook; nothing publishes it today `code` |

`ReasoningStreamer` gives the reasoning/content split by token id where the
vocabulary spells a tag as one special token (`code`); the event mapping uses
the same split as `stream_chat`, so text and Responses cannot disagree.

## State (point 4)

A stored response, in `src/api/responses_state.{h,cpp}`, one `ResponseStore` per
process (`Context` owns it), guarded by a mutex:

| field | content |
|---|---|
| `id`, `created_at`, `status`, `completed_at` | the resource envelope (`paper` ResponseResource) |
| `lane`, `slot`, `previous_response_id` | affinity and the chain (`code`: `Lane`, `SlotPool`) |
| `chat` | the `ChatRequest` after mapping (messages, tools, tool_choice, sampler, thinking) — the input half of the spec's concatenation |
| `output_items` | the spec items the turn produced (message/reasoning/function_call), for the output half |
| `prompt_tokens`, `prompt_ids` | the prompt as served; ids required for the token-mode concat, tokens for the affinity gate |
| `prefix_hash` | sha256 of the rendered prompt (text mode) or of the id vector (token mode) — a mismatch means the render can no longer reproduce the prefix, and the continuation is refused |
| `renderer_hash` | the artifact/template material hash of the turn (Token mode) |
| `usage`, `error`, `incomplete_details` | the terminal accounting |
| `cancel` | `shared_ptr<std::atomic<bool>>`, also held by the in-flight registry |

Semantics, decided:

- **`store` default `true`** (OpenAI convention; the spec does not state a
  default `paper`). `store=true` retains the entry in RAM until evicted.
  `store=false` retains nothing; the ID is registered only while the request is
  in flight (for cancel) and removed when it ends, so a later
  `previous_response_id` gets `previous_response_not_found`.
- **Limits, per lane**: 32 entries and 16 MiB of retained state (the prompt ids
  dominate: a 100k-token id vector is ~400 KB `code`), whichever binds first;
  LRU eviction by `last_used`. TTL 24 h so an abandoned chain frees itself. No
  disk, no restart survival. Operator flags
  `--responses-store-entries`/`--responses-store-bytes`/`--responses-store-ttl`,
  defaults above, `/props` reports the live count.
- **`previous_response_id` resolution**: lookup is lane-scoped. Entry found in
  the same lane → continue. Not found, wrong lane, or `prefix_hash` mismatch →
  **400 `previous_response_not_found`, param `previous_response_id`** (`paper`
  code; the same code the spec names). A continuation over a stored entry
  rebuilds `ChatRequest` = stored messages + the stored assistant turn + the
  new mapped items, re-renders with the same template, and serves
  `stored.prompt ∪ stored output ∪ new input` in that order (`paper`
  §`previous_response_id`).
- **Eviction on error**: a continuation turn that ends 4xx/5xx or `failed`
  evicts the referenced entry, so a retry cannot silently continue from a state
  the server already treated as broken (`paper` §WebSocket Continuation, applied
  to HTTP too).
- **Slot affinity**: the entry records the slot (the lease index, or the named
  lane's sequence). A continuation first tries that slot non-blockingly
  (`SlotPool::try_acquire(index)`, new, ~15 lines next to `acquire_for`
  318-336), then falls back to the lane's ordinary `acquire_for(queue_timeout)`.
  It never takes another lane (`DESIGN §4.1`, `code`). A different slot means
  the prompt re-prefills fully; the response says so through the usual
  `cache_hit_tokens` number. Gate: affinity path
  `cache_hit_tokens >= prompt_tokens - 1` (`code`: `backend_llama.cpp`
  1110-1129 reuses the longest common prefix and always re-runs one token).
- **No persistence** unless the spec forces it. It does not: the state
  machines are explicitly not storage (`paper` §Items are state machines).

## Cancel (point 5)

The spec defines **no cancel endpoint and no cancel state** — the word does not
occur in the 2026-04-24 text or the OpenAPI document (`paper`, checked). The
only spec-defined way to stop an in-flight streamed response is the client
closing the connection, which arcint already handles at every prefill ubatch
boundary (`backend_llama.cpp:1951`) and every 0.5 s of decode (`peer_gone`
160-172; `kPeerCheckSeconds` 166) (`code`). Background mode, which OpenAI pairs
with cancel, is not in 1.3.0. Cancel by id is therefore an **arcint extension**:

- `POST /v1/responses/{id}/cancel` → `200` with the current `ResponseResource`
  snapshot plus `arcint.cancelling: true`. Unknown id → `404` error envelope,
  code `response_not_found`. Terminal response → `409`, code
  `response_not_cancellable`.
- The flag is an `std::atomic<bool>` registered with the response ID **before**
  the first byte (`response.created` is what gives the client the id). The
  route's `peer_alive` closure ORs it with the socket probe:
  `[sock, cancel]{ return !cancel->load() && peer_probe(sock)(); }`. Every
  boundary check the engine already makes therefore sees a cancel exactly as it
  sees a close. **No new check in any loop; the `mtpw` trace writes around the
  step loop are not touched.** (`code` for the hook; the closure is the design.)
- Naming the cause, so the slot line does not lie: `AbortCause::Cancelled` is
  added to `src/exec/backend.h` (~390), `GenerationInput::cancel` next to
  `peer_alive`, and `abort_cause_of` (`backend_llama.cpp:150-152`) reads it
  after the stop flag; the stub's `cause` (`backend_stub.cpp:216-220`) mirrors
  it. These are the only engine-file edits. Token mode does not change them.
- What the client receives: `arcint:response.cancelled` then
  **`response.incomplete`** with `incomplete_details.reason:"cancelled"`; the
  open item closes `incomplete` and is last (`paper` item rule). A non-streamed
  request that is cancelled returns `200` with the same incomplete resource.
- Voice/speech and transcription stay uncancellable, unchanged (`DESIGN §3.7`,
  `code`).

**Measurement plan** (W4; no number claimed before it runs): a stub arm proves
the path device-free (`--stub`, with the test constructing
`make_stub_backend(n_ctx, delay_ms, {}, {}, {}, N)` as
`tests/test_disconnect.cpp:177` already does), then one card
window at a 100k-token prefill on the card host with `--prefill-chunk 512`. Two
arms: (a) today's control, client closes the socket; (b) cancel by id from a
second connection. Clocked at the endpoint: issue → the request's slot line
(`aborted: cancel requested`) and → `response.incomplete` arrival; plus
`prompt_tokens_prefilled - tokens_at_issue`. The expected bound is **one
ubatch** for both arms (`code`: the boundary sits at 1951; the chunk is the
ubatch). Decode-side cancel is bounded by `kPeerCheckSeconds = 0.5 s` (`code`),
not a ubatch; state that, too.

## Extension namespace (point 6)

The spec's MUST is a colon form and the implementor slug, not a dot:
byte-prefixed `implementor_slug:event` (`paper` §Extending streaming events,
§Items are extensible). So the namespace is **`arcint:`**, not
`arcint.loop_guard`; the body's `type` carries it and the SSE `event:` line
matches it. Versioning:

- Requests: one optional top-level object `"arcint": {"v": 1, ...}`. `v` is the
  extension schema version (integer, current 1). An unknown `v` → 400
  `unsupported_parameter`, param `arcint.v`. Additive optional fields do not
  bump `v`; a semantic change, a removal or a rename bumps it. This keeps all
  arcint-only input under one key, so a strict client sees only standard fields
  (`paper` §Extending existing schemas: keep standard fields, prefer optional).
- Events: `arcint:token`, `arcint:span`, `arcint:loop_guard`,
  `arcint:response.cancelled`, `arcint:rotation`. Each body carries `"v": 1`
  and a `sequence_number` shared with the standard stream (`paper` requires
  `type` + `sequence_number` for extended events).
- Extended event fields never change the standard stream's semantics: token
  order, item lifecycle and the terminal event remain intact, and the terminal
  `ResponseResource` carries the full answer, so a client that ignores
  `arcint:*` can reconstruct everything (`paper` §Extending: clients MUST be
  able to ignore safely).
- Items: none in 1.3.0. Loop-guard and rotation are observability, not output
  items; inventing an item type would put a non-portable record in `output`.
- A one-line statement in `/props`:
  `"feyntalk": {"renderer": "...", "schema": "feyntalk-render/1",
  "token_mode": true, "arcint_v": 1}`.

## Errors (point 7)

The spec envelope is `{"error":{"message","type","param","code"}}` with types
`server_error|invalid_request|not_found|model_error|too_many_requests`
(`paper` §Errors). `error_body` already emits that shape but with the chat
spellings (`invalid_request_error`, `internal_error`; `src/api/error.h`,
`code`). The Responses route gets its own `responses_error(message, type, code,
param)` so the chat and completions routes keep their bytes unchanged.

| condition | HTTP | type | code / param | evidence |
|---|---|---|---|---|
| body not JSON, bad field, unsupported optional field | 400 | `invalid_request` | `invalid_request` / field path | `paper`; existing 400s `code` |
| unknown model / named lane | 404 | `not_found` | `model_not_found` / `model` | `resolve_lane`, `code` |
| no route | 404 | `not_found` | `not_found` | `server.cpp:99-108`, `code` |
| `previous_response_id` unknown / different lane / hash mismatch | 400 | `invalid_request` | `previous_response_not_found` / `previous_response_id` | `paper` names this code |
| context overflow | 400 | `invalid_request` | `context_length_exceeded` / `input`, plus `prompt_tokens`, `n_ctx` at the top level (extension keys) | `DESIGN §3.8`, `code` |
| all lanes busy after the queue timeout | **429** | `too_many_requests` | `all_lanes_busy` / null | `paper` maps this type to 429; chat keeps 503, `code` |
| engine exception before/inside a turn (non-stream) | 500 | `server_error` | `internal_error` | `guarded`, `code` |
| engine exception while streaming | 200 stream | `server_error` | `error` event then `response.failed` | `paper` §Errors |
| `background`, `allowed_tools`, json format, `max_tool_calls`, unknown `arcint.v` | 400 | `invalid_request` | `unsupported_parameter` / the field | Scope |
| cancel unknown / terminal | 404 / 409 | `not_found` / `invalid_request` | `response_not_found` / `response_not_cancellable` | extension |

`dump_json` keeps replacing invalid UTF-8 with U+FFFD (`src/api/error.h`,
`code`); an error body still serialises.

## Token mode (point 10)

Decided as an **option in 1.3.0**, off by default; text stays the default until
a shared renderer (feyntalk-render) is proven against a served unit.

**Request.** In the `arcint` object:
`{"v":1,"token_mode":true,"renderer_hash":"sha256:…","input_ids":[…]}`.
`stream:true` is required (token mode without a stream → 400
`unsupported_parameter`, param `stream`). `arcint.input_ids` is **required** in
token mode: the renderer tokenizes and renders the prompt, the server does not
render it — that is what makes the prefix byte-identical and the slot binding
deterministic. On a first turn, `input_ids` is the whole rendered prompt; on a
`previous_response_id` turn it is only this turn's rendered delta (the prior
assistant turn's template suffix, plus the new items and the next assistant
opener), and the server concatenates `stored.prompt_ids ∪ stored output_ids ∪
input_ids`. The standard `input` items are still required and are what the
resource records; the server does not verify them against `input_ids` (the
renderer test does). `arcint.renderer_hash` is required; a mismatch with the
served one → **409 `renderer_mismatch`** before any token, so a stale renderer
cannot drive a prompt the server would tokenise differently. Text-only: token
mode with an image part → 400 `unsupported_parameter`.

**Events.** Standard lifecycle events stay exactly as in text mode; in
addition, one `arcint:token` per accepted output token:

```json
{"type":"arcint:token","v":1,"sequence_number":41,
 "response_id":"resp_…","item_id":"msg_…","output_index":0,
 "content_index":0,"field":"reasoning","token_index":7,
 "token_id":248069}
```

`field` is `reasoning` or `text` (the token's side of the think split; a
function call's tokens are `text` and are delimited by the tool-call span).
`token_index` is the 0-based position in the accepted output stream (reasoning,
content and tool-call tokens share one index space); under speculation only
accepted tokens are counted, because `on_piece` is called for emitted tokens
only (`code`: the plain loop breaks on `is_stop` before emitting 1184-1197; the
verify walk emits kept tokens 1826-1834). The server also keeps emitting the
standard text deltas as a **compatibility shadow** from the existing
UTF-8 streamer; a renderer ignores them, a strict client reconstructs from
them, and a test asserts the shadow equals `render(tokens)` byte for byte. That
keeps every spec MUST satisfied and leaves the token seam non-authoritative.

**Span events.** One type, a `span` discriminator, `phase` start/end (or
`event` for a point signal), and a `token_index`:

```json
{"type":"arcint:span","v":1,"sequence_number":52,"span":"thinking",
 "phase":"start","token_index":0,"item_id":"rs_…","output_index":0,
 "detail":{}}
{"type":"arcint:span","v":1,"sequence_number":78,"span":"tool_call",
 "phase":"start","token_index":31,"item_id":"fc_…","output_index":1,
 "call_id":"call_0","detail":{"name":"read_file"}}
{"type":"arcint:span","v":1,"sequence_number":90,"span":"loop_guard",
 "phase":"event","token_index":44,"detail":{"rung":"door","match":"near"}}
```

Sources, all outside the engine step loop: thinking start is `prep.think_open`
at index 0 or the first reasoning piece; thinking end is the `ReasoningStreamer`
closer; tool-call start is the token whose piece completed the call opener
(one special token on the coder vocabulary, `measured-here`, clean-turns
2026-10-21) and end is the parser's last emitted index; loop-guard is an
**event sink** in `SamplerParams` (`src/core/sampling.h:166`) that
`src/core/sampler.cpp` fires at its own increments — detection 580, nudge 779,
slap 788, abort 813 — with the API's stream stamping `token_index` from its
accepted-token counter (same thread, no race). The sampler already receives the
params from `GenerationInput::sampler`, so no backend loop call site changes.

**Renderer material.** `GET /v1/responses/renderer`, no auth (like `/props`),
returns a bundle conforming to **`feyntalk-render/1`**:

```json
{"schema":"feyntalk-render/1","hash":"sha256:…",
 "model":{"served_id":"…","canonical_id":"…","quant":"…","arch":"…"},
 "tokenizer":{"kind":"bpe","tokens":[],"merges":[],"token_types":[],
              "added":[],"eos":248044},
 "chat_template":{"text":"…","source":"gguf"},
 "special_tokens":{"think":[{"open_ids":[248068],"close_ids":[248069]}],
                   "tool_call":{"open_ids":[…],"close_ids":[…]},
                   "eog_ids":[…]},
 "parser":{"reasoning":"tags/1","tool_calls":"qwen3-coder/1","grammar":"gbnf/1"}}
```

`hash` = sha256 over the canonical JSON (sorted keys, no insignificant
whitespace) of `tokenizer`, `chat_template`, `special_tokens`, `parser`,
excluding the volatile model names. Published in `/props`, in every
`response.created` as `arcint.renderer_hash`, and in the terminal resource.
The material is produced by a new `Backend::renderer_material()` (default:
empty, so the stub has no token mode), implemented in `backend_llama.cpp` next
to `template_caps` 1032 and `special_ids` 326 — load-time, not the step loop.
feyntalk-render lives outside this repository; this repository owns the
material contract and the hash only.

**Renderer, the translator lives in the gateway (operator, 2026-10-31).**
feyntalk-render is versioned independently (schema 1) and one implementation
lives in the gateway/proxy in front of the engine, not in the agents. The proxy
renders the prompt, parses the answer and owns the token→text seam; agents
receive text plus the `arcint:` span events through the proxy. The agent-side
extension is only for WebSocket and `previous_response_id`, so no second
renderer has to agree with the first. Two modes: **tokens** (arcint now; vLLM
and llama.cpp later, both take ids) and **text** (cloud providers only deliver
text; the renderer still renders the prompt and parses the answer). The proxy
refuses on a hash mismatch before consuming a token. Incremental byte-BPE
detokenisation (split UTF-8 sequences, leading spaces, byte fallbacks) is the
one piece that needs a clean test set once; the server's shadow stream is the
oracle for it.
Why this exists: the recurring defect class sits at the token→text seam — stop
strings swallowing a closer id, opener literals in the chat path, UTF-8 splits
in think deltas, `reasoning` vs `reasoning_content`, tool-call salvage in the
proxy (`docs/campaigns/clean-turns.md`, `loop-guard.md`, `code`).

## Test plan (point 8)

| level | what | gate | evidence |
|---|---|---|---|
| unit, stub | `tests/test_responses.cpp`: request mapping (every item type, flat tools, tool_choice object, refusals, sampler, instructions, extension `v`); a required-key set test for `ResponseResource`; event writer (order, `event:`==type, `sequence_number` contiguous from 1, item/part lifecycle, reasoning→content switch, tool item added with name before arguments); terminal mapping (`Length`→incomplete last item, `Loop`→incomplete reason `loop_guard_abort`, exception→error+failed); state (32/16 MiB LRU, TTL, `store=false` not stored, `previous_response_not_found`, eviction on a failed continuation, hash mismatch); token mode (indexes contiguous from 0, span indices exact on the stub's fixed reply, shadow text == `render(tokens)` via a test renderer, `arcint.input_ids` required, images refused); cancel (stub with `prefill_ubatches=4` and `delay_ms`: cancel aborts at the next ubatch, returns incomplete, frees the slot; terminal→409, unknown→404) | `ctest -L unit`, 0 failed, red-first per case | `code` |
| round-trip | `tests/roundtrip.sh`: non-stream resource and SSE framing on `--stub` | 0 failed | `code` |
| compliance | the spec CLI against a stub/CPU build: runnable HTTP non-stream + SSE tests | 0 failed, exit 0; skipped = the named Scope list | `paper` |
| live, text | pi (`openai-responses`, SSE) multi-turn tool task on a test port, card host, one window | the Gate items 4/5 | `measured-here` |
| live, token | the pi provider extension + feyntalk-render in tokens mode against the same test port; the harness byte-compares rendered text to the server's shadow per turn and fails on any hash mismatch | task passes, 0 mismatches; a precondition before the operator enables the option for a unit | `measured-here` |

The compliance id list cannot be enumerated from the specification checkout
available here: the CLI imports `../src/lib/compliance-tests`, which is not in
that checkout — only `bin/compliance-test.ts` is (`code`: directory listing).
Two ids are named in the specification README (`basic-response`,
`streaming-response`, `paper`). See Open questions.

## Work packages (point 9)

| # | package | files | gate |
|---|---|---|---|
| W1 | **responses seam**: split `prepare_chat`; extract `drive`/`log_stats`; parse + map the request; non-stream `ResponseResource`; `responses_error` helper | `src/api/handlers.{h,cpp}`, new `src/api/responses.{h,cpp}`, new `src/api/generate.{h,cpp}`, `src/http/server.cpp`, `tests/test_responses.cpp`, `CMakeLists.txt` | unit mapping + resource keys green on the stub, red before; roundtrip |
| W2 | **SSE event writer**: `sequence_number`, `event:` line, item/part lifecycle, reasoning/text/function-call deltas, terminal events, `[DONE]`, in-stream error + `failed` | `src/api/responses_stream.{h,cpp}`, `tests/test_responses.cpp`, `tests/roundtrip.sh` | unit event order/sequence/lifecycle green; stub curl SSE green; compliance HTTP+SSE runnable subset 0 failed |
| W3 | **state + continuation**: `ResponseStore`, limits/LRU/TTL, `store` semantics, `previous_response_id`, `prefix_hash`, `SlotPool::try_acquire`, eviction on error | `src/api/responses_state.{h,cpp}`, `src/api/handlers.{h,cpp}` (`SlotPool`), `src/api/responses.cpp`, `src/config.{h,cpp}`, `/props`, `tests/test_responses.cpp` | unit store/continuation/eviction green; stub continuation `cache_hit_tokens >= prompt_tokens - 1`; compliance `previous_response_id` tests |
| W3.2 | **reasoning replay**: `ChatMessage::reasoning`, Responses reasoning-item pairing, chat `reasoning_content`, the stub template's Qwen3 rule, stored-continuation assistant turns per output item, think state keyed to the generation start | `src/core/chat.{h,cpp}`, `src/exec/chat_json.h`, `src/api/responses.cpp`, `tests/test_chat.cpp`, `tests/test_responses.cpp`, `tests/golden/feyntalk-w1/` | unit mapping/rendering/continuation red-first; think-state guard with replayed closed reasoning; 12 golden captures (10 old byte-identical); roundtrip |
| W4 | **cancel by id**: extension route, in-flight registry, `GenerationInput::cancel`, `AbortCause::Cancelled`, `abort_cause_of` + stub `cause`, `arcint:response.cancelled` + incomplete(cancelled) | `src/http/server.cpp`, `src/api/responses_state.{h,cpp}`, `src/exec/backend.h` (~390), `src/exec/backend_llama.cpp:150-152`, `src/exec/backend_stub.cpp:216-220`, `tests/test_responses.cpp`, a cancel measurement script under `tools/` | unit cancel paths red-first; stub cancel at 4 ubatches; card window: cancel-to-abort within one ubatch at 100k, control arm the same |
| W5 | **gate + live leg**: vendor the compliance library at the pinned release into `tests/feyntalk/`, wire a runner, add the pi multi-turn leg | `tests/feyntalk/`, `tests/CMakeLists.txt` or `CMakeLists.txt`, `tools/feyntalk_live.py` | compliance 0 failed, skipped = the named list; live leg passes Gate 4 |
| W6 | **token mode + spans + material**: `arcint.token_mode`/`input_ids`/`renderer_hash`, `arcint:token`, `arcint:span`, sampler event sink, `Backend::renderer_material()`, `GET /v1/responses/renderer`, hash | `src/api/responses*.{h,cpp}`, `src/core/sampler.{h,cpp}` (580/779/788/813), `src/core/sampling.h:166`, `src/exec/backend.h` + `backend_llama.cpp` (326/1032 area), `src/api/handlers.cpp` (`/props` 514), `tests/test_responses.cpp` | unit token/span/shadow equality green; hash stable across two loads; off by default; live token leg (Gate 5) |
| W7 | **WebSocket (1.3.x)**: `response.create`, one in-flight response per connection, sequential turns, connection-local `store=false`, 60-min limit, WS error envelope | `src/http/server.cpp` + a new WS framing unit, `src/api/responses_state.{h,cpp}`, `tests/test_responses.cpp` | WS compliance tests pass; connection-limit and `previous_response_not_found` after eviction |
| W9 | **bundle** (after W6/W7, operator decision 2026-10-31): ship the chain an interested user can install end to end — engine >= 1.3, the gateway with the token→text translator, the agent extension (WebSocket + `previous_response_id`), and a wiring guide | packaging + `contrib/` bundle, the wiring guide, the gateway and agent-extension repos outside this tree | a clean host installs the bundle and runs the live task (Gate 4's class, text and token) |
| W8 | **compact + background (1.3.x)**: `/v1/responses/compact`, `background:true`, background cancel, `allowed_tools`, `text.format` json, reasoning summary/logprobs | new compaction + job modules, `src/api/responses*.{h,cpp}` | each part's compliance test; `background` cancel; no card gate before design |

Order: W1 before W2 (the resource and the event writer share the snapshot
builder); W2 before W3 (a continuation is observable in the stream); W3 before
W4 (cancel must resolve an id a store knows); W4 before W5 (the live leg
includes a cancel); W5 closes 1.3.0; W6 is the option, after the text surface is
green; W7 and W8 are 1.3.x; W9 is the bundle after W6/W7. Each package fits
one coding session; W1's refactor
is the riskiest (it touches the chat path's prepare), and its gate is that the
existing unit suite and roundtrip stay green unchanged.

## Open questions

1. **Compliance test ids.** *Resolved by review R1 (ids enumerated, option (a)).* The specification checkout here has
   `bin/compliance-test.ts` but not `src/lib/compliance-tests`, so the ids and
   the per-test assertions cannot be enumerated in this note (`code`). Options:
   (a) vendor the library at the 2026-04-24 release into `tests/feyntalk/` and
   freeze the id list in this campaign in W5 (recommended: the gate then names
   ids, not capabilities); (b) keep a capability-defined skip list. Until (a)
   happens the gate is capability-defined, which is weaker. Vendoring needs the
   specification repository's licence checked and recorded in `THIRD_PARTY.md`
   first (not read here).
2. **Prior reasoning replay.** *Decided (operator, 2026-10-31: "do as the
   models expect").* `ChatMessage::reasoning` exists; the Responses mapping
   pairs a reasoning item with the FOLLOWING assistant message or
   `function_call` (several reasoning items join with a newline; any other item
   between drops the pending reasoning), and the chat route maps
   `reasoning_content` on assistant messages onto the same field. It is passed
   to the template as `reasoning_content` and the TEMPLATE decides what to
   show: the Qwen3.5-4B family template (read at source in the llama.cpp
   template directory, `code`: assistant branch lines 90-104) renders the block
   only for an assistant turn after `last_query_index` (the last user message)
   and drops it for earlier turns; Qwen3-Coder does not reference the field at
   all. The stub template mirrors the Qwen3 rule so the path is unit-tested.
   The stored continuation rebuilds one assistant `ChatMessage` per output item
   (reasoning paired the same way), so its prompt equals a full-history request's
   on the same items; what can remain is the template's own formatting and the
   client's item shape (a client that sends a message and a function_call as one
   combined chat turn, or sends a different content form, renders differently —
   that is the template's difference, not arcint's policy). No "keep the last N"
   and no truncation on the server side.
3. **`allowed_tools`.** A spec MUST when present (`paper`). Options: (a) 400
   `unsupported_parameter` in 1.3.0 and enforce in 1.3.x (recommended: loud,
   no silent weakening); (b) enforce now by narrowing the tool grammar when it
   is on and returning non-listed calls as raw text when it is off — a
   half-enforcement that looks like compliance.
4. **WebSocket engine.** cpp-httplib 0.18.3 has no WebSocket support (`code`).
   Options: (a) a small in-repo RFC6455 server (handshake, text frames, pong,
   no permessage-deflate) reusing the event writer (recommended: no new
   dependency; ~a few hundred lines and its own unit); (b) vendor a WS library
   (Boost.Beast or similar) — an operator dependency decision under
   `THIRD_PARTY.md`.
5. **Spec tension on errors.** "Any error while streaming is followed by
   `response.failed`" and "an incomplete item MUST be the last item and the
   response MUST be incomplete" conflict when an error truncates an open item
   (`paper`). We choose `failed` and close the item `incomplete`; a proposal to
   the TSC is out of scope. Unchanged if the next release resolves it.

## Review (second reader, 2026-10-31)

Checked at source against the full specification repository (`code`/`paper` as
marked); the note's decisions stand except where a row says otherwise.

| # | finding | disposition |
|---|---|---|
| R1 | Open question 1 is an artifact of an incomplete checkout: the full repository has `src/lib/compliance-tests.ts` (1,784 lines, Apache-2.0, `code`). It defines **17 ids**: `basic-response`, `assistant-phase`, `response-output-phase-schema`, `streaming-response`, `system-prompt`, `tool-calling`, `image-input`, `multi-turn`, seven `websocket-*`, `compact-response`, `compact-missing-model`. | **Resolved, option (a).** 1.3.0 gate = `basic-response`, `streaming-response`, `system-prompt`, `tool-calling`, `multi-turn`, `assistant-phase`, `response-output-phase-schema` (7). `image-input` runs only where the unit has a projector (skipped on the stub, named). `websocket-*` (7) → W7, `compact-*` (2) → W8. Licence row in `THIRD_PARTY.md` in W5. |
| R2 | `assistant-phase` only sends assistant history with `phase:"commentary"` and checks acceptance; `response-output-phase-schema` checks the output schema (`code`). | Move `phase` out of the skip list: accept it on input items (ignored by the template), emit no `phase` on output (optional field). W1. |
| R3 | `incomplete_details.reason` is a free string in the OpenAPI document (no enum, `paper`); the spec text does not mention cancel anywhere (0 occurrences, `paper`). | Confirms `"cancelled"` and `"loop_guard_abort"` are spec-legal values and cancel-by-id must be an extension. No change. |
| R4 | `store` carries no default in the OpenAPI document (`paper`). With default `true`, every prompt of every Responses turn is retained in RAM for up to 24 h. | Keep `true` (client convention), but `--responses-store-entries 0` must disable retention entirely (in-flight registry only), and `/props` reports entries and bytes. W3 unit case for the 0 setting. |
| R5 | W1 splits `prepare_chat`, which is the chat route every production unit serves. A green suite does not prove identical bytes. | W1 gate adds a **golden capture**: the existing chat and completions routes on `--stub`, non-stream and SSE, byte-identical before/after the refactor. W1 merges only in agreement with the engine maintainer's release plan. |
| R6 | 429 on busy lanes (Responses) vs 503 (chat) is right per the spec's type table; clients retry 429 with backoff. | Keep. Record the test client's retry behaviour in the W5 live leg. |
| R7 | `output_tokens_details.reasoning_tokens = 0` although the reasoning split is token-exact where the tag is one special token. | Count it in W2 from the streamer's side per emitted piece; 0 only where the split is textual. |
| R8 | feyntalk-render "lives outside this repository" — its home is not named. | Operator decision before W6 (own repository; public or private). Not a 1.3.0 text-mode blocker. |

## Current state

W1–W4 and W3.2 are built and accepted (2026-10-31); W5's fix round landed: the request mapping and the
non-stream/non-SSE `ResponseResource` (`src/api/responses.*`), the SSE event
writer (`responses_stream.*`) with the item/part lifecycle and the terminal
events, the in-memory state and continuation (`responses_state.*`) with
limits, LRU/TTL, `previous_response_id`, prefix hash and slot affinity, and
cancel by id (`POST /v1/responses/{id}/cancel`, an arcint extension) with
`AbortCause::Cancelled` in the engine seam. W3.2 replays prior reasoning through
the template's own `reasoning_content` rule (open question 2, `code`) and the
stored continuation rebuilds one assistant `ChatMessage` per output item so its
prompt equals a full-history request's with the same items. `include` is accepted and ignored
(the live-client fix); a Responses refusal writes one `req:` warning line with
its code and numbers; the chat route has the same gap for prepare-time
refusals and was left unchanged (its bytes are a gate).

Gate numbers (`measured-here`): `ctest -L unit` 7/7, 432 cases 0 failed 0
skipped on the stub build with `-Werror`; the feyntalk-golden captures of the
chat/completions routes (12: the 10 original byte-identical plus two carrying
`reasoning_content`) stay byte-identical; `tests/roundtrip.sh` green
including the two-turn continuation and cancel-by-id checks; the 7 named
compliance ids (`basic-response`, `streaming-response`, `system-prompt`,
`tool-calling`, `multi-turn`, `assistant-phase`,
`response-output-phase-schema`) pass against the stub (second-reader run,
node+tsx; the vendor + runner stays W5).

Live leg (`measured-here`, 2026-10-31, Arc A770, the coder model, serving build
83d4e26, pi with provider type `openai-responses`): PASS — 4 tool calls
(read, read, write, read), 0 malformed, the expected answer right. The
continuation shared 93–97 % of the prior prompt; every turn was 4 tokens below
`prior prompt_tokens - 1`, the same property as the chat route (the prior
turn's generation prefix is re-rendered differently in history). Cancel at a
~14–15k-token prefill, issued 10 s in: cancel-by-id aborted at 14,336
prefilled after 1.28 s, socket-close control at 14,336 after 1.25 s, and
`response.created` reached the client after 0.01 s — ~1.6 ubatches for both
arms, the existing boundary check's property, not one.

W3.2's replay was measured on the same live leg's history shape: pi sent the
reasoning item of every past assistant turn to a unit that dropped it, and the
prior generation prefix rendered differently (the 4 tokens below
`prior prompt_tokens - 1`, and with thinking on the whole reasoning block would
re-prefill). The template's rule now keeps the reasoning only for assistant
turns after the last user message (the running tool loop), so inside that loop
the replayed block renders as it was generated; earlier turns stay dropped by
the template's own rule. What remains is the template's formatting and a
client's item shape — a clean re-run of Gate 4 is owed with W3.2 merged, and
the number is not claimed here.

W5 still owes the compliance vendor + runner and the live token-mode leg is
W6. `GET /v1/responses/{id}` is not in the 2026-04-24 OpenAPI document and
was not added. No WebSocket, no compact/background, no W6 option work.

## Where it lives

- `src/http/server.cpp` — routes (112-250), `peer_probe` (41-46); the
  `/v1/responses` routes, `/v1/responses/{id}/cancel`,
  `/v1/responses/renderer` join here.
- `src/api/handlers.{h,cpp}` — `SlotPool` (301-360), `prepare_chat` (746),
  `drive` (250), `stream_chat` (1119), `sse_send` (182), `/props` (514); split
  and reused, not duplicated.
- new `src/api/responses.{h,cpp}`, `responses_stream.{h,cpp}`,
  `responses_state.{h,cpp}`, `generate.{h,cpp}`.
- `src/exec/backend.h` (`GenerationInput`, `AbortCause`, `Backend`) and
  `backend_llama.cpp` only at `abort_cause_of` 150-152 and the load-time
  material area 326/1032; `backend_stub.cpp` 216-220. The step loop, and the
  `mtpw` trace writes near :1782 and :1871, are untouched.
- `src/core/{chat,toolcall,reasoning,sampler,sampling}.{h,cpp}` — the mapping,
  the split, and the sampler's event sink (580/779/788/813, params at
  `sampling.h:166`).
- `src/api/error.h` — the spec-typed `responses_error` next to `error_body`.
- `tests/test_responses.cpp`, `tests/roundtrip.sh`, `tests/feyntalk/` (vendored
  compliance library + runner), `tools/feyntalk_live.py`,
  `tools/feyntalk_cancel.py`.
