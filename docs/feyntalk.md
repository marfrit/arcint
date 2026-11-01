# feyntalk: token mode, end to end

feyntalk is the way arcint, a gateway and a client agree on **what the model actually produced**: token ids
and their spans (reasoning, answer, tool call), instead of text that has to be parsed back. Three pieces make
the chain; this page is the one place that ties them together. The design record is
[`docs/campaigns/feyntalk.md`](campaigns/feyntalk.md).

| Piece | Version | Role |
|---|---|---|
| **arcint** (this repository) | 1.4.0 | the engine: generates, and emits `arcint:token` events with span marks |
| **feyntalk-render** | 1.0.0 | Python library: renders the chat template exactly like the engine's vendored minja, hashes the prompt, detokenises and parses spans |
| **llm-proxy** | 1.3.0 | the gateway: renders the prompt with feyntalk-render, sends it in token mode, translates tokens back into plain OpenAI text for the client |

The versions above are the ones released together and tested as a chain. Each piece's CHANGELOG names the other
two. <!-- links filled in at release: arcint tag, feyntalk-render tag, llm-proxy tag -->

## How a turn runs

1. The gateway fetches the engine's **renderer material** once (`GET /v1/responses/renderer`: vocabulary, chat
   template, special tokens, and its `renderer_hash`; behind the same `--api-key` as every route) and caches it by
   hash.
   `/props` reports `token_mode: "ready"` when the unit can serve it.
2. For each request it renders the prompt locally with feyntalk-render and sends an Open Responses request with
   `arcint: {v: 1, token_mode: true, renderer_hash, render: {prompt_sha256}}` and `stream: true`.
3. The engine renders the same request with its own template engine and compares the hash. **Mismatch = 409**
   (`arcint.render.prompt_sha256`); the gateway retries that turn once in text mode and logs it. A match means
   both sides hold the identical prompt, byte for byte.
4. The engine streams `arcint:token` events. The gateway detokenises them, splits reasoning, answer and tool
   calls by the engine's spans (no text parsing of think or tool markers), and streams ordinary Chat Completions
   or Responses events to the client.

Token mode is stream-only and text-only: the engine refuses an image part with token mode, and the gateway sends
such a request in text mode (log: `request stays text (images are not served in token mode)`).

## Thinking replay (`preserve_thinking`)

The Qwen3.6 coder template replays an assistant turn's reasoning only after the last user message. An agent's
**steering message** (a user message in the middle of a tool loop) therefore drops the whole loop's reasoning
from the prompt and with it the engine's prefix cache (measured-here, the 1.4.0 UAT on the coder: one steering
turn hit the cache for 10.6 % of the prompt and re-prefilled for 53 s; the next turn was back at 99.4 %). The
template's own switch `preserve_thinking` keeps it:

- engine: `chat_template_kwargs.preserve_thinking` (Chat) / `arcint.preserve_thinking` (Responses), strictly
  boolean; `/props` → `chat_template_caps.supports_preserve_thinking` says whether the template reads it.
  Other `chat_template_kwargs` keys pass through to the template unchanged (llama-server's dialect), except the
  names the template context owns (`messages`, `tools`, `add_generation_prompt`, `bos_token`, `eos_token`,
  `strftime_now`) and the template engine's builtins (`namespace`, `range`, `tojson`, …): those are a 400. A
  template that fails on what the request sent is a 400 too; an engine fault stays a 500. The gateway must
  render with exactly the same keys, or the token-mode hash is a 409.
- gateway policy: keep thinking by default; drop it only as the first compaction stage once the prompt passes
  `FEYNTALK_THINK_DROP_FRACTION` (0.7) of the context window. The request log shows `think=kept|dropped|off`.
- templates differ in their default, which is why the gateway always sends the flag explicitly: the Qwen3.6
  coder template keeps reasoning only when `preserve_thinking` is true; the Qwen3.8 templates keep it unless it
  is false. `true` and `false` mean the same in both.

## Setting it up

**Engine.** Nothing to switch on: every llama unit that is not `--tts` serves token mode (a GGUF without a chat
template gets the ChatML fallback and is ready too; the load log names the template's source). Check:
`GET /v1/responses/renderer` answers 200 with a hash (the first call builds the material; before it, `/props`'
`token_mode: "ready"` only says the load pre-flight passed), then `/props` shows `token_mode: "ready"` and the
load log has a `renderer material:` line.

**Renderer.** `pip install "feyntalk-render @ git+https://github.com/marfrit/feyntalk-render@v1.0.0"` (Python ≥ 3.11,
pins `jinja2==3.1.6`; a distribution package must provide exactly that version).


**Gateway.** Two ways to expose a model in token mode:

- the whole backend: `"feyntalk": "tokens"` in its `local-backends.json` entry;
- **beside the classic name** (recommended): an alias in `model-aliases.json`, e.g.
  `"qwen3.6-coder-ft": {"model": "qwen3.6-coder", "feyntalk": "tokens"}`. The alias shares the backend and its
  slots, inherits context, reasoning and output limits, and appears in `/v1/models` with `"feyntalk": "tokens"`
  and `"alias_of"`. Lightweight clients keep the plain name; agents use `-ft`.

The gateway does not use token mode (the request stays in text mode, one log line per reason) when the renderer
is older than 1.0.0, the engine does not report `token_mode: "ready"`, the material hash is unknown, or the
request carries something token mode does not serve (an image part, `n > 1`, a stop sequence); a prompt-hash
mismatch (409) is retried once in text mode. So a `-ft` request falls back to the same text path its classic
name takes, by mechanism.

**Telling it runs.** Gateway log: `feyntalk: arcint token mode, material sha256:…` once, then per request
`… -> arcint feyntalk chat …` or `… feyntalk responses …` with `think=…`. A turn that fell back says
`request stays text (…)` or `409 renderer mismatch; retrying this turn once in text mode`.

## Compatibility

Renderer and engine must agree byte for byte. The renderer's conformance suite compares its output against the
engine's template engine on real chat templates (Qwen 3/3.5/3.6/3.8 including the coder and Flash-Next
templates, and others); a template whose constructs the renderer cannot reproduce exactly is refused at load,
never rendered approximately.
