# lanes-agent-subagent — one process, one set of weights, an agent lane and a smaller subagent lane

**Open** (operator, 2026-10-07; the next release).

## Charter

The libllama engine serves two lanes of different context size from one
process and one copy of the weights. For example, `qwen3.8-agent` at 131,072
tokens and `qwen3.8-subagent` at 32,768. The request's `model` field picks
the lane. An agent and the subagents it spawns then share one card without a
second model load.

Operator decisions (2026-10-07):
- **Routing by name:** the `model` field picks the lane. An empty name goes
  to the first (agent) lane; an unknown name gets a 404. This amends DESIGN
  §4.2 for named lanes, where the model field was not binding.
- **The KV pool:** build the shared pool (the sum of the lanes' caps) and
  measure the subagent lane's decode against the agent lane's fill. If it
  drops below the operator's usability band (20 t/s), the fallback is one
  stream per lane at the largest cap, if that fits.

## Reference to follow

llama.cpp's server, at the pin (`code`):

- **`--parallel N --kv-unified`:** the slots share one KV pool
  (`llama_context_params.kv_unified`). One sequence can use the whole pool
  (`src/llama-context.cpp:293-294`); one stream (`llama-kv-cache.cpp:84`).
- **`--kv-unified-per-slot N`:** one cap for every slot. When `-c` is not
  given, the pool is `n_parallel x N` (`tools/server/server.cpp:170-177`;
  `n_ctx_slot()`, `server-context.cpp:4221-4229`). A prompt over the cap
  gets `ERROR_TYPE_EXCEED_CONTEXT_SIZE` (`:3329-3337`); generation stops at
  the cap (`:1940-1947`).
- **When the pool is exhausted:** the server purges idle slots, halves the
  batch, then fails the processing slots (`:3862-3907`). Not adopted here:
  arcint's caps sum to the pool, so it cannot overcommit (DESIGN §4.3, a lane
  is a memory reservation).
- **A cap per lane** goes beyond the reference, which has one cap for every
  slot. Recorded as such.

What carries over unchanged (`code`):

- **Recurrent state:** one row set per sequence, `n_seq_max x (1 +
  n_rs_seq)` (`llama-memory-recurrent.cpp:101`), whatever `kv_unified` is.
- **Flash-Next's QSA indexer cache** honours `kv_unified`
  (`llama-memory-hybrid-idx.cpp:70-72, 407-470`).
- **Interleaving:** arcint decodes one sequence per `llama_decode`
  (`backend_llama.cpp` `decode_locked`), so lanes interleave and are never
  batched together. The expert cache's decode classification (up to 8 tokens
  a ubatch) holds.

Known costs (`code`, to be measured):

- **Shared pool:** `n_kv` is the stream's padded `used_max_p1`
  (`llama-kv-cache.cpp:1260-1270`). A lane's attention, and on Flash-Next
  the indexer's scoring (`qwen4exp.cpp:828-832`), runs over the other lane's
  cells too, masked. `include/llama.h:425-428` warns of it.
- **The expert cache's usage table and adapt cadence** mix both lanes'
  routing, and its per-request log subtracts one process-wide snapshot.

## Gate

On the record before the work starts.

- **Answers:** right on each lane (capital, 20k needle, long answer), alone
  and while the other lane decodes. The acceptance task 10/10 on the agent
  lane. Mean KL of a lane against the one-lane baseline within CLAUDE.md's
  answer-level bar.
- **Speed** (B60, the served configuration):
  - the agent lane alone within the run-to-run spread of today's one-lane
    unit;
  - the subagent lane's decode at or above 20 t/s with the agent lane holding
    0, 64k and 120k cells (shared pool). Below that, the fallback arm decides.
- **Memory:** no eviction (each process's GTT at its bank throughout), both
  lanes filled to their cap minus one at once without a decode error.
- **Admission:**
  - a request over its lane's cap gets the §3.8 400 for that lane's size;
  - an unknown name gets a 404;
  - with the subagent lane busy, a second subagent request waits or gets
    the 503, but never takes the agent lane.

## Design

- **Flags:** `--served-model-name A,B` with `--lane-ctx CA,CB`, paired by
  index; the lane count is the number of names. Refuse `--parallel` when it
  disagrees, lists of different lengths, and caps that are not multiples of
  256 (`llama-context.cpp:291`). One name keeps today's behaviour.
- **The pool:** named lanes set `kv_unified = true`, `n_ctx = sum of caps`,
  `n_seq_max = lanes`, and assert `llama_n_ctx == sum`. The MTP draft context
  gets the same flag, else its per-sequence stream would be sum/2, under the
  agent lane's cap. Equal lanes (`--parallel` alone) stay on today's
  non-unified path.
- **Admission:**
  - the name resolves to its lane before the request is prepared, so §3.8's
    400 uses that lane's cap;
  - the slot pool becomes one per lane, with free counts per lane in
    `/health` and in the 503;
  - the backend's limits use `cap[seq]` where they use `n_ctx_` today.
- **`/v1/models`:** one entry per name with its own `n_ctx`; `/props` lists
  all names. The proxy reads `n_ctx` per entry, so the subagent appears at
  its size.
- **Checkpoints:** per lane, as now.
- **The expert-cache log:** its lines are labelled process-wide when lanes
  overlap.

## Current state

- Recon done (2026-10-07).
- Implementation: next.

## Where it lives

- `src/config.{h,cpp}`: the flags.
- `src/api/handlers.{h,cpp}` and `src/http/server.cpp`: the slot pool,
  admission, `/v1/models`, `/props`.
- `src/exec/backend_llama.cpp`: the context, the per-lane caps.
- `src/exec/llama_spec.cpp`: the MTP draft context.
- DESIGN §4.2 and §4.3.
