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

- **First window** (`measured-here`, 2026-10-08, B60; the dense Qwen3.8-27B
  Q4_K_M, q8_0 KV, MTP 5; lanes 131,072 + 32,768, shared pool of 163,840
  cells):
  - **Mechanics:**
    - `/v1/models` lists both names with their `n_ctx`;
    - an unknown name gets 404 `model_not_found`;
    - a 44,432-token prompt on the subagent gets the 400 at 32,768;
    - capital and the 20k needle are right on both lanes;
    - no eviction (KV 5.44 GiB; at a 68k agent fill, 379 MiB of VRAM free).
  - **Speed:** the same 300-token answer.

    | arm | decode | a verify |
    |---|---|---|
    | one lane, this build or 0.6.0 | 32.1 / 32.2 t/s | ~66 ms |
    | subagent, agent lane at ~20k / ~64k / ~122k | 7.7 / 7.3 / 6.5 t/s | ~300 ms |
    | agent lane, the other lane near empty | 8.4 t/s | ~280 ms |
    | both lanes at once | 4.2 + 4.2 t/s | each waits half the time |

  - **Not the shared pool's attention:** llama.cpp's KV debug shows n_kv at
    87-131 cells (padded to 256) while a verify still takes ~280 ms. The
    hypothesis that the pool's `used_max_p1` makes the subagent attend the
    agent's cells is withdrawn. What costs is lane mode itself; the split
    test (two sequences without `kv_unified`, lanes without MTP) locates it.
  - **Both lanes at once:** arcint decodes one sequence per `llama_decode`
    (`decode_locked`), so the lanes take turns. llama.cpp's server batches
    its slots into one decode: the reference for doubling the throughput.
- **Located** (`measured-here`, the split test and llama.cpp's decode
  timers; the dense agent, the same 300-token answer):

  | arm | decode |
  |---|---|
  | one lane, MTP 5 | 31.0-32.2 t/s |
  | `--parallel 2` (two streams, not unified), MTP 5 | 8.5 t/s |
  | named lanes, MTP 5 | 8.3-8.5 t/s |
  | one lane, no MTP | 18.6 t/s |
  | named lanes, no MTP | 18.6 t/s |

  - **Lanes without MTP cost nothing.** The cost is MTP with two
    sequences, whether the pool is unified or not. It predates this
    campaign: `--parallel 2` has it.
  - **The target context's verify decode** (LLAMA_DECODE_TIMING):
    - enqueue ("compute") is 12.4 against 12.3 ms;
    - the wait for its results ("post") is 55.6 against 262.7 ms;
    - the graphs are reused alike (97 of 100).
    So the card does ~5x the work per verify with two sequences.
  - **Next:** a per-kernel profile of one verify, one lane against two, to
    name the kernel that scales with the sequences.
- **The kernel** (`measured-here`, unitrace `-d --opencl`, the same
  300-token answer, one sequence against two configured with one busy):
  every kernel's device time is equal within 1 % except ggml-opencl's
  generic `kernel_cpy_f32_f32`:

  | | calls | time |
  |---|---|---|
  | one sequence | 2,734 | 8.6 ms |
  | two sequences | 8,926 | 27,366 ms |

  The 6,192 extra calls equal the `kernel_gated_delta_net_pp` calls.
- **The source** (`code`, with the arithmetic matching the trace):
  - **Retracted:** the `build_rs` extra-state copy. With one sequence in a
    ubatch `n_rs` is 1 and no extra rows are copied (`find_slot` takes
    min/max over the ubatch's own sequences).
  - **The cause:** the MTP snapshot write (`src/models/delta-net-base.cpp:617-627`,
    and `mamba-base.cpp:275-279` alike) copies K = 1 + n_rs_seq planes into a
    view `[D, n_seqs, K]` whose planes are `mem_size` rows apart. That view
    is contiguous only when `n_seqs == mem_size`: one sequence configured, or
    all of them busy. Two configured with one busy miss ggml-opencl's flat
    copy and run its generic `kernel_cpy_f32_f32`, one 64-item work-group a
    786,432-float row: ~4.4 ms x 48 GDN layers = ~211 ms a verify, against
    the measured 207 ms.
  - **Upstream:** at master the code is unchanged; no fix to follow.
  - **The fix** (llama.cpp patch 0025): a row-parallel f32 copy kernel in
    ggml-opencl for strided rows. Same rows, same addresses, so the
    semantics are unchanged.
- **Flash-Next with lanes** (5,500 and 7,500 MiB of slots): eviction both
  times; aborted, numbers void. llama.cpp reserves 6,679 MiB of OpenCL
  compute buffer: the 2,048-token prefill ubatch against the 163,840-cell
  pool, two sequences. On top come the doubled recurrent state (675 MiB)
  and the MTP context in the same mode.

## Where it lives

- `src/config.{h,cpp}`: the flags.
- `src/api/handlers.{h,cpp}` and `src/http/server.cpp`: the slot pool,
  admission, `/v1/models`, `/props`.
- `src/exec/backend_llama.cpp`: the context, the per-lane caps.
- `src/exec/llama_spec.cpp`: the MTP draft context.
- DESIGN §4.2 and §4.3.
