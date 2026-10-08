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
- **Fixed** (llama.cpp patch 0025, `measured-here`, B60, the dense agent,
  MTP 5, the same 300-token answer):

  | arm | decode |
  |---|---|
  | one lane | 41.0 t/s |
  | two lanes, one busy, before | 10.9 t/s |
  | two lanes, one busy, after | 40.9 t/s |
  | both lanes at once, after | 18.3 + 20.1 t/s |

  - **Text:** byte-identical one lane / two lanes before / after, on two
    questions. With both lanes at once the subagent lane is identical to
    its run alone; the agent lane diverges at char 372 of 1,613 into an
    equivalent sentence (a near-tie). Likely the shared pool's
    interleaved cells changing the attention's reduction order; not
    measured. Answer-level: right; the KL with both lanes busy is owed.
  - **Next** (prior art: `research-agent-lanes.md`): batch both lanes
    into one decode with drafts across the sequences (llama.cpp server
    `update_slots`); decode before prefill; a per-lane draft budget as
    the priority knob. Then Flash-Next's lane memory.
- **The gate after 0025** (`measured-here`, 2026-10-08, B60, the
  test-drive unit: lanes 131,072 + 32,768, shared pool, MTP 5; the
  repository's source as context, cold prefixes, 400 tokens greedy; one
  lane decoding at a time, in this order):

  | cell | prefill | decode |
  |---|---|---|
  | agent 4k, pool empty | 824 t/s | 36.8 t/s |
  | subagent 4k, agent holds 4k | 707 t/s | 31.8 t/s |
  | subagent 29k | 527 t/s | 25.6 t/s |
  | agent 4k, subagent holds 29k | 606 t/s | 28.3 t/s |
  | agent 130k, subagent holds 29k | 191 t/s | 14.3 t/s |
  | subagent 4k, agent holds 130k | 339 t/s | 14.2 t/s |
  | agent 4k, subagent holds 4k | 782 t/s | 36.7 t/s |

  The single-lane window, the same build and prompts:
  - 4k: 36.6 t/s;
  - 29k: 32.3 t/s;
  - 130k: 19.3 t/s, prefill 252 t/s.

  Acceptance is 32-38 % in every cell, so the drafts do not explain the
  gap.
  - **The gate fails:** with the agent lane 130k deep, the subagent decodes
    14.2 t/s, under 20. A lane's speed follows the pool's total fill, not
    its own.
  - **The record agrees:** on the served unit before this window, with
    80k of agent context left in the pool, the agent lane decoded a 4k
    cell at 24.5 t/s and a 29k cell at 25.2 t/s.
  - **Hypothesis, not measured:** the `n_kv = used_max_p1` cost listed
    under Known costs. The subagent's cells sit above the agent's in the
    one stream, so its attention spans both (`code`). The KV debug that
    withdrew this earlier ran with an empty pool. The next check is n_kv
    in the "agent holds 130k" cell.
  - **The fallback is out:** one stream per lane at the largest cap needs
    2 x 131,072 cells of q8_0 KV. That is about 3.3 GiB more than the shared
    pool's 163,840 cells (5.44 GiB, scaled). The card has 22.7 GiB usable,
    and the single-lane unit already peaks at 21.4.
  - **For the operator:** pin the subagent lane to the pool's low cells, so
    its n_kv stays under its cap whatever the agent holds. Or build
    per-sequence attention ranges. Either goes beyond the reference, whose
    unified pool has the same cost.
  - **Operator, 2026-10-08:** per-lane ranges (a window per lane), then
    the release.
- **The cause, measured** (`measured-here`, a lanes window with llama.cpp's
  KV debug, `-vv`): the subagent lane holds 4,508 cells.
  - With the pool empty, its decode reads `n = 4,508` and runs 38.4 t/s
    (prefill 772 t/s).
  - With the agent lane holding 29k, the same request reads `n = 33,741` and
    runs 30.3 t/s (prefill 606 t/s).
- **The fix: llama.cpp patch 0026, cell windows** (`code`;
  `contrib/llama.cpp/README.md`). Lane i finds its cells in its own range
  of the pool and attends over that range only. arcint sets the windows
  (`llama_memory_seq_windows`) on the target and the MTP draft context when
  lanes are named; Flash-Next's indexed memory refuses them and keeps the
  whole-pool views.
- **After 0026** (`measured-here`, B60, the first build):

  | cell | before | after |
  |---|---|---|
  | subagent 4k, pool empty | 38.4 t/s | 38.0 t/s |
  | agent 29k (one lane: 32.3) | 30.6 t/s | 32.2 t/s |
  | subagent 4k, agent holding 29k | 30.3 t/s | 37.5 t/s |

  The lanes probe (`lanes_probe.py`, a 300-token answer on the subagent):
  - with the agent lane at about 20k / 64k / 118k: 28.7 / 28.5 / 28.3 t/s,
    the same text each time. The first window, before 0025: 7.7 / 7.3 / 6.5;
  - 404 and 400 as before; the capital and the 20k needle right on both
    lanes;
  - both lanes at once: 14.7 + 14.5 t/s. The lanes take turns, and the sum
    equals one lane alone (28.3-31.6 on this prompt);
  - the acceptance task on the agent lane: 8/10 greedy, as the one-lane unit
    scores; 10, 10, 10 sampled.
- **The gate on the final build** (`measured-here`, after the review
  fixes, `6cb712afef80c7a0`; the same probe):
  - the subagent's answer with the agent lane at about 20k / 64k / 118k:
    32.1 / 32.0 / 32.0 t/s, the same text each time and the same as the
    agent lane's alone (32.1 t/s). The first build's 28.x carried the
    search-head defect: the subagent's request landed above its earlier
    20k needle;
  - both lanes at once: 15.9 + 15.8 t/s;
  - the capital and the needle right, 404 and 400 as before;
  - the acceptance task on the agent lane: 8/10 greedy; 10, 8, 10 sampled.
  The speed gate holds: the subagent stays at 32 t/s whatever the agent
  lane holds, against a bar of 20. The answers gate's "acceptance task
  10/10 on the agent lane" is **not met** as written: 8/10 greedy, the same
  score and lost cases' count as the one-lane unit (four runs of four,
  `docs/llama-engine.md`). Parity with the one-lane unit as the bar is the
  operator's decision, asked at the release.
  - **Operator, 2026-10-08:** parity accepted for the time being.
  - **Evidence** (`measured-here`): the greedy answer is byte-identical
    across the agent lane on the first 0026 build, on the final one, and
    on the deployed one-lane 0.7.0 unit. It loses the same two cases (CRLF
    and LF-only input: the last field and row repeated at the end of
    input), as at 0.5.6. A greedy score is one trajectory and flips with
    the configuration at equal KL; the sampled means are the measure.
- **KL** (`measured-here`, `llama-perplexity` with 0026's test switches,
  16 chunks of 512 against the Q8 reference, q8_0 KV): one sequence 0.004099
  (top-1 97.745 %). Four sequences decoded one at a time in a shared pool,
  arcint's path, give the same values with windows and without, on the
  first build and after the review fixes. The answer-level bar holds.
- **Review** (an outside model, before the push):
  - the window's search head was never reset, so a lane's fresh request
    after a long one landed above the old high-water mark. The red case, an
    agent request of 29k and then a fresh 4k one: 34.1 t/s and 22.7 s before
    the fix, 37.8 t/s and 15.6 s after;
  - Flash-Next's QSA k-pool selects cells by absolute index, so windows are
    refused for that memory;
  - `seq_cp` across windows and a whole-cache restore are refused; the
    relative-position buckets read from the offset;
  - accepted: a lane switch rebuilds the graph, since its K/V views carry
    the window's offset. With both lanes busy the sum is one lane's rate, as
    measured above.
- **Found on the way, not the lanes' path:** a batch whose ubatches mix
  sequences (llama.cpp's own batching: `llama-perplexity` with several
  sequences, `llama-server -np N`) scores KL 0.2438, top-1 81.96 %. This
  happens with windows or without, unified or in separate streams.
  - The bare pin and pin + 0001-0020 score the same as one sequence.
    Pin + 0001-0022 (0.6.0's series) does not.
  - Turning off 0022's default-on paths (`F16_XMX`, `F32_SKINNY`,
    `FUSE_SCALE_ACT`) does not fix it.
  - arcint decodes one sequence a `llama_decode` and is not affected.
  - **Bisection** (`measured-here`, the same KL setup, four sequences in
    mixed ubatches):
    - pin + 0001-0021 is right (0.004183, as its one-sequence arm), so 0022
      brings it;
    - none of 0022's switches fixes it: `F16_XMX`, `F32_SKINNY`,
      `FUSE_SCALE_ACT`, `CPY_FLAT`, `DISABLE_FUSION`, `GDN_INTEL`
      (0.2428-0.2444);
    - moving ops to the CPU (`GGML_OPENCL_OPFILTER`): `CONCAT` alone
      (0.004099), `GATED_DELTA_NET` alone (0.004093) or `SSM_CONV` alone
      (0.004185) fixes it; `REPEAT`, `FILL` or `FLASH_ATTN_EXT` alone do
      not;
    - `test-backend-ops` passes `CONCAT`, `SSM_CONV`, `GATED_DELTA_NET`,
      `REPEAT` and `FILL` on the B60.
    0022 does not touch those three ops (`code`), and each of them alone
    fixes it. So the fault is likely elsewhere in 0022: an out-of-bounds
    write whose victim moves when the graph splits, or a missing
    synchronisation that a split hides. Not localised; open. Next: a
    per-node comparison of the mixed graph against the CPU.
  - **Found and fixed (2026-10-08, llama.cpp patch 0027):** not a kernel.
    0022's flat-columns path multiplies a 3D `src1` through a stack copy whose
    address repeats, and the Intel activation cache keys on that address. A
    mixed ubatch's `final_output` (3D) took that path in every GDN layer, and
    layer N+1 reused layer N's converted activation. Evidence
    (`measured-here`):
    - `GGML_OPENCL_KQ_DEDUP=0` alone fixes it;
    - per-node dumps hide it, and the KL fell with the share of graphs
      dumped (0.2438 -> 0.2401 -> 0.1789). This retracts the first dump's
      "not a race" reading: it had covered only the warm-up graph;
    - a slot log shows `linear_attn_out-9` hitting the slot
      `linear_attn_out-8` had filled.
    With 0027, every mixed arm scores 0.004099 / 97.745 %, and Flash-Next's
    answers are byte-identical before and after.
    - **The shape:** in a mixed ubatch of four sequences of 128 tokens,
      `final_output` is [6144, 128, 4]; one sequence gives [6144, 512, 1] and
      never takes the flat path.
    - **Flash-Next cells:** B60, the served flags at 32,768 tokens; the
      capital, the 20k needle and a 300-token answer, byte-identical with and
      without 0027; decode 23.4 against 23.3 t/s.
    - **Review:** the Adreno broadcast loop builds the same kind of stack
      temporary (not compiled into arcint). The clearing is a helper
      (`ggml_cl_kq_forget`) called after both, and the Adreno build compiles.
    - **Dense agent cells** (the review asked: the MTP draft context could
      take the flat path): one lane, q8_0 KV, MTP 5, 32k. 0.7.0 and the
      0027 build give the same text on two prompts, the same draft
      acceptance (172/634; 1159/2005) and the same acceptance-task code,
      10/10. The mixed-batch KL on the revised patch: 0.004099.
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
