# research-agent-lanes — serving an agent and its subagents from one model, one process, one card

Written for `lanes-agent-subagent.md`. The question: what other engines and
the research do when one copy of the weights serves an agent lane and a
smaller subagent lane. Questions: (1) per-lane context caps of different
sizes; (2) routing a request to a lane; (3) one shared KV pool against
per-lane pools, eviction; (4) batching lanes in one forward with speculative
decoding per sequence; (5) priority and preemption; (6) prefix sharing
between agent and subagent; (7) agent-aware scheduling.

Accessed 2026-10-08. Pins: llama.cpp `bed0a85` (upstream master,
2026-10-02), vLLM `81198e97ba`, SGLang `ac656ee79a`, ExLlamaV3 `151539c77a`,
ollama `e3cddc3e89`, MLC-LLM and mistral.rs at `main`/`master` of that day.

## Summary table

`—` means the system has no mechanism for that question.

| system | (1) caps | (2) routing | (3) pool | (4) batch + spec | (5) priority | (6) prefix sharing | (7) agent-aware | evidence |
|---|---|---|---|---|---|---|---|---|
| llama.cpp server | one cap for all slots (`--kv-unified-per-slot`) | `id_slot`, LCP similarity, LRU | unified or per-slot streams; under pressure purge idle slots, halve the batch | all slots in one decode, drafts batched | — (FIFO) | LCP reuse, host-RAM prompt cache, `seq_cp` | — | code |
| llama.cpp router | per model instance | model name and aliases to child processes | one process per model, LRU unload at `models_max` | — | priority PR #22851 closed unmerged | — | — | code |
| ollama | one `num_ctx` per loaded runner; a different value reloads | model name | `num_ctx x NUM_PARALLEL` | llama.cpp's | — | — | — | code |
| LM Studio | "varying request sizes" under unified KV | model name | Unified KV Cache (default on) | llama.cpp continuous batching | — | — | — | docs |
| KoboldCpp | one context | — | — | serial; `--multiuser N` is a queue depth | — | fast-forward over the common prefix | — | code |
| llama-swap | per model entry | aliases; `setParamsByID` rewrites the request body per alias without a reload | groups (`swap`, `exclusive`) | — | — | — | — | docs |
| vLLM | global `max_model_len`; per request only `max_tokens` | `--served-model-name a,b`: aliases of one model; LoRA adapters as names | one paged pool | continuous batching with spec decode | `--scheduling-policy priority`: waiting heap on (priority, arrival); lowest priority preempted when blocks run out | hash-based block prefix cache; hybrid models via `mamba_cache_mode="align"` | — | code |
| SGLang | global | model name; LoRA | one pool, radix tree, LRU over leaves | continuous batching | priority sort, plus `preempt_to_schedule` when the gap exceeds a threshold | RadixAttention; `MambaCheckpointPool` for GDN/Mamba states | LPM / DFS-weight cache-aware ordering | code + paper |
| ExLlamaV3 / TabbyAPI | per job: prompt + `max_new_tokens` reserved | — | one paged pool, 256-token pages, content-hashed, ref-counted | dynamic batching; the docstring warns spec decode with many jobs may not pay | `max_skips`: a large job can be overtaken a bounded number of times | page dedup by chained hash | — | code |
| MLC-LLM | global | model name | paged | auto spec decode: draft length 4 → 0 as the batch grows | — | radix prefix cache, fork from a running sequence | — | code |
| mistral.rs | global | model name | paged, `max_num_seqs` | continuous batching | — (no priority field in the scheduler config) | prefix cacher | — | code (scheduler config only) |
| TGI | global `max_total_tokens` | model name | paged | continuous batching | — | prefix caching | — | docs (maintenance mode since 2025-12) |
| Parrot | — | — | — | — | per-DAG latency/throughput objectives | shared prompt regions deduplicated | Semantic Variables expose the request DAG | paper |
| Autellix | — | — | — | — | PLAS / ATLAS: priority by service a program has already received; anti-starvation | — | program-level, critical path for multi-threaded programs | paper |
| Andes | — | — | — | — | token-level preemption by QoE gain | — | — | paper |
| InferCept | — | — | preserve / discard / swap during tool pauses | — | — | — | the tool pause as a first-class state | paper |
| Continuum | — | — | KV pinned with a TTL across tool calls | — | keeps a multi-turn job's order | — | multi-turn agent jobs | paper |
| KVFlow | — | — | eviction by steps-to-execution, prefetch from host | — | — | tree-structured shared prefixes | Agent Step Graph | paper |
| Pie | — | — | the application owns its KV pages | — | — | application-level fork | inferlets (Wasm) drive generation | paper |
| Preble | — | — | — | — | priority by prefix-hit ratio, fairness | prefix-aware placement | — | paper |
| CacheWise, CacheScout, TOPAS (2026) | — | — | reuse-aware eviction | — | TOPAS: aging | agent prompt prefixes | agent traces | paper |

## llama.cpp server (the reference we follow)

- **Caps.** `--kv-unified-per-slot N` (upstream #24124, 2026-08-27) is one
  cap for all slots: `n_ctx_slot()` = min(`llama_n_ctx_seq`, N, n_ctx_train)
  (`tools/server/server-context.cpp:4221-4229`); without `-c` the pool is
  `n_parallel x N` (`tools/server/server.cpp:168-178`). Discussion #22658
  asked for a request cap independent of the pool; the maintainer's answer
  was `--no-kv-unified`. A different cap per slot exists nowhere upstream.
- **Slot selection** (`server-context.cpp:1602-1714`): an explicit `id_slot`
  first; else, with `--slot-prompt-similarity`, the idle slot with the best
  longest-common-prefix fraction above the threshold; else LRU by
  `t_last_used`. If the chosen slot would keep less than half its cache, it
  is saved to the host-RAM prompt cache (`--cache-ram`) first, and the new
  prompt is loaded from that cache by LCP (`server-task.cpp:1804ff`).
  `--cache-idle-slots` saves idle slots and, under `kv_unified`, clears them
  (`:2553-2568`).
- **Queue.** No slot: the task is deferred; on release the first deferred
  task that asked for that slot, else the oldest, goes next
  (`server-queue.cpp:90-110`). No priority field.
- **Batching with spec decode** (`server-context.cpp:3088-3240`): every
  generating slot that `can_batch_with` the first (same task type, same
  LoRA, `:452-457`) joins one batch; each slot's draft parameters are set,
  then one `common_speculative_draft()` drafts for all slots, and the target
  verifies every slot's sampled + draft tokens in one decode. The MTP
  drafter (`common/speculative.cpp:1648ff`) adds one token per drafting
  sequence per step and runs one `llama_process` per draft step for all of
  them. Pending prompts fill the rest of `n_batch` after the generating
  tokens (`:3234ff`), so decode goes first and prefill takes what is left.
- **Prefix sharing.** A shared prompt prefix inside one request is evaluated
  once and copied to the child slots with `seq_cp`
  (`server-context.cpp:3588-3603`, grouping in `server-decision.cpp:673-700`).
  In one stream, `seq_cp` only tags the cells with the second sequence
  (`src/llama-kv-cache.cpp:451-476`); for recurrent layers it shares the
  source's tail state, which is valid only at the source's end position
  (`src/llama-memory-recurrent.cpp:261-296`). Per-slot context checkpoints
  (`--ctx-checkpoints`) are what make an earlier position of a hybrid model
  restorable.

## ollama, LM Studio, KoboldCpp, llama-swap

- **ollama** (`server/sched.go`): context = `num_ctx x numParallel`
  (`:812-813`); `needsReload` compares runner options, `NumCtx` included
  (`:1405-1450`), so a different `num_ctx` reloads the model; the
  qwen35/qwen3next families are forced to one sequence (`:517-519`).
- **LM Studio** (blog 0.4.0): Max Concurrent Predictions (default 4) and a
  Unified KV Cache "not hard-partitioned per concurrent request" — llama.cpp's
  `kv_unified`.
- **KoboldCpp:** `--multiuser N` is a queue depth (`koboldcpp.py:13571`).
- **llama-swap** (`docs/config.example.yaml:451-474`): `setParamsByID` maps
  an alias to request-body overrides without a reload; on llama-server it
  can pin an alias to an `id_slot`, not give it a cap.

## vLLM

- `max_model_len` is global (`config/model.py:245`), requests are clamped to
  it (`v1/core/sched/scheduler.py:704-711`). Several `--served-model-name`
  values are aliases of one model (`config/model.py:329-336`).
- **Priority** (`config/scheduler.py:160-166`, `v1/request.py:357-367`):
  lower value first, ties by arrival. When a running request cannot get
  blocks, the worst (priority, arrival) is preempted by recompute
  (`scheduler.py:784-831`, `:1569-1612`). A waiting high-priority request
  that does not fit is skipped, not given room (`:1243-1262`).
- **Hybrid prefix caching:** `mamba_cache_mode="align"` caches GDN/Mamba
  state at block-aligned positions, optionally also where an MTP/EAGLE
  sibling resumes (`config/cache.py:189-202`).

## SGLang

- `srt/managers/schedule_policy.py`: cache-aware LPM and DFS-weight
  (`:162-175`); priority then arrival (`:494-500`); `preempt_to_schedule`
  preempts running requests when the priority gap exceeds a threshold
  (`:1531-1600`).
- `mem_cache/mamba_checkpoint_pool.py`: one int8 GDN/Mamba state per radix
  node, dequantized into a fresh slot on a hit.
- Paper (arXiv 2312.07104): RadixAttention with LRU over leaves; `fork`.

## ExLlamaV3 / TabbyAPI, MLC-LLM

- **ExLlamaV3** (`exllamav3/generator/`): a job reserves prompt +
  `max_new_tokens` + draft (`job.py:1250-1271`). Pages are 256 tokens, keyed
  by a chained blake2b hash, ref-counted; unreferenced pages stay as an LRU
  prefix cache by `access_serial` (`pagetable.py:23-28, 126-132, 415-456`).
  `max_skips` bounds how often a large job is overtaken (`job.py:84-88`,
  `generator.py:1447-1475`). The `draft_model` docstring: spec decode "with
  many parallel jobs is likely not advantageous" (`generator.py:79`).
- **MLC-LLM:** `cpp/serve/engine_actions/auto_spec_decode.cc:52-67` sets
  the draft length from the running batch: 4 below 10 sequences, 3 below
  20, 2 below 30, then 0. `cpp/serve/prefix_cache.h` is a radix prefix
  cache that forks from a running sequence and recycles old ones.

## Research

Sources: Autellix arXiv 2502.13965; Parrot 2405.19888 (OSDI'24);
InferCept 2402.01869 (ICML'24); Continuum 2511.02230; KVFlow 2507.07400
(NeurIPS'25); Andes 2404.16283; Preble 2407.00023 (ICLR'25); Pie 2510.24051
(SOSP'25); CacheWise 2606.16824; CacheScout 2608.14624; TOPAS 2608.25523.
The mechanisms are in the table. None of them gives lanes different context
caps: all assume one pool and a global limit.

## Harnesses (what the server has to expose)

- **Claude Code** (docs, `code.claude.com/docs/en/sub-agents`): a subagent
  starts with a fresh context; a *fork* inherits the whole conversation and
  reuses the parent's prompt cache. The model resolves from the call's
  `model`, the definition's `model`, `CLAUDE_CODE_SUBAGENT_MODEL`, then the
  parent's. Default concurrent subagent limit: 20.
- **Codex** (docs, `learn.chatgpt.com/docs/agent-configuration/subagents`):
  custom agents in TOML with `model`;
  `agents.max_concurrent_threads_per_session`.
- **OpenCode** (docs, `opencode.ai/docs/agents`): `mode: subagent`,
  `model: provider/model`; a child session without one inherits the
  parent's.

All three pick the subagent's model by name.

## What arcint should take (ranked)

1. **Batch the lanes into one decode, with drafts batched across
   sequences.** Reference: llama.cpp `update_slots`
   (`server-context.cpp:3088-3240`) and the MTP drafter
   (`common/speculative.cpp:1648ff`). Today the lanes take turns (4.2 + 4.2
   t/s, `measured-here`). Precondition: patch 0025 (the strided
   `kernel_cpy_f32_f32`). The contiguous view also holds when every
   configured sequence is in the ubatch, but one idle lane breaks it again.
2. **Decode before prefill, and a per-lane draft budget as the priority
   knob.** Reference: llama.cpp fills `n_batch` with generating tokens first
   (`:3234ff`); MLC shrinks the draft as the batch grows
   (`auto_spec_decode.cc:52-67`). For two lanes: the agent keeps its MTP
   depth, the subagent's draft shrinks when the agent is generating. Gate
   it on the agent lane's decode spread.
3. **Priority without memory preemption.** The caps sum to the pool (DESIGN
   §4.3), so vLLM's and SGLang's KV preemption (`scheduler.py:784-831`,
   `schedule_policy.py:1531-1600`) has nothing to free. Take only the
   ordering: when lanes compete for a step or a prefill chunk, the agent
   goes first. Autellix's critical-path point applies: the agent waits on a
   foreground subagent, so do not deprioritise a subagent while the agent
   lane is idle.
4. **Keep subagent prefixes across subagents.** Several subagents share one
   lane and the same system prompt and tool block. Reference: llama.cpp's
   LCP selection plus the host-RAM prompt cache (`--cache-ram`,
   `server-task.cpp:1804ff`) and context checkpoints for the GDN state;
   ExLlamaV3's hashed pages show the page-granular alternative.
5. **Fork from the agent's prefix: ask the operator first.** Reference:
   `seq_cp` within one stream (`llama-kv-cache.cpp:451-476`) and the
   shared-prefix path (`server-context.cpp:3588-3603`); for the recurrent
   half, a state at the fork position (vLLM `align`, SGLang
   `MambaCheckpointPool`, llama.cpp checkpoints). Shared cells would count
   against two lanes at once, which conflicts with a lane as a memory
   reservation (DESIGN §4.3). Record the conflict and its price; do not
   build it before the answer.
6. **Routing stays by model name.** The harnesses select subagents by
   `model` (docs); vLLM's multi-name is alias-only and llama-swap's
   `setParamsByID` shows name-to-parameters mapping. Exposing `n_ctx` per
   entry in `/v1/models` matches what the clients read. No header routing.
7. **Not now:** workflow-graph eviction (KVFlow, TOPAS), TTL pinning
   (Continuum), tool-pause swap (InferCept). With fixed lane reservations
   the agent's state is never evicted; they become relevant only if lanes
   stop being reservations.

Deviations recorded: a different cap per lane goes beyond every reference
(llama.cpp has one cap, vLLM/SGLang a global limit). The no-preemption
choice in (3) rests on DESIGN §4.3.
