<!-- Campaign static-partition-cold-start (docs/campaigns/static-partition-cold-start.md),
design pass 2026-09-05. An experiment design: which of three recomputed
kinds of work owns a cold sequence's first-process cost, measured with
instruments that already exist, before any lever is built. -->

# Design note: separating the cold start's three candidate owners

Spec: `docs/campaigns/static-partition-cold-start.md` — a cold sequence's
first tier-ON processes loaded in 215–585 s against tier OFF's 30–45 s and
served their first request at 0.1–4.2 t/s decode against a warm 16.4
(DESIGN §7.0.2ai), and the three candidate owners were never separated.
The campaign's ladder is the plan; this note fixes the instruments, the
process sequence, and what each delta may be read as.

## 0. What is actually true today

- **The compute runtime's on-disk program cache is on, populated, and
  writing** in the dev container: the default directory under the user's
  cache holds 5,019 files, 517 MB against a 1 GiB default cap, and grew
  today when the plugin unit tests ran. Both production units run with an
  empty unit environment, so they use the same default. The campaign's
  ladder step 1 is answered without a card: the mechanism exists and is
  hit by ordinary use; whether the tier-ON per-expert kernels hit it is
  what step 2 measures.
- **The engine's `warmup()` is one forward of one token on lane 0**
  (`src/exec/backend_ov.cpp`, `warmup()`), run after every compile. It
  proves the compiled graph works; it touches the experts one token
  routes to, a handful per layer, not the resident set and not the host
  set.
- **Pinned slots are reserved at `bind()`, not filled.** Patch 0018's
  `bind()` calls `reserve_pinned()` per resident expert and nothing else;
  the fill happens on the expert's first acquisition (`acquire_one` /
  `try_acquire_simultaneous`: probe reports the slot unfilled, then
  `fill_weights_memory` reads it from the weight file). So the first
  request that routes to a resident expert pays its upload, and every
  host-tier expert is read through the plugin's mmap of the same file on
  every token that routes to it.
- **The 35B artifact's language-model weights are 18.6 GB in one file on
  a ZFS dataset**, read through the plugin's positional reads and mmap.
  The host's ARC is at its cap (40 GiB, size ≈ c), and the two production
  models it serves beside any test process are 12 and 14 GB on disk. A
  test process's first touch of an expert that is not in the ARC is a
  disk read; `fincore` cannot see the ARC, but `/proc/spl/kstat/zfs/
  arcstats` (`misses`, `demand_data_misses`, `size`) counts exactly those
  reads, and a timed sequential read of the file measures its state.
- **The tier cell's runner emits only second-request metrics**; the
  first request's rates are printed but not measured. Step 5 of the
  ladder (outputs compared byte for byte per process) is already what
  the runner's identity groups do for two processes; the experiment
  extends it to every process it starts.

## 1. Instruments (all existing; nothing new is built for the window)

| term | instrument | read from |
|---|---|---|
| load time | `language model ready in N s` | the server log |
| first / second request rates | `slot N: prefill … t/s`, `slot N: decode … t/s` lines 1 and 2 | the server log |
| kernel JIT | `created_onednn_kernels` in the last `[OTD_PERF]` line; the file count of a fresh runtime cache directory after the process | the server log; the directory |
| disk reads | `arcstats` `misses` / `demand_data_misses` / `size` before and after each process | the host's kstat, readable in the container |
| file cache state | a timed sequential read of the weight file (seconds, GB/s) | `dd` to `/dev/null` |
| first-use fills | `gpu_misses` / `gpu_hits` in `[OTD_PERF]`; first against second request in one process | the server log |
| outputs | sha256 of every request's greedy text | the runner's files |

## 2. The process sequence (one card window, the 16 GiB card, `+p4`)

Fixed for every process: the tier reference cell's own configuration
(35B int4, ratio 50, 8 GiB device pool, u8 KV, one lane, n_ctx 65,536,
prefix cache off, the prompt sized once by a sizing server to a
1,198-token target (1,167 measured),
64 greedy tokens, `MOE_OTD_PERF_LOG=1`), two requests per process, ARC
counters read before and after each process, every output hashed.

| step | process | what differs | the delta reads as |
|---|---|---|---|
| P0 | sizing server, tier OFF | — | not a sample; it sizes the prompt and is reported |
| P1 | tier ON | the host as found (the "cold sequence" first process) | the whole cold cost, with its ARC miss count |
| R | timed read of the 18.6 GB weight file | — | warms the file; its own seconds say what state P1 left it in |
| P2 | tier ON | file warm, runtime cache intact | P1 − P2 = the disk term (with the ARC counters saying how many bytes) |
| P3 | tier ON | file warm, runtime cache pointed at an empty directory | P3 − P2 = the kernel JIT term; the directory's file count afterwards = kernels compiled |
| P4 | tier ON | file warm, cache intact, one pre-warm request (a filler prefill sized to a 256-token target, 222 measured, plus four tokens) before the timed first request | P2's first request − P4's first request = what a pre-warm buys; P4's second request against its first = what remains |

Within every process, request 2 against request 1 is the first-use-fill
term at that process's cache state; `gpu_misses` says how many slots
were filled during request 1.

## 3. Reading the result

- If P1 − P2 carries most of the 215→45 s and the first-request cliff,
  and P1's ARC misses are of the order of the resident half of the file:
  the owner is the file cache. No file can hold it on this host (the ARC
  is already the cache, and it is full); the lever is a pre-warm at
  service start that touches every expert once, kept out of the gate's
  timing, gated on the tier being on.
- If P3 − P2 carries it: the owner is kernel JIT, the runtime cache is
  missing for these kernels, and the lever is making it hit (a stable
  directory, or a process-variant compile input to find and remove).
- If neither and P2's request 1 against request 2 still shows the cliff:
  first-use fills own it, and the same pre-warm lever applies.
- If P1's text differs from P2's or P3's for the same request: a §3.4
  finding of a new kind (history on the host's state), before any lever.

Every number in the record names the card, the artifact, the flags, the
pool, the runtime's patch level, and which of the caches was warm.

## 4. Out of scope here

The prefill fallback's own cost (`static-partition-prefill`), seeding the
partition (`partition-seeding`), any change to the ranking rule; and the
model blob cache (`cache_dir`), off by design on the paged path and a
different cache from every one above.
