# prefill-slot-loan — lend the expert cache's slots to a prompt, give them back for decode

**Status: open (2026-10-29).** Flash-Next on the B60. Lever, not defect.

## The lever

A prompt and decode want the same VRAM for different things. Decode wants expert slots (the share of routed
experts served from the card); a prompt wants a large ubatch, whose compute buffer arcint reserves for the
whole session at load (`src/exec/backend_llama.cpp:577-588`: with the expert cache the ubatch is the prefill
chunk). Today the split is static, so a larger chunk costs decode its slots for good.

Measured (`measured-here`, 2026-10-29, `research-concept-mining.md`, Wave 1 window 2: the B60, Flash-Next
IQ3_S with the unit's flags, a 19,196-token prompt, one fresh process an arm):

- chunk 2,048 at the unit's 11,600 MiB of slots: prefill 442.1 t/s, peak 23,519 MiB;
- chunk 8,192 at 8,000 MiB: **569.9 / 572.1 t/s (+29 %)**, peak 23,632 MiB; chunk 4,096 at 9,600 MiB: 532.9
  (+20.5 %); the same chunks with more slots sit at the card's edge (24,400-24,437 MiB of 24,480) and scatter
  or lose (393-499 t/s, 364.1 t/s);
- the static price: decode on the six answers at 8,000 MiB of slots 31.08 t/s against 34.30 at 11,600
  (-9.4 %; routed experts on the card 71.5 against 81.2 %).
- why the larger chunk pays: the grouped GEMM's 32-token tiles hold +396.5 % (gate/up, bank experts) and
  +192.5 % (slot experts) more rows than were routed at 2,048, +115.8 % and +48.6 % at 8,192.

The prize is the larger chunk's prefill with the unit's decode: borrow slots for the prompt, refill them
before decode.

## The reference

Strata's prompt path borrows the tail slots of the GPU expert cache, a "lend region", for its buffers, and
refills them after the prompt (`code`, `~/src/Strata-ref/src/core/expert_source.cpp:247` (the lend region's
start), `:3571` (experts taken from the last slot down, the order the borrowing uses), `:3649-3655` (every
part's cache lends its tail; the refill copies them back)). History (`paper`): 68072d30 "refill the prompt
path's lent cache slots with queued copies and one wait"; d5733321 the refill from the GGUF mapping's pieces;
2838cbae "never write into the borrowable cache slots while laying out (a resident expert was corrupted)";
219ae436 Battlemage's borrowing configuration with a 8,192 prefill (B70 16K 1,222 -> 1,347 t/s, 32K 1,187 ->
1,361). Strata picks the largest chunk the lent bytes allow (`--prefill auto`).

## arcint today (`code`)

- The slots: one buffer a cache, `bslots` (`llama-src/src/llama-expert-cache.cpp:471`), three tensors a layer;
  the bank (host) beside it (`:473`). The cache's `verify` compares slot bytes with the file
  (`:132-140`).
- The compute buffer: reserved once for the prefill chunk and the reserve cells (patch 0052,
  `--llama-reserve-cells`); decode needs ~93-770 MiB of it (`paper`, the Flash-Next record, stage 3c).
- MTP's draft context reserves its own buffers beside it.

## Design constraints

- **Exact:** a lent slot is never read as an expert while lent (the cache treats it as empty: misses go to
  the CPU rows or the bank stream), and after the refill every lent slot's bytes equal the file's. The
  hazard Strata recorded (2838cbae) is the red case.
- **Whole graphs:** the loan is taken before a prompt's first ubatch and returned after its last; the cache's
  adapt is paused while the loan is out, and decode starts only after the refill's wait.
- **No eviction thrash:** the prefill graph's peak plus the remaining slots stay under the unit's measured
  peak (23,519 MiB); the record shows what the edge costs.
- **One lane** (Flash-Next serves one), so no second request can see a lent slot.

## Operator decisions

- 2026-10-29: a poisoned or broken loan state (a failed resident re-reserve, a refill that does not verify, an undo
  whose tables still differ) ends the process with an ERROR line naming the problem and a non-zero exit, so the unit
  manager restarts it clean; no serving in that state.

## Gate (on the record before the work)

Against the unit's configuration (chunk 2,048, 11,600 MiB), same binary family, same window, alternated fresh
processes, `tools/fn_sweep.py`:

1. prefill at the 19,196-token prompt **>= +20 %**;
2. decode on the six answers within the unit arm's run-to-run spread (window 1: 0.0-1.7 %);
3. the answers: the capital and the 20k needle right; the answer-level bar (mean KL at most 0.03 nats worse
   than the baseline arm against the reference, argmax agreement at most 1 point lower); the chunk change is
   rounding-only;
4. the slots' bytes equal the file's after every refill (the cache's verify, exact);
5. peak VRAM at or under the unit arm's; the refill's wall time reported (estimate ~0.25 s for ~3.6 GiB at
   the B60's 14.3 GB/s, inferred).

## Stages

0. Design note: where the prefill graph's compute buffer lives during a loan (a buffer over the lent slot
   range, or a second reservation the scheduler switches to), and how the scheduler's reserve stays valid for
   decode. Size the loan from the measured peaks.
1. The loan in the cache: lend a contiguous tail range, mark it empty, pause adapt, refill with queued copies
   and one wait, verify. Red cases: a graph that writes into a lent range is refused (mutation: the bound
   check removed); a lent slot not refilled fails the verify (mutation: one slot skipped).
2. The prefill chunk chosen from the lent bytes (Strata's `--prefill auto`), with the unit's chunk as the
   floor.
3. The served A/B against the gate.

## Implementation

DeepSeek implements (patch 0068+ and `src/exec/backend_llama.cpp`), Claude runs the card windows and the A/B.

## First served run (2026-10-29, `measured-here`)

The B60, Flash-Next IQ3_S with the unit's flags, the round-6 build (`088df89`, recipe), `tools/fn_sweep.py`, an
18,802-token prompt, one fresh process an arm, two rounds:

| arm | prefill | decode, six answers | answers (md5) |
|---|---|---|---|
| the unit's binary | 436.7 / 436.7 t/s | 34.65 / 34.67 | identical |
| new, loan off | 436.9 / 436.9 | 34.67 / 34.65 | identical |
| new, loan on (`--llama-slot-loan 6400 --llama-loan-chunk 8192 --prefill-chunk 8192`) | 485.2 / 485.0 (+11.1 %) | 34.62 / 34.58 | identical |

- The loan lends 26 layers (6.20 GiB), its table check and the refill's verify are exact (0 of 192, 0 of 9,360),
  also after decode's adapt swaps (a second prompt 501.0 t/s); 4,608 MiB is refused (the buffers need 4.55 GiB).
- Red switches: a corrupt refill and a failed return re-reserve exit with the cause (exit 70); a corrupt lend table
  undoes the lend and the prompt runs resident; the reserve-failure switch (3) did not trigger (open).
- The refill takes 2.66-2.86 s for 6.2 GiB (~2.3 GB/s against the link's 14.3 GB/s): ~7 % of the prompt's time.
- Peak VRAM 24,242 MiB (the unit 23,519 MiB of 24,480): 238 MiB below the card's limit.

**Disposition (operator, 2026-10-29): +11 % is adopted, not discarded.** The +20 % bar above was Claude's, not the
project's rule (a measured gain in one phase without a regression in the other beyond the spread, at the
answer-level bar, is adopted). Before the unit takes it: the KL check, and the peak's margin at depth. Next: the
refill at the link's speed (~0.5 s would bring the prompt near +18 %, inferred), the reserve-failure switch.

## The refill at the link's speed (2026-10-29, `measured-here`, round 7's `a1a9a91`)

The same configuration and prompt (18,802 tokens); the loan off in the same window: 436.6 t/s.

| arm | refill | verify | prefill |
|---|---|---|---|
| loan, verify against the file | 0.541 s (write 0.525 from 78 tensors, wait 0.015) | 3.295 s, 0 of 9,360 rows differ | 498.4 t/s (+14.2 %) |
| loan, verify off | 0.514 s | off | 546.4 / 546.6 t/s (+25.2 %) |

The refill went from 2.7 s (~2.3 GB/s) to 0.51-0.54 s (~12 GB/s on the 14.3 GB/s link): one region copy a tensor
from the lend bank's USM. The lend bank is checked against the file once at load (0 of 9,282 rows, 1.5 s). The
verify's cost now decides the gain: verify off reaches +25.2 %, near the static arm's +29 %; round 8 makes the
default verify (against the lend bank) correct for IQ4_NL and cheap. The capital was right and the six answers decoded
at 34.40 t/s in the verified arm. The refill red switch (1) crashed inside the driver instead of corrupting a data
row: fixed in round 8.

## With the default verify (2026-10-29, `measured-here`, round 8's `6709c19`)

Same configuration and prompt; the loan off in the same window 436.7 t/s. Four loan runs: **517.3 / 516.7 / 518.0 /
517.7 t/s (+18.5 %)** with every refill verified against the lend bank (refill 0.51 s, verify 1.9 s, 0 of 9,360 rows
differ; the bank itself checked against the file at load). Red switch 1 (one corrupted byte) is caught and exits 70
with the cause and leaves no GPU fault (the queue drain works there). Red switch 4 (a failed re-reserve at the return)
exits with the cause but the B60 faulted at the exit (-ENOENT, compute and copy engines reset): queued work addresses
the freed tail buffer on that path; open. Red switch 3 does not fire yet. Peak VRAM 24,229 MiB (the unit 23,492).
