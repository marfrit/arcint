# host-expert-bank — the CPU tier reads its experts from RAM, filled at load at bulk disk speed

Charter: 0.5.x performance (operator direction 2026-09-27: "a performance, not
a measurement project"; "we need to get closer to the disk bulk read speed").

## The defect, as measured

The served Flash-Next hybrid decodes on the per-expert dispatch route with a
CPU tier. That tier read every expert in place from a shared mapping of the
artifact's `.bin` (`ParallelWeightReader::mapped`, `code`).

- **The host tier does not fit the page cache.** Expert constants are
  56.4 GiB (`measured-here`, the IR's `512,…` constants). At ratio 78 the host
  tier is ~47 GiB; the dev container has 44 GiB (52 GiB since 2026-09-27).
- **The misses land on the workers.** Every miss arrives as 128 KiB
  page-fault reads on the worker that touches it. Thread-state sampling of the
  B60 decode (d48s2, ratio 78 + VRAM slot pool + census112): the 7 tier
  workers are R 36 % / D 15–17 % / S 47 %, every D sample in
  `folio_wait_bit_common`. Decode reads 170–260 MB/s at 1.5–1.9k major
  faults/s (`measured-here`).
- **The disk rates.** The NVMe (PCIe 3.0 x4) reads 3.0–3.2 GB/s sequential
  and 1.0–1.2 GB/s at 288 KiB random, the size of one expert tensor span
  (`measured-here`, fio on the artifact file). One expert is 6 spans across 6
  per-tensor arrays.
- **Disk is ~40 % of cold decode.** The same prompt served twice in one
  process decodes 48.5 s -> 29 s (`measured-here`): the D fraction
  understates it, because idle workers wait at the join for the one in IO.

## Known against hypothesised

**Known** (`measured-here`, B60 unless stated):
- `madvise(WILLNEED)` per span at dispatch: no effect, 56.04 vs 56.27 s. The
  kernel caps one call at `max(io_pages, ra_pages)` = 128 KiB (`code`:
  `force_page_cache_ra`).
- `read_ahead_kb` 4096: WILLNEED pays (47.55 s) but the plain path loses
  (64.73 s). Fault readaround pulls 4 MiB of neighbours: ~108 MB read per
  token against ~50.
- 16 host CPUs + 52 GiB (from 8 / 44): 56.27 -> 48.98 s.
- n-gram staging reads are ~25 rows per token (90 B each), not a factor.

**Hypothesised, then measured:** a RAM bank filled at load by sequential
reads removes the fault stalls. See Status.

## Gate

The first 257-token answer of a fresh process on the configuration above
decodes at least 10 % faster than the mapping arm, with greedy text
byte-identical to it on two prompts (`measured-here`, one card window).

## Entry criteria

Met: the defect measured (above), the disk rates measured, the host RAM
raised by the operator (ARC 16 -> 4 GiB, container 44 -> 52 GiB).

## Scope — in / out

In: plugin patch 0072 (`moe/host_expert_bank.hpp`, the tier's desc/dispatch/
join, a provider accessor for the static-partition members), its cells.
Out: an expert-major store (one contiguous record per expert); a layer-ahead
router prefetch; the arcint-side flag and fit-ledger term (the bank is
env-driven for now); making the budget follow MemAvailable.

## Where it lives

`moe/host_expert_bank.hpp`; `moe_3gemm_swiglu_opt.cpp` (`cpu_tier_bank`,
`queue_cpu_bank_fill`, `cpu_expert_spans`, `build_cpu_expert_desc`,
`dispatch_cpu_tier`, `cpu_tier_join`); `expert_weight_providers.{hpp,cpp}`
(`static_partition_members`); cells
`tests/unit/test_cases/moe_host_expert_bank_test.cpp`. Knobs:
`MOE_CPU_BANK_BYTES`, `MOE_CPU_BANK_SEED`, `MOE_CPU_BANK_FILL_PER_LAYER`,
`MOE_CPU_BANK_IO_THREADS`. Counters: the `[MOE_BANK]` line at exit.

## Pipeline for this campaign

Recon (above) -> red-first cells with mutants -> one B60 window -> review
before commit -> a DESIGN record and a CHANGELOG line.

## Invariants

DESIGN §3.4: the bank serves the file's bytes, so greedy output cannot
change. A slot is a hit only for the identical span list. Everything else
(no slot, a record being read, a failed read) reads the mapping.

## Status

- 2026-09-28. Patch 0072 built and applied on the series (0003–0072
  reproduces the built tree). Ten cells, each red on a named mutant (listed
  in the patches README). B60 window (fresh process, "Hello" then a second
  prompt, 257 greedy tokens each, `measured-here`):

  | arm | decode "Hello" | decode 2nd prompt |
  |---|---|---|
  | mapping | 48.55 s | 34.62 s |
  | bank 40 GiB | 40.42 s | 34.11 s |
  | bank 46 GiB | 38.18 s | 33.77 s |
  | bank 46 GiB, reviewed build | 39.66 s | 34.72 s |
  | bank 46 GiB, final build | 38.90 s | 33.77 s |

  Text is byte-identical across arms. At 46 GiB: ~16,000 experts filled,
  2,161–2,506 demand reads, 0 evictions, minimum MemAvailable 5.6–7.4 GB.
  The fill read 59.8 GiB sequentially: whole per-tensor ranges, card-resident
  bytes included, ~21 s added to the load. **Gate met** (−18 to −21 %).
- 2026-09-28. **Decode-only tier cost with every expert in RAM**
  (`measured-here`, counter difference between a one-request and a
  two-request process on the same prompt): 1.14 ms per layer call for 7.57
  experts. That is the tier bench's figure, so the served "2x bench" gap was
  the disk. A token now costs ~118 ms (2.46 ms per layer), about half CPU tier
  and half GPU plus sync.
- **Open:** in that all-in-RAM repeat the bank is slightly slower than the
  mapping: 29.48 s (32 GiB) and 30.39 s (46 GiB) against 28.92 s. The
  mechanism is not measured (candidates: huge-page coverage of the bank,
  memory pressure at a 46 GiB reservation).
- Review (Fable, before commit): no blockers. Should-fix items 1–3 (drain
  leftover leases and callbacks, no queue drain at exit, construction failure
  logged once) and 4–5 (doc overclaim, fill read volume, OOM note) are applied.
- 2026-09-28. **Closed.** DESIGN §7.0.2da records the defect, the levers that
  did not work, the mechanism, the ten cells, the B60 window table and the
  review; CHANGELOG carries the `+p23` line. The all-in-RAM repeat where the
  bank is slightly slower than the mapping (29.48/30.39 s against 28.92 s)
  stays an open finding with an unmeasured mechanism, recorded rather than
  smoothed.

- 2026-09-30. **The same-host llama.cpp bar at 20,085 tokens is not runnable
  on this host** (`measured-here`). llama.cpp (`origin/master`,
  `src/models/qwen4exp.cpp`) was built with Vulkan and loads the same
  UD-Q3_K_XL GGUF. Both arms -- CPU-only (`-ngl 0`) and Vulkan with all MoE on
  CPU (`-ngl 99 -ncmoe 99`) -- thrash the page cache: the model is 83.80 GiB
  against the container's 52 GiB and `/models` reads at ~170 MB/s. The Vulkan
  arm reached `read_bytes` 98.5 GB in 100 min (past the model's own 90 GB, i.e.
  re-reading) and the CPU-only arm 81.7 GB in 48 min, both pinned in `D` state;
  neither reached llama-bench's table. A 64-token CPU-only probe did complete:
  pp64 4.26 t/s, tg4 0.75 t/s. The bar needs a host that can hold the model
  resident (>= 128 GiB) or a reduced depth; page-cache pressure is the first
  fact of any such comparison, and no per-token CPU-expert number was obtained.

- 2026-10-01. **The llama.cpp CPU-expert tier as a kernel microbenchmark**
  (`measured-here`, one <= 30 min leg, services stopped, B60 box).
  `test-backend-ops` has no IQ3_XXS/IQ4_NL + 512-expert/8-used MUL_MAT_ID
  case, so the exact decode shapes were run through a small standalone harness
  on the qwen4exp build's CPU backend: `ggml_mul_mat_id`, `as` (k, m, 512),
  `ids` view (8, 1), `b` (k, 8, 1), one token, 200 timed graph computes after
  4 warmups. Expert types read off the shard with `GgufFeed.gguf_type`.
  - **Types (48 layers):** `ffn_gate_exps` = 47 IQ3_XXS + 1 IQ4_XS,
    `ffn_up_exps` = 47 IQ3_XXS + 1 IQ4_XS, `ffn_down_exps` = 43 IQ4_NL +
    5 Q8_0.
  - **Shapes:** hidden 2,560, expert width 640, 512 experts, 8 used, one token.
  - **Per MUL_MAT_ID** (8 used, one token; `taskset -c 0-7` = 8 physical cores
    vs `0-15` = 16 threads):

    | op | 8 threads | 16 threads |
    |---|---|---|
    | gate/up IQ3_XXS (k 2560, m 640) | 166.7 / 177.7 us | 164.5 us |
    | down IQ4_NL (k 640, m 2560) | 108.6 / 107.4 us | 102.5 us |

    A layer (gate + up + down) is **~444 us** at 8 threads; 16 threads move it
    by < 6 %. The reference's ~20 % pinning claim is not reproduced at this
    shape -- the work is bandwidth-bound over the expert bytes.
  - **Against the tier.** The recorded tier figure is **1.14 ms per layer call
    for 7.57 experts, all in RAM** (2026-09-28). llama.cpp is **~2.6x faster
    per layer** (444 us vs 1,140 us); if the tier's 1.14 ms is a single gemm
    the gap is larger. Either way it is >= 1.5x. (Not re-taken: the tier
    figure is the recorded all-in-RAM number; a fresh take needs a served run
    with a full bank, outside this box.)
  - **Decision.** The next build is the tier's hot loop: read ggml's AVX2
    `vec_dot` for IQ3_XXS/IQ4_NL first (source read before implementing), then
    the call's threading and pinning and its per-call overheads. The gap to
    the reference is **compute, not RAM residency**.

- 2026-10-01. **Harness v2: cold-DRAM decode and a prefill row.** The first
  microbench held the same 8 expert ids every iteration, so its decode row was
  L3-hot. Re-run with ids rotating over all 512 per iteration (coverage
  verified 512/512) and a prefill-shaped row (512 tokens, 7 used, 322 distinct
  experts -> 3,584 pairs vs the measured 3,227), 8 physical cores vs 16
  threads, services stopped.
  - **Decode** (8 used, 1 token), per MUL_MAT_ID:

    | op | 8 threads | 16 threads |
    |---|---|---|
    | gate/up IQ3_XXS | 180.2 us | 233.7 us |
    | down IQ4_NL | 179.2 us | 196.0 us |

    Layer = **~540 us** at 8 threads. Cold DRAM makes 16 threads (SMT)
    **10-30 % SLOWER** than 8 physical cores -- the reference's pinning claim
    is real at cold DRAM, reversing the L3-hot reading above.
  - **Prefill** (512 tokens, 7 used, 322 distinct), per MUL_MAT_ID:
    IQ3_XXS 37.77 ms (8t) / 36.89 ms (16t); IQ4_NL 40.95 ms (8t) / 41.19 ms
    (16t). Layer = **~116.5 ms** at 8 threads.
  - **Against the tier.** Decode: 540 us vs the tier's 1.14 ms -> **2.1x**.
    Prefill: 116.5 ms vs the recorded ~152 ms per chunk-512 call -> **1.30x**,
    i.e. at the 1.3x gate boundary and not better.
  - **Cause (source).** The tier's AVX2 path (`moe_cpu_expert_avx2.cpp`)
    dequantises IQ3_XXS/IQ4_NL rows to f32 (`decode_*_row_avx2`) and then does
    an f32 dot. ggml dots in the quantised domain with the activation
    quantised once per token to Q8_K/Q8_0 (`vec_dot_iq3_xxs_q8_K`,
    `vec_dot_iq4_nl_q8_0`, `vec_dot_q8_0_q8_0`); that is where the gap lives.

- 2026-10-01. **Tier hot loop: the native expert dot runs in the quantised
  domain (patch 0074, `measured-here`).** ggml's approach, not its code: the
  activation is quantised once per stage to int8 with one scale per 32 values,
  and `maddubs`/`madd` dot the packed weight directly, so the f16-to-f32 row
  decode (the tier's dominant cost) disappears.
  - **Measured** (dirac, services stopped; 8 experts x one job, IQ3_XXS
    gate/up + IQ4_NL down, H=2560 I=640): **651 us per layer call** on the
    split plan {4,2}. The whole-expert schedule is 1007 us and the old
    dequant path ~1.4 ms. Against llama.cpp's `mul_mat_id` at ~540 us that is
    **1.21x**, inside the 1.3x microbench gate.
  - **Numerics** change: the int8 activation replaces the f16->f32 decode. The
    cell `moe_cpu_expert_native.quantised_dot_matches_the_f32_tier_within_bound`
    holds a scale-relative bound of 0.06 against the f32 tier at the served
    shape (achieved ~0.003) and is deterministic across runs. The one real bug
    found and fixed: `_mm256_sign_epi8` ZEROES the source where its sign byte
    is 0 (PSIGNB semantics), so the IQ3_XXS sign vector must carry +1 for a
    clear bit, not 0.
  - **Dispatch**: the AVX2 path now accepts the native formats (IQ3_XXS /
    IQ4_NL / IQ4_XS / Q8_0); IQ2_S keeps the scalar reference.
    `MOE_CPU_TIER_Q8_DOT=0` restores the pre-0074 routing (the A/B used here).
  - **Open**: the served pair (dense arm; decode and prefill at 20k) and the
    window-0 KL gate have not been run.
