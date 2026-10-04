# kernel-autotune-ga — search the llama engine's kernel parameters, scored at the endpoint

**Open.** First results in `contrib/llama.cpp` 0014 (2026-10-04).

## Charter

The Intel kernels of the llama engine (`contrib/llama.cpp`) carry
hand-chosen tiles, shapes and intervals. Search them instead:
- a genetic algorithm (GA) over the joint parameter space;
- scored by the served path's own numbers on the model's own shapes;
- held to the same correctness and answer-level gates as a hand change.

## Reference to follow

Kernel Tuner (KernelTuner/kernel_tuner, read at a467bdc, `code`):
- its GA is `kernel_tuner/strategies/genetic_algorithm.py`;
- defaults: population 26, up to 90 generations;
- the start configuration as `population[0]`, so the default is seeded;
- rank-biased selection (a beta distribution over the sorted population);
- single-point crossover by default;
- a mutation for 1 child in 55, to a random Hamming neighbour;
- invalid children repaired to the nearest valid neighbour;
- no elitism; evaluations cached per configuration.

Its measured effect on these kernels is not known here: Kernel Tuner
reports results for CUDA/OpenCL kernels of its own test set, not read.
Its strategy list also has differential evolution, PSO, simulated
annealing, Bayesian optimization and random sampling. CLTune and OpenTuner
are the other tuners in this line (not read here).

tools/kq_tune deviates from Kernel Tuner's GA in five places, chosen, not
measured against it:
- tournament selection of 3;
- elitism (2-3);
- crossover by kernel role, not by position;
- per-gene mutation, mostly ±1 steps on ordered knobs;
- invalid genomes rejected and redrawn rather than repaired.

Running Kernel Tuner's own GA on the same fitness is open.

The tuning discipline both follow:
- three disjoint sets:
  - **TUNE** drives selection;
  - **GATE** checks correctness, with a probe that must fail;
  - **REPORT** gives the only published number, never seen by the search;
- a fixed reference configuration re-measured during the search, against
  clock and thermal drift;
- invalid configurations repaired or rejected before any run.

## Gate

A searched configuration replaces a default only if all of these hold:
- MUL_MAT (and MUL_MAT_ID where the kernel serves it) pass in full on the
  card;
- its TUNE gain survives an interleaved re-measurement against the default;
- the REPORT set (other prompt lengths, another ubatch) moves the same way;
- the answer-level bar holds: KL, the acceptance task.

## Current state (`measured-here`, 2026-10-04)

Harness: `tools/kq_tune/` (README there). Measurement configuration:
- the A770's GT clock pinned at 2,000 MHz;
- llama-bench `-fa 1 -r 2`, its default f16 KV, one sequence;
- test-backend-ops perf us/run, the activation conversion included.

**Does the GA work? Tested against ground truth.** The B60's 2D-GEMM knobs
(token tile, sub-groups, activation read rows, barrier interval, int8 on or
off) are program-wide, so they form one joint space: 448 genomes. Fitness is
test-backend-ops us/run on the dense 27B's own shapes, weighted by tensor
counts.
- Exhaustive search: the optimum is 0.999 of 0013's defaults, and the
  median genome is 1.287. The space is hostile, and the hand defaults were
  already within 0.1 % of the optimum.
- Unseeded GA, three seeds, ~65 evaluations each: ranks 1, 2 and 1 of 448.
  Within 1 % of the optimum after 29, 58 and 28 evaluations.
- Random search at the same budget: median gap +2.1 %; within 1 % in 27 %
  of draws.
- The seeded runs (the default in the first population) said nothing,
  because the seed was already within the noise (~1 %).

**End to end, where exhaustive search is impossible.** Fitness is
llama-bench pp512; GA and random search get 120 (B60) or 100 (A770)
evaluations each.
- **B60, dense 27B.** Per-type programs (`GGML_OPENCL_KQ_2D_T<i>`), ~1.3e7
  genomes.
  - Two GA seeds: 1,033.9 and 1,032.8 t/s against the default's 1,028.5.
  - Random search: nothing better than the default.
  - Re-measured on the harness (the genome set through the environment) and
    reported: +0.5 % at pp4096 (929.0 -> 933.4), +0.6 % at pp2048 with
    ubatch 1,024.
  - The change is the int8 Q4_K kernel's barrier interval, 16 -> 32.
- **A770, coder.** The MoE GEMM's ID2 switch and shape, the tile kernel's
  tile, the plain GEMM's tile. llama-bench pp512 at the defaults read
  1,607.5-1,613.5 over 39 re-measures.
  - GA seed 1: ID2 shape 24 x 32, +5.6 %.
  - GA seed 2: ID2 **off**, with the tile kernel's tile retuned (2,4,2,2)
    and the plain tile 4,4,2,8, +7.9 %. ID2 had replaced the tile kernel
    because it beat it at the tile kernel's old tile. Random search did
    not find it (nothing better than the default in 100 evaluations); a
    per-knob sweep was not run.
  - Gates, on the harness (genome set through the environment):
    - interleaved TUNE re-measure 1,608-1,612 -> 1,737-1,740 t/s;
    - MUL_MAT 1,123/1,123 and MUL_MAT_ID 338/338;
    - REPORT pp2048 at ubatch 1,024, a GEMM shape the search never ran:
      1,659 -> 1,882 t/s (+13.4 %).

Both are defaults in 0014. Re-measured on the built patch, two interleaved
repeats each:

| model, card | REPORT pp4096 | TUNE pp512 |
|---|---|---|
| coder, A770 | 1,345-1,346 -> 1,431-1,432 t/s (+6.4 %) | 1,613 -> 1,739-1,742 (+7.9 %) |
| dense 27B, B60 | 930.1-930.4 -> 934.5-934.8 (+0.5 %) | 1,028.3-1,028.6 -> 1,033.1-1,033.8 |

The rest of the gate:
- MUL_MAT 1,123/1,123 on both cards and MUL_MAT_ID 338/338 on the A770
  (the B60 shows the pin's own 74 MXFP4 MUL_MAT_ID failures, 0013);
- the coder's KL 0.007043 -> 0.007040, top-1 95.88 -> 96.08 %;
- the acceptance task, served (`--engine llama`, MTP and the 40,960-id
  draft head): the coder 10/10 at temperature 0 and 3 of 3 sampled; the
  dense model 10/10 at temperature 0;
- decode unchanged: dense 52.5 t/s, coder 77-80 t/s;
- the B60 change moves no arithmetic (the 2D kernels use no local memory;
  a barrier interval only), `code`.

No fault was injected into 0014's tiles. The suite's power to fail rests on
the faults injected for 0011 and 0013 (`contrib/llama.cpp/README.md`).

Open:
- the decode kernels (matvecs, decode attention), with decode t/s as
  fitness;
- the IQ kernels on Flash-Next;
- **code-level genetic optimization (priority, operator 2026-10-04):** genes
  are structural variants of the kernel source, not only its parameters:
  - load paths per operand (direct, staged in local memory, 2D block);
  - staging tiles, prefetch depth, barrier placement;
  - which products run on XMX, and row/column ownership per sub-group.

  They sit behind compile-time switches and are recombined by the GA. Later,
  source mutations are proposed by an editor model and admitted only after
  FLASH_ATTN_EXT / MUL_MAT pass on the card. The first target is the
  decode/verify attention of `gqa-small-t-decode.md`, whose hand ablation
  already has this form (K and V staged or direct, BK, the products), with a
  measured cost per switch. Reference to read before building: program
  search for kernels (AlphaEvolve-style evolution, `paper`); Kernel Tuner's
  structural tunables (`code`).

## Code-level search: the decode/verify attention kernel (2026-10-04, `measured-here`)

The first structural search (operator's priority, 2026-10-04). Target:
0016/0017's `flash_attn_gqa_dpas.cl` on the B60. Driver:
`tools/kq_tune/fagqa.py`.

The genes are source alternatives behind compile-time switches, plus one
host knob:
- SIDES: sub-groups sharing an 8-row tile;
- KMODE: K read direct or staged;
- BK: keys a staged tile;
- KVPS: the kernel's own split size, `GGML_OPENCL_FA_GQA_KV_PER_SPLIT`;
- two genes written for the search:
  - SSPLIT (`GQA_S_SPLIT`): the sides of a tile split QKᵀ by key group and
    exchange S through local memory;
  - PF (`GQA_PF`): the next V tile prefetched into registers.

Fitness: the geometric mean of one attention layer at 4, 6 and 8 rows and
32k and 131k keys (dense 27B geometry, f16), against the shipped genome,
re-measured every 8 evaluations (drift under 0.4 %).

- **Exhaustive, 90 genomes** (SIDES × KMODE × BK × KVPS 64-1,024):
  - best 0.8626, the shipped structure at KVPS 512;
  - median 1.356, worst 3.53;
  - the shipped KVPS 128 scores 0.9997.
- **GA with SSPLIT** (216 genomes, budget 70, 59 evaluated): best 0.7834,
  KVPS 512 with SSPLIT, found at evaluation 45.
- **GA with prefetch** (60 evaluated): best 0.798, prefetch 8 without
  S-split.
- **Local enumeration of S-split × prefetch × split** (18 genomes):
  - S-split 0.784, prefetch 8 0.798;
  - both together 2.44, prefetch 4 about 2.0. IGC compiles those into a
    rolled form of a third the length (3,181-3,383 assembly lines against
    10,176-12,453) that keeps the per-tile arrays in memory (ocloc dumps).
- **Gate:** FLASH_ATTN_EXT at head sizes 256 and 128 with the kernel taking
  every row count, 860 of 861 (the pin's softcap case) for the exhaustive
  best, the three best S-split genomes and the local enumeration's two best.
- **End to end** (0018 = S-split + split 512; q8_0 KV, 32k depth): dense 27B
  6 rows 52.6 -> 64.2 t/s, 8 rows 62.1 -> 78.5. Cydonia 6 rows 42.3 -> 58.8.
  Served agent at 62.6k depth 15.6 -> 16.4 t/s. KL 0.003589 -> 0.003587.
- **Reproducing:** the exhaustive run used the first four genes with KVPS
  64-1,024. `fagqa.py` has since grown KVPS 2,048, SSPLIT and PF, so
  `exhaustive` now enumerates 648 genomes. The results files hold no binary
  sha: start a new file after a rebuild.

## Where it lives

`tools/kq_tune/` (the drivers and the analyses). The search hooks are in
`contrib/llama.cpp` 0014:
- `GGML_OPENCL_KQ_2D_T<i>`, `GGML_OPENCL_KQ_2D_OPTS`;
- the dense model's shapes in test-backend-ops perf;
- the A770 tiles through the existing `GGML_OPENCL_KQ_MM_*` switches.
