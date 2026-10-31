# kq_tune — parameter search for the llama engine's Intel kernels

Drivers for `docs/campaigns/kernel-autotune-ga.md`. They run against a
llama.cpp tree carrying `contrib/llama.cpp/patches` through 0014, on the
card the kernels serve. Each driver writes one JSON file, rewritten after
every evaluation. A rerun with the same file resumes and skips genomes
already measured.

Environment:
- `ARCINT_TUNE_LLAMA_BIN`: the tree's `build/bin` (test-backend-ops,
  llama-bench);
- `ARCINT_TUNE_MODEL`: the GGUF scored by the end-to-end drivers. Put it on
  a file system whose page cache keeps it: each evaluation reloads it.
- `ARCINT_TUNE_PLATFORM`: ggml-opencl's platform index of the card (default
  0 for the B60 drivers, 1 for the A770's). A run on another device is
  rejected: the drivers check the device name in the log.

The drivers drop every `GGML_OPENCL_KQ_*` variable from the caller's
environment, and they reject a run in which the host ignored a setting.
Otherwise a fallback to the defaults would be scored as the genome that
asked for it.

| driver | card | genome | fitness (TUNE) |
|---|---|---|---|
| `kqtune.py exhaustive\|ga <type\|all> out.json` | B60 | the 2D GEMM's TM, WG, AT, KSYNC, int8 on/off (program-wide) | test-backend-ops us/run on the dense 27B's shapes, weighted by tensor count |
| `kqga_e2e.py ga\|random\|eval out.json` | B60 | the same, per type (Q4_K, Q5_K, Q6_K) | llama-bench pp512, dense 27B |
| `kqga_a770.py ga\|random\|eval out.json` | A770 | MoE ID2 on/off and XT, XSG; MoE tile; plain tile | llama-bench pp512, coder |

GA options:
- `--seed`, `--budget` (evaluations), `--pop`;
- `--noseed` starts from random genomes only. Without it, the current
  default is seeded into the first population, so the search cannot end
  worse than the default.

How it searches:
- **Selection:** tournament of 3, elitism, crossover by kernel role (a
  type's or a kernel's block of genes comes whole from one parent).
- **Mutation:** mostly ±1 steps on ordered knobs.
- **Validity:** genomes that break the host's constraints are rejected
  before any run (A770: local memory, work-group size, ID2 divisibility;
  B60: TM divisible by AT). Whatever the host still refuses, a shape that
  does not build or fit, is caught in its log and scores nothing.
- **Drift:** a reference genome is re-measured every 8 evaluations; each
  score is a ratio to the preceding re-measure. The end-to-end drivers use
  the shipping defaults (0014). kqtune.py uses 0013's, because its one
  KSYNC gene cannot express 0014's split (16 for the fp16 kernels, 32 for
  the int8 one).
- **Failures:** a run that fails or times out scores nothing; 5 in a row
  abort.

Analyses:
- `ga_analysis.py exhaustive.json ga1.json ...`: GA runs against the
  exhaustive table. Reports the rank found and the evaluations until within
  1 % / 2 % of the optimum, plus random search at the same budget, by
  resampling.
- `e2e_analysis.py run1.json ...`: best so far at equal budgets, and the
  top genomes.

A winner is a candidate, not a result. Re-measure it interleaved with the
default, run the GATE and REPORT sets, and take the answer-level bar before
it becomes a default (the campaign's gate).
