# The OpenVINO era (0.1-0.7.12)

arcint's first engine served OpenVINO IRs on a patched OpenVINO GPU plugin.
From 0.5.5 the libllama engine took the services over one by one (the agent
in 0.5.6, Flash-Next in 0.6.0, the coder on 2026-10-05), and on 2026-10-15
the operator retired the OpenVINO engine, its runtime package
`marfrit-openvino` and its tools (charter item A6,
`docs/milestone-1.0.0.md`; DESIGN §6 has the measured reasons).

Everything of that era is kept at the tag **`openvino-history`** (commit
`308547e`, the last tree that carried both engines):

    git show openvino-history:DESIGN.md
    git show openvino-history:docs/window-054.md
    git checkout openvino-history -- <path>      # to bring a file back

or browse <https://github.com/marfrit/arcint/tree/openvino-history>. A citation
"DESIGN §7.0.2x" anywhere in the record (the campaigns, the changelog, the
patch READMEs) points into that tag's `DESIGN.md`, whose §7.0.2 is the
record index of the era.

## What the era built, in one paragraph each

- **The engine** (`src/exec/backend_ov.cpp` and around it): a paged executor
  on OpenVINO's `SDPAToPagedAttention`, arcint owning the page tables, the
  GDN state rows and their checkpoints; an exact prefix cache (a keyed
  128-bit hash chain over token blocks) with a host tier for evicted
  prefixes; admission by measured reservation (the fit pass); two lanes
  behind a FIFO turnstile; the MTP head reconstructed for the dense 27B,
  DFlash2 block drafting, prompt lookup; an allowlist of IR artifacts
  pinned by hash.
- **The runtime** (`contrib/packaging/marfrit-openvino/`): the pinned
  OpenVINO nightly `71640275` with plugin patches 0003-0077 (`+p25`, `+p27`):
  the MoE device slot pool and CPU tier, the static partition and census
  seed, the K-quant kernels for GGUF weights on a template IR, micro-SDPA
  for u8:i4 KV, native expert formats, the host expert bank, NVMe-to-VRAM
  pinned fill, Flash-Next's QSA.
- **The tools** (`tools/`): the IR exporters (`export_serving_artifact.py`,
  `tools/q4e/`, `export_mtp.py`, `export_dflash.py`), the reference forward
  and KLD capture against it, expert-placement and routing-census tools.
- **The measurements**: Flash-Next at full depth on the B60 with 75 % of
  the experts on the CPU tier, 6.5 t/s decode and 63 t/s prefill at 20k;
  the dense agent at 24.6 t/s decode with MTP and 1,141 t/s prefill at 4k;
  the coder at 43.9 t/s decode and 1,379 t/s prefill at 4k on the A770.

## The retired documents

The design notes, milestones, acceptance windows and campaigns of the era.
Each is at the tag under the same path.

**Design notes and reports** (`docs/`): `attribution-gap.md`,
`benchmark-served-services.md`, `design-expert-hot-set-lru.md`,
`design-fit-levers.md`, `design-gguf-native.md`,
`design-lyon-stateful-prefill.md`, `design-m8-asymmetric-kv.md`,
`design-m9-offload-v2.md`, `design-native-dpas-expert-kernel.md`,
`design-nvme-direct-expert-tier.md`, `design-ple-disk-backend.md`,
`design-qwen-flash-next.md`, `design-qwen35moe-serving-shape.md`,
`design-routing-aware-expert-execution.md`,
`design-served-prefill-determinism.md`,
`design-static-partition-cold-start.md`,
`design-static-partition-prefill.md`, `dflash-pairing-probe.md`,
`handoff-served-prefill-determinism.md`, `kernel-selectors.md`,
`model_requirements.md`, `moe-m2-path.md`, `mtp-head-card.md`,
`mtp-head-verification.md`, `prefill-baseline.md`,
`prefix-cache-production.md`, `serving-config-flash-next.md`,
`upstream-38099-comment.md`.

**Milestones and acceptance windows** (`docs/`): `milestone-0.3.0.md`,
`milestone-0.4.0.md`, `milestone-0.4.1.md`, `milestone-0.5.0.md`;
`window-050.md` (0.5.0), `window-051.md` (BERLIN), `window-052.md`
(VENICE), `window-053.md` (LISBON), `window-054.md` (LYON).

**Outside reviews** (`docs/reviews/`): `secondopinion.md`,
`thirdopinion.md`, `additionalscope.md`.

**Campaigns** (`docs/campaigns/`), all on the OpenVINO engine's MoE tier,
KV or test ladder: `dense-q8-flash-next.md`, `expert-hot-set-lru.md`,
`host-expert-bank.md`, `hybrid-expert-fetch.md`, `kquant-host-storage.md`,
`nvme-direct-expert-tier.md`, `partition-seeding.md`,
`ple-disk-backend.md`, `prefill-fallback-tristate.md`, `qsa.md`,
`served-prefill-determinism.md`, `serving-shape-logits.md`,
`static-partition-cold-start.md`, `static-partition-prefill.md`,
`sub4bit-vram-kernel.md`, `test-ladder-close.md`,
`tier-handoff-doorbell.md`, `tier-served-gap.md`,
`turnstile-wall-time.md`, `u8i4-deep-prefill-fault.md`,
`u8i4-prefill-price.md`. Their mechanisms that the references use were
rebuilt on the libllama engine: the adaptive expert cache (0021, 0042,
0055), the CPU's share of the misses through a doorbell (0056), prefill
streaming of the bank (0032, 0044), the n-gram table read lazily.

**Other**: `FURTHER-READING.md` (the era's long README),
`contrib/packaging/marfrit-openvino/patches/README.md` (the per-patch record
of plugin patches 0003-0077), `kernels/README.md` (the custom-kernel
experiment).

## Upstream reports from the era

openvino.genai#4367 (continuous batching against the stateful path under
greedy), openvino#37607 (blob import with an uninitialised MoE weight
provider), openvino PR #37878 (micro-SDPA prefetch out of bounds),
openvino#38099 (a sibling comment: the B60's GDN run-to-run
nondeterminism), compute-runtime#948 (a comment: the direct-submission
fault).
