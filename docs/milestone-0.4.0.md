# 0.4.0 — open and run GGUF models

Recorded 2026-09-05 as the next feature line after 0.3.x's campaigns. The
charter is one sentence: **arcint opens a GGUF checkpoint and serves it**,
under the same gates as an OpenVINO IR directory. It is not gated by
hardware — the format question is answered on the host, the card only has
to run what comes out — and plugin patches are in scope, as they were for
0.3.0 (`contrib/packaging/marfrit-openvino/patches/` is the series this
would add to).

## Where the repository stands

- The artifact contract is an **OpenVINO IR directory, nothing else**
  (`src/core/artifact.cpp`, `docs/model_requirements.md` §1): GGUF and
  safetensors are "wrong format, not loaded" (§6). The allowlist names one
  model family — hybrid GatedDeltaNet + full attention, two MoE and one
  dense checkpoint, one shared tokenizer — and every IR came from
  optimum-intel's export plus NNCF weight compression (int4 AWQ, §2).
- Everything downstream of `load_artifact` assumes that IR's graph shape:
  the paged-attention ports the executor audits after compile, the GDN
  state ledger, the logits slice, the MTP and DFlash drafter contracts, the
  MoE lowering the exporter chooses (`--moe-lowering tiled`). A GGUF path
  has to produce that graph, not merely those weights.
- Sub-4-bit and K-quant storage are already two campaigns
  (`sub4bit-vram-kernel`, `kquant-host-storage`, `docs/campaigns/`): the
  pinned runtime's MoE fusion matches u4 only, and there is no in-kernel
  dequant for anything else. A GGUF whose expert tensors are Q4_K/Q5_K/
  Q6_K/IQ-anything lands exactly on those campaigns' open questions.
- The tokenizer and chat template live in the artifact (DESIGN §3.7); GGUF
  carries both as metadata (`tokenizer.ggml.*`, `tokenizer.chat_template`),
  which the OpenVINO tokenizer extension does not read directly.

## What "open" can mean — three routes, to be decided by recon, not here

1. **Offline import** (`tools/import_gguf.py`): read the GGUF (the
   reference `gguf-py` reader is Python, MIT), map every tensor onto the
   IR the exporter would have produced for the same architecture, write an
   IR directory the existing `load_artifact` accepts unchanged.
   Lossless where the block format has an exact OpenVINO equivalent
   (Q8_0 → u8 per-32 symmetric; Q4_0/Q4_1 → u4 per-32 with scale or
   scale+min); a stated, measured *re-quantisation* everywhere else. Cheapest
   route to "opens"; the tokenizer becomes its own sub-problem (the
   extension wants a HF `tokenizer.json`, which the GGUF's vocab/merges can
   regenerate).
2. **In-process graph build**: arcint constructs the `ov::Model` from the
   GGUF at load, the way OpenVINO GenAI's own GGUF reader is understood
   to do for a few dense families (to be confirmed in recon against its
   release notes and source — which families, which block types, and
   that its builder covers neither GatedDeltaNet nor this MoE lowering). No
   intermediate directory; the import logic lives in C++ and every
   architecture is a builder to maintain.
3. **Native block formats in the plugin**: a K-quant constant type with
   in-kernel dequant, so a GGUF's expert tensors are served at their own
   bit width. This is `sub4bit-vram-kernel`'s route under another name, and
   the only one that pays M10's original claim; it is also the largest.

Recon decides between 1 and 2 for the *opening* half and records whether 3
is needed for any GGUF worth serving; the likely shape is 1 first (with
Q4_0/Q8_0 exact, K-quants re-quantised and measured), 3 as the campaign it
already is.

## Ground rules (unchanged from 0.3.0)

Every item ends in a measurement naming card, depth, KV precision and
configuration; red case first; the §5 gates apply; a plugin patch needs a
fusion-impact profile, not a kernel micro-benchmark; no "as good as
llama.cpp on this file" without a recorded survey of what llama.cpp's own
Arc backends (SYCL, Vulkan) do on the same checkpoint and card.

## Gate

- **Opens:** a GGUF of a checkpoint in the served family loads through
  `arcint --model <file.gguf>` (or the imported directory) with the same
  banner, `/props` and refusals an IR gets; a GGUF outside the family is
  refused by name, not by crash.
- **Runs:** Prüfstand through the served endpoint on the coder family at
  the GGUF's own bit width, scored the same way; a Q4_0/Q8_0 file whose
  blocks convert exactly serves **byte-identically** to the IR built from
  the same weights (the equivalence bar of `CLAUDE.md`); a re-quantised
  K-quant file names its measured loss against the IR artifact's 10/10.
- **Determinism:** importing the same file twice yields byte-identical IRs;
  cold and warm prefix cache stay byte-identical on the GGUF path exactly
  as on the IR path (DESIGN §3.4).
- **Not a regression:** the unit set and every acceptance cell on the IR
  path unchanged.

## Entry criteria

`docs/model_requirements.md` current; the campaign survey
`docs/campaigns/research-kv-quantisation.md`'s "what transfers" read; a GGUF
of the coder family on the dev host (Q4_K_M and Q8_0 at least, so both the
exact and the re-quantised branches have a file); the `kquant-host-storage`
and `sub4bit-vram-kernel` campaign documents re-read so this milestone does
not re-plan them.

## Pipeline

Recon (the three routes against the real GGUF's tensor inventory: which
block types, which tensors, what the tokenizer metadata carries) → design
note `docs/design-gguf-import.md` (route, tensor map, what is exact and
what is re-quantised, the tokenizer plan) → red-first implementation (a
reader that refuses a wrong file; an importer whose IR the existing loader
accepts; the byte-identity cell) → one card window at the end → review →
DESIGN `§7.0.2x` record, CHANGELOG, `model_requirements.md` §1/§6 rewritten
to say what is now true.

## Size

Medium–large for route 1 with the exact branch only; large once K-quants
are served natively (route 3 = the kernel campaign). Not a single session.

## Status

- 2026-09-05 — recorded; nothing started. Independent of 0.5.0.
- 2026-09-06 — discussion closed, decisions recorded (operator, 2026-09-05/06):
  the allowlist keeps its families; the point is comparing quantisations
  of the same model, so the K-quant tensors are served **natively** (a
  new block format with in-kernel dequant, the route the sub4bit-vram-
  kernel campaign owns), not re-quantised; the engine opens the file **in
  process**; the served IR's chat template governs a GGUF-opened model for
  now (the file's own differs — Unsloth's merges leading system messages
  and drops the no-user-query exception); the order is the dense 27B at
  Q4_K_M, then the pruned coder's MoE file, then the sub-4-bit set. The
  gate stays as written above. Reference files exist for every stage in
  the operator's store, all `qwen35`/`qwen35moe`, each carrying its MTP
  layer as block 64/40. **Tokenizer measured identical** between the
  dense Q4_K_M file and the served IR: tokens equal over the IR's 248,077
  entries (the file pads to 248,320), merges equal (247,587), the
  pre-tokenizer is the same regex (llama.cpp's `qwen35` type quotes it
  from `tokenizer.json`), eos 248046 in both; only the pad id differs.
  One Q4_K_M file is four block types (Q4_K, Q5_K, Q6_K, Q8_0), so the
  first kernel set is four decoders in the fully-connected path plus the
  embedding gather; the MoE fusion is the second; IQ4_XS/IQ3_S/Q3_K the
  third. Next artifact: `docs/design-gguf-native.md` from two recon
  passes (the converter's tensor map and block layouts; the plugin's
  compressed-weight kernels and what they accept).
- 2026-09-06 — recon done and the design note written:
  `docs/design-gguf-native.md` (template IR read with mapped weights, the
  GGUF's bytes replacing each decompression subgraph as tagged u8
  constants; a plugin op `FullyConnectedKQuant` with K-quant unpack in
  the fully-connected kernel, an embedding gather and the expert fusion;
  exactness in three layers; four stages with gates). Findings that
  shaped it: the converter's V-head reorder and `−exp(A_log)` / `+1`
  norm transforms must be undone at open; the plugin's compressed-weight
  path accepts only integer zero points and a flat group size, its
  expert fusion matches twelve u4 constants and nothing else, and no
  patch in the series touches the weight format. Open points are §6 of
  the note.
- 2026-09-06 — stage 0 landed: a memory-mapped GGUF v3 reader
  (`src/core/gguf.*`), host reference dequantizers for Q8_0, Q4_K, Q5_K
  and Q6_K transcribed from ggml (`src/core/gguf_dequant.*`), a generated
  fixture (`tests/fixtures/qwen35-tiny.gguf`, its reference dequantization
  from gguf-py beside it; `tools/gguf_fixture.py` regenerates both,
  seeded), twelve unit cases (metadata, tensor table, exact equality of
  every quantized tensor against gguf-py's decoder, both branches of the
  6-bit scale unpack, three refusals by name) — red first with stub
  implementations, 426 cases green after. The generator caught its own
  Q5_K field-order mistake through gguf-py's decoder before the fixture
  was written. Still owed for stage 0: the comparison against a real
  Q4_K_M tensor on the dev host (`tools/gguf_dequant_check.py` is ready
  for it). Kernel side (plugin patch 0021, the fully-connected decoder)
  in progress.
- 2026-09-06 — stage 1 served (DESIGN §7.0.2ay): `--gguf` opens the dense
  Qwen3.8-27B Q4_K_M on the dense IR template, 497 projections from the
  file, Prüfstand 10/10 through the GGUF-opened model on the 24 GB card;
  plugin patch 0021 (`+p7` recipe, package not built) decodes Q4_K/Q5_K/
  Q6_K/Q8_0 in the fully-connected kernel. The first serve was 0/10
  because the template's AWQ activation scales compensated weights now
  raw; the pass sets them to one. The kernel climbed a measured ladder
  of eight versions from 28.8 / 3.5 t/s (prefill / decode) to the
  benchmark the operator asked for, against Intel's own int4 IR export
  on the same card: 213 / 9.9 at 856 prompt tokens and 174 / 8.5 at
  71.7k, the IR 1,609 / 23.1 and 552 / 16.5 (chunk 256 against 2048,
  the fit's choice per arm). Owed within stage 1: the embedding gather
  kernel, the MTP layer from the file, the decode kernel at bandwidth
  and 2-D block loads for the prefill tile, the activation reservation
  at chunk 256; then stage 2.
