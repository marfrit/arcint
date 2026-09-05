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
