#!/usr/bin/env python3
"""Export Qwen/Qwen3.8-Flash-Next (model_type qwen4_exp) to OpenVINO IR.

Unlike export_mtp.py / export_dflash.py, this is not a hand-reconstructed
head: qwen4_exp is a full 125B-total MoE backbone (config delta recorded in
docs/design-qwen-flash-next.md, FIX B), so hand-writing the graph is not the
plan. The plan is the ordinary path every other served architecture in this
repo's allowlist went through -- optimum-intel's OVModelForCausalLM export,
which needs transformers to recognise the architecture first.

Two upstream gaps block this as of 2026-09-09: optimum-intel 2.1.0 pins
transformers<5.6, which excludes the version that carries qwen4_exp
(5.17.0+), and even force-upgraded, optimum-intel's own imports break
against transformers 5.17.0 (VisionRotaryEmbedding removed). Both gaps are
upstream's, not arcint's. This script checks both and refuses with a named
reason when either blocks, so the refusal is legible to a script
(tools/watch_flash_next_export.py) and to a person without reading a
traceback.

    python3 tools/export_flash_next.py --out /models/ov/qwen38-flash-next

Today this exits 1 with a named-reason assertion. When both blockers clear,
the same invocation should proceed straight to the optimum-intel export
below -- if it needs more than a version bump at that point (a new config
key optimum-intel's OpenVINO config class does not map yet, say), that is
a small, ordinary follow-up, not a rewrite of this file.
"""
import argparse
import importlib
import sys

CHECKPOINT = "Qwen/Qwen3.8-Flash-Next"
MODEL_TYPE = "qwen4_exp"
ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
MODELING_MODULE = f"transformers.models.{MODEL_TYPE}"

KNOWN_BLOCKER = (
    "known upstream blocker (docs/design-qwen-flash-next.md, FIX A; "
    "HANDOFF-0.5.0.local.md FIX A) -- not an arcint defect, nothing to fix "
    "here until upstream ships it"
)


def check_transformers_support():
    """Import transformers and check for the qwen4_exp modeling module.

    Returns the installed transformers version on success. Raises
    AssertionError, with the exact reason named, on either failure mode:
    transformers itself absent, or present but without qwen4_exp.
    """
    try:
        import transformers
    except ImportError as exc:
        raise AssertionError(
            f"transformers is not installed in this interpreter at all "
            f"({exc}) -- cannot even check for {MODEL_TYPE} support. This "
            f"is an environment problem, not the {KNOWN_BLOCKER}."
        ) from exc

    version = transformers.__version__

    try:
        importlib.import_module(MODELING_MODULE)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"transformers {version} does not carry {MODELING_MODULE} "
            f"(architecture {ARCHITECTURE}, checkpoint {CHECKPOINT}): "
            f"{exc}. This is the {KNOWN_BLOCKER}."
        ) from exc

    return version


def check_optimum_intel_compat():
    """Check that optimum-intel is installed and compatible with the current
    transformers. Returns (optimum_intel_version, transformers_version) on
    success. Raises AssertionError naming the exact incompatibility on
    failure -- the two known failure modes as of 2026-09-09:

      1. optimum-intel pins transformers<5.6, which excludes the version
         that carries qwen4_exp.
      2. Even with transformers force-upgraded, optimum-intel 2.1.0's own
         imports break (VisionRotaryEmbedding removed in transformers 5.x).
    """
    try:
        import importlib.metadata
        oi_version = importlib.metadata.version("optimum-intel")
    except Exception:
        oi_version = "unknown"

    try:
        from optimum.intel import OVModelForCausalLM  # noqa: F401
    except ImportError as exc:
        raise AssertionError(
            f"optimum-intel {oi_version} fails to import with the current "
            f"transformers: {exc}. This is the {KNOWN_BLOCKER}."
        ) from exc

    import transformers
    return oi_version, transformers.__version__


def export(checkpoint, out_dir):
    """The actual export, reached only once both check_transformers_support()
    and check_optimum_intel_compat() return cleanly. Same shape as any other
    allowlisted-architecture export in this repo's chain: optimum-intel
    builds the OpenVINO IR straight from the HF checkpoint, no hand-written
    graph."""
    from optimum.intel import OVModelForCausalLM

    model = OVModelForCausalLM.from_pretrained(
        checkpoint,
        export=True,
        trust_remote_code=False,
    )
    model.save_pretrained(out_dir)
    print(f"exported {checkpoint} ({ARCHITECTURE}) to {out_dir}")


def emit_ngram_table(out_path, n_rows, n_cols, fmt, seed):
    """FIX D Link 1: synthetic per_layer_token_embd stand-in. The real
    generation procedure ("learned from the model's embeddings + n-gram
    statistics as documented") has no procedure documented in
    docs/design-qwen-flash-next.md -- the doc records what the table IS
    (shape, byte layout, lines 662-701; the config keys at lines 108-111)
    but not how it is produced from a source checkpoint. Until upstream
    documents a procedure OR a decision memo names one here, the emitter
    below produces a synthetic stand-in with the same byte layout, which
    is what links 2 and 3 need to build against. See
    tools/synthetic_ngram_table.py for the layout and its refusal rules."""
    from synthetic_ngram_table import emit_synthetic_ngram_table
    written, blocks_per_row = emit_synthetic_ngram_table(
        out_path, n_rows=n_rows, n_cols=n_cols, fmt=fmt, seed=seed)
    print(f"emitted {written} bytes to {out_path} "
          f"({n_rows} rows x {blocks_per_row} blocks x "
          f"{written // n_rows // blocks_per_row} bytes/block, fmt={fmt})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT,
                     help="HF repo id (default: the pinned Flash-Next checkpoint)")
    ap.add_argument("--out", default="/models/ov/qwen38-flash-next")
    ap.add_argument("--emit-ngram-synthetic", action="store_true",
                    help="Emit a synthetic per_layer_token_embd table to --out "
                    "and exit; skips the transformers-support / optimum-intel "
                    "checks (this path does not need either). See "
                    "tools/synthetic_ngram_table.py for the layout.")
    ap.add_argument("--n-rows", type=int, default=1024,
                    help="Rows in the synthetic table (default: 1024).")
    ap.add_argument("--n-cols", type=int, default=160,
                    help="Row width, must be a multiple of 32 "
                    "(default: 160, the design doc's own row width).")
    ap.add_argument("--fmt", choices=("q4_0", "q4_1", "q8_0"), default="q4_0",
                    help="Quantisation format (default: q4_0, the shipped GGUF's).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Deterministic seed (default: 0).")
    args = ap.parse_args(argv)

    if args.emit_ngram_synthetic:
        emit_ngram_table(args.out, args.n_rows, args.n_cols, args.fmt, args.seed)
        return 0

    try:
        version = check_transformers_support()
    except AssertionError as exc:
        print(f"REFUSED (transformers): {exc}", file=sys.stderr)
        return 1

    print(f"transformers {version} carries {MODELING_MODULE}")

    try:
        oi_version, tf_version = check_optimum_intel_compat()
    except AssertionError as exc:
        print(f"REFUSED (optimum-intel): {exc}", file=sys.stderr)
        return 1

    print(f"optimum-intel {oi_version} + transformers {tf_version} "
          f"-- attempting the export")
    export(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
