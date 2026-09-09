#!/usr/bin/env python3
"""Export Qwen/Qwen3.8-Flash-Next (model_type qwen4_exp) to OpenVINO IR.

Unlike export_mtp.py / export_dflash.py, this is not a hand-reconstructed
head: qwen4_exp is a full 125B-total MoE backbone (config delta recorded in
docs/design-qwen-flash-next.md, FIX B), so hand-writing the graph is not the
plan. The plan is the ordinary path every other served architecture in this
repo's allowlist went through -- optimum-intel's OVModelForCausalLM export,
which needs transformers to recognise the architecture first.

It does not, as of this writing: qwen4_exp / Qwen4ExpForConditionalGeneration
has no modeling module in the pinned transformers. That is upstream's gap,
not arcint's, and this script's whole job while the gap stands is to refuse
loudly and specifically, naming exactly what is missing and what version was
checked, so the refusal is legible to a script (tools/watch_flash_next_export.py)
and to a person without reading a traceback.

    python3 tools/export_flash_next.py --out /models/ov/qwen38-flash-next

Today this exits 1 with a named-reason assertion. When upstream lands
qwen4_exp, the same invocation should proceed straight to the optimum-intel
export below -- if it needs more than a version bump at that point (a new
config key optimum-intel's OpenVINO config class does not map yet, say),
that is a small, ordinary follow-up, not a rewrite of this file.
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
    "here until transformers ships it"
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


def export(checkpoint, out_dir):
    """The actual export, reached only once check_transformers_support()
    returns cleanly. Same shape as any other allowlisted-architecture
    export in this repo's chain: optimum-intel builds the OpenVINO IR
    straight from the HF checkpoint, no hand-written graph."""
    from optimum.intel import OVModelForCausalLM

    model = OVModelForCausalLM.from_pretrained(
        checkpoint,
        export=True,
        trust_remote_code=False,
    )
    model.save_pretrained(out_dir)
    print(f"exported {checkpoint} ({ARCHITECTURE}) to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT,
                     help="HF repo id (default: the pinned Flash-Next checkpoint)")
    ap.add_argument("--out", default="/models/ov/qwen38-flash-next")
    args = ap.parse_args()

    try:
        version = check_transformers_support()
    except AssertionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"transformers {version} carries {MODELING_MODULE} -- "
          f"attempting the optimum-intel export")
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
