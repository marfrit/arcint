#!/usr/bin/env python3
"""KLD-vs-BF16 acceptance harness for arcint weight-compression artifacts.

The WP3.5 nibble decision (docs/design-qwen-flash-next.md) fixes the served
Flash-Next backbone template to int4_asym and the artifact pipeline to
``nncf.compress_weights(INT4_ASYM, group_size=128, all_layers=True)`` on the
built backbone IR. "Compiles and serves" is not acceptance; "quantizes and
provably did not move" is. This harness is that proof instrument: it measures
the per-token KL divergence between a BF16/FP16 reference and the quantized
candidate over a text slice and gates on a fixed threshold.

Threshold: mean per-token KL(P_ref || P_cand), in NATS, must be <= 0.0599.
The UD-Q3_K_XL class was measured PASSING at .0399 against this .0599 bar
(DECIDED FACTS, 2026-09-10); narrow experts (Flash-Next width 640) tolerate
~3-4 bit, not 2-bit.

THE INSTRUMENT MUST BE ABLE TO GO RED. An acceptance check that cannot fail
measures nothing. ``--self-test`` drives the KL core with synthetic
distributions and asserts it reads 0 when nothing moved, stays green for a
sub-threshold perturbation, and goes RED for an over-threshold one. It is
torch-free and model-free, so it runs anywhere (the build host included) and is the
red-first gate for the whole harness. ``tools/test_kld_harness.py`` wraps the
same assertions for the unit ladder.

Three stages:
  --self-test                 red probe: KL core calibration, no model
  --quantize REF_XML          NNCF INT4_ASYM g128 all_layers -> candidate IR
  --ref REF_DIR --cand CAND_DIR --text SLICE   measure KLD + gate

The model runner feeds a single full-sequence causal forward (no KV cache
plumbing) and reads the [1, T, V] logits output, so it serves a stateless
(non-paged) export. Reference may be a BF16/FP16 IR directory or, with
--ref-torch, a local HF checkpoint run through transformers as the oracle.
"""
import argparse
import json
import math
import os
import sys

THRESHOLD_NATS = 0.0599


# --------------------------------------------------------------------------
# KL core -- pure numpy, no torch, no OpenVINO. This is the measured quantity
# and the part the red probe calibrates.
# --------------------------------------------------------------------------
def _log_softmax(logits):
    """Row-wise log-softmax, numerically stable. logits: [T, V] numpy array."""
    import numpy as np

    z = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return z - logsumexp


def kl_per_token(ref_logits, cand_logits):
    """Per-token KL(P_ref || P_cand) in nats, from raw logits [T, V].

    KL = sum_v P_ref(v) * (log P_ref(v) - log P_cand(v)). P_ref is the
    reference (BF16) distribution -- the quantized candidate is scored against
    the full-precision truth, never the other way round. Returns a [T] array.
    """
    import numpy as np

    ref_logits = np.asarray(ref_logits, dtype=np.float64)
    cand_logits = np.asarray(cand_logits, dtype=np.float64)
    if ref_logits.shape != cand_logits.shape:
        raise ValueError(
            f"logit shape mismatch: ref {ref_logits.shape} vs cand "
            f"{cand_logits.shape} -- the two models must share tokenizer, "
            f"vocab and token count")
    log_p = _log_softmax(ref_logits)
    log_q = _log_softmax(cand_logits)
    p = np.exp(log_p)
    return (p * (log_p - log_q)).sum(axis=-1)


def summarize_kl(per_token):
    import numpy as np

    per_token = np.asarray(per_token, dtype=np.float64)
    return {
        "tokens": int(per_token.shape[0]),
        "mean": float(per_token.mean()),
        "max": float(per_token.max()),
        "p95": float(np.percentile(per_token, 95)),
    }


def gate(summary, threshold=THRESHOLD_NATS):
    """PASS when mean per-token KL <= threshold. Returns (passed, verdict)."""
    passed = summary["mean"] <= threshold
    verdict = "PASS" if passed else "RED"
    return passed, verdict


# --------------------------------------------------------------------------
# RED PROBE -- the instrument's own falsifiability check.
# --------------------------------------------------------------------------
def self_test():
    """Drive the KL core with synthetic distributions and assert it can both
    read zero and go red. Deterministic, torch-free, model-free.

    Three cells:
      identical    -> KL == 0            (reads zero when nothing moved)
      small drift  -> 0 < KL < threshold (green, a real but tolerable move)
      large drift  -> KL > threshold     (RED: the instrument fires)
    A harness where the 'large drift' cell does not go red is not admitted.
    """
    import numpy as np

    rng = np.random.default_rng(20260910)
    T, V = 256, 4096
    ref = rng.standard_normal((T, V)) * 4.0  # peaky-ish logits

    failures = []

    # Cell 1: identical -> exactly zero (within fp tolerance).
    s_ident = summarize_kl(kl_per_token(ref, ref.copy()))
    passed, verdict = gate(s_ident)
    print(f"  self-test identical:   mean KL {s_ident['mean']:.6e} nats -> {verdict}")
    if not (s_ident["mean"] < 1e-12 and passed):
        failures.append("identical distributions did not read zero/PASS")

    # Cell 2: a small, calibrated perturbation that stays under the bar. A tiny
    # additive logit jitter moves the distribution a little; scaled to land
    # comfortably below 0.0599.
    small = ref + rng.standard_normal((T, V)) * 0.03
    s_small = summarize_kl(kl_per_token(ref, small))
    passed_small, verdict_small = gate(s_small)
    print(f"  self-test small drift: mean KL {s_small['mean']:.6e} nats -> {verdict_small}")
    if not (0.0 < s_small["mean"] < THRESHOLD_NATS and passed_small):
        failures.append(
            f"small-drift cell not in (0, {THRESHOLD_NATS}) & PASS: "
            f"{s_small['mean']:.6e}")

    # Cell 3: THE RED PROBE. A coarse perturbation that must exceed the bar.
    large = ref + rng.standard_normal((T, V)) * 0.6
    s_large = summarize_kl(kl_per_token(ref, large))
    passed_large, verdict_large = gate(s_large)
    print(f"  self-test RED PROBE:   mean KL {s_large['mean']:.6e} nats -> {verdict_large}")
    if not (s_large["mean"] > THRESHOLD_NATS and not passed_large):
        failures.append(
            f"RED PROBE did not fire: mean KL {s_large['mean']:.6e} <= "
            f"{THRESHOLD_NATS}; the instrument cannot detect drift")

    # Cell 4: symmetry guard -- KL is directional; swapping args must change
    # the number (or the 'score the candidate against the truth' contract is
    # not actually implemented).
    k_fwd = summarize_kl(kl_per_token(ref, large))["mean"]
    k_rev = summarize_kl(kl_per_token(large, ref))["mean"]
    print(f"  self-test direction:   KL(ref||cand) {k_fwd:.6e} vs "
          f"KL(cand||ref) {k_rev:.6e}")
    if abs(k_fwd - k_rev) < 1e-9:
        failures.append("KL is symmetric here -- the directional contract is broken")

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELF-TEST PASSED: instrument reads zero, stays green under the bar, "
          "and goes RED above it.")
    return 0


# --------------------------------------------------------------------------
# NNCF quantization stage -- mirrors the WP3.5 decision exactly.
# --------------------------------------------------------------------------
def quantize_int4_asym(ref_xml, out_xml, group_size=128):
    """NNCF INT4_ASYM g128 all_layers=True, data-free RTN. Mirrors
    tools/export_dflash.py's proven call (that one is g64; WP3.5 fixes g128
    for the Flash-Next backbone). Returns the path written."""
    import nncf
    import openvino as ov

    core = ov.Core()
    core.set_property({"ENABLE_MMAP": False})  # decouple from the .bin lifetime
    model = core.read_model(ref_xml)
    print(f"compressing: mode=INT4_ASYM group_size={group_size} ratio=1.0 "
          f"all_layers=True dataset=None (data-free RTN)")
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        ratio=1.0,
        group_size=group_size,
        all_layers=True,
        dataset=None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    ov.save_model(compressed, out_xml)
    print(f"wrote {out_xml}")
    return out_xml


# --------------------------------------------------------------------------
# Model runner -- collect next-token logits over a text slice.
# --------------------------------------------------------------------------
def read_text_slice(path, max_chars):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.strip()
    if max_chars > 0:
        text = text[:max_chars]
    return text


def logits_from_ir(model_dir, token_ids, device="CPU"):
    """Single full-sequence causal forward; read [1, T, V] logits.

    For a stateless (non-paged) export: feed input_ids (and position_ids /
    attention_mask if the IR declares them) for the whole sequence in one
    infer and read the 'logits' output. No KV cache plumbing -- this is a
    measurement path, not a serving path.
    """
    import numpy as np
    import openvino as ov

    xml = os.path.join(model_dir, "openvino_language_model.xml")
    if not os.path.isfile(xml):
        # A bare .xml path was passed rather than an artifact dir.
        xml = model_dir
    core = ov.Core()
    model = core.read_model(xml)
    # INFERENCE_PRECISION_HINT f32 on any GPU device. The Intel GPU plugin
    # defaults this to float16 (measured, OV 2026.4 on both Arc cards:
    # `core.get_property("GPU", "INFERENCE_PRECISION_HINT")` -> float16, while
    # CPU reports float32), so an unconfigured GPU compile silently executes the
    # graph in f16. A KLD gate at 0.0599 nats cannot be read off an f16 forward
    # and be about the export. CPU is left alone -- it is already f32. Mirrors
    # tests/python/q4e_device.py, which carries the full note; this module is
    # product tooling and cannot import from tests/.
    cfg = ({"INFERENCE_PRECISION_HINT": "f32"}
           if str(device).upper().startswith("GPU") else {})
    compiled = core.compile_model(model, device, cfg)

    ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
    T = ids.shape[1]
    feed = {}
    for port in compiled.inputs:
        names = port.get_names()
        name = next(iter(names)) if names else ""
        if "input_ids" in name:
            feed[port] = ids
        elif "position_ids" in name:
            feed[port] = np.arange(T, dtype=np.int64).reshape(1, -1)
        elif "attention_mask" in name:
            feed[port] = np.ones((1, T), dtype=np.int64)
        elif "beam_idx" in name:
            feed[port] = np.zeros((1,), dtype=np.int32)
        else:
            # An unexpected required input (e.g. a KV/state port) means this IR
            # is not a stateless single-forward graph; refuse by name rather
            # than feed a zero tensor that silently changes the logits.
            raise RuntimeError(
                f"IR input {name!r} is not one of input_ids/position_ids/"
                f"attention_mask/beam_idx; {xml} is not a stateless forward "
                f"graph and needs the KV-cache runner, not this measurement path")
    req = compiled.create_infer_request()
    req.infer(feed)
    out = None
    for port in compiled.outputs:
        names = port.get_names()
        name = next(iter(names)) if names else ""
        if "logits" in name:
            out = req.get_tensor(port).data
            break
    if out is None:
        out = req.get_output_tensor(0).data
    logits = np.asarray(out)
    if logits.ndim == 3:
        logits = logits[0]  # [T, V]
    return logits


def logits_from_torch(checkpoint_dir, token_ids):
    """Reference oracle from a local HF checkpoint (BF16). transformers must be
    present and recognise the architecture. No network fetch."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    ids = torch.tensor(token_ids, dtype=torch.long).reshape(1, -1)
    with torch.no_grad():
        out = model(ids).logits  # [1, T, V]
    return np.asarray(out[0].float().numpy())


def tokenize(model_dir_or_ckpt, text, max_tokens):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir_or_ckpt, trust_remote_code=True)
    ids = tok(text, return_tensors=None)["input_ids"]
    if max_tokens > 0:
        ids = ids[:max_tokens]
    return ids


def measure(args):
    import numpy as np

    text = read_text_slice(args.text, args.max_chars)
    tok_source = args.tokenizer or args.cand
    token_ids = tokenize(tok_source, text, args.tokens)
    print(f"tokens: {len(token_ids)} (from {args.text})")

    if args.ref_torch:
        ref_logits = logits_from_torch(args.ref, token_ids)
    else:
        ref_logits = logits_from_ir(args.ref, token_ids, args.device)
    cand_logits = logits_from_ir(args.cand, token_ids, args.device)

    # Score next-token prediction: align logits at position t with the real
    # distribution at position t; drop the last position (no target).
    n = min(ref_logits.shape[0], cand_logits.shape[0]) - 1
    per_token = kl_per_token(ref_logits[:n], cand_logits[:n])
    summary = summarize_kl(per_token)
    passed, verdict = gate(summary, args.threshold)

    print("KLD table (nats, per-token KL(P_ref || P_cand)):")
    print(f"  tokens {summary['tokens']}  mean {summary['mean']:.4f}  "
          f"p95 {summary['p95']:.4f}  max {summary['max']:.4f}  "
          f"threshold {args.threshold}  -> {verdict}")
    print(json.dumps({"summary": summary, "threshold": args.threshold,
                      "verdict": verdict}))
    return 0 if passed else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="red probe: calibrate the KL core, no model needed")
    ap.add_argument("--quantize", metavar="REF_XML",
                    help="NNCF INT4_ASYM g128 all_layers -> --out")
    ap.add_argument("--out", help="output .xml for --quantize")
    ap.add_argument("--group-size", type=int, default=128)

    ap.add_argument("--ref", help="reference IR dir (BF16/FP16) or checkpoint dir")
    ap.add_argument("--ref-torch", action="store_true",
                    help="treat --ref as a local HF checkpoint, run via transformers")
    ap.add_argument("--cand", help="candidate IR dir (the quantized artifact)")
    ap.add_argument("--tokenizer", help="tokenizer source (default: --cand)")
    ap.add_argument("--text", help="text slice (wikitext-2 .txt)")
    ap.add_argument("--tokens", type=int, default=512, help="cap token count")
    ap.add_argument("--max-chars", type=int, default=0, help="cap chars read")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_NATS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.quantize:
        if not args.out:
            print("--quantize needs --out", file=sys.stderr)
            return 2
        quantize_int4_asym(args.quantize, args.out, args.group_size)
        return 0
    if args.ref and args.cand and args.text:
        return measure(args)
    print("nothing to do: pass --self-test, --quantize, or --ref/--cand/--text",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
