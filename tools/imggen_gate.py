#!/usr/bin/env python3
"""The image gate: a candidate's latents and image against the CPU reference.

Phase 1 of `docs/milestone-1.1.0.md` (charter §3). There are no logits, so there
is no KL; the answer-level equivalent is the image, but comparing the final
image alone does not attribute a failure to a kernel, because denoising
amplifies small errors over the steps.

Reference: sd.cpp on the CPU (F32/BF16) with the same file, seed, steps and
sampler as the candidate. For a fixed seed the noise is made on the CPU and is
the same on every backend (`code`, milestone §0.2; to be measured in Phase 0).

Per step: cosine similarity and MSE between the candidate's and the reference's
previews. sd.cpp writes a `PREVIEW_PROJ` preview per step -- the latent
projected to RGB, the closest the CLI comes to the latent itself -- so this is
a latent proxy, not the raw F32 latent. At the end: PSNR and a global SSIM on
the decoded image.

Comparative, as the release speed gate (`tests/acceptance/ab.py`): the
candidate is at most TOL worse than the baseline arm (the unchanged build on
the same card in the same window), never merely "close to" an absolute number.
A candidate is a pass when every per-step cosine is >= 1-TOL_COS and every
MSE <= ref MSE * (1+TOL_MSE), and the final PSNR is >= baseline PSNR - PSNR_TOL
and SSIM >= baseline SSIM - SSIM_TOL. The numbers are the operator's (charter
D6); the defaults below are placeholders and are printed with every result.

Exit status: 0 inside the tolerance; 1 otherwise; 2 a usage or decode error.

    imggen_gate.py run --sdcpp BIN --out DIR --diffusion F --llm F --vae F
                       --device NAME|cpu --prompt P [--steps 8 --seed 42 ...]
    imggen_gate.py compare --ref DIR --cand DIR [--baseline DIR]

The `run` arm writes `final.png` and `prev_%03d.png`, `run.log` (sd-cli's
output, the step times at `--log-level info`) and `arm.json` (the binary's
sha256 and every flag), so a later session can re-read what was run.
`compare` refuses arms whose `arm.json` is missing or whose image-defining
flags (`PARITY`) differ, unless `--allow-unverified` is given.

With `--baseline`, the baseline arm is measured against the same reference and
the candidate may be at most the tolerance worse than it: per-step MSE, final
PSNR and SSIM. Without one (a model the unchanged build cannot run), the final
image is held to the model's PSNR/SSIM floor (`MODEL_TOL`), since the per-step
previews never pass through the VAE. A missing final image or a preview step
present on one side only is a failure.

Only the standard library is used: the card host may have no numpy, PIL or
ffmpeg, and the gate must not need them.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zlib

# Placeholder tolerances -- the operator sets these (charter D6).
TOL_COS = 0.001      # per-step cosine may fall this much below 1.0
TOL_MSE = 0.05       # per-step MSE may grow this fraction on the reference's
PSNR_TOL = 1.0       # dB below the baseline arm's final PSNR
SSIM_TOL = 0.01      # below the baseline arm's final SSIM

# Per-model tolerances: the operator's D6 (2026-10-18): per-model TOL_COS, about
# 0.003 for the 1.1.1 models (Qwen-Image, FLUX.2 klein), whose CPU-vs-OpenCL
# per-step gaps measured 0.0013-0.0019; Z-Image Turbo keeps 0.001, its gap
# measured 0.00007 (1.1.0). Keyed by the diffusion model's file name -- arm.json's
# --diffusion-model basename, the key check_parity already compares. A model
# not listed keeps the placeholders above; an explicit --tol-cos / --psnr-floor
# / --ssim-floor overrides the table.
#
# psnr_floor / ssim_floor gate the final image against the CPU reference and
# apply only when there is no baseline arm: the per-step previews are the
# latent projected to RGB, so a defect in the VAE alone (0059's red) leaves
# every preview unchanged and the per-step cosine cannot see it. With a
# baseline arm the comparative PSNR_TOL / SSIM_TOL rule applies instead.
# "measured" is the evidence the values rest on; a size not listed there is
# printed as unmeasured.
MODEL_TOL = {
    # 1.1.0 C1: 1024² cos worst 0.999926, 42.59 dB, SSIM 0.999257; 512²
    # 0.999987, 46.54 dB. Reds: 0057 512² cos 0.890391 / 9.70 dB, 0058 1024²
    # 0.987219 / 19.56 dB, 512² 0.995502.
    "z_image_turbo-Q8_0.gguf": {
        "tol_cos": 0.001, "psnr_floor": 30.0, "ssim_floor": 0.99,
        "measured": ("1024x1024", "512x512")},
    # 1.1.1 §9: 512²/8 cos worst 0.998676, 26.26 dB, SSIM 0.9912. Red: 0059
    # previews cos 1.000000, final 8.70 dB / SSIM 0.0105 (vs the green arm).
    "Qwen_Image-Q4_K_M.gguf": {
        "tol_cos": 0.003, "psnr_floor": 20.0, "ssim_floor": 0.95,
        "measured": ("512x512",)},
    # 1.1.1 §9: 512²/4 text-to-image cos worst 0.998115, 25.31 dB (SSIM not
    # recorded); edit cos worst 0.999539, 32.43 dB, SSIM 0.9966. No red on
    # this model yet.
    "flux2-klein-4b-Q8_0.gguf": {
        "tol_cos": 0.003, "psnr_floor": 20.0, "ssim_floor": 0.95,
        "measured": ("512x512",)},
}


def model_tolerances(canddir):
    """(model file, size, entry or None) of the candidate arm, from arm.json."""
    try:
        cmd = arm_header(canddir).get("cmd") or []
    except ValueError:
        return None, None, None
    def flag(k):
        return cmd[len(cmd) - 1 - cmd[::-1].index(k) + 1] if k in cmd else None
    model = flag("--diffusion-model")
    model = os.path.basename(model) if model else None
    size = "%sx%s" % (flag("-W"), flag("-H")) if flag("-W") else None
    return model, size, MODEL_TOL.get(model)


# --- a PNG reader for exactly what sd.cpp writes (8-bit, non-interlaced) ---

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


_FILTERS = {0: lambda a, b, c: 0,
            1: lambda a, b, c: a,
            2: lambda a, b, c: b,
            3: lambda a, b, c: (a + b) >> 1,
            4: _paeth}


def read_png(path):
    """Return (width, height, channels, bytes) for an 8-bit non-interlaced PNG."""
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s: not a PNG" % path)
    pos = 8
    width = height = channels = None
    idat = bytearray()
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if typ == b"IHDR":
            width, height, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", body)
            if depth != 8 or inter != 0:
                raise ValueError("%s: depth %d interlace %d unsupported" % (path, depth, inter))
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(height * stride)
    prev = bytearray(stride)
    ip = 0
    for y in range(height):
        ft = raw[ip]
        ip += 1
        row = bytearray(raw[ip:ip + stride])
        ip += stride
        fn = _FILTERS[ft]
        for x in range(stride):
            a = row[x - channels] if x >= channels else 0
            b = prev[x]
            c = prev[x - channels] if x >= channels else 0
            row[x] = (row[x] + fn(a, b, c)) & 0xff
        out[y * stride:(y + 1) * stride] = row
        prev = row
    return width, height, channels, bytes(out)


# --- metrics ---

def cosine_mse(pa, pb):
    a = read_png(pa)
    b = read_png(pb)
    if a[:3] != b[:3]:
        raise ValueError("shape mismatch %s %s" % (pa, pb))
    n = len(a[3])
    sa = sb = sab = 0.0
    d2 = 0.0
    for i in range(n):
        x, y = a[3][i], b[3][i]
        sa += x * x
        sb += y * y
        sab += x * y
        d = x - y
        d2 += d * d
    cos = sab / math.sqrt(sa * sb) if sa and sb else 0.0
    return cos, d2 / n


def psnr_mse(pa, pb):
    a = read_png(pa)
    b = read_png(pb)
    if a[:3] != b[:3]:
        raise ValueError("shape mismatch %s %s" % (pa, pb))
    n = len(a[3])
    d2 = 0.0
    for i in range(n):
        d = a[3][i] - b[3][i]
        d2 += d * d
    mse = d2 / n
    return (float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)), mse


def ssim_global(pa, pb):
    """A single-window SSIM over the whole image; a cheap global companion to
    PSNR, not the 8x8-window form. Reads as ints, ignores channels' RGB mixing."""
    a = read_png(pa)
    b = read_png(pb)
    n = len(a[3])
    mx = my = 0.0
    for i in range(n):
        mx += a[3][i]
        my += b[3][i]
    mx /= n
    my /= n
    vx = vy = cov = 0.0
    for i in range(n):
        dx = a[3][i] - mx
        dy = b[3][i] - my
        vx += dx * dx
        vy += dy * dy
        cov += dx * dy
    vx /= n - 1 if n > 1 else 1
    vy /= n - 1 if n > 1 else 1
    cov /= n - 1 if n > 1 else 1
    L = 255.0
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# --- the arms ---

def read_dir(d):
    prev = {}
    for fn in sorted(os.listdir(d)):
        if fn.startswith("prev_") and fn.endswith(".png"):
            idx = int(fn[5:8])
            prev[idx] = os.path.join(d, fn)
    final = os.path.join(d, "final.png")
    if not os.path.exists(final):
        # accept the sd.cpp default name too, and the run arm's own name
        for cand in ("result.png", "output.png"):
            if os.path.exists(os.path.join(d, cand)):
                final = os.path.join(d, cand)
                break
    return prev, (final if os.path.exists(final) else None)


def per_step(ref, cand):
    common = sorted(set(ref) & set(cand))
    idx_missing = sorted(set(ref) ^ set(cand))
    rows = []
    for i in common:
        cos, mse = cosine_mse(ref[i], cand[i])
        rows.append((i, cos, mse))
    return rows, idx_missing


# The flags that make two arms the same image: a comparison across arms that
# differ in any of them measures the inputs, not the backend (a no-prompt arm
# against a prompted one was once read as a defect, docs/imggen.md).
PARITY = ("--diffusion-model", "--llm", "--vae", "-p", "--cfg-scale", "--steps",
          "-W", "-H", "--seed", "--sampling-method", "--scheduler",
          "--flow-shift", "--ref-image", "--mask", "--upscale-model",
          "--upscale-repeats", "--init-img", "--strength")
MODEL_FLAGS = ("--diffusion-model", "--llm", "--vae")


def arm_header(d):
    try:
        return json.load(open(os.path.join(d, "arm.json")))
    except (OSError, ValueError):
        raise ValueError("%s has no readable arm.json: its parameters are unverifiable" % d)


def arm_params(d):
    cmd = arm_header(d).get("cmd") or []
    # The last occurrence: sd-cli's parser keeps the last value, so a flag
    # repeated through --extra-args is what the arm actually ran.
    p = {k: (cmd[len(cmd) - cmd[::-1].index(k)] if k in cmd else None) for k in PARITY}
    # A model is its file, not its directory: the reference may read the
    # mirror and the card arm a sha-checked copy on another disk.
    for k in MODEL_FLAGS:
        if p[k]:
            p[k] = os.path.basename(p[k])
    return p


def arm_mechanism(d):
    """What may differ between arms: the binary, the other flags, the environment."""
    h = arm_header(d)
    cmd = h.get("cmd") or []
    rest, i = [], 1
    while i < len(cmd):
        if cmd[i] in PARITY or cmd[i] in ("--preview-path", "-o"):
            i += 2
            continue
        rest.append(cmd[i])
        i += 1
    return {"sha256_16": h.get("sha256_16"), "flags": " ".join(rest), "env": h.get("env", "(not recorded)")}


def check_parity(dirs):
    params = [(d, arm_params(d)) for d in dirs]
    d0, p0 = params[0]
    for d, p in params[1:]:
        diff = [k for k in PARITY if p[k] != p0[k]]
        if diff:
            raise ValueError("%s and %s differ in %s: not the same image" % (
                d0, d, ", ".join("%s (%r vs %r)" % (k, p0[k], p[k]) for k in diff)))
    # Everything else is the mechanism under test: printed, so a reader sees
    # what separated the arms next to the verdict.
    for d in dirs:
        print("  arm %s: %s" % (d, json.dumps(arm_mechanism(d))))


def compare_ref(refdir, canddir, basedir=None, tol_cos=None, tol_mse=TOL_MSE,
                psnr_tol=PSNR_TOL, ssim_tol=SSIM_TOL, allow_unverified=False,
                psnr_floor=None, ssim_floor=None):
    if not allow_unverified:
        check_parity([refdir, canddir] + ([basedir] if basedir else []))
    model, size, entry = model_tolerances(canddir)
    entry = entry or {}
    if tol_cos is None:
        tol_cos = entry.get("tol_cos", TOL_COS)
    if psnr_floor is None:
        psnr_floor = entry.get("psnr_floor")
    if ssim_floor is None:
        ssim_floor = entry.get("ssim_floor")
    ref_prev, ref_final = read_dir(refdir)
    cand_prev, cand_final = read_dir(canddir)
    rows, miss = per_step(ref_prev, cand_prev)
    if not rows:
        raise ValueError("no common preview steps between %s and %s" % (refdir, canddir))
    mean_cos = sum(r[1] for r in rows) / len(rows)
    worst_cos = min(r[1] for r in rows)
    worst_mse = max(r[2] for r in rows)
    failures = []
    for i, cos, mse in rows:
        if cos < 1.0 - tol_cos:
            failures.append("step %d cosine %.6f < %.6f" % (i, cos, 1.0 - tol_cos))
    if miss:
        # An arm that stopped early (or crashed after sampling) is not a pass.
        failures.append("preview step(s) only on one side: %s" % miss)

    final = {}
    if ref_final and cand_final:
        ps, pmse = psnr_mse(ref_final, cand_final)
        final["psnr_ref"] = ps
        final["mse_ref"] = pmse
        final["ssim_ref"] = ssim_global(ref_final, cand_final)
    else:
        # A missing final image is a failed decode (0059 off crashes in the
        # VAE after every preview was written), never an ungated pass.
        failures.append("final image missing: ref %s, cand %s" % (ref_final, cand_final))
    if not basedir and final:
        if psnr_floor is not None and final["psnr_ref"] < psnr_floor:
            failures.append("final PSNR %.2f < floor %.2f (%s, no baseline arm)" %
                            (final["psnr_ref"], psnr_floor, model))
        if ssim_floor is not None and final["ssim_ref"] < ssim_floor:
            failures.append("final SSIM %.4f < floor %.4f (%s, no baseline arm)" %
                            (final["ssim_ref"], ssim_floor, model))

    baseline = None
    if basedir:
        # The baseline arm's own distance from the reference is the allowance:
        # the candidate may be at most the tolerance worse than it (charter §3).
        bprev, bfinal = read_dir(basedir)
        baseline = {}
        brows, _ = per_step(ref_prev, bprev)
        if brows:
            baseline["step_cosine_worst"] = min(r[1] for r in brows)
            baseline["step_mse_worst"] = max(r[2] for r in brows)
            if worst_mse > baseline["step_mse_worst"] * (1 + tol_mse):
                failures.append("step MSE %.4f > baseline %.4f * (1+%.3f)" %
                                (worst_mse, baseline["step_mse_worst"], tol_mse))
        if bfinal and ref_final:
            baseline["psnr_ref"], _ = psnr_mse(ref_final, bfinal)
            baseline["ssim_ref"] = ssim_global(ref_final, bfinal)
        if bfinal and cand_final:
            baseline["psnr_cand_vs_baseline"], _ = psnr_mse(bfinal, cand_final)
        if baseline.get("psnr_ref") is not None and final.get("psnr_ref") is not None:
            if final["psnr_ref"] < baseline["psnr_ref"] - psnr_tol:
                failures.append("final PSNR %.2f < baseline %.2f - %.2f" %
                                (final["psnr_ref"], baseline["psnr_ref"], psnr_tol))
            if final["ssim_ref"] < baseline["ssim_ref"] - ssim_tol:
                failures.append("final SSIM %.4f < baseline %.4f - %.4f" %
                                (final["ssim_ref"], baseline["ssim_ref"], ssim_tol))

    print("ACCEPTANCE-METRIC step_cosine_mean %.6f" % mean_cos)
    print("ACCEPTANCE-METRIC step_cosine_worst %.6f" % worst_cos)
    print("ACCEPTANCE-METRIC step_mse_worst %.6f" % worst_mse)
    if final:
        print("ACCEPTANCE-METRIC psnr_ref %.4f" % final["psnr_ref"])
        print("ACCEPTANCE-METRIC ssim_ref %.6f" % final["ssim_ref"])
    for i, cos, mse in rows:
        print("  step %-3d cos %.6f mse %.6f" % (i, cos, mse))
    if miss:
        print("  step(s) only on one side: %s" % miss)
    if basedir:
        print("  baseline %s: %s" % (basedir, json.dumps({k: round(v, 6) for k, v in baseline.items()})))
    elif psnr_floor is not None or ssim_floor is not None:
        print("  no baseline arm: cosine per step, final PSNR/SSIM against the model's floor")
    else:
        print("  no baseline arm and no floor for %s: cosine is gated; MSE/PSNR/SSIM are"
              " reported, not gated -- a VAE-only defect cannot fail this verdict" % model)
    tol = {"tol_cos": tol_cos, "tol_mse": tol_mse, "psnr_tol": psnr_tol, "ssim_tol": ssim_tol,
           "psnr_floor": psnr_floor, "ssim_floor": ssim_floor}
    print("  tolerances (operator, D6) for %s at %s: %s" % (model, size, json.dumps(tol)))
    if entry and size not in entry.get("measured", ()):
        print("  WARNING: %s's tolerances were measured at %s, not %s" %
              (model, "/".join(entry["measured"]), size))
    if failures:
        for f in failures:
            print("  FAIL %s" % f)
        return 1
    print("  PASS")
    return 0


# --- run arm (thin: sd-cli is the engine) ---

def run_arm(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    cmd = [a.sdcpp,
           "--diffusion-model", a.diffusion,
           "--llm", a.llm,
           "--vae", a.vae,
           "-p", a.prompt,
           "--cfg-scale", str(a.cfg_scale),
           "--steps", str(a.steps),
           "-W", str(a.width), "-H", str(a.height),
           "--seed", str(a.seed),
           "--sampling-method", a.sampling_method,
           "--backend", a.device,
           "--preview", "proj",
           "--preview-interval", str(a.preview_interval),
           "--preview-path", os.path.join(out, "prev_%03d.png"),
           "-o", os.path.join(out, "final.png"),
           "--disable-image-metadata",
           "--log-level", a.log_level]
    if a.scheduler:
        cmd += ["--scheduler", a.scheduler]
    if a.flow_shift is not None:
        cmd += ["--flow-shift", str(a.flow_shift)]
    if a.ref_image:
        cmd += ["--ref-image", a.ref_image]
    if a.mask:
        cmd += ["--mask", a.mask]
    if a.upscale_model:
        cmd += ["--upscale-model", a.upscale_model]
        if a.upscale_repeats:
            cmd += ["--upscale-repeats", str(a.upscale_repeats)]
    if a.extra_args:
        cmd += a.extra_args
    sha = hashlib.sha256(open(a.sdcpp, "rb").read()).hexdigest()[:16] if os.path.isfile(a.sdcpp) else "?"
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GGML_", "SD_"))}
    header = {"sdcpp": a.sdcpp, "sha256_16": sha, "cmd": cmd, "env": env,
              "diffusion_sha256": a.diffusion_sha, "llm_sha256": a.llm_sha, "vae_sha256": a.vae_sha}
    print("run: %s" % " ".join(cmd))
    json.dump(header, open(os.path.join(out, "arm.json"), "w"), indent=2)
    with open(os.path.join(out, "run.log"), "wb") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--sdcpp", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--diffusion", required=True)
    r.add_argument("--llm", required=True)
    r.add_argument("--vae", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--prompt", required=True)
    r.add_argument("--steps", type=int, default=8)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--cfg-scale", type=float, default=1.0)
    r.add_argument("-W", "--width", type=int, default=512)
    r.add_argument("-H", "--height", type=int, default=512)
    r.add_argument("--sampling-method", default="euler")
    r.add_argument("--scheduler")  # omitted: sd.cpp's model-specific default
    r.add_argument("--flow-shift", type=float, default=None)  # Qwen/VL flow shift
    r.add_argument("--ref-image", default=None)               # edits: the source image
    r.add_argument("--mask", default=None)                    # edits: the mask
    r.add_argument("--upscale-model", default=None)           # ESRGAN
    r.add_argument("--upscale-repeats", type=int, default=0)
    r.add_argument("--preview-interval", type=int, default=1)
    r.add_argument("--diffusion-sha", default="")
    r.add_argument("--llm-sha", default="")
    r.add_argument("--vae-sha", default="")
    r.add_argument("--log-level", default="info")  # info carries the step times
    r.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    r.set_defaults(func=run_arm)

    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--baseline")
    c.add_argument("--tol-cos", type=float, default=None,
                   help="per-step cosine tolerance (default: MODEL_TOL, else %g)" % TOL_COS)
    c.add_argument("--tol-mse", type=float, default=TOL_MSE)
    c.add_argument("--psnr-tol", type=float, default=PSNR_TOL)
    c.add_argument("--ssim-tol", type=float, default=SSIM_TOL)
    c.add_argument("--psnr-floor", type=float, default=None,
                   help="final PSNR floor vs the reference without a baseline (default: MODEL_TOL)")
    c.add_argument("--ssim-floor", type=float, default=None,
                   help="final SSIM floor vs the reference without a baseline (default: MODEL_TOL)")
    c.add_argument("--allow-unverified", action="store_true",
                   help="compare arms without arm.json or with differing flags (reported, never a gate)")
    c.set_defaults(func=lambda a: compare_ref(a.ref, a.cand, a.baseline,
                                             a.tol_cos, a.tol_mse, a.psnr_tol, a.ssim_tol,
                                             a.allow_unverified, a.psnr_floor, a.ssim_floor))

    a = ap.parse_args()
    try:
        return a.func(a)
    except ValueError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
