"""Device selection and compile configuration shared by the q4e parity suites.

WHY THIS FILE EXISTS (carry-forward closed 2026-09-12). Six suites had each
rolled their own `_device_params()` and every one of them called
`core.compile_model(model, device)` with NO config. On the Intel GPU plugin that
is not neutral:

    OV 2026.4.0-22849-71640275d29, dirac
    GPU INFERENCE_PRECISION_HINT default = <Type: 'float16'>
    CPU INFERENCE_PRECISION_HINT default = <Type: 'float32'>

So every GPU leg was silently executing the graph in FLOAT16 while being judged
against an f64 reference at an f32 floor. f16 carries ~3 decimal digits; the
gates in these suites live at 1e-7. That is sufficient on its own to fail every
GPU parity column, independently of any plugin defect, and it is the first thing
to rule out before blaming a transform or a kernel. RECONCILE's 2026-09-11
GPU.1 window ("ALL 22 GPU.1 legs fail") predates this being wired.

`compile_for` therefore pins f32 on any GPU device and leaves CPU alone (it is
already f32, and passing the hint there would only add noise to the diff). The
value is the string "f32": measured against the installed plugin, "f32" and
`ov.Type.f32` are both accepted and "float32" is REJECTED --
    Wrong value float32 for property key INFERENCE_PRECISION_HINT.
    Supported values: bf16, f16, f32, undefined
-- so the string form that matches the plugin's own supported-values list is the
one used here.

This does NOT claim the GPU legs pass once the hint is set. It claims they were
never being asked the question the assertions assumed. What the hint changes is
measured per leg, on the card, and recorded in RECONCILE; nothing here predicts
it.
"""
import os

# The one place the precision decision is written down.
GPU_INFERENCE_PRECISION = "f32"


def device_params():
    """CPU always, plus whatever Q4E_GPU names as a comma list. The GPU legs are
    opt-in because loading a graph next to a resident service fails with
    allocation errors at best: a card has to be free first."""
    devs = ["CPU"]
    extra = os.environ.get("Q4E_GPU", "").strip()
    if extra:
        devs += [d for d in (s.strip() for s in extra.split(",")) if d]
    return devs


def compile_config(device):
    """The per-device compile config. GPU gets f32 pinned; CPU gets nothing."""
    if str(device).upper().startswith("GPU"):
        return {"INFERENCE_PRECISION_HINT": GPU_INFERENCE_PRECISION}
    return {}


def compile_for(core, model, device):
    """compile_model with this suite's device config. Use everywhere instead of
    a bare compile_model, so a GPU leg cannot quietly run in f16 again."""
    return core.compile_model(model, device, compile_config(device))


def effective_precision(compiled):
    """What the compiled model actually reports, for pasting next to a number.
    A leg that prints this cannot claim f32 while running f16."""
    try:
        return str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception as exc:                      # pragma: no cover
        return f"<unavailable: {exc}>"
