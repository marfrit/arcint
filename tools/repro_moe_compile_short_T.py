"""MINIMAL REPRODUCER: the emitted MoE block SEGFAULTS the OpenVINO plugin at
two short sequence lengths.

FOUND 2026-09-12, on the dev host, while adding the CF-TIESENT degenerate-row
cell -- which wanted a short T so the per-row selection table would be readable,
and got a SIGSEGV instead. Every numeric cell in the q4e MoE suites runs at
T=64 or T=96, so nothing in tree could see this.

MEASURED, `ov.Core().compile_model(build_moe_model(...), "CPU")`, OV
2026.4.0-22849, E=16 k=4 H=256 I=64 Is=64:

    T =  2  4 | 6      | 7  9 10 12 14 16 20 24 32 64 96
        OK OK | SIGSEGV | OK OK OK OK OK OK OK OK OK OK OK
                 ^ and T=8, also SIGSEGV

(A shell reports that as rc 139; `subprocess.returncode` reports -11. Same
signal, and the gate below checks the return code rather than the number.)

Only T=6 and T=8 crash. The crash is in COMPILE, not inference: the child
prints "built" and dies inside `compile_model`. It is deterministic -- T=8
repeated three times, killed each time.

IT IS THE WHOLE BLOCK, not a piece. At T=8 each sub-model this repository also
emits compiles cleanly on its own:

    build_router_model        T=4  8 12 16  -> all OK
    build_shared_expert_model T=4  8 12 16  -> all OK
    build_experts_chunk_model T=4  8 12 16  -> all OK
    build_moe_model           T=4     12 16 -> OK;  T=8 -> SIGSEGV

so it is an interaction in the assembled 421-op graph (the dense 16-expert loop, the
router's TopK -> Slice -> ScatterElementsUpdate chain and the shared expert in
one model), not any one of them.

WHY IT IS TORCH-FREE. The state dict is seeded numpy, built here, and the
config is a plain object carrying the six fields `q4e.moe` reads. So this file
imports openvino and numpy and nothing else: it is a reproducer somebody can
hand to a plugin maintainer, and it is fast enough for the gate in
tests/python/test_moe_block.py to shell out to it per T.

Usage:
    python3 tools/repro_moe_compile_short_T.py <T> [device] [E] [H] [I]
    -> exit 0 on a clean compile; killed by SIGSEGV on the defect.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


class MoeConfig:
    """Exactly the fields `q4e.moe._moe_subgraph` and `_router_gate` read --
    no transformers import, so the reproducer stands alone."""

    def __init__(self, hidden_size, num_experts, num_experts_per_tok,
                 moe_intermediate_size, shared_expert_intermediate_size,
                 norm_topk_prob=True):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob


def make_state(config, seed=0):
    """A seeded random checkpoint over every key `build_moe_model` consumes.

    Scale 0.05 is the q4e MoE fixture's (test_moe_block.py:154): it keeps every
    gemv output O(1) in f32. The VALUES are irrelevant to a compile crash --
    the shapes are what the plugin sees -- but a seeded state keeps the
    reproducer byte-reproducible if the failure ever turns out to be data
    dependent after all.
    """
    rng = np.random.default_rng(seed)
    H = config.hidden_size
    E = config.num_experts
    I = config.moe_intermediate_size
    Is = config.shared_expert_intermediate_size

    def n(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    return {
        "gate.weight": n(E, H),
        "experts.gate_up_proj": n(E, 2 * I, H),
        "experts.down_proj": n(E, H, I),
        "shared_expert.gate_proj.weight": n(Is, H),
        "shared_expert.up_proj.weight": n(Is, H),
        "shared_expert.down_proj.weight": n(H, Is),
        "shared_expert_gate.weight": n(1, H),
    }


def compile_at(T, device="CPU", E=16, H=256, I=64, Is=64, k=4, verbose=True):
    """Build and compile the MoE block at sequence length `T`. Returns the
    compiled model; dies on SIGSEGV on the defect this file reproduces."""
    import openvino as ov

    from q4e.moe import build_moe_model

    config = MoeConfig(hidden_size=H, num_experts=E, num_experts_per_tok=k,
                       moe_intermediate_size=I,
                       shared_expert_intermediate_size=Is)
    model = build_moe_model(config, make_state(config), seq_len=T)
    if verbose:
        print(f"T={T} device={device} E={E} H={H} I={I}: built "
              f"({len(model.get_ordered_ops())} ops), compiling", flush=True)
    compiled = ov.Core().compile_model(model, device)
    if verbose:
        print(f"T={T} device={device}: COMPILED OK", flush=True)
    return compiled


if __name__ == "__main__":
    a = sys.argv[1:]
    T = int(a[0]) if a else 8
    device = a[1] if len(a) > 1 else "CPU"
    E = int(a[2]) if len(a) > 2 else 16
    H = int(a[3]) if len(a) > 3 else 256
    I = int(a[4]) if len(a) > 4 else 64
    compile_at(T, device=device, E=E, H=H, I=I)
