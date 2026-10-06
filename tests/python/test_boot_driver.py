"""tools/boot_serving_shape.py, device-free (REVIEW MEDIUM-2 rider, 2026-09-13):
the boot driver's own behaviours, run on the CPU plugin over the suite's
reduced geometry (`--tiny`, q4e.serving_shape.tiny_config) -- every stage the
served path has, at a width a test can afford. Never the served path; the
driver says so in its first line.

Each cell names the driver commit it covers. Red first: every cell below was
run against the driver at 5afec70 (before the six commits and before
`--tiny`): `--tiny` is an unknown option there, and the cut leg dies in
Shape([.., -1, -1, -1]) on the KV pools -- the collection is red as a whole,
the per-commit lines are red one by one once `--tiny` is grafted.

NAMED NO-TEST: b1fdb0b (the zero row takes the request tensor's own element
type) fixed a GPU-only mismatch -- under an f32 INFERENCE_PRECISION_HINT the
GPU plugin hands back an f32 state tensor for a port the driver bound as f16.
The CPU plugin keeps the port's type, so no device-free cell can reproduce
the mismatch; `test_fresh_requests_bind_state_rows_in_the_ports_own_type`
covers the invariant that replaced the hard-coded f16 (the port's type is
the request's type) and the GPU leg of window-050 §4.11.1 (L3, f32) stays the
device witness.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openvino")

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools" / "boot_serving_shape.py"
TINY = ["--tiny", "--layers", "4", "--device", "CPU", "--ids", "1,2,3,4,5"]


def run(*extra):
    env = dict(os.environ)
    env.pop("Q4E_GPU", None)
    p = subprocess.run([sys.executable, str(DRIVER), *TINY, *extra],
                       capture_output=True, text=True, timeout=600, env=env)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    return out


@pytest.fixture(scope="module")
def same_request_x3():
    return run("--repeat", "3")


def test_the_tiny_leg_is_labelled_and_runs_every_served_stage(same_request_x3):
    out = same_request_x3
    assert "TINY GEOMETRY" in out and "NOT the served path" in out
    assert "BOOT [pass] OK" in out
    assert "BOOT [compile] OK" in out
    assert "SERVED PATH: every name the C++ feeds was accepted" in out
    assert "BOOT [forward] INFER OK" in out


def test_the_state_rows_are_zeroed_before_every_forward__c836354(same_request_x3):
    zeroed = re.findall(r"BOOT \[state\] (\S+ ?#?\d?): (\d+) state table\(s\) zeroed",
                        same_request_x3)
    labels = [z[0] for z in zeroed]
    counts = {int(z[1]) for z in zeroed}
    assert labels == ["forward #1", "repeat #2", "repeat #3"], labels
    # 4 tiny layers: 3 GDN layers, each a conv table and a GDN table
    assert counts == {6}, counts


def test_every_repeat_carries_a_digest_and_the_vs_previous_reading__2ce179b(same_request_x3):
    out = same_request_x3
    assert re.search(r"BOOT \[repeat\] #1 digest [0-9a-f]{12} ", out)
    for k in (2, 3):
        m = re.search(rf"BOOT \[repeat\] #{k} \(same request\) INFER OK [\d.]+s: (.*)", out)
        assert m, f"repeat #{k} line missing"
        line = m.group(1)
        assert "BIT-IDENTICAL to #1" in line, line       # the CPU plugin is deterministic
        assert re.search(r"digest [0-9a-f]{12}; identical to the previous", line), line


def test_fresh_requests_bind_state_rows_in_the_ports_own_type_and_repeat_bit_identically():
    out = run("--repeat", "2", "--fresh")
    m = re.search(r"state rows bound: (\d+) \((f16|f32), rows=3\)", out)
    assert m and int(m.group(1)) == 6, out[-2000:]
    assert re.search(r"#2 \(fresh request\) INFER OK [\d.]+s: BIT-IDENTICAL to #1", out), out[-2000:]


def test_a_plugin_prop_reaches_the_compile_and_a_refused_key_fails_by_name__bd214a0():
    ok = run("--stage", "compile", "--plugin-prop", "NUM_STREAMS=1")
    assert "'NUM_STREAMS': '1'" in ok and "BOOT [compile] OK" in ok
    bad = run("--stage", "compile", "--plugin-prop", "NOT_A_PLUGIN_KEY=1")
    m = re.search(r"BOOT \[compile\] FAIL after .*", bad)
    assert m and "NOT_A_PLUGIN_KEY" in m.group(0), bad[-2000:]


def test_a_cut_builds_and_compiles_device_free_up_to_the_request():
    """The cut's graph half (`--cut layerN/out`: one Result on the named
    node, the rest dropped) is device-free through the compile.

    NAMED NO-TEST for cc28ce3 (KV ports left unreachable by the cut skipped
    by name) and e53d75b (zero_state skips a state port left unbound by a
    cut): both live AFTER `create_infer_request`, and the CPU plugin refuses
    the cut graph there -- measured 2026-09-13 on the dev host, this file's
    first run:

        RuntimeError: Exception from src/inference/src/cpp/compiled_model.cpp:128:
        Check 'm_element_type.is_static()' failed at src/inference/src/dev/make_tensor.cpp:58

    (an unreachable parameter of the cut model keeps an undefined element
    type; the GPU plugin creates the request and declares the port, which is
    the situation the two commits handle). No device-free form exists without
    pruning the unreachable parameters out of the cut, which would change the
    reviewed instrument; the device witness stays window-050 §4.11.1 chain 3b
    and window-051 row (b)'s cut table."""
    out = run("--stage", "compile", "--cut", "layer1/out")
    assert "LOCALISER: graph cut after 'layer1/out' (Add, [1, -1, 1024])" in out
    assert re.search(r"; \d+ ops remain; NOT the served path", out)
    assert "BOOT [compile] OK" in out
    # the unreachable KV pools are still DECLARED by the cut model, every
    # dim and the element type dynamic -- the port cc28ce3 skips on the GPU,
    # and the undefined type the CPU plugin refuses one stage later
    assert re.search(r"key_cache\.\d+\[-1, -1, -1, -1\]:dynamic", out), out[-1500:]


def test_a_pruned_cut_declares_only_the_ports_the_cut_graph_reaches():
    """`--cut-prune` (2026-09-18, campaign serving-shape-logits): the cut
    keeps only the Parameters the cut node reaches. Opt-in, so the reviewed
    default above stays what it was. Why it exists: on the full-depth artifact
    a `layer0/out` cut still DECLARED the seven `ngram_table.K` ports (the PLE
    sits at layer 1, after the cut), the driver bound the real 26.8 GiB table
    for a graph that never reads it, the container swapped and the next tiny
    device allocation failed with `[GPU] out of GPU resources` (measured on
    the dev host, `~/wp/logs/cut48b/l0.log`). At `layer0/out` the reachable
    ports are the embedding, the positions, the mask and layer 0's own state
    rows; the tables, the KV pools and every later layer's state are gone.
    Red first: `--cut-prune` unknown."""
    out = run("--stage", "compile", "--cut", "layer0/out", "--cut-prune")
    m = re.search(r"LOCALISER: graph cut after 'layer0/out' .*; (\d+) unreachable "
                  r"parameter\(s\) pruned, (\d+) kept", out)
    assert m and int(m.group(1)) > 0, out[-2000:]
    ports = re.search(r"BOOT \[compile\] ports: (.*)", out)
    assert ports, out[-2000:]
    names = re.findall(r"([A-Za-z_][\w.]*)\[", ports.group(1))   # shapes carry ", " too
    assert "inputs_embeds" in names and "conv_mask" in names, names
    assert not [n for n in names if n.startswith(("ngram_table.", "key_cache.", "value_cache."))], names
    assert len(names) == int(m.group(2)), (names, m.group(2))
    assert "BOOT [compile] OK" in out, out[-2000:]


@pytest.mark.xfail(strict=True, reason=(
    "since the emitter matches the tiled MoE pattern (2026-09-17) the CPU "
    "plugin's own ConvertTiledMoeBlockToGatherMatmuls fires on the ported "
    "build too (it accepts any weight producer) and its GatherMatmul refuses "
    "Parameter weights: 'Only constant weights are supported for GatherMatmul "
    "operation'. The ported route is dead since window-051 B.3; the cell is "
    "kept as the record of what it proved and of why it no longer runs."))
def test_expert_ports_bind_every_body_and_the_forward_repeats_bit_identically():
    """`--expert-ports` (the segmented route's expert bodies as u8 PORTS with
    the in-graph unpack): every body the sink declares is bound by name --
    3 kinds x 4 MoE layers at the tiny geometry -- and the forward runs and
    repeats bit-identically on the CPU plugin. What stays device-only: the
    USM-host sharing and the unpack's device memory (GPU_MEMORY_STATISTICS
    lines), which the card leg reads. Red first: `--expert-ports` unknown."""
    out = run("--repeat", "2", "--expert-ports")
    m = re.search(r"bodies bound as PORTS: (\d+) u8 port\(s\), ([\d,]+) B", out)
    assert m and int(m.group(1)) == 12, out[-2000:]
    assert "bytes ZEROS; the unpack runs in-graph" in out
    assert "declared by the sink, not by the compiled model" not in out
    assert "BOOT [forward] INFER OK" in out
    assert re.search(r"#2 \(same request\) INFER OK [\d.]+s: BIT-IDENTICAL to #1", out), out[-2000:]


def test_the_rewrite_flag_is_idempotent_on_a_fixed_build_and_the_census_prints():
    """`--rewrite-tiled-moe` on the fixed emitter's build rewrites nothing and
    the walker matches every MoE layer; `--census` prints the runtime graph's
    primitive types (on the CPU plugin no MoE-typed primitive exists, and the
    line must say so rather than stay silent -- that silence is the defect
    class the flag was written for)."""
    out = run("--stage", "compile", "--rewrite-tiled-moe", "--census")
    m = re.search(r"BOOT \[rewrite\] tiled MoE blocks rewritten (\d+) \(swish (\d+), "
                  r"chains (\d+)\); walker matched (\d+)", out)
    assert m and tuple(int(g) for g in m.groups()) == (0, 0, 0, 4), out[-2000:]
    assert "BOOT [compile] OK" in out, out[-2000:]
    # the moe-typed field is `NONE` or a dict repr (with spaces) on a device
    # where the MoE fuses; match either, assert the CPU plugin's NONE
    c = re.search(r"BOOT \[census\] exec nodes (\d+); moe-typed (.+?); top \[", out)
    assert c and int(c.group(1)) > 0, out[-2000:]
    assert c.group(2).strip() == "NONE", c.group(2)
