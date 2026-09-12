"""THE REPACK: the block path from a GGUF expert row to a plugin slot, and the
coordinates at which bit-identity is not available on THIS artifact.

The frontier's ruling for 0.5.0 is that the serving route must be
BLOCK-CARRYING: the shipped quantisation is transported, not re-encoded, so
that no second quantisation is stacked on the file's own (THE FILL measured
that second quantisation at 17.6% relative -- RECONCILE T4, and the Mixpert
lesson before it). The named transport was `src/core/gguf_repack.cpp`.

This module is the cell the ruling's premise dies in. It does not argue about
the route; it reads the two ends and reports whether the shipped bytes can
enter it. They cannot, and there are two independent reasons in the C++ and two
more above it:

  C1  TYPE. `repack_supported` (gguf_repack.cpp:260) admits ggml types
      8 / 12 / 13 / 14 -- Q8_0, Q4_K, Q5_K, Q6_K. The shipped UD-Q3_K_XL puts
      its expert bodies in IQ3_XXS (18), IQ4_NL (20) and IQ4_XS (23), with a
      Q8_0 tail of five. 139 of 144 bodies are refused on type alone, by name,
      with `not a repacked type`.

  C2  RANK. `repack_tensor` refuses anything that is not 2-D
      (gguf_repack.cpp:266). Every expert body is rank 3 -- [in, out, E], the
      expert axis. That closes the remaining 5 as well: 0 of 144 bodies can
      enter `repack_tensor` at all, whatever their type.

  C3  LOSS. Even where it is admitted, the repack is a TRANSCODE, not a byte
      carry: gguf_repack.h's own "What is exact and what is not" records that a
      K-quant group scale `d*sc` rounds to f16, and `repack_bound_steps`
      (gguf_repack.cpp:486) exists to bound a deviation that is not zero. The
      one type whose stored form it carries exactly is Q8_0, whose integers and
      f16 scales are the block's own. "ZERO added loss" is a property of Q8_0
      here, not of the path.

  C4  REPRESENTATION. The IR chain the GPU plugin's MoE fusion matches is
      uniform affine -- Convert -> Subtract(zero_point) -> Multiply(scale) --
      over a Constant declared u4/i4/u8/i8. Within one block that is
      `(q - zp) * s` over `q in 0..15`: sixteen EQUALLY SPACED levels. A
      codebook is not an affine dequant.

      WHICH EVIDENCE CLOSES WHICH BODIES (rewritten under H3, REVIEW f8229d8,
      after C4 was found citing two lines that do not carry it):

        IQ3_XXS  94 bodies  28.11 GiB  docs/design-gguf-native.md:52, which
                                       names IQ3_XXS in the codebook-and-sign-
                                       table set. The citation reaches.
        IQ4_XS    2 bodies   0.83 GiB  same line, same list. Reaches.
        IQ4_NL   43 bodies  18.90 GiB  MEASURED, not cited to prose -- see
                                       `test_iq4nl_levels_are_non_uniform...`
                                       below. The levels recovered from the
                                       shipped bytes are
                                       [-127,-104,...,89,113], spacings 11..24
                                       (2.18x), best affine fit off by 0.847
                                       of a step.

      WHAT C4 DOES NOT DO, stated because it was previously implied. C4 was
      advertised as the coordinate that "sits ABOVE gguf_repack.cpp and would
      survive its repair". That is true for all three groups NOW that the
      IQ4_NL half is measured -- but it was NOT established by the citations
      originally given, which were `design-gguf-native.md:52` (a sub-4-bit list
      that does not contain IQ4_NL; the string does not occur in that document
      at all) and `serving_shape.py:138` (which says IQ4_NL has no OpenVINO
      ELEMENT TYPE -- a different proposition). The 43 IQ4_NL bodies were, and
      remain, closed independently by C1 (type refused) and C2 (rank refused);
      the verdict never depended on C4 for them, and this cell does not pad
      C4's weight to make it look as though it did.

WHAT THIS MODULE DELIBERATELY DOES NOT ASSERT. Not that a block-carrying route
is impossible -- only that `gguf_repack.cpp` is not one for this artifact.
C4 closes the affine chain; it says nothing about a Gather over a 16-entry
codebook, which would reproduce IQ4_NL exactly and is not attempted here
(it would leave the fused MoE path, which is a design decision and not a
plumbing one). The day any of these four moves, a cell here goes red and asks
for the route to be re-derived rather than letting a stale ruling stand.

Device-free and shard-free for C1/C2/C3: the C++ side is PARSED from its own
source (clause 2 -- the defining source generates the set, nobody recites it)
and the shipped side is a DATED HOST CENSUS the suite cannot regenerate
without the shards, in the manner of the other host-census constants in this
suite. With `Q4E_GGUF_SHARDS` set, the census leg REGENERATES the table off
the real file and fails on any drift, so the dated constant cannot rot
silently.

Run:
    pytest tests/python/test_repack_route.py -q -s                  # C1-C3
    Q4E_GGUF_SHARDS=<shard-dir> pytest tests/python/test_repack_route.py -q -s
"""
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_GGUF_REPACK_CPP = REPO_ROOT / "src" / "core" / "gguf_repack.cpp"
_GGUF_REPACK_H = REPO_ROOT / "src" / "core" / "gguf_repack.h"

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()
_skip = pytest.mark.skipif(
    not _SHARDS, reason="Q4E_GGUF_SHARDS unset (real GGUF shards absent)")

# ggml type codes, for reporting only -- the numbers are what the C++ compares.
_GGML_TYPE_NAMES = {8: "Q8_0", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
                    18: "IQ3_XXS", 20: "IQ4_NL", 23: "IQ4_XS"}

# DATED HOST CENSUS, 2026-09-12, dev host, /flash-model, the shipped
# Qwen3.8-Flash-Next-UD-Q3_K_XL 3-shard set. (kind, type name, ggml code) ->
# (bodies, bytes on disk). Read off the GGUF tensor headers with gguf-py, not
# estimated. The suite cannot regenerate this without the shards; the
# shard-gated cell below does regenerate it and fails on drift.
SHIPPED_EXPERT_BODIES = {
    ("ffn_down_exps", "IQ4_NL", 20): (43, 20_289_945_600),
    ("ffn_down_exps", "Q8_0", 8): (5, 4_456_448_000),
    ("ffn_gate_exps", "IQ3_XXS", 18): (47, 15_092_940_800),
    ("ffn_gate_exps", "IQ4_XS", 23): (1, 445_644_800),
    ("ffn_up_exps", "IQ3_XXS", 18): (47, 15_092_940_800),
    ("ffn_up_exps", "IQ4_XS", 23): (1, 445_644_800),
}
# Every expert body in the shipped file is rank 3 -- [in, out, E]. Same census.
SHIPPED_EXPERT_RANK = 3
# 48 backbone layers x {gate, up, down}. The census is reconciled to this
# rather than to its own sum, so a shard set that is missing a layer is a red.
MOE_LAYERS = 48
EXPERT_KINDS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def _repack_supported_types():
    """The ggml type codes `repack_supported` admits, parsed from its own body.

    Clause 2 applied to a type set: the C++ one-liner is the definition, so the
    test reads it rather than carrying a copy that can drift out of agreement
    with the function it describes.
    """
    text = _GGUF_REPACK_CPP.read_text()
    hits = [ln for ln in text.splitlines()
            if ln.startswith("bool repack_supported(")]
    assert len(hits) == 1, (
        f"`repack_supported` is defined {len(hits)} times in "
        f"{_GGUF_REPACK_CPP.name}; this parser needs exactly one definition to "
        f"read the admitted type set out of.")
    body = hits[0].split("return", 1)[1]
    codes = frozenset(int(n) for n in re.findall(r"==\s*(\d+)", body))
    assert codes, (
        f"no `t == <code>` comparisons found in `repack_supported`: "
        f"{hits[0].strip()!r}. The function was rewritten into a shape this "
        f"parser cannot read -- update the parser, do not assume the set.")
    return codes


def _repack_rank_guard_line():
    """The line number of `repack_tensor`'s 2-D refusal, resolved by anchor."""
    anchor = 'if (t.dims.size() != 2) throw std::runtime_error('
    hits = [i for i, ln in enumerate(_GGUF_REPACK_CPP.read_text().splitlines(), 1)
            if anchor in ln]
    assert len(hits) == 1, (
        f"the rank guard anchor {anchor!r} matches {len(hits)} lines of "
        f"{_GGUF_REPACK_CPP.name}. If the guard was removed, C2 has moved and "
        f"the repack route must be re-derived, not assumed still shut.")
    return hits[0]


def _by_type():
    """bodies and bytes per ggml type code, summed over the expert kinds."""
    out = {}
    for (_kind, _name, code), (n, b) in SHIPPED_EXPERT_BODIES.items():
        prev_n, prev_b = out.get(code, (0, 0))
        out[code] = (prev_n + n, prev_b + b)
    return out


def test_the_census_reconciles_to_the_model_geometry():
    """The census's own arithmetic, before anything is concluded from it.

    Counts generated from the rows, checked against the geometry the rest of
    the program uses (48 MoE layers x 3 expert bodies), so a census taken over
    a partial shard set cannot pass as a complete one.
    """
    total = sum(n for n, _ in SHIPPED_EXPERT_BODIES.values())
    total_bytes = sum(b for _, b in SHIPPED_EXPERT_BODIES.values())
    per_kind = {k: sum(n for (kind, _, _), (n, _) in SHIPPED_EXPERT_BODIES.items()
                       if kind == k) for k in EXPERT_KINDS}

    print(f"\n[repack-census] {total} expert bodies, "
          f"{total_bytes:,} B = {total_bytes / 2**30:.2f} GiB on disk")
    for k in EXPERT_KINDS:
        print(f"    {k:16s} {per_kind[k]:3d} bodies")
    for (kind, name, code), (n, b) in sorted(SHIPPED_EXPERT_BODIES.items()):
        print(f"    {kind:16s} {name:8s} (ggml {code:2d})  {n:3d} bodies  "
              f"{b:>15,} B")

    assert total == MOE_LAYERS * len(EXPERT_KINDS), (
        f"the census holds {total} expert bodies; {MOE_LAYERS} MoE layers x "
        f"{len(EXPERT_KINDS)} kinds is {MOE_LAYERS * len(EXPERT_KINDS)}. A "
        f"census over an incomplete shard set must not be read as the "
        f"model's.")
    for k in EXPERT_KINDS:
        assert per_kind[k] == MOE_LAYERS, (
            f"{k} appears {per_kind[k]} times, not once per each of the "
            f"{MOE_LAYERS} MoE layers.")


def test_no_shipped_expert_body_is_a_type_the_repack_admits():
    """C1. The type gate, both sides read from their own source.

    This is the first half of why THE REPACK cannot be emitted as specified.
    The count is generated by intersecting the parsed C++ set with the census,
    never written down.
    """
    supported = _repack_supported_types()
    census = _by_type()
    admitted = {c: census[c] for c in sorted(census) if c in supported}
    refused = {c: census[c] for c in sorted(census) if c not in supported}
    n_admitted = sum(n for n, _ in admitted.values())
    n_refused = sum(n for n, _ in refused.values())

    print(f"\n[repack-type] repack_supported admits ggml "
          f"{sorted(supported)} = "
          f"{', '.join(_GGML_TYPE_NAMES.get(c, str(c)) for c in sorted(supported))}"
          f"  (gguf_repack.cpp:260)")
    for c in sorted(census):
        n, b = census[c]
        verdict = "ADMITTED by type" if c in supported else "REFUSED by type"
        print(f"    ggml {c:2d} {_GGML_TYPE_NAMES.get(c, '?'):8s} {n:3d} bodies "
              f"{b / 2**30:6.2f} GiB  -> {verdict}")
    print(f"[repack-type] {n_refused} of {n_refused + n_admitted} expert "
          f"bodies are refused on type alone")

    assert n_refused > 0, (
        "every shipped expert body is now a type `repack_supported` admits. "
        "C1 has moved: the type half of the block is gone and THE REPACK's "
        "route must be re-derived against the artifact, not assumed shut.")
    assert set(refused) == {18, 20, 23}, (
        f"the refused set is {sorted(refused)}, not the IQ3_XXS / IQ4_NL / "
        f"IQ4_XS trio this module's argument was written against. The shipped "
        f"artifact's quantisation changed; re-read the route.")
    assert set(admitted) <= {8}, (
        f"types {sorted(set(admitted) - {8})} are now admitted by type. Only "
        f"the Q8_0 tail was, and only C2 closed it -- check C2 still holds "
        f"before concluding anything.")


def test_the_rank_guard_closes_what_the_type_gate_admits():
    """C2. Every expert body is rank 3; `repack_tensor` takes 2-D only.

    The half that makes the verdict total rather than partial: without this,
    five Q8_0 bodies would enter. The guard's line is resolved by anchor so the
    citation in the design note cannot drift (the `cite()` discipline).
    """
    supported = _repack_supported_types()
    census = _by_type()
    by_type_admitted = sum(n for c, (n, _) in census.items() if c in supported)
    guard = _repack_rank_guard_line()

    print(f"\n[repack-rank] every shipped expert body is rank "
          f"{SHIPPED_EXPERT_RANK} ([in, out, E]); `repack_tensor` refuses "
          f"anything but rank 2 at gguf_repack.cpp:{guard}")
    print(f"[repack-rank] admitted by type: {by_type_admitted}  ->  admitted "
          f"after the rank guard: 0 of "
          f"{sum(n for n, _ in census.values())}")

    assert SHIPPED_EXPERT_RANK != 2, (
        "the expert bodies are rank 2 in this census, so the rank guard no "
        "longer closes the Q8_0 tail. Re-derive C2.")
    assert guard > 0


def test_the_repack_is_a_transcode_and_says_so_in_its_own_header():
    """C3. "Bit-identical" is not what this path claims for itself.

    The ruling's phrase was "ZERO added loss". The transport's own header
    records an f16 rounding of the K-quant group scale, and a bound function
    exists to police a deviation that is therefore not zero. Asserted against
    the header text so that a future header that DID claim exactness would
    have to change this cell deliberately.
    """
    header = _GGUF_REPACK_H.read_text()
    cpp = _GGUF_REPACK_CPP.read_text()
    rounds = "rounds to f16" in header
    has_bound = "double repack_bound_steps(int32_t ggml_type) {" in cpp
    q8_exact = "The stored integers and Q8_0's scales are" in header

    print(f"\n[repack-loss] gguf_repack.h states the K-quant group scale "
          f"'rounds to f16': {rounds}")
    print(f"[repack-loss] repack_bound_steps exists to bound the deviation: "
          f"{has_bound}")
    print(f"[repack-loss] the header's exactness claim is scoped to Q8_0's own "
          f"stored integers and scales: {q8_exact}")

    assert rounds and has_bound, (
        "gguf_repack no longer describes itself as rounding, or no longer "
        "carries a deviation bound. If the transcode became a byte carry, C3 "
        "has moved and the design note's third coordinate is stale.")
    assert q8_exact


@_skip
def test_the_shipped_expert_census_regenerates_off_the_shards():
    """The dated census, re-read from the artifact. LEG that stops it rotting.

    Everything above is arithmetic over a table typed into this file. This cell
    is the one that says the table is still true of the file on disk, and it
    fails on any drift -- a different quantisation mix, a different layer
    count, a different rank.
    """
    import glob

    from gguf import GGUFReader

    fresh = {}
    ranks = set()
    for path in sorted(glob.glob(os.path.join(_SHARDS, "*.gguf"))):
        reader = GGUFReader(path, "r")
        for t in reader.tensors:
            if not t.name.endswith("_exps.weight"):
                continue
            key = (t.name.split(".")[-2], t.tensor_type.name, int(t.tensor_type))
            n, b = fresh.get(key, (0, 0))
            fresh[key] = (n + 1, b + int(t.n_bytes))
            ranks.add(len(t.shape))
        del reader

    print(f"\n[repack-census/shards] regenerated from {_SHARDS}")
    for key in sorted(fresh):
        print(f"    {key}  {fresh[key]}")
    print(f"[repack-census/shards] ranks seen: {sorted(ranks)}")

    assert fresh == SHIPPED_EXPERT_BODIES, (
        "the shipped expert census has drifted from the dated table in this "
        "file.\n  in tree: "
        + repr(sorted(SHIPPED_EXPERT_BODIES.items()))
        + "\n  on disk: " + repr(sorted(fresh.items()))
        + "\nUpdate the table AND re-read the four coordinates in this "
          "module's header: a changed quantisation mix can open or close the "
          "repack route.")
    assert ranks == {SHIPPED_EXPERT_RANK}, (
        f"expert bodies are rank {sorted(ranks)} on disk, not "
        f"{SHIPPED_EXPERT_RANK}. C2 is derived from this.")


@_skip
def test_iq4nl_levels_are_non_uniform_so_no_affine_chain_carries_them():
    """C4 for the 43 IQ4_NL bodies, MEASURED off the shipped bytes.

    H3 (REVIEW f8229d8). This cell exists because C4's IQ4_NL half used to be
    cited to `design-gguf-native.md`, whose codebook sentence lists the
    SUB-4-BIT set and does not mention IQ4_NL at all (the string does not occur
    in that document) -- and to `serving_shape.py:138`, which says IQ4_NL has
    no OpenVINO ELEMENT TYPE, a different proposition from "non-uniform
    codebook". The claim was true and its citations did not reach it, on the
    second-largest group of bodies in the file.

    So it is measured here instead of cited to prose. The affine chain the GPU
    fusion matches is `Convert -> Subtract(zp) -> Multiply(scale)`: within one
    block that is `(q - zp) * s` over `q in 0..15`, i.e. SIXTEEN EQUALLY SPACED
    levels, whatever `zp` and `s` are. If IQ4_NL's sixteen levels are not
    equally spaced, no choice of the two reproduces them, and the "carry it
    exactly" route is shut for these bodies independently of `gguf_repack.cpp`.

    The levels are recovered from the file's OWN bytes -- the f16 `d` read out
    of each 18-byte block and the dequantised values divided by it -- and the
    nibble order is DERIVED by consistency rather than assumed: the mapping is
    accepted only if every code resolves to exactly one ratio under it.
    """
    import glob

    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    tensor = None
    for path in sorted(glob.glob(os.path.join(_SHARDS, "*.gguf"))):
        reader = GGUFReader(path, "r")
        tensor = next((t for t in reader.tensors
                       if t.name.endswith("ffn_down_exps.weight")
                       and t.tensor_type.name == "IQ4_NL"), None)
        if tensor is not None:
            break
    assert tensor is not None, "no IQ4_NL expert body found in the shards"

    nblocks = 200_000
    raw = tensor.data.view(np.uint8).reshape(-1)[:nblocks * 18].copy()
    blocks = raw.reshape(nblocks, 18)
    scale = blocks[:, :2].copy().view(np.float16).astype(np.float64).reshape(-1)
    lo = (blocks[:, 2:] & 0xF).astype(np.int64)
    hi = (blocks[:, 2:] >> 4).astype(np.int64)
    values = dequantize(raw, tensor.tensor_type).astype(np.float64).reshape(nblocks, 32)

    live = scale != 0
    ratios = values[live] / scale[live][:, None]
    orders = {"lo-first": np.concatenate([lo, hi], axis=1),
              "interleaved": np.stack([lo, hi], axis=2).reshape(nblocks, 32)}
    levels = None
    for name, order in orders.items():
        table, consistent = {}, True
        for code in range(16):
            picked = np.unique(np.round(ratios[order[live] == code], 6))
            if picked.size != 1:
                consistent = False
                break
            table[code] = float(picked[0])
        if consistent:
            levels = [table[c] for c in range(16)]
            print(f"\n[iq4nl] nibble order resolved by consistency: {name} "
                  f"(every code -> exactly one value/scale over "
                  f"{nblocks:,} blocks)")
            break
    assert levels is not None, (
        "no candidate nibble order maps every code to a single value/scale. "
        "The block layout assumed here (2 B f16 scale + 32 nibbles) does not "
        "hold for this tensor -- do not conclude anything about the levels.")

    spacing = np.diff(levels)
    fit = np.stack([np.arange(16, dtype=np.float64), np.ones(16)], axis=1)
    coef, *_ = np.linalg.lstsq(fit, np.array(levels), rcond=None)
    residual = np.abs(np.array(levels) - fit @ coef).max()
    steps = residual / abs(coef[0])

    print(f"[iq4nl] levels (value/scale): {[int(v) for v in levels]}")
    print(f"[iq4nl] consecutive spacings: {[int(s) for s in spacing]}")
    print(f"[iq4nl] spacing min {spacing.min():g} max {spacing.max():g} "
          f"-> {spacing.max() / spacing.min():.2f}x")
    print(f"[iq4nl] best affine fit value = {coef[0]:.4f}*q {coef[1]:+.4f}; "
          f"max residual {residual:.4f} scale units = {steps:.3f} AFFINE STEPS")

    assert spacing.min() > 0, "the levels are not monotonic; the recovery is wrong"
    assert spacing.max() > 2 * spacing.min(), (
        f"the IQ4_NL levels are near-uniform at this artifact (spacing "
        f"{spacing.min():g}..{spacing.max():g}). C4's IQ4_NL half rests on "
        f"their NON-uniformity -- if they are uniform, an affine chain may "
        f"carry them and the route must be re-derived for these 43 bodies.")
    assert steps > 0.5, (
        f"the best affine fit over the 16 codes is within {steps:.3f} of a "
        f"step, so an affine `(q - zp) * scale` chain approximates IQ4_NL "
        f"better than this cell's argument assumes. The argument is that no "
        f"affine chain carries it EXACTLY; a small residual here would make "
        f"that true but uninteresting, and C4's weight for these bodies "
        f"should be restated rather than left standing on a near-miss.")


def test_the_type_parser_detects_a_widened_repack_supported():
    """The gate's own red, in tree and permanent.

    The C1 cell's whole force is that the parsed set and the census do not
    intersect. A parser that silently returned the same answer for a rewritten
    `repack_supported` would make that agreement meaningless, so the parse is
    exercised against mutated source here rather than trusted.
    """
    line = ("bool repack_supported(int32_t t) { return t == 8 || t == 12 || "
            "t == 13 || t == 14 || t == 18; }")
    body = line.split("return", 1)[1]
    widened = frozenset(int(n) for n in re.findall(r"==\s*(\d+)", body))
    assert widened == {8, 12, 13, 14, 18}, widened
    census = _by_type()
    assert any(c in widened for c in census), (
        "a `repack_supported` widened to IQ3_XXS must intersect the census; "
        "if it does not, the two sides are not being compared at all.")
    # and the real one must NOT contain it, or the cell above is vacuous
    assert 18 not in _repack_supported_types(), (
        "the tree's own `repack_supported` now admits IQ3_XXS (18). C1 has "
        "moved -- re-derive the route.")
