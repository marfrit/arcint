#!/usr/bin/env python3
"""Generate FIX D Link 3 row_ids() test vectors from the FreeToken reference.

Emits ``tests/ngram_row_ids_vectors.h`` -- the deterministic ground-truth
vectors that ``tests/test_ngram_row_ids.cpp`` checks arcint's C++ ``row_ids``
against. Re-run this only when the reference is present; the emitted header is
committed and is what the test actually reads, so the C++ test has no runtime
dependency on the reference checkout.

Provenance and licence (see docs/research-freetoken.md "Code-side ground truth"
and HANDOFF-0.5.0.local.md):

  - The reference repo FlashML-org/FreeToken is Apache-2.0. Apache-2.0 would
    permit copying its ``tests/models/qwen4_exp/test_ple.py`` vectors verbatim
    under the attribution terms, but that carries a permanent notice obligation
    into this public repository. arcint therefore does NOT copy the reference
    test vectors.
  - Instead this generator INVOKES the reference's own deterministic
    hash-constant code -- ``derive_ngram_hash_constants`` (pure integer
    arithmetic, torch-free) -- from the pinned checkout, and TRANSCRIBES the
    documented row-index mixing (ple.py NGramEmbedding.row_ids /
    _shift_ignore_eos, commit 505477ab). The resulting integers are computed
    facts, not copied expression.

The reference package cannot be imported directly (its ``__init__`` chain drags
in torch, absent on the build host). This script slices ONLY
the two pure, torch-free regions of ple.py -- the module hash constants and the
four pure functions ``_splitmix64`` / ``_is_prime`` / ``_nth_prime_after`` /
``derive_ngram_hash_constants`` -- and exec's them in an isolated namespace. It
asserts the sliced text contains neither ``torch`` nor ``Protocol`` before
running it, so a reference edit that moved torch into that region would fail
loudly rather than import silently.

Usage:
    python3 tools/gen_ngram_vectors.py [--ref-ple PATH] [--out PATH] [--check]

  --ref-ple  path to the reference ple.py (default: $FREETOKEN_REF_PLE, else a
             sibling ``FreeToken-ref`` checkout next to this repository)
  --out      output header (default: tests/ngram_row_ids_vectors.h)
  --check    regenerate into a temp buffer and diff against the committed
             header; exit non-zero if they differ (drift guard, no reference
             checkout writes).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Tuple

REF_COMMIT = "505477ab4429579e552adedd165f3cd6dbd40200"
MASK64 = (1 << 64) - 1


def _default_ref_ple() -> str:
    env = os.environ.get("FREETOKEN_REF_PLE")
    if env:
        return env
    # A sibling checkout of the reference repo next to this repository's root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(repo_root), "FreeToken-ref", "python",
                        "freetoken", "models", "qwen4_exp", "ple.py")


def load_reference_derive(ref_ple: str):
    """Return the reference's own ``derive_ngram_hash_constants``, torch-free.

    Slices the two pure regions of ple.py and exec's them in isolation. Raises
    if the file is missing or if a pure region has grown a torch dependency.
    """
    if not os.path.isfile(ref_ple):
        raise SystemExit(
            f"reference ple.py not found at {ref_ple}\n"
            f"  set --ref-ple or FREETOKEN_REF_PLE to the pinned checkout "
            f"(commit {REF_COMMIT}); the committed header stands until then."
        )
    text = open(ref_ple, encoding="utf-8").read()
    consts = text[
        text.index("_MASK64 = ") : text.index("_PLE_LAYER_PRIME = 10007")
        + len("_PLE_LAYER_PRIME = 10007")
    ]
    funcs = text[
        text.index("def _splitmix64") : text.index("@dataclass\nclass PLEMetadata")
    ]
    region = consts + "\n\n\n" + funcs
    for name in ("_splitmix64", "_is_prime", "_nth_prime_after",
                 "derive_ngram_hash_constants"):
        if name not in region:
            raise SystemExit(f"reference pure region is missing {name}; ple.py changed shape")
    if "torch" in region or "Protocol" in region:
        raise SystemExit("reference pure region now references torch/Protocol; refusing to exec")
    ns = {"math": math, "List": List, "Tuple": Tuple}
    exec(compile(region, "<freetoken-ref-pure>", "exec"), ns)  # noqa: S102 - pinned, guarded slice
    return ns["derive_ngram_hash_constants"]


def _s64(x: int) -> int:
    """Interpret the low 64 bits of ``x`` as a signed int64 (two's complement)."""
    x &= MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def shift_ignore_eos(packed: List[int], ngram_size: int, eos: int) -> List[List[int]]:
    """Transcription of ple.py NGramEmbedding._shift_ignore_eos for one request.

    ``out[s][p]`` is the token ``s`` positions left of ``p``, or ``eos`` when
    the window would cross a boundary (an eos between them) or run off the
    start. ``out[0]`` is ``packed`` itself.
    """
    width = len(packed)
    # prev_eos[p] = largest index j < p with packed[j] == eos, else -1
    prev_eos = [-1] * width
    last = -1
    for p in range(width):
        prev_eos[p] = last
        if packed[p] == eos:
            last = p
    in_segment = [p - prev_eos[p] - 1 for p in range(width)]
    shifted = [list(packed)]
    for s in range(1, ngram_size):
        row = []
        for p in range(width):
            src = p - s
            if src >= 0 and in_segment[p] >= s:
                row.append(packed[src])
            else:
                row.append(eos)
        shifted.append(row)
    return shifted


def row_ids(
    multipliers: List[int],
    sizes: List[int],
    offsets: List[int],
    ngram_size: int,
    heads_per_ngram: int,
    eos: int,
    context: List[int],
    tokens: List[int],
) -> List[List[int]]:
    """Transcription of ple.py NGramEmbedding.row_ids for one request.

    ``context`` is the ``ngram_size - 1`` tokens immediately before ``tokens[0]``
    (all ``eos`` for a fresh sequence). Returns ``[len(tokens)][num_heads]``
    int64 global row ids. The int64 wrap and floor-mod match torch exactly (see
    docs/research-freetoken.md); the C++ row_ids reproduces this byte for byte.
    """
    ctx_len = ngram_size - 1
    assert len(context) == ctx_len, (ctx_len, context)
    packed = list(context) + list(tokens)
    shifted = shift_ignore_eos(packed, ngram_size, eos)
    # select this forward's tokens (columns ctx_len ..)
    per_shift = [[row[ctx_len + i] for i in range(len(tokens))] for row in shifted]
    out: List[List[int]] = []
    for i in range(len(tokens)):
        heads: List[int] = []
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * heads_per_ngram
            mixed = _s64(per_shift[0][i] * multipliers[0])
            for position in range(1, ngram):
                term = _s64(per_shift[position][i] * multipliers[position])
                mixed = _s64((mixed & MASK64) ^ (term & MASK64))
            for h in range(start, start + heads_per_ngram):
                r = mixed % sizes[h]  # python % is floor mod; matches torch.remainder for size>0
                heads.append(r + offsets[h])
        out.append(heads)
    return out


# --- fixtures ---------------------------------------------------------------
# Each fixture pins one geometry + input. ple_layer_index selects which PLE
# layer's constants (base_seed = seed + 10007*index; per-head prime index uses
# index*num_heads + head), so different indices exercise different multipliers,
# vocab sizes and offsets.
FIXTURES = [
    dict(name="basic_2gram_only", vocab_size=257, ngram_size=2, heads_per_ngram=3,
         ngram_vocab_size_base=17, ple_layer_index=1, eos=0,
         context=[5], tokens=[11, 12, 13, 14]),
    dict(name="trigram_no_eos", vocab_size=257, ngram_size=3, heads_per_ngram=2,
         ngram_vocab_size_base=17, ple_layer_index=1, eos=0,
         context=[7, 9], tokens=[21, 22, 23, 24, 25]),
    dict(name="trigram_eos_midseq", vocab_size=257, ngram_size=3, heads_per_ngram=2,
         ngram_vocab_size_base=17, ple_layer_index=2, eos=0,
         context=[3, 4], tokens=[31, 0, 33, 34]),  # eos at position 1 breaks windows
    dict(name="trigram_fresh_context", vocab_size=257, ngram_size=3, heads_per_ngram=2,
         ngram_vocab_size_base=17, ple_layer_index=0, eos=0,
         context=[0, 0], tokens=[41, 42, 43]),      # fresh: all-eos context
    dict(name="decode_single_token", vocab_size=257, ngram_size=3, heads_per_ngram=4,
         ngram_vocab_size_base=17, ple_layer_index=1, eos=0,
         context=[51, 52], tokens=[53]),            # L=1 (decode path)
    # Realistic geometry: Qwen3.8's 16 heads (8x2-gram + 8x3-gram), a
    # vocab-scale hash space, a realistic eos id, and token ids at the vocab
    # ceiling. NOTE: `mixed` is provably non-negative here and everywhere --
    # the reference's `half_bound` bounds every token*multiplier strictly below
    # 2**63, so each product has bit 63 clear and the XOR keeps it clear. The
    # C++ floor-mod-for-negative branch is therefore defensive, never taken
    # under spec; no reference-derived vector can exercise it.
    dict(name="qwen_scale_16head", vocab_size=151936, ngram_size=3, heads_per_ngram=8,
         ngram_vocab_size_base=8209, ple_layer_index=3, eos=151643,
         context=[151935, 151934], tokens=[151935, 100000, 1]),
]


def build_fixtures(derive):
    built = []
    for fx in FIXTURES:
        num_heads = (fx["ngram_size"] - 1) * fx["heads_per_ngram"]
        mult, sizes, offsets = derive(
            vocab_size=fx["vocab_size"],
            ngram_size=fx["ngram_size"],
            num_ngram_heads=num_heads,
            ngram_vocab_size_base=fx["ngram_vocab_size_base"],
            ple_layer_index=fx["ple_layer_index"],
        )
        rows = row_ids(mult, sizes, offsets, fx["ngram_size"], fx["heads_per_ngram"],
                       fx["eos"], fx["context"], fx["tokens"])
        total_rows = offsets[-1] + sizes[-1]
        for r in rows:
            for v in r:
                assert 0 <= v < total_rows, (fx["name"], v, total_rows)
        built.append(dict(fx=fx, num_heads=num_heads, mult=mult, sizes=sizes,
                          offsets=offsets, rows=rows, total_rows=total_rows))
    return built


def _i64_list(name: str, xs: List[int]) -> str:
    return "    {%s}, // %s" % (", ".join(str(x) for x in xs), name)


def render_header(built) -> str:
    L = []
    A = L.append
    A("// GENERATED by tools/gen_ngram_vectors.py -- DO NOT EDIT BY HAND.")
    A("//")
    A("// FIX D Link 3 (docs/design-qwen-flash-next.md): ground-truth row_ids()")
    A("// vectors for tests/test_ngram_row_ids.cpp. The per-head hash constants")
    A("// (multipliers / vocab sizes / offsets) are produced by the FreeToken")
    A("// reference's own derive_ngram_hash_constants (Apache-2.0, pinned commit")
    A("// %s); the expected row ids are the" % REF_COMMIT)
    A("// documented XOR-multiply mixing transcribed in the generator. Neither")
    A("// is copied from the reference test suite (see the generator's header).")
    A("//")
    A("// Regenerate: python3 tools/gen_ngram_vectors.py   (needs the pinned")
    A("// reference checkout; --check diffs without writing).")
    A("#pragma once")
    A("#include <cstdint>")
    A("#include <vector>")
    A("")
    A("namespace lgc::ngram::test_vectors {")
    A("")
    A("struct RowIdsVector {")
    A("    const char*          name;")
    A("    int                  ngram_size;")
    A("    int                  heads_per_ngram;")
    A("    int64_t              eos_token_id;")
    A("    std::vector<int64_t> layer_multipliers;       // [ngram_size]")
    A("    std::vector<int64_t> ngram_heads_vocab_sizes;  // [num_ngram_heads]")
    A("    std::vector<int64_t> ngram_heads_offsets;      // [num_ngram_heads]")
    A("    std::vector<int64_t> context;                  // [ngram_size-1]")
    A("    std::vector<int64_t> tokens;                   // [T]")
    A("    int64_t              total_rows;               // global row-space size")
    A("    std::vector<int64_t> expected_row_ids;         // row-major [T * num_ngram_heads]")
    A("};")
    A("")
    A("inline std::vector<RowIdsVector> row_ids_vectors() {")
    A("    return {")
    for b in built:
        fx = b["fx"]
        flat = [v for r in b["rows"] for v in r]
        A("        RowIdsVector{")
        A('            "%s",' % fx["name"])
        A("            %d, %d, %d," % (fx["ngram_size"], fx["heads_per_ngram"], fx["eos"]))
        A("            {%s}," % ", ".join(str(x) for x in b["mult"]))
        A("            {%s}," % ", ".join(str(x) for x in b["sizes"]))
        A("            {%s}," % ", ".join(str(x) for x in b["offsets"]))
        A("            {%s}," % ", ".join(str(x) for x in fx["context"]))
        A("            {%s}," % ", ".join(str(x) for x in fx["tokens"]))
        A("            %d," % b["total_rows"])
        A("            {%s}," % ", ".join(str(x) for x in flat))
        A("        },")
    A("    };")
    A("}")
    A("")
    A("}  // namespace lgc::ngram::test_vectors")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-ple", default=_default_ref_ple())
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "ngram_row_ids_vectors.h"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    derive = load_reference_derive(args.ref_ple)
    built = build_fixtures(derive)
    header = render_header(built)

    if args.check:
        if not os.path.isfile(args.out):
            print("FAIL: committed header %s missing" % args.out, file=sys.stderr)
            return 1
        current = open(args.out, encoding="utf-8").read()
        if current != header:
            print("FAIL: regenerated header differs from committed %s" % args.out, file=sys.stderr)
            return 1
        print("OK: committed header matches the reference-derived vectors")
        return 0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header)
    n = sum(len(b["rows"]) for b in built)
    print("wrote %s: %d fixtures, %d token rows" % (args.out, len(built), n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
