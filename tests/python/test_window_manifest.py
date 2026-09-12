"""CF-MANIFESTSHA: a `RUN` marker in the window manifest must name its commit.

WHY THIS FILE EXISTS (REVIEW e78812d, finding F7, 2026-09-12):

    "`docs/window-050.md` records 'CPU-only control, for the same tree (`RUN`,
    2026-09-12): 101 passed with shards, 33 passed / 59 skipped device-free'.
    At the tip those are 112 and 42/70. Both figures are true of their own
    trees and a reader cannot tell them apart -- which is the b929924 gap one
    level up, in the tracked operating document. The manifest already knows the
    rule; it applies it to the reserved coherence row ('in the same commit as
    the measurement') but not to its own RUN markers."

The manifest is the document a window operator executes. A number in it that
cannot be tied to a tree is the same defect as a measurement read off an
unproved staging directory, one level up, and it is the more dangerous of the
two because it outlives the session that made it.

THE GATE, three cells:

  1. Every `RUN` marker in the manifest body carries an id: `RUN@<sha>`,
     `RUN@wt+<sha>` (a working tree at <sha> with uncommitted deltas) or
     `RUN@unrecorded` (provenance never recorded and no longer establishable).
     The header, where the marker vocabulary is defined, is excluded by
     construction -- it is scanned separately for the definitions themselves.
  2. Every `<sha>` an id names resolves to a real commit in this repository,
     when the check runs inside a git work tree. A staged tree has no `.git`,
     so that leg SKIPS rather than failing there -- and says which.
  3. `RUN@unrecorded` is a RATCHET. The count may go down and never up. A new
     measurement whose tree was not recorded is not a thing this repository
     accepts, and a ceiling is the only form of that rule a test can enforce
     without knowing which markers are new.

Device-free, no shards, no card: it reads one markdown file.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "window-050.md"

# The ratchet. 0 as of the commit that introduced this cell: every marker in
# the manifest was traced to a tree (the GPU-night tables to the working tree
# at 2e99661, from the window timestamps against `git log --format=%cI`).
# LOWER THIS when a marker's provenance is recovered. NEVER RAISE IT.
MAX_UNRECORDED = 0

_ID = r"(?:[0-9a-f]{7,40}|wt\+[0-9a-f]{7,40}|unrecorded)"
_MARKER = re.compile(r"RUN(@" + _ID + r")?")


def _split_header(text):
    """The marker vocabulary lives above the first `---` rule; the manifest
    proper is everything after it. Splitting means the legend can spell the
    word `RUN` without the gate mistaking a definition for a claim."""
    parts = text.split("\n---\n", 1)
    assert len(parts) == 2, "the manifest has no header rule; the split is wrong"
    return parts[0], parts[1]


def _body_markers():
    header, body = _split_header(MANIFEST.read_text())
    offset = len(header) + len("\n---\n")
    out = []
    for m in _MARKER.finditer(body):
        line = body.count("\n", 0, m.start()) + 1
        out.append((m.group(0), line, body[max(0, m.start() - 60):m.start() + 40]))
    return out, offset


def test_the_manifest_exists_and_is_scanned():
    """A scan of nothing passes every other cell in this file vacuously."""
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    markers, _ = _body_markers()
    assert len(markers) >= 20, (
        f"only {len(markers)} RUN markers found in the manifest body -- the "
        f"split or the pattern is wrong, not the document")


def test_every_run_marker_in_the_body_names_its_commit():
    """CF-MANIFESTSHA itself."""
    markers, _ = _body_markers()
    bare = [(line, ctx.replace("\n", " ")[-70:])
            for txt, line, ctx in markers if txt == "RUN"]
    assert not bare, (
        "RUN marker(s) with no commit id in docs/window-050.md (body line "
        "numbers, counted from the header rule):\n"
        + "\n".join(f"  line {ln}: ...{ctx}" for ln, ctx in bare)
        + "\n\nUse RUN@<sha>, RUN@wt+<sha> or RUN@unrecorded. A figure that "
          "cannot be tied to a tree is the b929924 defect in a tracked document."
    )


def test_unrecorded_markers_do_not_exceed_the_ratchet():
    """The count of provenance-less markers may fall and must never rise."""
    markers, _ = _body_markers()
    n = sum(1 for txt, _, _ in markers if txt == "RUN@unrecorded")
    sys.stdout.write(f"\n[manifest-sha] RUN@unrecorded {n} "
                     f"(ratchet {MAX_UNRECORDED})\n")
    assert n <= MAX_UNRECORDED, (
        f"{n} RUN@unrecorded markers, ratchet is {MAX_UNRECORDED}. A NEW "
        f"measurement whose tree was not recorded is not acceptable; record "
        f"the tree instead of raising the ceiling.")
    assert n == MAX_UNRECORDED or n < MAX_UNRECORDED, "unreachable"


def _in_git_worktree():
    try:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse",
                            "--is-inside-work-tree"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


@pytest.mark.skipif(not _in_git_worktree(),
                    reason="not a git work tree (a staged tree has no .git); "
                           "the sha-resolution leg needs the object store")
def test_every_named_sha_resolves_to_a_commit():
    """An id that names nothing is worse than no id: it looks like provenance."""
    markers, _ = _body_markers()
    shas = set()
    for txt, line, _ in markers:
        m = re.fullmatch(r"RUN@(?:wt\+)?([0-9a-f]{7,40})", txt)
        if m:
            shas.add((m.group(1), line))
    assert shas, "no sha-bearing markers found; the pattern is wrong"
    bad = []
    for sha, line in sorted(shas):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-t", sha],
                           capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip() != "commit":
            bad.append((sha, line, (r.stdout + r.stderr).strip()[:60]))
    sys.stdout.write(f"[manifest-sha] {len(shas)} distinct commit ids named, "
                     f"{len(bad)} unresolvable\n")
    assert not bad, "manifest names ids that are not commits in this repo: " + \
        "; ".join(f"{s} (line {ln}): {why}" for s, ln, why in bad)


def test_the_header_still_defines_every_marker_form_the_body_uses():
    """The legend is the reader's contract. A form used in the body and absent
    from the table is an undocumented marker, which is the state this file was
    written to end."""
    header, _ = _split_header(MANIFEST.read_text())
    markers, _ = _body_markers()
    forms = set()
    for txt, _, _ in markers:
        if txt == "RUN":
            forms.add("RUN")
        elif txt == "RUN@unrecorded":
            forms.add("RUN@unrecorded")
        elif txt.startswith("RUN@wt+"):
            forms.add("RUN@wt+<sha>")
        else:
            forms.add("RUN@<sha>")
    missing = [f for f in sorted(forms) if f"`{f}`" not in header]
    assert not missing, (
        f"marker form(s) used in the body but not defined in the header table: "
        f"{missing}")
