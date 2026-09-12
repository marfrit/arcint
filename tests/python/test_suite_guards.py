"""STRUCTURAL GUARD ON THE GUARDS: a declared skip marker that is never applied.

WHY THIS FILE EXISTS (REVIEW e78812d, finding F4 leg 2, 2026-09-12). The
`test_size_ledger.py` defect was not "a decorator was forgotten". It was that
NOTHING IN THE TREE COULD SEE a forgotten decorator. The marker was declared at
module level, every cell needed it, and the shards leg -- the only leg anyone
ran -- was 112-green throughout, because with the shards present the guard is a
no-op whether or not it is applied. The defect was visible for exactly one
reason: somebody happened to run the device-free leg, where the four unguarded
cells errored in fixture setup instead of skipping.

The review's words for this are the specification of this file:

    "Nothing asserts that a declared `skipif` is applied to anything. The
    `_skip` defect was caught only because a device-free leg happened to be
    run; the shards leg was 112-green throughout and could never see it. The
    next unapplied guard is invisible again."

The review proposed the cheap closure: "a meta-cell asserting every module-level
`pytest.mark.skipif` is referenced at least once in the file". That leg is here
(LEG 1) and it is necessary, but it was MEASURED INSUFFICIENT before this file
was committed: staged tree `~/stage-cf1-red`, one of the four `@_skip`
decorators deleted, LEG 1 alone -> 29 passed. Three applications remain, the
marker is still "referenced at least once", and the cell that lost its guard
errors in setup exactly as before. Reporting only that leg would have shipped a
gate that cannot see a partial regression of the very defect it names.

So there is a second, stronger leg. LEG 2 is a taint analysis over the module's
own fixture graph, and it closes the class rather than the instance:

    a module-level marker's CONDITION names a resource (`_SHARDS`);
    a fixture is TAINTED if its body reads that resource, or if it requests a
      tainted fixture (transitively, to a fixpoint);
    a test cell that requests a tainted fixture MUST carry that marker.

A cell that requests a tainted fixture and does not carry the guard is not a
style problem: it is the 2026-09-12 defect, and it will error in fixture setup
on the leg where the resource is absent. LEG 2 fires on the one-decorator
mutation above, which is the whole reason it exists.

Both legs are device-free, need no shards and no card: they read source with
`ast`. They cost milliseconds and they run on every leg, which is the point --
the guard on the guards must not itself be skippable.

The scanner is proved able to fail in-tree and permanently:
`test_the_scanner_detects_an_unapplied_marker`,
`test_the_scanner_detects_the_unapplied_one_of_two` and
`test_the_taint_scanner_detects_an_unguarded_cell` feed it synthetic modules
carrying each defect shape, and `test_the_scanner_accepts_each_application_form`
plus `test_the_taint_scanner_accepts_a_clean_module` feed it the legitimate
forms, because a gate that false-alarms gets weakened and then guards nothing.
A checker with no negative cell is the same species of defect it is here to
catch.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every directory that holds pytest cells. Kept explicit rather than globbed
# from the repo root so that a stray `test_*.py` in a build tree cannot make
# this cell fail for an unrelated reason.
_SUITE_DIRS = (
    REPO_ROOT / "tests" / "python",
    REPO_ROOT / "tools",
)


def suite_files():
    """Every pytest file this repository owns, sorted, as (relpath, Path)."""
    out = []
    for d in _SUITE_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("test_*.py")):
            out.append((str(p.relative_to(REPO_ROOT)), p))
    return out


def _is_pytest_mark_call(node):
    """True for `pytest.mark.<anything>(...)` and for `pytest.mark.<anything>`.

    Both forms produce a marker object: `skipif` takes arguments, `skip` and
    `xfail` are usable bare. Either can be bound to a module-level name and
    then forgotten.
    """
    target = node.func if isinstance(node, ast.Call) else node
    # pytest.mark.NAME  ->  Attribute(attr=NAME, value=Attribute(attr='mark',
    #                                 value=Name(id='pytest')))
    if not isinstance(target, ast.Attribute):
        return False
    inner = target.value
    return (
        isinstance(inner, ast.Attribute)
        and inner.attr == "mark"
        and isinstance(inner.value, ast.Name)
        and inner.value.id == "pytest"
    )


def declared_markers(tree):
    """Module-level names bound to a pytest marker object.

    Only module level: a marker built inside a function is applied at the point
    it is built or not at all, and cannot be silently orphaned the way a
    module-level one can.
    """
    names = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not _is_pytest_mark_call(node.value):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                names[tgt.id] = node.lineno
    return names


def _root_name(node):
    """The leading identifier of a possibly-dotted expression, or None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def applied_names(tree):
    """Every name that is USED as a marker somewhere in the module.

    Three application forms are recognised, because all three are real ways to
    apply a marker and a scanner that knows only decorators would produce false
    alarms the next time somebody uses one of the others:

      1. a decorator on a function or a class          -> `@_skip`
      2. a module-level or class-level `pytestmark`    -> `pytestmark = [_skip]`
      3. a `marks=` argument, as in `pytest.param(x, marks=_skip)` or
         `pytest.mark.parametrize(..., [pytest.param(..., marks=[_skip])])`
    """
    used = set()

    for node in ast.walk(tree):
        # form 1 -- decorators
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                base = dec.func if isinstance(dec, ast.Call) else dec
                nm = _root_name(base)
                if nm:
                    used.add(nm)

        # form 2 -- pytestmark assignment, scalar or sequence
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                    for sub in ast.walk(node.value):
                        if isinstance(sub, ast.Name):
                            used.add(sub.id)

        # form 3 -- marks=
        if isinstance(node, ast.keyword) and node.arg == "marks":
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Name):
                    used.add(sub.id)

    return used


def unapplied_markers(source, filename="<string>"):
    """LEG 1. Declared module-level markers that are never applied.
    Returns a sorted list of (name, lineno)."""
    tree = ast.parse(source, filename=filename)
    declared = declared_markers(tree)
    used = applied_names(tree)
    return sorted((n, ln) for n, ln in declared.items() if n not in used)


# ---------------------------------------------------------------------------
# LEG 2 -- the taint analysis
# ---------------------------------------------------------------------------

def _module_level_names(tree):
    """Every name bound by a module-level assignment."""
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def _marker_resources(tree):
    """For each module-level marker name, the module-level names its CONDITION
    reads. `_skip = pytest.mark.skipif(not _SHARDS, ...)` -> {"_skip": {"_SHARDS"}}.

    A marker with no readable condition (a bare `pytest.mark.slow`, or a
    condition that is a literal) contributes an empty set and therefore taints
    nothing -- correctly: it guards no resource.
    """
    modnames = _module_level_names(tree)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not _is_pytest_mark_call(node.value):
            continue
        res = set()
        if isinstance(node.value, ast.Call) and node.value.args:
            for sub in ast.walk(node.value.args[0]):
                if isinstance(sub, ast.Name) and sub.id in modnames:
                    res.add(sub.id)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                out[tgt.id] = res
    return out


def _is_fixture(fn):
    for dec in fn.decorator_list:
        base = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(base, ast.Attribute) and base.attr == "fixture":
            return True
        if isinstance(base, ast.Name) and base.id == "fixture":
            return True
    return False


def _self_guards(fn):
    """A fixture that skips on its own does not need the caller to be marked."""
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr in ("skip", "importorskip", "xfail"):
                return True
    return False


def _params(fn):
    a = fn.args
    return [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]


def _reads(fn, names):
    """The subset of `names` the function body reads (parameters excluded --
    a parameter shadows the module-level name)."""
    shadow = set(_params(fn))
    hit = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Name) and sub.id in names and sub.id not in shadow:
            hit.add(sub.id)
    return hit


def _decorator_names_of(fn):
    out = set()
    for dec in fn.decorator_list:
        base = dec.func if isinstance(dec, ast.Call) else dec
        nm = _root_name(base)
        if nm:
            out.add(nm)
    return out


def unguarded_resource_cells(source, filename="<string>"):
    """LEG 2. Test cells that reach a guarded resource without carrying a guard.

    Returns a sorted list of (test_name, lineno, resource, marker_choices).
    """
    tree = ast.parse(source, filename=filename)
    resources_by_marker = _marker_resources(tree)
    all_resources = set().union(*resources_by_marker.values()) if resources_by_marker else set()
    if not all_resources:
        return []

    # module-level pytestmark applies to every cell in the file
    module_marks = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                    for sub in ast.walk(node.value):
                        if isinstance(sub, ast.Name):
                            module_marks.add(sub.id)

    # collect functions at module level and inside classes, remembering the
    # class's own decorators (a class-level guard covers its methods)
    funcs = []          # (fn, inherited_marks)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append((node, set(module_marks)))
        elif isinstance(node, ast.ClassDef):
            cls_marks = set(module_marks) | _decorator_names_of(node)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.append((sub, cls_marks))

    fixtures = {fn.name: fn for fn, _ in funcs if _is_fixture(fn)}

    # taint to a fixpoint
    tainted = {}                                   # fixture name -> resources
    changed = True
    while changed:
        changed = False
        for name, fn in fixtures.items():
            if _self_guards(fn):
                continue
            res = set(_reads(fn, all_resources))
            for p in _params(fn):
                res |= tainted.get(p, set())
            if res and res != tainted.get(name, set()):
                tainted[name] = res
                changed = True

    findings = []
    for fn, inherited in funcs:
        if not fn.name.startswith("test_") or _is_fixture(fn):
            continue
        if _self_guards(fn):
            continue
        res = set(_reads(fn, all_resources))
        for p in _params(fn):
            res |= tainted.get(p, set())
        if not res:
            continue
        carried = _decorator_names_of(fn) | inherited
        covered = set()
        for m in carried:
            covered |= resources_by_marker.get(m, set())
        missing = res - covered
        if missing:
            choices = sorted(m for m, r in resources_by_marker.items() if r & missing)
            findings.append((fn.name, fn.lineno, sorted(missing), choices))
    return sorted(findings, key=lambda f: f[1])


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relpath,path", suite_files(),
                         ids=[rp for rp, _ in suite_files()])
def test_every_declared_marker_is_applied(relpath, path):
    """No suite file may declare a module-level pytest marker it never applies.

    This is the cell that would have caught the 2026-09-12 `_skip` defect on
    the SHARDS leg, where the behavioural symptom does not exist.
    """
    orphans = unapplied_markers(path.read_text(), filename=relpath)
    assert not orphans, (
        f"{relpath}: declared pytest marker(s) never applied -- "
        + ", ".join(f"{n} (line {ln})" for n, ln in orphans)
        + ". A declared-and-unapplied guard is invisible on the leg where the "
          "guard is a no-op; that is exactly how the size-ledger cells errored "
          "in fixture setup on the device-free leg."
    )


@pytest.mark.parametrize("relpath,path", suite_files(),
                         ids=[rp for rp, _ in suite_files()])
def test_no_cell_reaches_a_guarded_resource_unguarded(relpath, path):
    """LEG 2, the strong one. A cell that requests a resource-dependent fixture
    must carry the marker that guards that resource.

    This is the leg that fires on a PARTIAL regression -- one decorator deleted
    out of four -- which LEG 1 cannot see and which was measured not to fire
    before this cell existed.
    """
    bad = unguarded_resource_cells(path.read_text(), filename=relpath)
    assert not bad, (
        f"{relpath}: cell(s) reach a guarded resource without the guard -- "
        + "; ".join(
            f"{name} (line {ln}) reads {res} but carries none of {choices}"
            for name, ln, res, choices in bad)
        + ". Without the marker this cell ERRORS in fixture setup on the leg "
          "where the resource is absent, instead of skipping."
    )


def test_the_suite_file_list_is_not_empty():
    """A scanner that scanned nothing would pass the gate above vacuously."""
    files = suite_files()
    assert len(files) >= 10, f"expected the q4e suites, found {len(files)}: {files}"
    names = {rp for rp, _ in files}
    for expected in ("tests/python/test_size_ledger.py",
                     "tests/python/test_attention_piece.py",
                     "tests/python/test_gguf_feed.py"):
        assert expected in names, f"{expected} not scanned; got {sorted(names)}"


# ---------------------------------------------------------------------------
# THE SCANNER'S OWN RED -- permanent, in-tree
# ---------------------------------------------------------------------------

_ORPHANED = '''
import os
import pytest
_shards = os.environ.get("SHARDS", "")
_skip = pytest.mark.skipif(not _shards, reason="needs shards")

def test_one():
    assert True
'''

_MISSED_ONE = '''
import pytest
_a = pytest.mark.skipif(True, reason="a")
_b = pytest.mark.skipif(True, reason="b")

@_a
def test_one():
    assert True

def test_two():
    assert True
'''


def test_the_scanner_detects_an_unapplied_marker():
    """The exact shape of the 2026-09-12 defect, reduced to eight lines."""
    found = unapplied_markers(_ORPHANED, "<orphaned>")
    assert found == [("_skip", 5)], found


def test_the_scanner_detects_the_unapplied_one_of_two():
    """Half-applied is the likelier real-world shape and must not pass."""
    found = unapplied_markers(_MISSED_ONE, "<missed-one>")
    assert [n for n, _ in found] == ["_b"], found


@pytest.mark.parametrize("form,source", [
    ("decorator", '''
import pytest
_skip = pytest.mark.skipif(True, reason="r")

@_skip
def test_one():
    assert True
'''),
    ("pytestmark-scalar", '''
import pytest
_skip = pytest.mark.skipif(True, reason="r")
pytestmark = _skip

def test_one():
    assert True
'''),
    ("pytestmark-list", '''
import pytest
_skip = pytest.mark.skipif(True, reason="r")
pytestmark = [_skip]

def test_one():
    assert True
'''),
    ("marks-kwarg", '''
import pytest
_skip = pytest.mark.skipif(True, reason="r")

@pytest.mark.parametrize("x", [pytest.param(1, marks=_skip)])
def test_one(x):
    assert True
'''),
    ("marks-kwarg-list", '''
import pytest
_skip = pytest.mark.skipif(True, reason="r")

@pytest.mark.parametrize("x", [pytest.param(1, marks=[_skip])])
def test_one(x):
    assert True
'''),
    ("on-a-class", '''
import pytest
_skip = pytest.mark.skipif(True, reason="r")

@_skip
class TestGroup:
    def test_one(self):
        assert True
'''),
    ("bare-marker-object", '''
import pytest
_slow = pytest.mark.slow

@_slow
def test_one():
    assert True
'''),
])
def test_the_scanner_accepts_each_application_form(form, source):
    """False alarms are as fatal as misses: a gate that fires on a legitimate
    application form gets weakened or deleted, and then it guards nothing."""
    assert unapplied_markers(source, f"<{form}>") == [], form


# --- LEG 2's own red/green: the exact size-ledger shape, reduced ---

_TAINT_BAD = '''
import os
import pytest
_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()
_skip = pytest.mark.skipif(not _SHARDS, reason="needs shards")

@pytest.fixture(scope="module")
def feed():
    return open(_SHARDS)

@pytest.fixture(scope="module")
def measured(feed):
    return {"x": 1}

@_skip
def test_guarded(measured):
    assert measured

def test_unguarded(measured):
    assert measured
'''

_TAINT_OK = '''
import os
import pytest
_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "").strip()
_skip = pytest.mark.skipif(not _SHARDS, reason="needs shards")

@pytest.fixture(scope="module")
def cfg():
    return {"n": 1}

@pytest.fixture(scope="module")
def feed():
    return open(_SHARDS)

@pytest.fixture(scope="module")
def measured(feed):
    return {"x": 1}

@_skip
def test_guarded(measured):
    assert measured

def test_device_free(cfg):
    assert cfg
'''


def test_the_taint_scanner_detects_an_unguarded_cell():
    """`test_unguarded` requests `measured`, which requests `feed`, which reads
    `_SHARDS`. Two hops -- the size-ledger shape exactly."""
    bad = unguarded_resource_cells(_TAINT_BAD, "<taint-bad>")
    assert [(n, res, ch) for n, _, res, ch in bad] == [
        ("test_unguarded", ["_SHARDS"], ["_skip"])], bad


def test_the_taint_scanner_accepts_a_clean_module():
    """An unguarded cell that only touches an untainted fixture is legitimate --
    `test_baked_rope_tables_are_the_pin_rotary_module` in the attention suite is
    exactly this shape, and a gate that fired on it would be deleted within a
    week."""
    assert unguarded_resource_cells(_TAINT_OK, "<taint-ok>") == []


def test_the_taint_scanner_honours_a_self_guarding_fixture():
    """A fixture that calls pytest.skip() itself needs no marker at the call
    site; flagging it would be a false alarm."""
    src = '''
import os
import pytest
_SHARDS = os.environ.get("S", "")
_skip = pytest.mark.skipif(not _SHARDS, reason="r")

@pytest.fixture(scope="module")
def feed():
    if not _SHARDS:
        pytest.skip("no shards")
    return open(_SHARDS)

def test_one(feed):
    assert feed
'''
    assert unguarded_resource_cells(src, "<self-guard>") == []


def test_the_taint_scanner_honours_a_class_level_guard():
    src = '''
import os
import pytest
_SHARDS = os.environ.get("S", "")
_skip = pytest.mark.skipif(not _SHARDS, reason="r")

@pytest.fixture(scope="module")
def feed():
    return open(_SHARDS)

@_skip
class TestGroup:
    def test_one(self, feed):
        assert feed
'''
    assert unguarded_resource_cells(src, "<class-guard>") == []


def test_the_scanner_ignores_a_marker_built_inside_a_function():
    """Only module-level bindings can be orphaned invisibly; a local one is
    applied where it is built or it is dead code the linter sees."""
    src = '''
import pytest

def _build():
    m = pytest.mark.skipif(True, reason="r")
    return m
'''
    assert unapplied_markers(src, "<local>") == []


if __name__ == "__main__":                                  # pragma: no cover
    # Usable as a standalone auditor, so the gate can be run without pytest
    # (for instance from a hook or from a staged tree before the suite starts).
    rc = 0
    for relpath, path in suite_files():
        src = path.read_text()
        for name, lineno in unapplied_markers(src, relpath):
            print(f"UNAPPLIED  {relpath}:{lineno}  {name}")
            rc = 1
        for name, lineno, res, choices in unguarded_resource_cells(src, relpath):
            print(f"UNGUARDED  {relpath}:{lineno}  {name}  reads {res}  "
                  f"needs one of {choices}")
            rc = 1
    print("scanned", len(suite_files()), "files;", "CLEAN" if rc == 0 else "DEFECTS")
    sys.exit(rc)


# ---------------------------------------------------------------------------
# K2 (REVIEW 9162ac9): THE COUNT GATES ARE ENUMERATED, so two honest readings
# of "LEG A" can be reconciled without anyone opening a gate to make the
# numbers match.
# ---------------------------------------------------------------------------

# Every switch that can move this suite's passed/skipped split. Three env vars
# and one property of the checkout. A reading of LEG A is only comparable to
# another reading at the SAME point in this space, which is why the close-out
# and window-050 record the command AND the env beside every count.
#
#   Q4E_GPU            comma list of devices; empty = CPU only
#   Q4E_GGUF_SHARDS    path to the real GGUF shards; unset = every real-weight
#                      cell skips by name
#   Q4E_SERVING_FULL   1 = run the 48-layer keystone build (deliberately not
#                      on every pass)
#   a git work tree    NOT an env var, and the one that caught us out: a bare
#                      `git archive` extract has no .git, so test_citations'
#                      LEG 2 (`git ls-files`) skips. A clone and a tarball of
#                      the SAME COMMIT therefore report different splits, and
#                      both are correct.
#   Q4E_GDN_UT_MODE    the GDN unit-test emit mode, read by tools/q4e/gdn.py.
#                      RECORDED BECAUSE THE CENSUS READS IT (REVIEW fe68342,
#                      M2), not because it was caught moving a count: it is a
#                      switch the suite obeys through an IMPORTED MODULE rather
#                      than through a test file, which is the shape the first
#                      census could not see. "Measured count-harmless today" is
#                      a date, not a property.
_COUNT_GATES = frozenset({"Q4E_GPU", "Q4E_GGUF_SHARDS", "Q4E_SERVING_FULL",
                          "Q4E_GDN_UT_MODE"})

# The files whose CELLS are scanned for checkout-shaped gates: the suite's own
# test modules, plus any conftest (a fixture there gates every cell under it).
_SUITE_FILES = sorted((REPO_ROOT / "tests" / "python").glob("test_*.py")) + [
    REPO_ROOT / "tools" / "test_export_qwen4_exp.py"] + sorted(
    (REPO_ROOT / "tests").rglob("conftest.py"))

# The files scanned for `Q4E_*` READS. Wider than the cell scan on purpose
# (REVIEW fe68342, M2): a gate does not have to live in a test file to move the
# split. `tools/q4e/gdn.py` reads `Q4E_GDN_UT_MODE`, and every cell that emits
# a GDN block obeys it through the import; `tests/python/q4e_device.py` reads
# `Q4E_GPU` for the whole suite. Scanning only `test_*.py` censused neither.
_CENSUS_FILES = sorted(
    set(_SUITE_FILES)
    | set((REPO_ROOT / "tests" / "python").glob("*.py"))
    | set((REPO_ROOT / "tools" / "q4e").glob("*.py")))


def test_the_suite_declares_no_count_gate_outside_the_recorded_set():
    """The reconciliation K2 asked for, as an assertion rather than prose.

    Two readings of LEG A at the same commit differed (194/84 against 192/86)
    and both were honest: one was a clone, one a `git archive` extract, and the
    citation scan's `git ls-files` leg is gated on that difference. The fix is
    not to pick a number; it is to make the SPACE the numbers live in closed,
    so a future disagreement is always attributable to a named coordinate and
    nobody ever reconciles two counts by quietly exporting a variable.

    So: every `Q4E_*` environment variable the suite reads must be one of the
    recorded gates. A new one is a new axis in that space, and it goes red here
    until the recorded matrix grows to cover it.

    The census reads `_CENSUS_FILES`, which is WIDER than the test modules: a
    variable read by an imported tooling module gates the suite exactly as much
    as one read by a cell (REVIEW fe68342, M2).
    """
    import re

    found = {}
    for path in _CENSUS_FILES:
        if not path.is_file():
            continue
        for name in re.findall(r"Q4E_[A-Z0-9_]+", path.read_text()):
            found.setdefault(name, set()).add(path.name)

    print(f"\n[count-gates] recorded {sorted(_COUNT_GATES)}")
    for name in sorted(found):
        print(f"  {name:<20} read by {len(found[name])} file(s): "
              f"{', '.join(sorted(found[name])[:4])}")

    undeclared = sorted(set(found) - _COUNT_GATES)
    assert not undeclared, (
        f"the suite reads {undeclared}, which is not in the recorded count-gate "
        f"set. Every such variable can move the passed/skipped split, so a "
        f"recorded count becomes ambiguous the moment one exists undeclared. "
        f"Add it to _COUNT_GATES and to the env matrix in the close-out and "
        f"window-050 -- do NOT reconcile two counts by setting it.")
    unused = sorted(_COUNT_GATES - set(found))
    assert not unused, (
        f"_COUNT_GATES lists {unused}, which no suite file reads any more. A "
        f"stale gate makes the matrix over-report the space and hides that the "
        f"two axes left are the only ones that matter.")




# ---------------------------------------------------------------------------
# THE CHECKOUT-SHAPED GATES -- the axis that is not an environment variable.
# K2 (REVIEW 9162ac9) recorded it per FILE with a regex. REVIEW fe68342 (B3)
# showed that promise was not kept, so it is an ast scan over CELLS now.
# ---------------------------------------------------------------------------
#
# WHAT THE REVIEWER DID, AND WHY THE OLD SHAPE DESERVED TO LOSE. The docstring
# promised "a third one appearing silently would make the recorded env matrix
# wrong without anything going red". The reviewer wrote a third checkout-gated
# cell the ordinary other way --
#
#     def test_something(...):
#         if not _in_git_worktree():
#             pytest.skip("needs a git work tree")
#
# -- and the guard stayed GREEN while the clone-minus-tarball delta moved from
# two to three. The regex `skipif\([^)]*(?:ls-files|_in_git_worktree|...)` sees
# ONE syntactic shape: a decorator. An in-body `pytest.skip()` is the same gate
# with the same effect on the count.
#
# TWO THINGS ARE WIDENED, not one:
#
#   1. the FORM. A cell is checkout-gated if a `skipif` it carries is
#      checkout-flavoured, OR its body calls `pytest.skip()` on a
#      checkout-flavoured path, OR it requests a fixture in the same file that
#      does, OR its module gates itself wholesale (`pytestmark`, or a
#      module-level `pytest.skip(..., allow_module_level=True)`), OR a
#      `marks=` argument carries the gate onto one parametrised id. Helper
#      functions are followed to a fixpoint, so gating on `_in_git_worktree()`
#      is seen wherever the helper is named.
#      THE LAST THREE OF THOSE ARE NOT THE REVIEWER'S SHAPE. They were found
#      by asking, before re-measuring, which OTHER ordinary spellings of the
#      same gate this scan would still miss -- the same question the reviewer
#      asked of the regex, asked of its replacement.
#   2. the GRANULARITY. The recorded set is CELLS, not files. The old set
#      compared file names, so a second gated cell added to a file already in
#      the set -- `test_citations.py`, say -- moved the delta without moving
#      the set. The count of gated cells IS the clone-minus-tarball delta, so
#      that is what gets recorded.
#
# WHAT STILL SLIPS, stated rather than implied. The scan is per-file and
# reads `_SUITE_FILES` (the suite's test modules and any conftest), so two
# gates cross a file boundary and are invisible to it:
#
#   * a fixture defined in an IMPORTED helper module -- `q4e_device.py`, or
#     anything under `tools/q4e/` -- that skips on the checkout's shape;
#   * a conftest fixture that skips: the conftest IS scanned, but the cells it
#     gates live in other files, so the gate is seen and the cells are not.
#
# Neither exists in the tree today -- measured, not assumed: `pytest.skip(`
# appears in no non-test module and in no conftest (there is no conftest at
# all). Filed as CF-CHECKOUTGATE-IMPORT, DONE-WHEN the scan resolves fixtures
# across the import graph or the suite gains a conftest, whichever comes
# first. "Nothing does this today" is a date, not a property.

_CHECKOUT_STRINGS = (
    "ls-files", "is-inside-work-tree", "rev-parse", "--git-dir", ".git",
    "work tree", "worktree",
)
_CHECKOUT_NAMES = re.compile(r"in_git|git_worktree|worktree|git_dir", re.I)


def _checkout_flavoured(node):
    """True if this subtree mentions the shape of the checkout: a name like
    `_in_git_worktree`, or a string naming a git plumbing call or `.git`.

    Deliberately generous. A false positive here fails an equality assertion
    loudly and gets recorded; a false negative is the B3 defect, silent.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and _CHECKOUT_NAMES.search(sub.id):
            return True
        if isinstance(sub, ast.Attribute) and _CHECKOUT_NAMES.search(sub.attr):
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            low = sub.value.lower()
            if any(tok in low for tok in _CHECKOUT_STRINGS):
                return True
    return False


def _called_names(node):
    """The root identifier of every call made anywhere in this subtree."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            nm = _root_name(sub.func)
            if nm:
                out.add(nm)
    return out


def _flavoured_helpers(tree):
    """Module functions that reach the checkout, transitively.

    `test_citations._tracked_files` runs `git ls-files`; `_in_git_worktree`
    runs `git rev-parse --is-inside-work-tree`. A cell that calls either and
    then skips is checkout-gated even though the cell itself spells no git.
    """
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    flavoured = {name for name, fn in funcs.items()
                 if _CHECKOUT_NAMES.search(name) or _checkout_flavoured(fn)}
    changed = True
    while changed:
        changed = False
        for name, fn in funcs.items():
            if name in flavoured:
                continue
            if _called_names(fn) & flavoured:
                flavoured.add(name)
                changed = True
    return flavoured


def _reaches_checkout(node, helpers):
    return _checkout_flavoured(node) or bool(_called_names(node) & helpers)


def _calls_pytest_skip(fn):
    """`pytest.skip(...)` called in this function's own body."""
    for sub in ast.walk(fn):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "skip" and _root_name(sub.func) == "pytest"):
            return True
    return False


def _mentions_gating_marker(node, markers):
    """True if any name in this subtree is bound to a checkout-shaped marker.

    Walking every name rather than reading the root one covers `marks=`:
    `@pytest.mark.parametrize("x", [pytest.param(1, marks=_needs_git)])` gates
    one id of a cell, which moves the count as surely as gating the cell.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and markers.get(sub.id):
            return True
    return False


def _decorator_gate(declist, helpers, markers):
    """True if any decorator in this list is a checkout-flavoured skip gate,
    written inline as `pytest.mark.skipif(...)`, bound to a module name, or
    carried on a `marks=` argument."""
    for dec in declist:
        if isinstance(dec, ast.Call) and _is_pytest_mark_call(dec):
            if _reaches_checkout(dec, helpers):
                return True
        if _mentions_gating_marker(dec, markers):
            return True
    return False


def checkout_shaped_gated_cells(source, filename="<string>"):
    """Every test cell in this module that skips when the tree has no `.git`.

    Returns a sorted list of `"<basename>::<cell>"`. Its LENGTH is the
    clone-minus-tarball delta for this file.
    """
    tree = ast.parse(source, filename=filename)
    base = Path(filename).name
    helpers = _flavoured_helpers(tree)

    # module-level markers, and whether each one is a checkout gate
    markers = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and _is_pytest_mark_call(node.value):
            flav = _reaches_checkout(node.value, helpers)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    markers[tgt.id] = flav

    # Whole-module gates, which cost every cell in the file rather than one:
    #   pytestmark = _needs_git                        (a bound marker)
    #   pytestmark = pytest.mark.skipif(not _in_git_worktree(), ...)   (inline)
    #   if not _in_git_worktree():
    #       pytest.skip("...", allow_module_level=True)
    # The last one is the cheapest way to gate a whole file and it is the one
    # a cell-shaped scan would be most embarrassed to miss.
    module_gated = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                    if (_mentions_gating_marker(node.value, markers)
                            or _reaches_checkout(node.value, helpers)):
                        module_gated = True
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "skip"
                        and _root_name(sub.func) == "pytest"
                        and _reaches_checkout(node, helpers)):
                    module_gated = True

    # (function, inherited-gate) for module-level functions and class methods
    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append((node, module_gated))
        elif isinstance(node, ast.ClassDef):
            cls_gated = module_gated or _decorator_gate(
                node.decorator_list, helpers, markers)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.append((sub, cls_gated))

    # a fixture that skips on the checkout gates every cell that requests it,
    # and every fixture that requests IT -- to a fixpoint, as in LEG 2
    fixtures = {fn.name: fn for fn, _ in funcs if _is_fixture(fn)}
    gating = {name for name, fn in fixtures.items()
              if _calls_pytest_skip(fn) and _reaches_checkout(fn, helpers)}
    changed = True
    while changed:
        changed = False
        for name, fn in fixtures.items():
            if name in gating:
                continue
            if set(_params(fn)) & gating:
                gating.add(name)
                changed = True

    out = []
    for fn, inherited in funcs:
        if not fn.name.startswith("test_") or _is_fixture(fn):
            continue
        gated = (
            inherited
            or _decorator_gate(fn.decorator_list, helpers, markers)
            or (_calls_pytest_skip(fn) and _reaches_checkout(fn, helpers))
            or bool(set(_params(fn)) & gating)
        )
        if gated:
            out.append(f"{base}::{fn.name}")
    return sorted(out)


# The cells that skip on the CHECKOUT's shape rather than on an env var: they
# need a git work tree. Written down because this is the axis that produced
# K2's disagreement, and because the NUMBER OF THESE CELLS IS the
# clone-minus-tarball delta.
#
# THE FILE-LEVEL VERSION OF THIS SET WAS WRONG TWICE, and both corrections are
# on the record rather than edited away. First it said "only test_citations may
# do this" and went red on its first run naming test_window_manifest.py -- two
# cells, not one, which is exactly the 194/84 (clone) against 192/86 (tarball)
# gap K2 was asked to reconcile. Then REVIEW fe68342 (B3) showed that a set of
# FILE NAMES policed by a decorator regex cannot keep the promise its docstring
# made: a third gated cell written as an in-body `pytest.skip()` left it green.
_CHECKOUT_GATED_CELLS = frozenset({
    "test_citations.py::test_every_resolvable_citation_points_at_a_line_that_exists",
    "test_window_manifest.py::test_every_named_sha_resolves_to_a_commit",
})


def test_the_checkout_shaped_gates_are_exactly_the_recorded_cells():
    """The non-env gate, pinned to the CELLS that have it.

    A cell gated on a git work tree skips in a tree with no `.git` -- which is
    what a `git archive` extract is. That is legitimate, but the SET of such
    cells must be closed, because their number is the whole difference between
    a clone reading and a tarball reading of the same commit. A third one
    appearing silently would make the recorded env matrix wrong without
    anything going red -- and that is now true of a gate written in the body of
    a cell, which is how the reviewer proved the previous version of this cell
    did not mean it.

    Parametrised cells would break the cell-count-equals-delta identity (one
    cell, many ids); none of the recorded ones is parametrised, and a new one
    that is would show up here as a set difference first.
    """
    found = []
    for path in _SUITE_FILES:
        if not path.is_file():
            continue
        found.extend(checkout_shaped_gated_cells(path.read_text(), str(path)))
    found = sorted(found)

    print(f"\n[count-gates] checkout-shaped (git work tree) gated cells "
          f"({len(found)}): {found}")
    assert set(found) == set(_CHECKOUT_GATED_CELLS), (
        f"the cells gating a skip on the checkout's shape are {found}, not the "
        f"recorded {sorted(_CHECKOUT_GATED_CELLS)}. Their COUNT is the "
        f"clone-minus-tarball delta in the recorded env matrix, so this set "
        f"changing means that matrix is stale -- re-measure it and update both, "
        f"and do not reconcile two readings by running one of them in the "
        f"other's tree.")


def test_the_count_gate_census_reaches_past_the_test_files():
    """M2 (REVIEW fe68342): the census population, asserted.

    `tools/q4e/gdn.py` reads `Q4E_GDN_UT_MODE` and every cell that emits a GDN
    block obeys it through the import; `tests/python/q4e_device.py` reads
    `Q4E_GPU` for the whole suite. A census of `test_*.py` alone saw neither,
    so narrowing it back would silently re-open the hole rather than fail.
    """
    scanned = {str(p.relative_to(REPO_ROOT)) for p in _CENSUS_FILES}
    for expected in ("tools/q4e/gdn.py", "tests/python/q4e_device.py",
                     "tests/python/test_citations.py",
                     "tools/test_export_qwen4_exp.py"):
        assert expected in scanned, (
            f"{expected} is not in the count-gate census population; a "
            f"`Q4E_*` read there would move the split unseen. Scanned: "
            f"{sorted(scanned)}")


# ---------------------------------------------------------------------------
# THE CHECKOUT SCANNER'S OWN RED AND GREEN -- permanent, in-tree.
# The first of these is the reviewer's B3 mutation, reduced.
# ---------------------------------------------------------------------------

_GATE_IN_BODY = '''
import pytest

def _in_git_worktree():
    return False

def test_one():
    if not _in_git_worktree():
        pytest.skip("needs a git work tree")
    assert True
'''

_GATE_DECORATOR = '''
import pytest

def _in_git_worktree():
    return False

@pytest.mark.skipif(not _in_git_worktree(), reason="no .git")
def test_one():
    assert True
'''

_GATE_VIA_FIXTURE = '''
import subprocess
import pytest

@pytest.fixture()
def tracked():
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no index")
    return r.stdout.split()

def test_one(tracked):
    assert tracked
'''

_GATE_VIA_MARKER_NAME = '''
import pytest

def _worktree():
    return False

_needs_git = pytest.mark.skipif(not _worktree(), reason="r")

@_needs_git
def test_one():
    assert True
'''

_GATE_VIA_PARAM_MARKS = '''
import pytest

def _in_git_worktree():
    return False

_needs_git = pytest.mark.skipif(not _in_git_worktree(), reason="r")

@pytest.mark.parametrize("x", [1, pytest.param(2, marks=_needs_git)])
def test_one(x):
    assert x
'''

_SKIP_UNRELATED = '''
import os
import pytest

_SHARDS = os.environ.get("Q4E_GGUF_SHARDS", "")

def test_one():
    if not _SHARDS:
        pytest.skip("needs the real shards")
    assert True
'''

_GATE_WHOLE_MODULE_PYTESTMARK = '''
import pytest

def _in_git_worktree():
    return False

pytestmark = pytest.mark.skipif(not _in_git_worktree(), reason="r")

def test_one():
    assert True

def test_two():
    assert True
'''

_GATE_WHOLE_MODULE_SKIP = '''
import subprocess
import pytest

if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"]).returncode:
    pytest.skip("not a git work tree", allow_module_level=True)

def test_one():
    assert True

def test_two():
    assert True
'''

_MODULE_SKIP_UNRELATED = '''
import os
import pytest

if not os.environ.get("Q4E_GGUF_SHARDS"):
    pytest.skip("needs the real shards", allow_module_level=True)

def test_one():
    assert True
'''


@pytest.mark.parametrize("form,source", [
    ("in-body-skip", _GATE_IN_BODY),
    ("skipif-decorator", _GATE_DECORATOR),
    ("fixture-that-skips", _GATE_VIA_FIXTURE),
    ("module-level-marker", _GATE_VIA_MARKER_NAME),
    ("param-marks", _GATE_VIA_PARAM_MARKS),
])
def test_the_checkout_scanner_sees_every_gate_form(form, source):
    """`in-body-skip` IS the B3 mutation: the shape that stayed green.

    A scanner that knows only decorators re-opens the hole the moment somebody
    writes the gate the other ordinary way.
    """
    assert checkout_shaped_gated_cells(source, f"<{form}>.py") == \
        [f"<{form}>.py::test_one"], form


@pytest.mark.parametrize("form,source", [
    ("pytestmark", _GATE_WHOLE_MODULE_PYTESTMARK),
    ("allow-module-level-skip", _GATE_WHOLE_MODULE_SKIP),
])
def test_the_checkout_scanner_sees_a_whole_module_gate(form, source):
    """A module that gates itself costs EVERY cell in it, so it is the largest
    possible error in the clone-minus-tarball delta and the one a cell-shaped
    scan is likeliest to walk past."""
    assert checkout_shaped_gated_cells(source, f"<{form}>.py") == [
        f"<{form}>.py::test_one", f"<{form}>.py::test_two"], form


@pytest.mark.parametrize("form,source", [
    ("in-body-skip", _SKIP_UNRELATED),
    ("module-level-skip", _MODULE_SKIP_UNRELATED),
])
def test_the_checkout_scanner_ignores_a_skip_that_is_not_checkout_shaped(form, source):
    """False alarms are as fatal as misses. A shards-gated skip does NOT move
    the clone-minus-tarball delta and must not be counted in it."""
    assert checkout_shaped_gated_cells(source, f"<{form}>.py") == [], form


def test_the_checkout_scanner_finds_the_two_recorded_cells_by_name():
    """The scanner is run against the real modules it polices, so a rename or
    a deletion of either cell shows up here as well as in the set equality."""
    for base, cell in (("test_citations.py",
                        "test_every_resolvable_citation_points_at_a_line_that_exists"),
                       ("test_window_manifest.py",
                        "test_every_named_sha_resolves_to_a_commit")):
        path = REPO_ROOT / "tests" / "python" / base
        found = checkout_shaped_gated_cells(path.read_text(), str(path))
        assert f"{base}::{cell}" in found, (base, found)
