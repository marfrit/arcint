"""Every patch in the repository must parse.

The build applies the patch series under contrib/ to upstream trees, and a
corrupt patch is a build failure that nothing else checks: in September three
patches of one series landed corrupt and git apply rejected them with
"corrupt patch at line N" weeks after they landed. The only prior check
verified that the files existed and matched a mirror, not that they parse.

git apply --numstat parses a patch without needing a target tree and without
changing anything, so it is the cheapest possible parse check. A patch fails
if git exits non-zero.

The earlier version of this checker also failed a patch that exited zero but
reported zero files. That second condition was unreachable and has been
deleted: with this exact invocation (git apply --numstat, no --allow-empty)
no input makes git exit 0 with empty stdout. The evidence, measured on git
2.47.3:

  empty file / commit-header-only / whitespace-only /
  diff --git-lines-only / a real empty-commit format-patch
      -> exit 128, "No valid patches in input (allow with --allow-empty)"
  binary-patch header, mode-change-only, rename/copy at 100 % similarity,
  new-file header with no hunk, delete with no hunk
      -> exit 0, but stdout names the file ("0 0 f", "- - img")
  every one of the 63 tracked *.patch files in this repository
      -> exit 0 with at least one file line

The only way to reach exit 0 with empty stdout is to pass --allow-empty,
which the checker does not. A check no test can turn red is untested, so it
is removed rather than kept.

The second test proves the checker can go red by running it on a temporary
directory with a known-good and four known-bad patches. The bad fixtures are
chosen so that each one has empty stdout, which is what makes them
catchable by the exit-code condition alone: an empty file, a real patch
truncated in the middle of a hunk, a commit-message header with no diff, and
a hunk header whose line counts are wrong.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_patches(paths):
    """Parse each patch with git apply --numstat.

    Returns a list of (path, reason) for every patch that does not parse.
    A patch fails if git exits non-zero: a corrupt patch, an empty file, or
    an input with no valid patches all do. (An exit-zero run always names at
    least one file, so there is no separate zero-files condition to check;
    see the module docstring for the evidence.)
    """
    failures = []
    for path in paths:
        r = subprocess.run(
            ["git", "apply", "--numstat", str(path)],
            capture_output=True, text=True)
        if r.returncode != 0:
            lines = (r.stderr or r.stdout).strip().splitlines()
            reason = lines[0] if lines else "git apply exited %d" % r.returncode
            failures.append((path, reason))
    return failures


def test_every_tracked_patch_parses():
    r = subprocess.run(
        ["git", "ls-files", "*.patch"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    patches = [ROOT / p for p in r.stdout.splitlines() if p.strip()]
    # a test that checks zero files and passes is worthless: fail loudly if
    # the enumeration came up empty (wrong cwd, broken glob, ...)
    assert patches, "no tracked *.patch files found under %s" % ROOT
    failures = check_patches(patches)
    assert not failures, (
        "%d of %d tracked patches do not parse:\n%s" % (
            len(failures), len(patches),
            "\n".join("  %s: %s" % (p.relative_to(ROOT), reason)
                      for p, reason in failures)))


def _tracked_patches_under(subdir):
    r = subprocess.run(
        ["git", "ls-files", subdir + "/*.patch"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [ROOT / p for p in sorted(r.stdout.splitlines()) if p.strip()]


def _truncate_mid_hunk(text):
    """Cut a patch inside its first hunk: header plus only part of the body."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("@@"):
            cut = i + 3
            if cut >= len(lines):
                cut = max(1, len(lines) // 2)
            return "".join(lines[:cut])
    return "".join(lines[:max(1, len(lines) // 2)])


def test_the_checker_catches_corrupt_patches(tmp_path):
    # (a) a valid small patch: must be reported OK
    valid = tmp_path / "valid.patch"
    valid.write_text(
        "diff --git a/hello.txt b/hello.txt\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        "+++ b/hello.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+hello\n"
        "+world\n")

    # (b) a real patch truncated in the middle of a hunk: must be FAILED
    real = _tracked_patches_under("contrib/llama.cpp/patches")
    assert real, "no tracked patches under contrib/llama.cpp/patches"
    truncated = tmp_path / "truncated.patch"
    full = real[0].read_text()
    cut = _truncate_mid_hunk(full)
    assert len(cut) < len(full), "truncation did not shorten the patch"
    truncated.write_text(cut)

    # (c) an empty .patch file: must be FAILED. git exits 128 with no
    # stdout, so this is caught by the exit-code condition alone.
    empty = tmp_path / "empty.patch"
    empty.write_text("")

    # (d) a commit-message header with no diff: must be FAILED. Like (c),
    # git exits 128 with no stdout, so the exit-code condition alone
    # catches it.
    header = tmp_path / "header.patch"
    header.write_text(
        "commit 0123456789abcdef0123456789abcdef01234567\n"
        "Author: Someone <someone@example.com>\n"
        "Date:   Mon Jan 1 00:00:00 2026 +0000\n"
        "\n"
        "    subject line only, no diff body\n")

    # (e) a hunk header whose line counts are wrong: @@ -1,3 +1,3 @@ over a
    # 2-line hunk: must be FAILED
    badheader = tmp_path / "badheader.patch"
    badheader.write_text(
        "diff --git a/f.txt b/f.txt\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/f.txt\n"
        "@@ -1,3 +1,3 @@\n"
        "+a\n"
        "+b\n")

    # Every bad fixture above has empty stdout, so the exit-code condition
    # is the only thing that can catch each one. That is the point: the
    # test is red if the exit-code check is removed.
    failures = check_patches([valid, truncated, empty, header, badheader])
    failed = {p.name for p, _ in failures}
    assert "valid.patch" not in failed, "a valid patch was reported corrupt"
    for name in ("truncated.patch", "empty.patch", "header.patch",
                 "badheader.patch"):
        assert name in failed, "%s was not reported corrupt" % name
    assert failed == {"truncated.patch", "empty.patch", "header.patch",
                      "badheader.patch"}, (
        "unexpected failure set: %s" % sorted(failed))
