"""The materialisation instrument (tools/bigalloc.c) must attribute what it claims.

A driver allocates known buffers through four routes -- malloc, posix_memalign,
anonymous mmap, operator-new-sized calloc -- frees one of them, and exits. The
dump has to report each buffer against its own stack, the freed one with no
live bytes, and the peak snapshot holding exactly the buffers that were live at
the peak. Needs a C compiler; skipped without one.
"""

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MiB = 1 << 20

DRIVER = textwrap.dedent(
    r"""
    #define _GNU_SOURCE
    #include <stdlib.h>
    #include <string.h>
    #include <sys/mman.h>
    __attribute__((noinline)) void* site_malloc(void) { void* p = malloc(64u << 20); memset(p, 1, 4096); return p; }
    __attribute__((noinline)) void* site_align(void) { void* p = 0; posix_memalign(&p, 4096, 32u << 20); return p; }
    __attribute__((noinline)) void* site_mmap(void) {
        return mmap(0, 48u << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); }
    __attribute__((noinline)) void* site_calloc(void) { return calloc(1, 16u << 20); }
    int main(void) {
        void* a = site_malloc();   /* live 64          */
        void* b = site_align();    /* live 96          */
        free(b);                   /* live 64          */
        void* c = site_mmap();     /* live 112 = peak  */
        munmap(c, 48u << 20);      /* live 64          */
        void* d = site_calloc();   /* live 80          */
        for (int i = 0; i < 1000; ++i) free(malloc(100));
        (void)a; (void)d;
        return 0;
    }
    """
)


def _parse(text):
    head = text.splitlines()[0]
    fields = dict(re.findall(r"(\w+)=([\d.]+)", head))
    stacks = []
    for block in re.split(r"^S ", text, flags=re.M)[1:]:
        lines = block.splitlines()
        rec = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", lines[0])}
        rec["frames"] = [ln.strip()[2:] for ln in lines[1:] if ln.strip().startswith("F ")]
        stacks.append(rec)
    return fields, stacks


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("bigalloc")
    so = d / "bigalloc.so"
    exe = d / "driver"
    subprocess.run([cc, "-O2", "-fPIC", "-shared", "-o", str(so), str(ROOT / "tools" / "bigalloc.c"),
                    "-ldl", "-lpthread"], check=True)
    (d / "driver.c").write_text(DRIVER)
    # -rdynamic so dladdr can name the driver's own sites in the frames
    subprocess.run([cc, "-O0", "-rdynamic", "-o", str(exe), str(d / "driver.c")], check=True)
    out = d / "trace.txt"
    env = dict(os.environ, LD_PRELOAD=str(so), BIGALLOC_OUT=str(out), BIGALLOC_MIN=str(8 * MiB),
               BIGALLOC_STEP=str(16 * MiB), BIGALLOC_PERIOD="60")
    subprocess.run([str(exe)], check=True, env=env)
    return _parse(out.read_text())


def _site(stacks, name):
    hits = [s for s in stacks if any(name in f for f in s["frames"][:3])]
    assert len(hits) == 1, (name, [s["frames"][:3] for s in stacks])
    return hits[0]


def test_every_route_is_attributed_to_its_own_site(dump):
    _, stacks = dump
    for name, size in (("site_malloc", 64), ("site_align", 32), ("site_mmap", 48), ("site_calloc", 16)):
        s = _site(stacks, name)
        assert s["total"] == size * MiB and s["count"] == 1, (name, s)


def test_freed_buffers_leave_no_live_bytes(dump):
    fields, stacks = dump
    assert _site(stacks, "site_align")["live"] == 0
    assert _site(stacks, "site_mmap")["live"] == 0
    assert _site(stacks, "site_malloc")["live"] == 64 * MiB
    assert int(fields["live"]) == 80 * MiB


def test_peak_snapshot_holds_what_was_live_at_the_peak(dump):
    fields, stacks = dump
    assert int(fields["peak"]) == 112 * MiB
    assert _site(stacks, "site_malloc")["at_peak"] == 64 * MiB
    assert _site(stacks, "site_mmap")["at_peak"] == 48 * MiB
    assert _site(stacks, "site_align")["at_peak"] == 0
    assert _site(stacks, "site_calloc")["at_peak"] == 0


def test_small_allocations_are_counted_not_recorded(dump):
    fields, stacks = dump
    assert int(fields["small"]) >= 1000
    assert all(s["max"] >= 8 * MiB for s in stacks)


def test_only_filter_leaves_other_programs_untraced(tmp_path):
    """BIGALLOC_ONLY names the one program to trace; a launcher that inherits
    LD_PRELOAD must not write the dump."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    so = tmp_path / "bigalloc.so"
    subprocess.run([cc, "-O2", "-fPIC", "-shared", "-o", str(so), str(ROOT / "tools" / "bigalloc.c"),
                    "-ldl", "-lpthread"], check=True)
    (tmp_path / "d.c").write_text(DRIVER)
    exe = tmp_path / "driver"
    subprocess.run([cc, "-O0", "-rdynamic", "-o", str(exe), str(tmp_path / "d.c")], check=True)
    for only, want in (("not-the-driver", False), ("driver", True)):
        out = tmp_path / f"trace-{only}.txt"
        env = dict(os.environ, LD_PRELOAD=str(so), BIGALLOC_OUT=str(out), BIGALLOC_ONLY=only,
                   BIGALLOC_PERIOD="60")
        subprocess.run([str(exe)], check=True, env=env)
        assert out.exists() == want, only


def test_a_failed_realloc_keeps_the_block_live(tmp_path):
    """realloc() returning NULL leaves the old block allocated; the tracer must
    still count it (it forgets the block before the call)."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    so = tmp_path / "bigalloc.so"
    subprocess.run([cc, "-O2", "-fPIC", "-shared", "-o", str(so), str(ROOT / "tools" / "bigalloc.c"),
                    "-ldl", "-lpthread"], check=True)
    (tmp_path / "r.c").write_text(textwrap.dedent(r"""
        #include <stdint.h>
        #include <stdlib.h>
        #include <string.h>
        int main(void) {
            void* p = malloc(64u << 20); memset(p, 1, 4096);
            void* q = realloc(p, SIZE_MAX / 2);          /* fails */
            return q == 0 ? 0 : 1;
        }
        """))
    exe = tmp_path / "r"
    subprocess.run([cc, "-O0", "-o", str(exe), str(tmp_path / "r.c")], check=True)
    out = tmp_path / "trace.txt"
    env = dict(os.environ, LD_PRELOAD=str(so), BIGALLOC_OUT=str(out), BIGALLOC_PERIOD="60")
    subprocess.run([str(exe)], check=True, env=env)
    fields, _ = _parse(out.read_text())
    # counted at malloc_usable_size (the requested size is not kept): up to a
    # page of allocator slack above the 64 MiB, and never 0
    assert 64 * MiB <= int(fields["live"]) <= 64 * MiB + 4096, fields
