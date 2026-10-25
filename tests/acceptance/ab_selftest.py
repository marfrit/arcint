#!/usr/bin/env python3
"""Device-free tests of tests/acceptance/ab.py (ctest label unit).

The decision function on synthetic numbers, the output parser on a captured
served_check output, and the whole script against a fake served_check. Each
assertion that guards a gate has its red case: the candidate slower beyond
the spread must exit 1, an answer FAIL must exit 1.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab  # noqa: E402

FAILED = []
TOTAL = 0
EMPTY = {"metrics": {}, "failures": [], "texts": []}


def check(label, got, expect):
    global TOTAL
    TOTAL += 1
    ok = got == expect
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok else f" (got {got!r}, expected {expect!r})"))
    if not ok:
        FAILED.append(label)


def runs(*vals, metric="decode", unit="t/s", failures=()):
    return [{"metrics": {metric: (v, unit)}, "failures": list(failures), "texts": ["aaaaaaaa"]} for v in vals]


def verdict(old, new, **kw):
    rows, failures = ab.decide(runs(*old), runs(*new), **kw)
    return rows[0]["ok"] and not failures


def near(a, b):
    return abs(a - b) < 1e-9


def main():
    check("equal arms pass", verdict((100, 100), (100, 100)), True)

    # spread per arm: old 100/102 -> 2/101; new constant -> 0; the window's is the larger
    rows, _ = ab.decide(runs(100, 102), runs(100, 100))
    r = rows[0]
    check("spread of the old arm is (max-min)/mean", near(r["spread_old"], 2 / 101), True)
    check("spread of a constant arm is 0", r["spread_new"], 0.0)
    check("the allowance is the old arm's spread", near(r["spread"], 2 / 101), True)
    check("means per arm", (r["mean_old"], r["mean_new"]), (101.0, 100.0))
    check("limit = mean(old) * (1 - spread)", near(r["limit"], 101 * (1 - 2 / 101)), True)
    rows, _ = ab.decide(runs(100, 100), runs(90, 94))
    check("the spread comes from the new arm too", near(rows[0]["spread_new"], 4 / 92), True)

    # red: the candidate slower beyond the spread fails
    check("RED: new 10 % slower, arms steady -> FAIL", verdict((100, 100), (90, 90)), False)
    check("RED: new 5 % slower, spread 1 % -> FAIL", verdict((100, 101), (95, 95)), False)
    # green: slower within the spread, or faster
    check("new 1.5 % slower, old spread 2 % -> pass", verdict((100, 102), (99.5, 99.5)), True)
    check("new faster -> pass", verdict((100, 100), (120, 120)), True)

    # the operator's rule (2026-10-16): the old arm's spread, at least FLOOR, is the
    # allowance; the candidate's own spread never widens it
    check("floor 0.5 % by default", ab.FLOOR, 0.005)
    rows, _ = ab.decide(runs(100, 100), runs(100, 104))
    check("a steady old arm: the allowance is the floor", near(rows[0]["spread"], 0.005), True)
    check("RED: a noisy candidate does not pass on its own noise (old 34.6/34.6, new 34.7/31.0)",
          verdict((34.6, 34.6), (34.7, 31.0)), False)
    check("RED: a candidate varying more than the old beyond the floor fails though faster",
          verdict((100, 100), (100, 110)), False)
    rows, _ = ab.decide(runs(100, 100), runs(100, 110))
    check("the unsteady verdict is named", (rows[0]["slow"], rows[0]["unsteady"]), (False, True))
    check("new spread within old + floor passes", verdict((100, 101), (100.5, 101.5)), True)
    check("GREEN: 0.4 % slower, steady arms: inside the floor", verdict((100, 100), (99.6, 99.6)), True)
    check("RED: 0.6 % slower, steady arms: outside the floor", verdict((100, 100), (99.4, 99.4)), False)
    check("1.0.0's Flash-Next decode A/B passes (old 34.2/35.0, new 34.5/34.9)",
          verdict((34.2, 35.0), (34.5, 34.9)), True)

    check("tolerance 0.0 by default", ab.TOLERANCE, 0.0)
    check("RED/GREEN: a 1 % shortfall fails at tolerance 0, passes at 0.02",
          (verdict((100, 100), (99, 99)), verdict((100, 100), (99, 99), tolerance=0.02)), (False, True))
    check("exactly at the limit passes", verdict((100, 100), (95, 95), tolerance=0.05, floor=0.0), True)

    rows, _ = ab.decide(runs(10, 10, unit="s"), runs(12, 12, unit="s"))
    check("RED: seconds, new 20 % longer -> FAIL", rows[0]["ok"], False)
    rows, _ = ab.decide(runs(10, 10, unit="s"), runs(9, 9, unit="s"))
    check("seconds, new shorter -> pass", rows[0]["ok"], True)

    # an answer FAIL fails whatever the numbers
    rows, failures = ab.decide(runs(100, 100), runs(100, 100, failures=["the capital"]))
    check("RED: an answer FAIL in the new arm fails the A/B despite equal rates",
          (rows[0]["ok"], len(failures)), (True, 2))
    _, failures = ab.decide(runs(100, 100, failures=["the needle"]), runs(100, 100))
    check("RED: an answer FAIL in the old arm fails the A/B", failures,
          ["old run 1: the needle", "old run 2: the needle"])

    old, new = runs(100, 100), runs(100, 100)
    del new[1]["metrics"]["decode"]
    check("RED: a metric a run did not print fails", ab.decide(old, new)[0][0]["ok"], False)
    check("RED: units differing between the arms fail",
          ab.decide(runs(100, 100), runs(100, 100, unit="s"))[0][0]["ok"], False)
    check("RED: no metric in any run is a failure", len(ab.decide([EMPTY, EMPTY], [EMPTY, EMPTY])[1]), 1)
    check("one run an arm: noted", ab.decide(runs(100), runs(100))[0][0]["note"],
          "one run an arm: no spread measured")

    check("texts identical", ab.compare_texts(runs(1, 1), runs(1, 1)).split(":")[1].strip().startswith("identical"), True)
    other = runs(1, 1)
    for x in other:
        x["texts"] = ["bbbbbbbb"]
    check("texts differing between the arms are reported", "DIFFER" in ab.compare_texts(runs(1, 1), other), True)

    # The decode answers (1.2.6, operator): the gate reads the runs whose decode
    # answer has the majority md5 across both arms. Every answer runs to its
    # token cap, so a run with another answer is another workload (another
    # draft acceptance), not a speed sample of the same one.
    def druns(*pairs, metric="decode"):
        out = []
        for v, m in pairs:
            out.append({"metrics": {metric: (v, "t/s")}, "failures": [], "texts": ["aaaaaaaa"],
                        "decode": {metric: {"md5": m, "tokens": 300, "accept": 65.0, "verifies": 116,
                                            "verify_s": 6.9}}})
        return out

    A, B = "d679e4cc", "4b3a6d8c"
    # 1.2.5's first window: the one slow run is the one run with another answer
    rows, _ = ab.decide(druns((37.9, A), (38.4, A), (38.2, A), (38.4, A)),
                        druns((38.2, A), (38.6, A), (37.3, B), (38.2, A)))
    r = rows[0]
    check("decode: a divergent slow run is not a speed sample -> pass", r["ok"], True)
    check("decode: the gate reads the majority runs", (len(r["used_old"]), len(r["used_new"])), (4, 3))
    check("decode: the divergent run is named", [(d[0], d[1], d[2]) for d in r["divergent"]], [("new", 3, B)])
    check("RED decode: the same rows read without the majority rule fail on spread",
          verdict((37.9, 38.4, 38.2, 38.4), (38.2, 38.6, 37.3, 38.2)), False)
    # 1.2.5's second window: slower on the same answer stays a FAIL
    rows, _ = ab.decide(druns((38.4, A), (38.6, A), (38.5, A), (38.6, "9b617370")),
                        druns((38.0, A), (38.2, A), (38.1, A), (37.9, B)))
    check("RED decode: slower on the majority answer still fails", (rows[0]["ok"], rows[0]["slow"]), (False, True))
    # too few majority runs in an arm: every run is gated, and the note says so
    rows, _ = ab.decide(druns((38.0, A), (38.0, B)), druns((38.0, A), (38.0, "cccccccc")))
    check("decode: an arm left with one majority run falls back to every run",
          (len(rows[0]["used_old"]), len(rows[0]["used_new"]), "every run" in rows[0]["note"]), (2, 2, True))
    # a tie between two answers has no majority: every run is gated
    rows, _ = ab.decide(druns((38.0, A), (38.0, B)), druns((38.0, A), (38.0, B)))
    check("decode: a tie has no majority answer, every run gated", len(rows[0]["used_new"]), 2)
    # runs without the decode lines (an older served_check) are gated as before
    rows, _ = ab.decide(runs(37.9, 38.4, 38.2, 38.4), runs(38.2, 38.6, 37.3, 38.2))
    check("decode: runs without decode lines are gated on every run", len(rows[0]["used_new"]), 4)

    # the parser, on a captured served_check output
    with open(os.path.join(HERE, "fixtures", "served-check-output.txt")) as f:
        text = f.read()
    p = ab.parse_run(text, 1)
    check("parser: metrics", p["metrics"], {"decode-a": (50.6, "t/s"), "decode-b": (47.2, "t/s"),
                                            "decode-c": (43.8, "t/s"), "prefill-41k": (724.6, "t/s")})
    check("parser: the FAIL gate", p["failures"], ["prefill-41k read 36000 tokens (band 38000-44000)"])
    check("parser: texts md5", p["texts"], ["aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee"])
    ok_text = "\n".join(line for line in text.splitlines() if "FAIL" not in line)
    check("parser: no FAIL line, exit 0 -> no failure", ab.parse_run(ok_text, 0)["failures"], [])
    check("parser: exit 1 without a FAIL line is a failure", len(ab.parse_run(ok_text, 1)["failures"]), 1)
    line = "ACCEPTANCE-DECODE decode md5 d679e4cc tokens 300 accept 65.8 verifies 116 verify_s 6.94\n"
    check("parser: a decode answer line", ab.parse_run(line)["decode"],
          {"decode": {"md5": "d679e4cc", "tokens": 300, "accept": 65.8, "verifies": 116, "verify_s": 6.94}})
    check("parser: a decode line without speculation",
          ab.parse_run("ACCEPTANCE-DECODE decode-a md5 aaaaaaaa tokens 300 accept - verifies 0 verify_s -\n")
          ["decode"]["decode-a"]["accept"], None)
    check("parser: a non-numeric metric is a failure",
          len(ab.parse_run("ACCEPTANCE-METRIC decode nan t/s\n")["failures"]), 1)

    # the whole script against a fake served_check
    fake = os.path.join(HERE, "fixtures", "fake_served_check.py")

    def a_b(old, new, rounds=2):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, AB_FAKE_STATE=d)
            os.makedirs(os.path.join(d, "bin"))
            old, new = os.path.join(d, "bin", old), os.path.join(d, "bin", new)
            for path in (old, new):
                with open(path, "w") as f:
                    f.write(os.path.basename(path))
            proc = subprocess.run([sys.executable, os.path.join(HERE, "ab.py"), "coder", "--old", old, "--new", new,
                                   "--gguf", "m.gguf", "--device", "GPU.1", "--assets", d, "--rounds", str(rounds),
                                   "--out", os.path.join(d, "out"), "--served-check", fake],
                                  capture_output=True, text=True, env=env)
        return proc.returncode, proc.stdout

    rc, out = a_b("100,101", "100,101")
    check("end to end: equal arms -> exit 0, PASS", (rc, "A/B PASS" in out), (0, True))
    rc, out = a_b("100,101", "90,91")
    check("RED end to end: new 10 % slower than the spread -> exit 1", (rc, "A/B FAIL" in out), (1, True))
    rc, out = a_b("100,104", "99,100")
    check("end to end: new slower within the spread -> exit 0", rc, 0)
    rc, out = a_b("100,101", "x")
    check("RED end to end: an answer gate FAIL in the new arm -> exit 1", rc, 1)
    rc, out = a_b("100,100", "100,110")
    check("RED end to end: a candidate varying more than the old -> exit 1",
          (rc, "varies more than the old" in out), (1, True))
    check("end to end: both binaries named with their sha256", out.count(" sha256 "), 4)
    proc = subprocess.run([sys.executable, os.path.join(HERE, "ab.py"), "coder", "--old", "/nonexistent/arcint",
                           "--new", "/nonexistent/arcint", "--gguf", "m", "--device", "GPU.1", "--assets", "."],
                          capture_output=True, text=True)
    check("RED: a binary that does not exist is refused before any run",
          (proc.returncode, "not found" in proc.stderr, "== old run" in proc.stdout), (2, True, False))
    rc, out = a_b("100,101", "100,101", rounds=1)
    check("end to end: one round runs old and new once each", out.count("== old run") + out.count("== new run"), 2)

    # end to end, the decode answers: 1.2.5's first window passes once the
    # divergent run is read as another workload, and the table names it
    rc, out = a_b("37.9:aaaaaaaa,38.4,38.2,38.4", "38.2:aaaaaaaa,38.6,37.3:4b3a6d8c,38.2", rounds=4)
    check("end to end: a divergent slow run -> exit 0, named in the table",
          (rc, "divergent: new run 3 4b3a6d8c 37.3" in out, "decode answers per run:" in out), (0, True, True))
    rc, out = a_b("38.4:aaaaaaaa,38.6,38.5,38.6:9b617370", "38.0:aaaaaaaa,38.2,38.1,37.9:4b3a6d8c", rounds=4)
    check("RED end to end: slower on the majority answer -> exit 1", rc, 1)

    if FAILED:
        print(f"\n{len(FAILED)} of {TOTAL} assertion(s) failed: {', '.join(FAILED)}", file=sys.stderr)
        return 1
    print(f"\nall {TOTAL} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
