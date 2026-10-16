#!/usr/bin/env python3
"""DFlash2 on the dense agent: the card legs (docs/campaigns/dflash2-agent.md,
measurement plan phase 1-2 and the red cases). Runs on the GPU host; nothing
here knows a host name. Stop the unit that holds the card first and restore
it afterwards (docs/sop-card-window.md); this script starts and stops only its
own servers.

  legs.py served <arcint> <agent.gguf> <drafter.gguf> <assets-dir> [options]
      The arms side by side, one server process each (a fresh one per deep
      cell): plain, MTP 5 (the agent unit's flags), DFlash2 7 and 5, DFlash2 7
      with the selector's stop at 0.3 and 0.5. Per arm: the capital, a code
      answer and a prose answer (400 tokens, two passes each), a 4k-token
      context, the window edge (prompts of ~2,047, ~2,049 and ~5,000 tokens),
      an edited message resumed from a checkpoint; for the --deep-arms (MTP 5
      and DFlash2 7: the design's two arms at depth, inside the 2 h budget)
      the needle at 40 % of each --depths prompt with a 200-word answer
      after it, a fresh process each, shallowest first. Per cell: prefill
      and decode t/s, accepted drafts, verifies, tokens a verify, draft and
      verify ms a cycle; per arm the peak VRAM and GTT (fdinfo every second)
      and the md5 of every answer (a tripwire, not a gate).

  legs.py red <red-dir> <arcint> <agent.gguf> <drafter.gguf> <assets-dir> [options]
      The red cases: <red-dir>/arcint-red-N, built by build-red.sh with one
      mutant each (src/exec/llama_dflash.cpp, ARCINT_DFLASH_RED), against the
      served binary's DFlash2 7 arm on the same cells. Each must fail where
      it says below; a red build that passes is a test that cannot fail.
        1 the target layers without the converter's + 1   tokens a verify falls (code: >= 1.0 below green)
        2 the injected rows shuffled                       tokens a verify falls (code: >= 1.0 below green)
        3 the block off by one position                    tokens a verify falls (code: >= 0.5 below green)
        4 the block's K/V left in the drafter              a request fails or the window check fires
        5 the rejected rows left in the drafter            a request fails or the window check fires
        6 checkpoints without the drafter's state          the resumed request accepts less (reported)
        7 drafts accepted without the target's verdict     an answer check fails or every answer differs
      and two equivalence arms, which must not fire:
        8 the reference's prefill (every prompt row)       answers right, tokens a verify within 0.15 of green
                                                           (code, prose, 4k context, the resumed request)
        9 the drafter loaded and injecting, no drafts      no more answers differing from plain's bytes than
                                                           plain's own repeat differs (the B60's run-to-run
                                                           determinism on this engine is not measured, DESIGN
                                                           §8.8): the target's layer inputs as graph outputs
                                                           leave its numerics alone
      Every red log must name its case (the mutant is in the loaded binary).

Options: --device GPU.0, --n-ctx 212992 (every arm), --arms a,b,..,
--deep-arms a,b, --depths 55000,130000,200000 (prompt tokens; the deepest the
context holds is the operator's pick), --red-depth 20000 (the red cases'
needle), --out DIR (default ./dflash-legs-<time>), --pruefstand
CMD (the operator's acceptance-task wrapper, run once per arm with
PRUEFSTAND_URL and PRUEFSTAND_MODEL set; it prints `... score N ...`).

The assets directory holds what the repository cannot ship (as
tests/acceptance/cells/served_check.py): corpus.txt (plain text, at least
800,000 characters for a 200k prompt) and mtp-vocab.json (the agent unit's
--llama-mtp-vocab).

Not here: the KL window of the verify rows against plain decode (the gate's
KL clause). The served engine exposes no logits rows; that cell needs a dump
the engine does not have.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.request

AGENT_FLAGS = ["--llama-kv", "q8_0:q4_0", "--queue-timeout", "30", "--repetition-penalty", "1.0"]
CODE = ("Write a Lua function parse_csv(text) that parses CSV text as RFC 4180 defines it: fields separated by "
        "commas, records by CRLF or LF, fields optionally enclosed in double quotes, a double quote inside a "
        "quoted field escaped by doubling it, and line breaks allowed inside quoted fields. Return a table of "
        "records, each a table of strings. Give the code only, with short comments.")
PROSE = "Explain in about 300 words how a lighthouse keeper's work changed when lighthouses were automated."
NOTE = "\nNOTE FOR THE READER: the secret passphrase for this archive is VIOLET-HARBOR-42. Remember it.\n"
CPT = 3.58   # corpus characters a token for the agent's tokenizer (served_check.py)


def arms_table(drafter, assets):
    d = ["--llama-dflash-gguf", drafter]
    return {
        "plain": [],
        "mtp5": ["--llama-mtp", "5", "--llama-mtp-vocab", os.path.join(assets, "mtp-vocab.json")],
        # 8-row verifies without the drafter: what DFlash2 7's verify shape does to the answers
        "mtp7": ["--llama-mtp", "7", "--llama-mtp-vocab", os.path.join(assets, "mtp-vocab.json")],
        "dflash7": d + ["--llama-dflash", "7"],
        "dflash5": d + ["--llama-dflash", "5"],
        # the reference's draft ubatch (the target's 512): its VRAM against the default 64's
        "dflash7-ub512": d + ["--llama-dflash", "7", "--llama-dflash-ubatch", "512"],
        "dflash7-p03": d + ["--llama-dflash", "7", "--llama-dflash-min-p", "0.3"],
        "dflash7-p05": d + ["--llama-dflash", "7", "--llama-dflash-min-p", "0.5"],
    }


class Server:
    def __init__(self, binary, gguf, device, n_ctx, flags, log_path):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.log_path = log_path
        cmd = [binary, "--engine", "llama", "--gguf", gguf, "--served-model-name", "m", "--device", device,
               "--host", "127.0.0.1", "--port", str(self.port), "--n-ctx", str(n_ctx)] + AGENT_FLAGS + flags
        with open(log_path, "w") as lf:
            lf.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self.proc = subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT)
        self.peak = [0, 0]
        self.offset = 0
        for _ in range(3600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read()
                break
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError("the server died at start:\n" + self.log()[-3000:])
                time.sleep(1)
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while self.proc.poll() is None:
            v = g = 0
            try:
                for fd in os.listdir(f"/proc/{self.proc.pid}/fdinfo"):
                    try:
                        t = open(f"/proc/{self.proc.pid}/fdinfo/{fd}").read()
                    except OSError:
                        continue
                    if "drm-driver" not in t:
                        continue
                    v += sum(int(x) for x in re.findall(r"drm-resident-vram0:\s+(\d+)", t))
                    g += sum(int(x) for x in re.findall(r"drm-resident-gtt:\s+(\d+)", t))
            except OSError:
                pass
            self.peak = [max(self.peak[0], v // 1024), max(self.peak[1], g // 1024)]
            time.sleep(1)

    def log(self):
        return open(self.log_path, errors="replace").read()

    def new_lines(self):
        t = self.log()
        out, self.offset = t[self.offset:], len(t)
        return out

    def chat(self, messages, n):
        b = {"model": "m", "messages": messages, "max_tokens": n, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}}
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(b).encode(),
                                   {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=7200))

    def props(self):
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/props", timeout=10))

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()


DECODE_RE = re.compile(r"slot \d+: decode +(\d+) tok in +([\d.]+) s \( *([\d.]+) t/s\)(.*)")
PREFILL_RE = re.compile(r"slot \d+: prefill +(\d+) tok in +([\d.]+) s \( *([\d.]+) t/s\)(.*)")
DRAFT_RE = re.compile(r"draft accept ([\d.]+)% \((\d+)/(\d+)\), propose ([\d.]+) s, verify ([\d.]+) s")
STEPS_RE = re.compile(r"(\d+) verifies \(([\d.]+) tokens a verify\)")


def cell_metrics(lines):
    """the request's prefill and decode lines (the last of each in `lines`)"""
    m = {}
    p = PREFILL_RE.findall(lines)
    if p:
        m.update(prompt_tokens=int(p[-1][0]), prefill_tps=float(p[-1][2]),
                 resumed="resumes from its checkpoint" in lines)
    d = DECODE_RE.findall(lines)
    if d:
        m.update(decode_tokens=int(d[-1][0]), decode_s=float(d[-1][1]), decode_tps=float(d[-1][2]))
        dr = DRAFT_RE.search(d[-1][3])
        st = STEPS_RE.search(d[-1][3])
        if dr:
            m.update(accept_pct=float(dr.group(1)), accepted=int(dr.group(2)), proposed=int(dr.group(3)))
        if st:
            n = int(st.group(1))
            m.update(verifies=n, tokens_a_verify=float(st.group(2)))
            if dr and n > 0:
                m.update(draft_ms_cycle=1000 * float(dr.group(4)) / n, verify_ms_cycle=1000 * float(dr.group(5)) / n)
    m["window_check"] = lines.count("its window and the target disagree")
    return m


def corpus(assets, n_chars):
    t = open(os.path.join(assets, "corpus.txt"), errors="replace").read()
    if len(t) < n_chars:
        raise SystemExit(f"corpus.txt holds {len(t)} characters, {n_chars} needed")
    return t[:n_chars]


def needle_prompt(assets, tokens):
    t = corpus(assets, int(tokens * CPT))
    k = t.find("\n", int(len(t) * 0.4))
    k = k if k >= 0 else int(len(t) * 0.4)
    return (t[:k] + NOTE + t[k:] + "\n\nFirst write the secret passphrase for this archive on its own line. Then "
            "summarize the last paragraph above in about 200 words.")


class Run:
    def __init__(self, a):
        self.a = a
        self.rows = []
        self.failed = []
        os.makedirs(a.out, exist_ok=True)

    def gate(self, ok, msg):
        print(f"  {'ok' if ok else 'FAIL'}   {msg}", flush=True)
        if not ok:
            self.failed.append(msg)
        return ok

    def ask(self, srv, arm, cell, messages, n, check=None):
        srv.new_lines()
        try:
            r = srv.chat(messages, n)
            text = r["choices"][0]["message"]["content"] or ""
            err = None
        except Exception as e:   # a red build may fail a request: recorded, not raised
            text, err = "", str(e)
        time.sleep(0.3)   # the log lines follow the response
        m = cell_metrics(srv.new_lines())
        m.update(arm=arm, cell=cell, md5=hashlib.md5(text.encode()).hexdigest()[:8], error=err,
                 answer_ok=(check(text) if check and err is None else None))
        self.rows.append(m)
        tv = m.get("tokens_a_verify")
        print(f"  {arm:<12} {cell:<14} prefill {m.get('prompt_tokens', '-'):>6} tok {m.get('prefill_tps', 0):7.1f} t/s"
              f" | decode {m.get('decode_tokens', '-'):>4} tok {m.get('decode_tps', 0):6.1f} t/s"
              f" | {('%.2f tok/verify' % tv) if tv else 'no drafts':<16} | {m['md5']}"
              f"{' | ERROR ' + err[:60] if err else ''}{' | answer ' + ('right' if m['answer_ok'] else 'WRONG') if check else ''}",
              flush=True)
        return text, m

    def fit(self, srv, target, suffix):
        """corpus characters whose chat prompt (with `suffix`) reads about `target` tokens, by the server's count"""
        lo, hi, best = int(target * 2.5), int(target * 5.0), None
        for _ in range(12):
            mid = (lo + hi) // 2
            r = srv.chat([{"role": "user", "content": corpus(self.a.assets, mid) + suffix}], 1)
            n = r["usage"]["prompt_tokens"]
            best = (mid, n) if best is None or abs(n - target) < abs(best[1] - target) else best
            if n == target:
                break
            lo, hi = (mid + 1, hi) if n < target else (lo, mid - 1)
        return best

    def short_cells(self, srv, arm, edge=True):
        u = lambda c: [{"role": "user", "content": c}]   # noqa: E731
        self.ask(srv, arm, "capital", u("What is the capital of France? Answer in one word."), 8,
                 lambda t: "paris" in t.lower())
        for i in (1, 2):
            self.ask(srv, arm, f"code-{i}", u(CODE), 400, lambda t: "function" in t)
            self.ask(srv, arm, f"prose-{i}", u(PROSE), 400, lambda t: len(t) > 400)
        self.ask(srv, arm, "ctx4k", u(corpus(self.a.assets, int(4000 * CPT)) +
                                      "\n\nSummarize the text above in about 300 words."), 400, lambda t: len(t) > 400)
        if edge:
            # the window edge: the OpenVINO head stopped drafting at its 2,048-row state
            suffix = "\n\nContinue the text above for about 200 words."
            for target in (2047, 2049, 5000):
                n_chars, n_tok = self.fit(srv, target, suffix)
                self.ask(srv, arm, f"edge-{n_tok}", u(corpus(self.a.assets, n_chars) + suffix), 300)
        # an edited message resumed from the lane's checkpoint near the prompt's end
        base = corpus(self.a.assets, int(6000 * CPT))
        self.ask(srv, arm, "ckpt-first", u(base + "\n\nQuestion: in one sentence, what is the text about?"), 64)
        self.ask(srv, arm, "ckpt-edited", u(base + "\n\nQuestion: write a Lua function that counts the words of a "
                                                   "string, with short comments."), 400, lambda t: "function" in t)

    def deep_cell(self, srv, arm, depth):
        self.ask(srv, arm, f"needle-{depth // 1000}k", [{"role": "user", "content": needle_prompt(self.a.assets, depth)}],
                 300, lambda t: "violet-harbor-42" in t.lower())

    def pruefstand(self, srv, arm):
        if not self.a.pruefstand:
            return
        env = dict(os.environ, PRUEFSTAND_URL=f"http://127.0.0.1:{srv.port}", PRUEFSTAND_MODEL="m")
        r = subprocess.run(self.a.pruefstand, shell=True, env=env, capture_output=True, text=True)
        s = re.findall(r"score\s+(\d+)", r.stdout)
        print(f"  {arm:<12} pruefstand     {s[-1] if s else '?'} (exit {r.returncode})", flush=True)
        self.rows.append({"arm": arm, "cell": "pruefstand", "score": int(s[-1]) if s else None})

    def serve(self, arm, binary, flags, cells, tag=""):
        log = os.path.join(self.a.out, f"{arm}{tag}.log")
        print(f"== {arm}: {os.path.basename(binary)} {' '.join(flags)}", flush=True)
        try:
            srv = Server(binary, self.a.gguf, self.a.device, self.a.n_ctx, flags, log)
        except RuntimeError as e:
            self.gate(False, f"{arm}: {e}")
            return None
        try:
            p = srv.props()
            print(f"  {arm:<12} props mtp {p['mtp']['enabled']} dflash {p.get('dflash', {}).get('enabled')}"
                  f" git {p['build']['git']}", flush=True)
            cells(srv)
        finally:
            srv.stop()
            print(f"  {arm:<12} peak VRAM {srv.peak[0]} MiB, GTT {srv.peak[1]} MiB", flush=True)
            self.rows.append({"arm": arm, "cell": "memory", "peak_vram_mib": srv.peak[0], "peak_gtt_mib": srv.peak[1],
                              "window_check_total": srv.log().count("its window and the target disagree"),
                              "red_banner": "THIS BUILD IS THE DFLASH RED CASE" in srv.log()})
        return srv

    def save(self):
        with open(os.path.join(self.a.out, "cells.json"), "w") as f:
            json.dump(self.rows, f, indent=1)
        print(f"== cells: {os.path.join(self.a.out, 'cells.json')}", flush=True)


def get(rows, arm, cell, key):
    for r in rows:
        if r.get("arm") == arm and r.get("cell") == cell:
            return r.get(key)
    return None


def served(a):
    run = Run(a)
    table = arms_table(a.drafter, a.assets)
    for arm in a.arms:
        flags = table[arm]
        srv = run.serve(arm, a.arcint, flags, lambda s, arm=arm: (run.short_cells(s, arm), run.pruefstand(s, arm)))
        if srv is not None:
            run.gate(not any(r.get("red_banner") for r in run.rows if r.get("arm") == arm), f"{arm}: not a red build")
        for depth in (a.depths if arm in a.deep_arms else []):   # a fresh process per deep cell, shallowest first
            run.serve(arm, a.arcint, flags, lambda s, arm=arm, depth=depth: run.deep_cell(s, arm, depth),
                      tag=f"-{depth // 1000}k")
    for r in run.rows:
        if r.get("answer_ok") is False:
            run.gate(False, f"{r['arm']} {r['cell']}: the answer")
        if r.get("error"):
            run.gate(False, f"{r['arm']} {r['cell']}: {r['error'][:80]}")
    for arm in a.arms:
        if arm.startswith("dflash"):
            run.gate((get(run.rows, arm, "code-1", "tokens_a_verify") or 0) > 1.0, f"{arm}: drafts accepted on the code answer")
            for r in run.rows:
                if r.get("arm") == arm and r["cell"].startswith("edge-"):
                    run.gate((r.get("tokens_a_verify") or 0) > 1.0, f"{arm} {r['cell']}: drafts accepted at the window edge")
                if r.get("arm") == arm and r["cell"].startswith("needle-"):
                    run.gate((r.get("tokens_a_verify") or 0) > 1.0, f"{arm} {r['cell']}: drafts accepted at depth")
            run.gate(bool(get(run.rows, arm, "ckpt-edited", "resumed")), f"{arm}: the edited message resumed from a checkpoint")
            run.gate(sum(r.get("window_check_total", 0) for r in run.rows if r.get("arm") == arm) == 0,
                     f"{arm}: the drafter's window and the target agreed before every draft")
    run.save()
    print(f"== {len(run.failed)} failed" if run.failed else "== all gates passed", flush=True)
    return 1 if run.failed else 0


def red(a):
    run = Run(a)
    flags = arms_table(a.drafter, a.assets)["dflash7"]

    def cells(arm):
        def f(s):
            run.short_cells(s, arm, edge=False)
            run.deep_cell(s, arm, a.red_depth)
        return f

    run.serve("green", a.arcint, flags, cells("green"))
    if os.path.exists(os.path.join(a.red_dir, "arcint-red-9")):   # build 9's reference: the served binary, no drafter
        run.serve("plain", a.arcint, [], cells("plain"))
        run.serve("plain-again", a.arcint, [], cells("plain-again"))
    for r in run.rows:   # the comparison needs a green arm that answers right
        if r.get("arm") == "green" and (r.get("answer_ok") is False or r.get("error")):
            run.gate(False, f"green {r['cell']}: {r.get('error') or 'the answer'}")
    g = {c: get(run.rows, "green", c, "tokens_a_verify") or 0 for c in ("code-1", "prose-1", "ckpt-edited")}
    green_md5 = {r["cell"]: r["md5"] for r in run.rows if r.get("arm") == "green" and "md5" in r}
    green_ok = {r["cell"]: r.get("answer_ok") for r in run.rows if r.get("arm") == "green" and "md5" in r}
    for n in range(1, 10):
        b = os.path.join(a.red_dir, f"arcint-red-{n}")
        if not os.path.exists(b):
            print(f"== red {n}: {b} missing, skipped", flush=True)
            continue
        arm = f"red{n}"
        run.serve(arm, b, flags, cells(arm))
        mine = [r for r in run.rows if r.get("arm") == arm]
        run.gate(any(r.get("red_banner") for r in mine), f"red {n}: the log names the red case (the mutant is loaded)")
        tv = get(run.rows, arm, "code-1", "tokens_a_verify") or 0
        errors = any(r.get("error") for r in mine)
        window = sum(r.get("window_check_total", 0) for r in mine)
        if n in (1, 2, 3):
            margin = 1.0 if n < 3 else 0.5
            fired = tv <= g["code-1"] - margin
            detail = f"code {tv:.2f} tokens a verify against green {g['code-1']:.2f} (needs {margin} below)"
        elif n in (4, 5):
            fired = errors or window > 0
            detail = f"request errors {errors}, window check {window}"
        elif n == 9:
            # no drafts: one-row steps as plain's, with the layer inputs extracted and injected
            md5 = lambda arm_: {r["cell"]: r["md5"] for r in run.rows if r.get("arm") == arm_ and "md5" in r}   # noqa: E731
            p1, p2 = md5("plain"), md5("plain-again")
            diff = [r["cell"] for r in mine if "md5" in r and p1.get(r["cell"]) not in (None, r["md5"])]
            noise = [c for c in p1 if p2.get(c) not in (None, p1[c])]
            run.gate(len(diff) <= len(noise), f"build 9 (no drafts) against plain: {len(diff)} answers differ "
                     f"({', '.join(diff) or '-'}); plain against itself: {len(noise)} ({', '.join(noise) or '-'})")
            continue
        elif n == 8:
            # the prefill's window-only injection against the reference's every row: the same drafter state
            cells8 = ("code-1", "prose-1", "ctx4k", "ckpt-edited")
            d8 = {c: (get(run.rows, arm, c, "tokens_a_verify") or 0) - (get(run.rows, "green", c, "tokens_a_verify") or 0)
                  for c in cells8}
            right = not any(r.get("answer_ok") is False or r.get("error") for r in mine)
            same = all(abs(v) <= 0.15 for v in d8.values())
            run.gate(right and same, "build 8 (every prompt row) matches green: answers right "
                     f"{right}, tokens a verify minus green's {', '.join(f'{c} {v:+.2f}' for c, v in d8.items())}")
            continue
        elif n == 6:
            tv6 = get(run.rows, arm, "ckpt-edited", "tokens_a_verify") or 0
            fired = tv6 < g["ckpt-edited"]
            detail = f"resumed request {tv6:.2f} tokens a verify against green {g['ckpt-edited']:.2f}"
            run.gate(bool(get(run.rows, arm, "ckpt-edited", "answer_ok")), "red 6: the resumed answer stays right")
        else:
            wrong = any(r.get("answer_ok") is False and green_ok.get(r["cell"]) for r in mine)
            differs = all(r.get("md5") != green_md5.get(r["cell"]) for r in mine if "md5" in r and r["cell"] in green_md5)
            fired = wrong or differs
            detail = f"an answer green has right is wrong {wrong}, every answer differs from green {differs}"
        run.gate(fired, f"red {n} fired: {detail}")
    run.save()
    print(f"== {len(run.failed)} failed" if run.failed else "== every red case fired, the equivalence arm matched",
          flush=True)
    return 1 if run.failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("served", "red"):
        p = sub.add_parser(mode)
        if mode == "red":
            p.add_argument("red_dir")
        p.add_argument("arcint")
        p.add_argument("gguf")
        p.add_argument("drafter")
        p.add_argument("assets")
        p.add_argument("--device", default="GPU.0")
        p.add_argument("--n-ctx", type=int, default=212992)
        p.add_argument("--out", default=f"dflash-legs-{time.strftime('%Y%m%d-%H%M%S')}")
        p.add_argument("--pruefstand", default="")
        if mode == "served":
            p.add_argument("--arms", default="plain,mtp5,mtp7,dflash7,dflash5,dflash7-ub512,dflash7-p03,dflash7-p05")
            p.add_argument("--depths", default="55000,130000,200000")
            p.add_argument("--deep-arms", default="mtp5,dflash7")
        else:
            p.add_argument("--red-depth", type=int, default=20000)
    a = ap.parse_args()
    if a.mode == "served":
        a.arms = [x for x in a.arms.split(",") if x]
        a.depths = [int(x) for x in a.depths.split(",") if x]
        a.deep_arms = [x for x in a.deep_arms.split(",") if x]
        unknown = [x for x in a.arms if x not in arms_table("", "")]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}")
        return served(a)
    return red(a)


if __name__ == "__main__":
    sys.exit(main())
