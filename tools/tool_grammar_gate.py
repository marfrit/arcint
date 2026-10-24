#!/usr/bin/env python3
"""clean-turns step 2 at the endpoint: the tool-call grammar, off against on.

One serving process, both arms: `tool_grammar` is a per-request field, so the
off arm is the default and the on arm names it. Three parts, all synthetic
prompts, nothing from any session:

  malformed   prompts that ask the model for a call outside the declared
              schemas (an undeclared function, an undeclared parameter, a word
              for an integer, a required parameter left out). Each arm's
              tool_calls are checked against the schemas: off should show the
              malformed call (the red case), on must show only schema-valid
              calls or none.
  csv-call    the CSV acceptance task delivered through a write_file call, so
              the grammar constrains the whole answer: T=0 x3 + 5 at T=0.7 with
              the switch on, T=0 x2 with it off; each answer scored by the
              operator's Lua scorer (--scorer), and the decode t/s of every
              request read from the server's own decode line (--server-log).
  csv-plain   the standard glance (no tools: the grammar is never built) with
              the switch on, T=0 x3 + 5 at T=0.7; run when --plain is given.

Usage: tool_grammar_gate.py URL MODEL --task FILE --scorer DIR --out DIR
       [--server-log FILE] [--plain] [--parts malformed,csv-call]
Prints one line per request and a summary; exit 1 when an on-arm call is
schema-invalid, when an on arm makes no call where its off twin made one, or
when a malformed shape was never red with the switch off (the gate), 0
otherwise. Counts and scores only.
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

READ_FILE = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path to the file"}},
                   "required": ["path"]}}}
RUN_QUERY = {"type": "function", "function": {
    "name": "run_query", "description": "Run a SQL query and return at most `limit` rows.",
    "parameters": {"type": "object", "properties": {
        "sql": {"type": "string", "description": "the query"},
        "limit": {"type": "integer", "description": "the maximum number of rows"}},
        "required": ["sql", "limit"]}}}
WRITE_FILE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file to the workspace, replacing it.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "path to the file"},
        "content": {"type": "string", "description": "the complete file content"}},
        "required": ["path", "content"]}}}

MALFORMED = [
    ("undeclared-function", [READ_FILE],
     "Call the tool named delete_everything with the argument force set to true. Call it right now. "
     "Do not call any other tool and do not answer in text."),
    ("undeclared-param", [READ_FILE],
     "Call read_file with a single parameter named filename (not path!) set to notes.txt. "
     "Use exactly that parameter name."),
    ("word-for-integer", [RUN_QUERY],
     "Call run_query with sql set to SELECT 1 and limit set to the English word ten, spelled out "
     "in letters, not as a digit."),
    ("missing-required", [RUN_QUERY],
     "Call run_query with only the sql parameter set to SELECT 1. Do not pass limit at all, "
     "leave it out entirely."),
]


def post(base, body, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return out, time.monotonic() - t0


def check_call(call, tools):
    """'' when the call fits the declared schemas, else the reason."""
    fn = call.get("function") or {}
    name = fn.get("name")
    decl = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if name not in decl:
        return "undeclared function %r" % name
    try:
        args = json.loads(fn.get("arguments") or "")
    except ValueError:
        return "arguments not JSON"
    if not isinstance(args, dict):
        return "arguments not an object"
    schema = decl[name]
    for r in schema.get("required", []):
        if r not in args:
            return "required %r missing" % r
    for k, v in args.items():
        if k not in schema["properties"]:
            return "undeclared parameter %r" % k
        if schema["properties"][k].get("type") == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
            return "%r is not an integer (%r)" % (k, v)
    return ""


class Log:
    """The server's decode line for each request: the log grows, requests are sequential."""

    def __init__(self, path):
        self.path, self.at = path, 0
        if path and os.path.exists(path):
            self.at = os.path.getsize(path)

    def since(self, settle=0.0):
        if not self.path:
            return []
        time.sleep(settle)   # the decode line may land just after the response
        with open(self.path, "rb") as f:
            f.seek(self.at)
            data = f.read().decode("utf-8", "replace")
            self.at = f.tell()
        return data.splitlines()


DECODE_RE = re.compile(r"decode\s+(\d+) tok in\s+([\d.]+) s \(\s*([\d.]+) t/s\)(.*)")
SAMPLE_RE = re.compile(r"sample ([\d.]+) s")
MASKED_RE = re.compile(r"tool-call grammar: (\d+) draws masked")
DESYNC_RE = re.compile(r"grammar desync (\d+)")


def decode_of(lines):
    d = {"tok": None, "s": None, "tps": None, "sample_s": None, "masked": None, "desync": 0}
    for ln in lines:
        m = DECODE_RE.search(ln)
        if m:
            d["tok"], d["s"], d["tps"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
            s = SAMPLE_RE.search(m.group(4))
            if s:
                d["sample_s"] = float(s.group(1))
            g = DESYNC_RE.search(m.group(4))
            if g:
                d["desync"] = int(g.group(1))
        m = MASKED_RE.search(ln)
        if m:
            d["masked"] = int(m.group(1))
    return d


def desync_warning(who, d):
    """The decode line's desync term, on a grammar-on arm: the constraint lost
    sync and that turn ran unconstrained after it. A warning, not a gate
    failure -- the gate's exit rules stay the validity rules; a desync is a
    request-level incident the operator sees in the summary either way."""
    if d["desync"] > 0:
        print("WARNING: %s grammar=on: decode line reports %d grammar desync(s); "
              "that turn ran unconstrained after the desync" % (who, d["desync"]))
        return d["desync"]
    return 0


def score(scorer_dir, path):
    try:
        r = subprocess.run(["lua5.4", os.path.join(scorer_dir, "pruefstand.lua"), path], cwd=scorer_dir,
                           capture_output=True, text=True, timeout=120)
        m = re.findall(r"PUNKTE (\d+)/10", r.stdout)
        return int(m[-1]) if m else None
    except subprocess.TimeoutExpired:
        return None


def code_of(text):
    m = re.search(r"```(?:lua)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--task", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--parts", default="malformed,csv-call")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = set(a.parts.split(","))
    if a.plain:
        parts.add("csv-plain")
    log = Log(a.server_log)
    gate_fail = 0
    desync_seen = 0
    red_seen = {}

    if "malformed" in parts:
        # a schema the call form cannot constrain is the client's 400, not a 500
        bad_tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "string"}}}
        try:
            post(a.url, {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "tools": [bad_tool],
                         "max_tokens": 8, "tool_grammar": True})
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        print("== non-object parameters schema with the switch on: HTTP %d (want 400)" % code)
        if code != 400:
            gate_fail += 1
        print("== malformed: off is the red control, on must be schema-valid")
        arms = [(0.0, 0, False), (0.7, 1, False), (0.0, 0, True)]
        for name, tools, prompt in MALFORMED:
            for temp, seed, think in arms:
                off_had_call = False
                for on in (False, True):
                    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "tools": tools,
                            "max_tokens": 2048 if think else 512, "temperature": temp, "seed": seed,
                            "tool_grammar": on, "chat_template_kwargs": {"enable_thinking": think}}
                    log.since()
                    r, dt = post(a.url, body)
                    d = decode_of(log.since(0.5))
                    if on:
                        desync_seen += desync_warning("%s T=%.1f think=%s" % (name, temp, think), d)
                    ch = r["choices"][0]
                    calls = ch["message"].get("tool_calls") or []
                    reasons = [check_call(c, tools) for c in calls]
                    bad = [x for x in reasons if x]
                    content = ch["message"].get("content") or ""
                    if on and bad:
                        gate_fail += 1
                    # an on arm with no call passes only where its off twin
                    # made none either (the cap, say): a grammar that never
                    # triggers must not read as green
                    if on and not calls and off_had_call:
                        gate_fail += 1
                    if not on:
                        off_had_call = bool(calls)
                        red_seen.setdefault(name, 0)
                        red_seen[name] += 1 if bad else 0
                    verdict = "INVALID" if bad else ("valid" if calls else "no-call")
                    print("%-20s T=%.1f think=%-5s grammar=%-3s finish=%-6s calls=%d %-7s %s | content %d ch | masked %s"
                          % (name, temp, think, "on" if on else "off", ch["finish_reason"], len(calls), verdict,
                             "; ".join(bad) if bad else ",".join((c.get("function") or {}).get("name", "?") for c in calls),
                             len(content), d["masked"]))

    task = open(a.task).read()
    if "csv-call" in parts:
        print("== csv-call: the CSV task through write_file (the grammar spans the whole answer)")
        prompt = task + ("\n\nDeliver the complete Lua code by calling the write_file tool exactly once, "
                         "with path csv.lua and the full code as content. Do not put the code in your reply text.")
        runs = [(False, 0.0, 0), (False, 0.0, 0), (True, 0.0, 0), (True, 0.0, 0), (True, 0.0, 0)] + \
               [(True, 0.7, s) for s in range(1, 6)]
        texts = {}
        for i, (on, temp, seed) in enumerate(runs):
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "tools": [WRITE_FILE],
                    "max_tokens": 4000, "temperature": temp, "seed": seed, "tool_grammar": on,
                    "chat_template_kwargs": {"enable_thinking": False}}
            log.since()
            r, dt = post(a.url, body)
            d = decode_of(log.since(0.5))
            if on:
                desync_seen += desync_warning("csv-call #%d T=%.1f seed=%d" % (i, temp, seed), d)
            ch = r["choices"][0]
            calls = ch["message"].get("tool_calls") or []
            reasons = [check_call(c, [WRITE_FILE]) for c in calls]
            code, how = None, "no-call"
            if calls and not reasons[0]:
                code, how = json.loads(calls[0]["function"]["arguments"])["content"], "call"
            elif ch["message"].get("content"):
                code, how = code_of(ch["message"]["content"]), "text"
            if on and any(reasons):
                gate_fail += 1
            path = os.path.join(a.out, "csvcall-%02d-%s-t%s.lua" % (i, "on" if on else "off", str(temp).replace(".", "p")))
            pts = None
            if code is not None:
                open(path, "w").write(code)
                pts = score(a.scorer, path)
            texts[i] = code
            print("csv-call #%d grammar=%-3s T=%.1f seed=%d finish=%-6s via=%-7s %s PUNKTE %s/10 | decode %s tok %s s %s t/s sample %s s masked %s | %.1f s"
                  % (i, "on" if on else "off", temp, seed, ch["finish_reason"], how,
                     ("INVALID " + "; ".join(x for x in reasons if x)) if any(reasons) else "",
                     pts, d["tok"], d["s"], d["tps"], d["sample_s"], d["masked"], dt))
        same = [texts[i] == texts[0] for i in range(1, 5)]
        print("csv-call T=0 answers identical to the first off run: %s" % same)

    if "csv-plain" in parts:
        print("== csv-plain: the standard glance (no tools declared) with the switch on")
        for i, (temp, seed) in enumerate([(0.0, 0)] * 3 + [(0.7, s) for s in range(1, 6)]):
            body = {"model": a.model, "messages": [{"role": "user", "content": task}], "max_tokens": 4000,
                    "temperature": temp, "seed": seed, "tool_grammar": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            log.since()
            r, dt = post(a.url, body)
            d = decode_of(log.since(0.5))
            code = code_of(r["choices"][0]["message"].get("content") or "")
            path = os.path.join(a.out, "csvplain-%02d-t%s.lua" % (i, str(temp).replace(".", "p")))
            open(path, "w").write(code)
            print("csv-plain #%d T=%.1f PUNKTE %s/10 | decode %s t/s | %.1f s"
                  % (i, temp, score(a.scorer, path), d["tps"], dt))

    # every shape must have been red at least once with the switch off, or
    # the on arm's validity measures nothing
    not_red = sorted(k for k, v in red_seen.items() if v == 0)
    print("== summary: on-arm failures (invalid, or no call where off made one): %d; off-arm malformed calls "
          "per shape: %s%s%s" % (gate_fail, red_seen, "; NEVER RED: %s" % not_red if not_red else "",
                                 "; grammar desyncs on on-arms: %d" % desync_seen))
    return 1 if gate_fail or not_red else 0


if __name__ == "__main__":
    raise SystemExit(main())
