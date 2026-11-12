#!/usr/bin/env python3
# dory_loops_spot.py <mode> <port> <pid> <outdir> [loops]: the served spot check of the Dory loop count per request
# (--llama-dory-max-loops, contrib/llama.cpp 0118) on one running arcint process on a card. Stdlib only; not part of
# the arcint build or its tests. Prompts are made up (a one-word capital question, a bicycle question, a 2,000-token
# text with a needle, a primes list, an arithmetic question with thinking), greedy.
#
# The card's PCI address (as the kernel driver prints it in the process's DRM fdinfo, `drm-pdev`) is required in the
# environment: ARCINT_SPOT_PDEV=<domain:bus:dev.fn>. Find it with `grep drm-pdev /proc/<pid>/fdinfo/*` on a process that
# holds the card, or `lspci -D | grep -i -E 'vga|display'`.
#
# Modes (run them against fresh processes, in this order; <outdir> carries the reference texts between them):
#   base <loops>: a process loaded at a fixed count (1, 2 or 4) WITHOUT the flag; the greedy texts of the prompt groups
#                 this count stands for (effort low at 1, unset and medium at 2, high at 4) go to <outdir>/texts-b<loops>.json.
#   dev: a process with --llama-dory-max-loops 4 and a unit count of 2; the same groups by reasoning effort, each compared
#        with the base texts (prompt tokens, answer, thinking text), the count each response names, the re-prefill on a
#        change inside one conversation, the Responses, chat-stream and token-mode routes, and /props.
#   red: the dev binary with the graph-reuse key bent (LLAMA_DORY_RED_REUSE=1): the high group after a medium request. It
#        did not fail on the served route (README, 0118 section): the CPU probe is the red case that fails.
# Output: PASS/FAIL lines, MEM lines (the process's card memory from fdinfo at marks, also written to <outdir>/mem-<tag>.jsonl),
# per-request wall times (single requests, one run each), and a final SPOT-OK / SPOT-FAILED line; the exit status is 0 only
# for SPOT-OK.
import json, os, sys, time, urllib.request as u
mode, port, pid, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
loops = int(sys.argv[5]) if len(sys.argv) > 5 else 0
PDEV = os.environ.get("ARCINT_SPOT_PDEV")
if not PDEV: sys.exit("set ARCINT_SPOT_PDEV to the card's PCI address (see the header)")
MIB = 1024.0 * 1024.0; UNIT = {"KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30}
EFF = {1: ["low"], 2: [None, "medium"], 4: ["high"]}
URL = "http://127.0.0.1:%s" % port
def fdinfo():
    cl = {}; d = "/proc/%d/fdinfo" % pid
    for f in os.listdir(d):
        try: t = open(d + "/" + f).read()
        except OSError: continue
        if "drm-client-id" not in t: continue
        kv = {}
        for line in t.splitlines():
            if ":" in line:
                k, v = line.split(":", 1); kv[k.strip()] = v.strip()
        cl[(kv.get("drm-pdev"), kv["drm-client-id"])] = kv
    tot = {}
    for (pdev, _), kv in cl.items():
        for k, v in kv.items():
            if k.startswith(("drm-total-", "drm-resident-")) and "cycles" not in k:
                p = v.split(); n = int(p[0]) * UNIT.get(p[1] if len(p) > 1 else "", 1)
                key = (pdev, k[4:]); tot[key] = tot.get(key, 0) + n
    return tot
def mark(label):
    time.sleep(1.0); cur = fdinfo(); r = cur.get((PDEV, "resident-vram0"), 0) / MIB; t = cur.get((PDEV, "total-vram0"), 0) / MIB
    print("  MEM %-5s %-14s vram resident %9.1f MiB total %9.1f" % (tag, label, r, t), flush=True)
    open(out + "/mem-%s.jsonl" % tag, "a").write(json.dumps({"leg": tag, "mark": label, "resident_mib": r, "total_mib": t}) + "\n")
def post(path, body, timeout=900):
    r = u.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(u.urlopen(r, timeout=timeout)); return d, time.time() - t
BASE = "def f(x):\n    return x * 2 + 1\n\n"
# (name, text, max_tokens, thinking): four cases with the thinking block off (short, the text is the answer) and one as served, thinking on
PROMPTS = [("paris", "What is the capital of France? One word.", 20, False),
           ("bike", "Write four sentences about how a bicycle stays upright.", 120, False),
           ("needle", "The secret code is ZEBRA7391.\n" + BASE * 140 + "\nWhat is the secret code? Answer with the code only.", 12, False),
           ("primes", "List the first eight prime numbers and say one thing about each.", 150, False),
           ("think", "What is 17 * 23? Reply with the number only.", 500, True)]
def chat(msgs, effort, max_tokens, thinking=False):
    body = {"model": "m", "messages": msgs, "max_tokens": max_tokens, "temperature": 0}
    if not thinking: body["chat_template_kwargs"] = {"enable_thinking": False}
    if effort: body["reasoning_effort"] = effort
    d, w = post("/v1/chat/completions", body); m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "finish": d["choices"][0].get("finish_reason"),
            "pt": d["usage"]["prompt_tokens"], "ct": d["usage"]["completion_tokens"], "wall": round(w, 2), "dory": d.get("dory"),
            "keys": sorted(d.keys())}
def group(effort):
    res = {}
    for name, text, mt, th in PROMPTS:
        r = chat([{"role": "user", "content": text}], effort, mt, th); res[name] = r
        print("  %-5s effort %-6s %-6s pt %5d ct %3d wall %6.2f s dory %s | %s" % (tag, effort, name, r["pt"], r["ct"], r["wall"], json.dumps(r["dory"]),
              (r["content"] or r["reasoning"]).strip().replace("\n", " ")[:70]), flush=True)
    return res
bad = 0
def check(ok, what):
    global bad
    print("  %s %s" % ("PASS" if ok else "FAIL", what), flush=True)
    if not ok: bad += 1
try:
    if mode == "base":
        tag = "b%d" % loops; mark("loaded"); res = {}
        for e in EFF[loops]:
            res[str(e)] = group(e); mark("group-%s" % e)
        json.dump(res, open(out + "/texts-%s.json" % tag, "w"))
        check("Paris" in res[str(EFF[loops][0])]["paris"]["content"], "base %d loops: Paris" % loops)
        check("ZEBRA7391" in res[str(EFF[loops][0])]["needle"]["content"], "base %d loops: the needle" % loops)
    elif mode == "dev":
        tag = "dev"; mark("loaded")
        props = json.load(u.urlopen(URL + "/props", timeout=30)).get("dory")
        print("  /props dory: %s" % json.dumps(props), flush=True)
        check(bool(props) and props.get("max_loops") == 4, "/props dory.max_loops is 4")
        B = {n: json.load(open(out + "/texts-b%d.json" % n)) for n in (1, 2, 4)}
        differ = 0
        for n, e in ((2, None), (2, "medium"), (1, "low"), (4, "high")):
            res = group(e); mark("group-%s" % e); ref = B[n][str(e)]
            for name, _, _, _ in PROMPTS:
                r, b = res[name], ref[name]
                check(r["dory"] is not None and r["dory"].get("loops") == n, "effort %s %s: the response names %d loops (%s)" % (e, name, n, json.dumps(r["dory"])))
                check(r["pt"] == b["pt"], "effort %s %s: prompt tokens %d = base %d" % (e, name, r["pt"], b["pt"]))
                same = r["content"] == b["content"] and r["reasoning"] == b["reasoning"]
                check(same, "effort %s %s: the greedy text equals the text of a load at %d loops (%d tokens)" % (e, name, n, r["ct"]))
            check(res["paris"]["dory"] is not None and res["paris"]["dory"].get("reprefill") == (e in ("low", "high")),
                  "effort %s: re-prefill flag on the group's first request is %s" % (e, e in ("low", "high")))
            check("Paris" in res["paris"]["content"] and "ZEBRA7391" in res["needle"]["content"], "effort %s: Paris and the needle" % e)
            print("  %-5s effort %-6s speed: needle prompt %d tokens in %.2f s; bike %d tokens in %.2f s" % (tag, e, res["needle"]["pt"], res["needle"]["wall"],
                  res["bike"]["ct"], res["bike"]["wall"]), flush=True)
        for a, b_ in ((1, 2), (2, 4), (1, 4)):
            ea, eb = EFF[a][0], EFF[b_][0]
            d = sum(1 for name, _, _, _ in PROMPTS if B[a][str(ea)][name]["content"] != B[b_][str(eb)][name]["content"]); differ += d
            print("  base texts that differ between %d and %d loops: %d of %d" % (a, b_, d, len(PROMPTS)), flush=True)
        check(differ > 0, "the base texts differ between counts (else the equality cells could not fail)")
        # one conversation, the level changed inside it
        m = [{"role": "user", "content": PROMPTS[1][1]}]
        def turn(label, effort, want_loops, want_re):
            r = chat(m, effort, 60)
            check(r["dory"] is not None and r["dory"].get("loops") == want_loops and r["dory"].get("reprefill") == want_re,
                  "conversation %s effort %s: loops %d reprefill %s (%s), pt %d wall %.2f s" % (label, effort, want_loops, want_re, json.dumps(r["dory"]), r["pt"], r["wall"]))
            return r
        r = turn("turn 1", "medium", 2, True)
        m += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": "Now the same in German."}]
        r = turn("turn 2", "medium", 2, False)
        r = turn("turn 2 again", "high", 4, True)
        m += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": "Shorter, please."}]
        r = turn("turn 3", "high", 4, False)
        r = turn("turn 3 again", None, 2, True)
        r = turn("turn 3 once more", "xhigh", 4, True)
        mark("conversation")
        # the Responses route and a chat stream
        d, w = post("/v1/responses", {"model": "m", "input": "What is the capital of France? One word.", "reasoning": {"effort": "low"}, "max_output_tokens": 200})
        ad = (d.get("arcint") or {}).get("dory"); txt = json.dumps(d.get("output"))[:300]
        check(ad is not None and ad.get("loops") == 1, "Responses effort low: arcint.dory %s | %s" % (json.dumps(ad), "Paris" in txt and "Paris" or txt[:120]))
        body = {"model": "m", "messages": [{"role": "user", "content": "What is the capital of France? One word."}], "max_tokens": 20, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high", "stream": True}
        raw = u.urlopen(u.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read().decode()
        sd = [json.loads(l[6:]).get("dory") for l in raw.splitlines() if l.startswith("data: {")]; sd = [x for x in sd if x]
        check(len(sd) == 1 and sd[0].get("loops") == 4, "chat stream effort high: one chunk names the count (%s)" % json.dumps(sd))
        # token mode (the route the gateway alias uses): the prompt as the engine renders it, its hash, effort low
        rb = {"model": "m", "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is the capital of France? One word."}]}], "reasoning": {"effort": "low"}}
        try:
            rh = json.load(u.urlopen(URL + "/v1/responses/renderer", timeout=120)); RH = rh.get("renderer_hash") or rh.get("hash")
            ids = post("/tokenize", rb)[0]["tokens"]; text = post("/detokenize", {"model": "m", "tokens": ids})[0]["content"]
            import hashlib
            tb = dict(rb, stream=True, max_output_tokens=300, temperature=0,
                      arcint={"v": 1, "token_mode": True, "renderer_hash": RH, "render": {"prompt_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest()}})
            raw = u.urlopen(u.Request(URL + "/v1/responses", json.dumps(tb).encode(), {"Content-Type": "application/json"}), timeout=300).read().decode()
            ev = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")]
            ntok = sum(1 for e in ev if e.get("type") == "arcint:token"); fin = [e for e in ev if e.get("type") in ("response.completed", "response.incomplete")]
            td = ((fin[-1].get("response", {}).get("arcint") or {}).get("dory")) if fin else None
            check(ntok > 0 and td is not None and td.get("loops") == 1, "token mode effort low: %d arcint:token events, the final event names %s" % (ntok, json.dumps(td)))
        except Exception as e:
            body = e.read()[:200].decode("utf-8", "replace") if hasattr(e, "read") else ""
            check(False, "token mode effort low: %s %s %s" % (type(e).__name__, str(e)[:120], body))
        mark("end")
    elif mode == "red":
        tag = "red"; mark("loaded"); B4 = json.load(open(out + "/texts-b4.json"))["high"]; B2 = json.load(open(out + "/texts-b2.json"))["medium"]
        group("medium"); res = group("high"); n4 = 0; n2 = 0
        for name, _, _, _ in PROMPTS:
            n4 += res[name]["content"] == B4[name]["content"]; n2 += res[name]["content"] == B2[name]["content"]
        print("  RED: high-group texts equal to the 4-loop base %d of %d, equal to the 2-loop base %d of %d (the response names %s)" % (n4, len(PROMPTS), n2, len(PROMPTS),
              json.dumps(res["bike"]["dory"])), flush=True)
        check(n4 < len(PROMPTS), "the red build's high group does NOT reproduce the 4-loop base (the equality cells can fail)")
    print("  %s SPOT-%s (%d failed)" % (tag, "OK" if bad == 0 else "FAILED", bad), flush=True)
    sys.exit(0 if bad == 0 else 2)
except Exception as e:
    print("  %s SPOT-ERROR %s: %s" % (mode, type(e).__name__, str(e)[:300]), flush=True); sys.exit(1)
