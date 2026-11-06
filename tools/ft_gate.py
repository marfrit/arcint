#!/usr/bin/env python3
"""The token-mode gate cell: real turns through the gateway on a -ft name at every effort, streamed.
A turn passes only if it is 200, the gateway says it ran on the token path, and the answer is right.
Usage: ft_gate.py BASE_URL MODEL [MODEL ...]"""
import json, sys, urllib.error, urllib.request as u

BASE = sys.argv[1].rstrip("/")
EFFORTS = [None, "none", "off", "minimal", "low", "medium", "high", "xhigh"]
TOOLS = [{"type": "function", "function": {"name": "get_time", "description": "The current time in a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def turn(model, effort, messages, tools=None, max_tokens=1200):
    b = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    if effort is not None:
        b["reasoning_effort"] = effort
    if tools:
        b["tools"] = tools
    q = u.Request(BASE + "/v1/chat/completions", json.dumps(b).encode(),
                  {"Content-Type": "application/json", "Authorization": "Bearer none"})
    try:
        r = u.urlopen(q, timeout=300)
    except urllib.error.HTTPError as e:
        return e.code, None, "", "", [], e.read()[:160]
    text, think, calls = "", "", []
    for line in r:
        if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
            continue
        for c in json.loads(line[5:]).get("choices", []):
            d = c.get("delta", {})
            text += d.get("content") or ""
            think += d.get("reasoning_content") or d.get("reasoning") or ""
            calls += [x for x in d.get("tool_calls") or []]
    return r.status, r.headers.get("X-Llm-Proxy-Path"), text, think, calls, b""


fails = 0
for model in sys.argv[2:]:
    for effort in EFFORTS:
        st, path, text, think, _c, err = turn(model, effort, [{"role": "user", "content": "What is the capital of France? Answer with one word."}])
        ok = st == 200 and path == "tokens" and "paris" in text.lower()
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {model} effort={effort} {st} path={path} think={len(think)} text={text.strip()[:24]!r} {err!r}" if err else
              f"{'ok  ' if ok else 'FAIL'} {model} effort={effort} {st} path={path} think={len(think)} text={text.strip()[:24]!r}", flush=True)
    # a conversation with a tool call, at the levels the template reads
    for effort in ("medium", "high"):
        st, path, text, think, calls, err = turn(model, effort, [{"role": "user", "content": "What time is it in Paris? Use the tool."}], TOOLS, 300)
        ok = st == 200 and path == "tokens" and bool(calls)
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {model} tool effort={effort} {st} path={path} calls={len(calls)} {err!r}", flush=True)
print(f"FT-GATE {'PASS' if not fails else 'FAIL'}: {fails} failed")
sys.exit(1 if fails else 0)
