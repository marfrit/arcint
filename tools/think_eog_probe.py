#!/usr/bin/env python3
"""Probe arm set for the empty-stop defect (clean-turns step 1).

Prompts make the model TALK ABOUT a chat-template control token so that it
emits the real special token instead of writing it as text. The prompts name
the tokens in words only; every token string is assembled from parts at run
time, so this file contains none.

Run ON the card host against the served unit (the unit needs no auth there).
Prints flags and counts only - no prompt, no answer text.
"""
import json
import sys
import urllib.request

LT, GT, SLASH, BAR = chr(60), chr(62), chr(47), chr(124)
THINK = "think"
IM = "im"
END = "end"
OFTEXT = "endoftext"
TOKENS = {
    "turn_terminator": LT + BAR + IM + "_" + END + BAR + GT,
    "of_text_marker": LT + BAR + OFTEXT + BAR + GT,
    "think_closer": LT + SLASH + THINK + GT,
    "think_opener": LT + THINK + GT,
}


def post(base, model, msgs, extra=None, max_tokens=256):
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def run(base, model, arm, prompt, extra=None, max_tokens=256):
    resp = post(base, model, [{"role": "user", "content": prompt}], extra, max_tokens)
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    # Whether the think block's closer was seen before the turn ended. None when
    # the unit does not report it (an older binary) or there was no reasoning.
    closed = msg.get("reasoning_closed")
    calls = msg.get("tool_calls") or []
    finish = ch.get("finish_reason")
    leaked = sorted(k for k, v in TOKENS.items() if v in (content + reasoning))
    empty = not content.strip() and not calls
    verdict = "answered"
    if empty:
        verdict = "EMPTY-STOP" if finish == "stop" else (
            "EMPTY-LENGTH" if finish == "length" else "EMPTY-" + str(finish))
    print("%-26s finish=%-8s content=%-4d reasoning=%-4d calls=%d "
          "closed=%s special_token_in_output=%s -> %s"
          % (arm, finish, len(content), len(reasoning), len(calls),
             str(closed).lower() if closed is not None else "?",
             ",".join(leaked) or "-", verdict))
    return {"verdict": verdict, "closed": closed, "content": len(content),
            "reasoning": len(reasoning), "calls": len(calls), "finish": finish,
            "leaked": leaked}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-coder"
    arm = sys.argv[3] if len(sys.argv) > 3 else "arm"
    for name, prompt in SHAPES.items():
        for temp in (0.0, 0.7):
            run(base, model, "%s %s T=%.1f" % (arm, name, temp), prompt,
                {"temperature": temp})
    return 0
# The prompt set. Each prompt refers to the token in WORDS only - never its
# spelling - so the model has to produce the special token itself.
SHAPES = {
    "explain-terminator":
        "In your reasoning, mention the ChatML end-of-turn marker by name and say "
        "one sentence about why emitting it inside an assistant message breaks an "
        "agent loop. Then answer with the single word OK.",
    "code-string":
        "Write one line of Python that binds a variable named MARKER to the "
        "ChatML end-of-turn marker as a string literal, spelled exactly the way "
        "the ChatML specification spells it. Then write the single word OK.",
    "template-task":
        "Write a short Jinja chat template fragment that renders a user turn and "
        "an assistant turn and closes every turn the way ChatML closes a turn. "
        "Keep it under five lines.",
    "repeat-task":
        "Say the name of the ChatML end-of-turn marker out loud in your reasoning, "
        "then reply with the single word OK.",
    # The shape that provokes the empty stop on the coder artifact (measured
    # 2026-10-20: finish=stop, content 0, ~600 characters of reasoning, think
    # block open, 2/2 arms at T=0 and T=0.7 at a 4096-token cap). The mention
    # shapes above do NOT provoke it on this artifact - they spell the marker as
    # text - and at a small cap they come back `length`, which is a different
    # defect. This one is the red case the gate is about.
    "empty-answer":
        "Reply with an empty answer.",
}


if __name__ == "__main__":
    sys.exit(main())
