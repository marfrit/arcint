"""The tool-call grammar gate's decode-line reading, without a card.

The gate's decode line carries the per-request desync count (review B4): an
arm with the grammar on whose constraint lost sync ran unconstrained after
the desync, and the operator should see a WARNING naming the request and a
count in the summary. That visibility must be proven before the gate is run
on a card seat, the way the step 1 gate's verdict logic is tested here
without one. The exit code rules stay the validity rules: a desync warns,
it does not fail the gate.
"""
import json
import http.server
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import tool_grammar_gate as gate  # noqa: E402


class _Server(http.server.BaseHTTPRequestHandler):
    """A unit that answers what the gate asks: the schema the call form
    cannot constrain is a 400, an off arm gets the malformed call the prompt
    asked for (the red control), an on arm gets a valid call -- and every
    answered arm appends its own decode line to the log the gate is pointed
    at, the way the real server does. `desync`, when set, puts the desync
    term on the decode line of every grammar-on arm."""

    log_path = None
    desync   = False

    def do_POST(self):
        n    = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if any(not isinstance((t.get("function") or {}).get("parameters"), dict)
               or (t.get("function") or {}).get("parameters", {}).get("type") != "object"
               for t in body.get("tools") or []):
            raw = json.dumps({"error": {"message": "the schema cannot be constrained",
                                        "type": "invalid_request"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        on  = bool(body.get("tool_grammar"))
        prompt = (body.get("messages") or [{}])[0].get("content") or ""
        fn = ("read_file" if "delete_everything" in prompt or "filename" in prompt
              else "run_query")
        if fn == "read_file":
            args = {"path": "notes.txt"}
        else:   # the run_query shapes need both required parameters
            args = {"sql": "SELECT 1", "limit": 10}
        call = {"id": "c0", "type": "function", "function": {"name": fn,
                                                             "arguments": json.dumps(args)}}
        if not on:
            # the red control: the malformed call the prompt asked for
            if fn == "read_file":
                call["function"] = {"name": "delete_everything", "arguments": json.dumps({"force": True})}
            elif "filename" in prompt:   # the declared parameter, misnamed
                call["function"]["arguments"] = json.dumps({"filename": "notes.txt"})
            elif "ten" in prompt:        # a word for an integer
                call["function"]["arguments"] = json.dumps({"sql": "SELECT 1", "limit": "ten"})
            else:                        # required parameter left out
                call["function"]["arguments"] = json.dumps({"sql": "SELECT 1"})
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        raw     = json.dumps({"choices": [{"finish_reason": "stop", "index": 0,
                                           "message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        if self.log_path:
            suffix = (" | grammar desync 1 (turn unconstrained after desync)"
                      if on and self.desync else "")
            with open(self.log_path, "a") as f:
                f.write("lgc  slot 0: decode    120 tok in   1.50 s ( 80.0 t/s)"
                        " | graph 1.40 s, embed 0.01 s, sample 0.02 s, emit 0.01 s, "
                        "wait 0.00 s, other 0.06 s%s\n" % suffix)

    def log_message(self, *a):
        pass


def _serve(handler, timeout=5.0):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler,
                                          bind_and_activate=False)
    srv.allow_reuse_address = True
    srv.socket = sock
    srv.server_activate()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    return srv, port


def _run_gate(port, log_path):
    # --task and --scorer are required, but with --parts malformed neither is
    # read; the stand-in never scores anything.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("a task the stand-in never runs")
        task = tf.name
    return subprocess.run(
        [sys.executable, str(TOOLS / "tool_grammar_gate.py"),
         "http://127.0.0.1:%d" % port, "fake",
         "--task", task, "--scorer", str(ROOT), "--out", str(ROOT / "build-utf8"),
         "--server-log", str(log_path), "--parts", "malformed"],
        capture_output=True, text=True, timeout=300)


def test_a_desync_on_a_grammar_on_arm_warns_and_counts():
    _Server.log_path, _Server.desync = None, True
    with tempfile.TemporaryDirectory() as out:
        log_path = str(Path(out) / "server.log")
        open(log_path, "w").close()
        _Server.log_path = log_path
        srv, port = _serve(_Server)
        try:
            r = _run_gate(port, log_path)
        finally:
            srv.shutdown()
    warnings = [ln for ln in r.stdout.splitlines() if ln.startswith("WARNING:")]
    # one WARNING per on arm (three configs per shape), each with a desync on
    # its decode line
    assert len(warnings) == 12, r.stdout
    # the warning names the request (shape and arm), not just the count
    assert any("undeclared-function" in w and "grammar=on" in w for w in warnings), r.stdout
    # and the summary carries the total across the on arms
    assert "grammar desyncs on on-arms: 12" in r.stdout, r.stdout
    # the desync warns, it does not fail the gate: every on arm was valid,
    # every shape was red off, the 400 check passed
    assert r.returncode == 0, r.stdout + r.stderr


def test_without_a_desync_the_line_is_quiet():
    _Server.log_path, _Server.desync = None, False
    with tempfile.TemporaryDirectory() as out:
        log_path = str(Path(out) / "server.log")
        open(log_path, "w").close()
        _Server.log_path = log_path
        srv, port = _serve(_Server)
        try:
            r = _run_gate(port, log_path)
        finally:
            srv.shutdown()
    assert not any(ln.startswith("WARNING:") for ln in r.stdout.splitlines()), r.stdout
    assert "grammar desyncs on on-arms: 0" in r.stdout, r.stdout
    assert r.returncode == 0, r.stdout + r.stderr
