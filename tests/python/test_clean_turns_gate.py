"""The step 1 gate's verdict logic, without a card.

The gate's whole value is that it can say FAILED and NOT SHOWN: a pair that
never provoked the empty stop proves nothing, and a pair that goes empty only
WITH the mask means the mask itself ends turns. Neither may look like a pass.
This tests the classifier and the exit code against a stand-in server, so the
gate is checked before anyone spends a card window on it.
"""
import json
import http.server
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import clean_turns_gate as gate  # noqa: E402


def test_only_an_empty_stop_counts_as_the_defect():
    assert gate.classify("EMPTY-STOP", "answered") == "fixed"
    assert gate.classify("EMPTY-STOP", "EMPTY-STOP") == "red"


def test_a_cap_ended_empty_is_not_the_defect():
    # The mask is about the turn-ending token winning a step, not max_tokens.
    assert gate.classify("EMPTY-LENGTH", "answered") == "neither"
    assert gate.classify("EMPTY-LENGTH", "EMPTY-LENGTH") == "neither"


def test_a_quiet_arm_proves_nothing():
    assert gate.classify("answered", "answered") == "neither"


def test_empty_only_with_the_mask_is_the_opposite_defect():
    assert gate.classify("answered", "EMPTY-STOP") == "flipped"


class _Server(http.server.BaseHTTPRequestHandler):
    """Answers whatever it is asked; `body` decides the shape from the request."""

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        content, finish = self.shape(body)
        raw = json.dumps({"choices": [{"finish_reason": finish, "index": 0,
                                       "message": {"role": "assistant",
                                                   "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def shape(self, body):
        return "always an answer", "stop"

    def log_message(self, *a):
        pass


def _serve(handler, timeout=5.0):
    # Bind once and hand the socket over: asking for port 0 twice would race
    # another test in the same run for the same port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler,
                                          bind_and_activate=False)
    srv.allow_reuse_address = True
    srv.socket = sock          # already bound: do not bind a second time
    srv.server_activate()      # ...but it still has to listen
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    return srv, port


def _run_gate(port):
    return subprocess.run(
        [sys.executable, str(TOOLS / "clean_turns_gate.py"),
         "http://127.0.0.1:%d" % port, "fake"],
        capture_output=True, text=True, timeout=120)


def test_the_gate_refuses_to_pass_when_nothing_was_provoked():
    srv, port = _serve(_Server)
    try:
        r = _run_gate(port)
    finally:
        srv.shutdown()
    assert r.returncode == 1, r.stdout + r.stderr
    assert "NOT SHOWN" in r.stdout, r.stdout


def test_the_gate_fails_when_the_mask_does_not_clear_the_stop():
    class AlwaysEmpty(_Server):
        def shape(self, body):
            return "", "stop"

    srv, port = _serve(AlwaysEmpty)
    try:
        r = _run_gate(port)
    finally:
        srv.shutdown()
    assert r.returncode == 1, r.stdout + r.stderr
    assert "GATE FAILED" in r.stdout, r.stdout


def test_the_gate_passes_when_the_pair_actually_pairs():
    class FixedByMask(_Server):
        def shape(self, body):
            # empty without the field, an answer with it: the red case and its
            # fix, served by one stand-in
            if body.get("think_eog_mask"):
                return "an answer", "stop"
            return "", "stop"

    srv, port = _serve(FixedByMask)
    try:
        r = _run_gate(port)
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GATE PASSED" in r.stdout, r.stdout
