#!/usr/bin/env python3
"""Focused test for scripts/api-constrained-smoke.sh.

The observed negative (published v8, both official layouts) was that the smoke
omitted thinking controls, so the server's default high-effort thinking consumed
the 96-token budget and returned empty content with finish_reason=length. The fix
sets `thinking: {type: "disabled"}` explicitly.

This test runs the real script against a stub OpenAI-compatible server that
reproduces that behaviour: it returns empty/length unless thinking is disabled.
It asserts (a) the fixed script passes, (b) a variant with the thinking field
stripped fails, proving the assertions were not weakened, and (c) all three
generation payloads still disable thinking.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "scripts" / "api-constrained-smoke.sh"

GOOD_CONTENT = '{"city":"Taipei","temperature":27,"condition":"sunny"}'
TOOL_ARGS = {"city": "Taipei", "units": "metric"}


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        thinking_disabled = request.get("thinking", {}).get("type") == "disabled"
        response_format = request.get("response_format", {})
        schema = response_format.get("json_schema", {})

        if schema.get("name") == "invalid_strict_schema":
            self._send_json(
                400,
                {"error": {"type": "invalid_request_error",
                           "param": "response_format.json_schema.schema",
                           "message": "unsupported schema"}},
            )
            return

        if request.get("stream"):
            if not thinking_disabled:
                self._sse([{"choices": [{"delta": {}, "finish_reason": "length"}]}])
                return
            first = json.dumps(TOOL_ARGS)[:9]
            second = json.dumps(TOOL_ARGS)[9:]
            self._sse([
                {"choices": [{"delta": {"tool_calls": [
                    {"function": {"arguments": first}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"function": {"arguments": second}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ])
            return

        if not thinking_disabled:
            self._send_json(200, {"choices": [{
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "",
                            "reasoning_content": "thinking used the budget"},
            }]})
            return

        self._send_json(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": GOOD_CONTENT},
        }]})


@pytest.fixture()
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def run_smoke(script: Path, url: str):
    return subprocess.run(["bash", str(script), url], capture_output=True, text=True)


pytestmark = pytest.mark.skipif(
    not (shutil.which("curl") and shutil.which("jq") and shutil.which("sed")),
    reason="curl, jq and sed are required by the smoke script",
)


def test_smoke_passes_with_thinking_disabled(stub_server):
    result = run_smoke(SMOKE, stub_server)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "json_schema=true" in result.stdout


def test_smoke_would_fail_without_the_fix(stub_server, tmp_path):
    """The original negative must still be reproducible: stripping the thinking
    control makes the exact-output assertions fail, so the fix is load-bearing
    rather than a weakened check."""
    original = SMOKE.read_text()
    stripped = original.replace('    thinking: {type: "disabled"},\n', "")
    assert stripped != original
    variant = tmp_path / "api-constrained-smoke-nothink.sh"
    variant.write_text(stripped)
    result = run_smoke(variant, stub_server)
    assert result.returncode != 0


def test_all_generation_payloads_disable_thinking():
    text = SMOKE.read_text()
    assert text.count('thinking: {type: "disabled"},') == 3
    for marker in ("schema_payload", "combined_payload", "tool_payload"):
        block = text.split(marker, 1)[1].split(")=\"", 1)[0]
        assert 'thinking: {type: "disabled"}' in block, marker


def test_checks_are_not_weakened():
    text = SMOKE.read_text()
    assert '.choices[0].finish_reason == "stop"' in text
    assert 'fromjson' in text
    assert "invalid_status" in text and '"$invalid_status" == 400' in text
    assert 'invalid_request_error' in text
    assert "tool_calls" in text
