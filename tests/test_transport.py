"""The real urllib transport against a throwaway HTTP server on 127.0.0.1 (no external network, no key)."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import plugin

ROUTES: dict[str, tuple[int, str]] = {
    "/api/cloud/memory/recall": (200, json.dumps({"results": [{"key": "k", "value": "v"}]})),
    "/api/cloud/memory/commit": (409, json.dumps({"error": "memory_exists", "message": "Memory already exists"})),
    "/api/cloud/memory/revise": (402, json.dumps({"error": "payment_required", "message": "Plan limit reached"})),
    "/api/cloud/memory/forget": (500, "<html>upstream error</html>"),
}


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        type(self).seen.append({"path": self.path, "headers": dict(self.headers),
                                "body": json.loads(self.rfile.read(length) or b"{}")})
        status, body = ROUTES.get(self.path, (404, "{}"))
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _client(base_url: str):
    return plugin._Client("bil_test_fake_key_for_unit_tests", base_url)


def test_success_parses_json(server):
    assert _client(server).recall("query", limit=3)["results"][0]["key"] == "k"
    request = _Handler.seen[-1]
    assert request["headers"]["Authorization"] == "Bearer bil_test_fake_key_for_unit_tests"
    assert request["body"] == {"query": "query", "profile": "balanced", "limit": 3}


@pytest.mark.parametrize(
    ("call", "status", "code"),
    [
        (lambda c: c.commit("k", "v", if_absent=True), 409, "memory_exists"),
        (lambda c: c.revise("k", "v"), 402, "payment_required"),
        (lambda c: c.forget("k", reason="r"), 500, None),
    ],
)
def test_http_errors_become_bilinc_errors(server, call, status, code):
    with pytest.raises(plugin.BilincError) as info:
        call(_client(server))
    assert info.value.status == status and info.value.code == code
    assert info.value.retryable is (status >= 500)


def test_refused_connection_is_retryable_error():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(plugin.BilincError) as info:
        _client(f"http://127.0.0.1:{port}").recall("query", limit=1, attempts=1, timeout=1.0)
    assert info.value.code == "connection_failed" and info.value.retryable
