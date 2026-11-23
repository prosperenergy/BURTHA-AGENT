"""Controllable local OpenAI-compatible endpoint for Burtha environments.

EXO (``127.0.0.1:52415``) is an Apple-Silicon cluster endpoint that is not
available in Cursor Cloud / Linux CI. This module replaces that dependency
with a process we own: a tiny OpenAI-compatible HTTP server suitable for
doctor probes, provider smoke tests, and agent bootstrapping when no remote
API key is configured.

Start:
    python -m hermes_cli.burtha_local_llm
    python -m hermes_cli.burtha_local_llm --port 18765

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (non-stream + stream=true)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
DEFAULT_MODEL = "burtha-local"
NOAUTH_PLACEHOLDER = "burtha-local-noauth"


def _chat_text(messages: list[dict[str, Any]]) -> str:
    """Deterministic, useful reply for smoke / doctor without a GPU."""
    last_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
                last_user = "\n".join(parts)
            else:
                last_user = str(content)
            break
    snippet = (last_user or "").strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:237] + "..."
    if not snippet:
        snippet = "(empty prompt)"
    return (
        "Burtha local endpoint online. "
        "EXO is not required in this environment. "
        f"Echo: {snippet}"
    )


def _models_payload() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": now,
                "owned_by": "burtha",
            }
        ],
    }


def _completion_payload(model: str, content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _sse_chunks(model: str, content: str) -> list[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    chunks = [
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or DEFAULT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or DEFAULT_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    lines.append("data: [DONE]\n\n")
    return lines


class BurthaLocalHandler(BaseHTTPRequestHandler):
    server_version = "BurthaLocalLLM/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in {"/v1/models", "/models"}:
            body = json.dumps(_models_payload()).encode("utf-8")
            self._send(200, body)
            return
        if path in {"/health", "/v1/health"}:
            self._send(200, b'{"ok":true,"service":"burtha-local"}')
            return
        self._send(404, b'{"error":{"message":"not found","type":"invalid_request_error"}}')

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send(404, b'{"error":{"message":"not found","type":"invalid_request_error"}}')
            return
        payload = self._read_json()
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        model = str(payload.get("model") or DEFAULT_MODEL)
        content = _chat_text(messages)
        if payload.get("stream"):
            chunks = "".join(_sse_chunks(model, content)).encode("utf-8")
            self._send(200, chunks, content_type="text/event-stream; charset=utf-8")
            return
        body = json.dumps(_completion_payload(model, content)).encode("utf-8")
        self._send(200, body)


class BurthaLocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, *, verbose: bool = False):
        super().__init__((host, port), BurthaLocalHandler)
        self.verbose = verbose


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, verbose: bool = False) -> BurthaLocalServer:
    server = BurthaLocalServer(host, port, verbose=verbose)
    thread = threading.Thread(target=server.serve_forever, name="burtha-local-llm", daemon=True)
    thread.start()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Burtha local OpenAI-compatible LLM endpoint")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    httpd = BurthaLocalServer(args.host, args.port, verbose=args.verbose)
    print(
        f"Burtha local LLM listening on http://{args.host}:{args.port}/v1 "
        f"(model={DEFAULT_MODEL})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Burtha local LLM.", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
