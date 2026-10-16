"""Multi-threaded HTTP/1.1 server for llm-router.

Supports both OpenAI-compatible and Anthropic-compatible endpoints,
unified (`/v1/...`) and provider-scoped (`/<provider>/v1/...`) routing,
chunked/content-length request bodies, and unbuffered SSE streaming.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from .anthropic_api import (
    HandlerResult,
    handle_anthropic_complete,
    handle_anthropic_count_tokens,
    handle_anthropic_messages,
    handle_anthropic_models,
    make_anthropic_error,
)
from .config import RouterConfig
from .openai_api import (
    handle_openai_models,
    handle_openai_request,
    make_openai_error,
)
from .router import (
    RouteClassification,
    authenticate_client,
    classify_request_path,
    resolve_target,
)


def dispatch_request(
    config: RouterConfig,
    method: str,
    raw_target: str,
    headers: Mapping[str, str],
    raw_body: bytes,
) -> HandlerResult:
    """Pure request dispatcher: takes method, URL target, headers, and body -> HandlerResult."""
    split = urlsplit(raw_target)
    path = split.path or "/"
    query = split.query or ""

    # CORS preflight
    if method.upper() == "OPTIONS":
        return HandlerResult(
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta, OpenAI-Beta",
            },
            body=b"",
        )

    classification = classify_request_path(path, config)

    # Health check endpoint (does not require bearer)
    if classification.kind == "health":
        return HandlerResult(
            status=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "status": "ok",
                    "providers": [p.name for p in config.providers],
                    "mock_anthropic": not config.has_live_anthropic_provider(),
                }
            ).encode("utf-8"),
        )

    # Authenticate client key (`Authorization: Bearer ...` or `x-api-key: ...`)
    client_name = authenticate_client(headers, config)
    if client_name is None:
        if classification.protocol == "anthropic":
            return make_anthropic_error(
                401,
                "Invalid or missing client API key.",
                "authentication_error",
            )
        return make_openai_error(
            401,
            "Invalid or missing client API key.",
            err_type="authentication_error",
            code="invalid_api_key",
        )

    if classification.kind == "not_found":
        return make_openai_error(
            404,
            f"Unmapped endpoint path: {path}",
            err_type="invalid_request_error",
            code="not_found",
        )

    subpath = classification.subpath.rstrip("/") or "/"
    upper_method = method.upper()

    # Parse JSON body if present
    parsed_body: Optional[Dict[str, Any]] = None
    if raw_body:
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except Exception:
            if classification.protocol == "anthropic":
                return make_anthropic_error(400, "request body is not a JSON object")
            return make_openai_error(400, "request body is not a JSON object", code="invalid_json")
        if not isinstance(decoded, dict):
            if classification.protocol == "anthropic":
                return make_anthropic_error(400, "request body is not a JSON object")
            return make_openai_error(400, "request body is not a JSON object", code="invalid_json")
        parsed_body = decoded

    # Anthropic endpoint dispatch
    if classification.protocol == "anthropic":
        if upper_method == "GET" and (
            subpath in ("/v1/models", "/api/v1/models", "/models")
            or subpath.startswith(("/v1/models/", "/api/v1/models/", "/models/"))
        ):
            return handle_anthropic_models(subpath)

        if upper_method == "POST" and subpath in (
            "/v1/messages/count_tokens",
            "/api/v1/messages/count_tokens",
            "/messages/count_tokens",
            "/anthropic/v1/messages/count_tokens",
        ):
            return handle_anthropic_count_tokens(parsed_body)

        if upper_method == "POST" and subpath in (
            "/v1/complete",
            "/complete",
            "/anthropic/v1/complete",
        ):
            return handle_anthropic_complete(parsed_body)

        if upper_method == "POST" and subpath in (
            "/v1/messages",
            "/api/v1/messages",
            "/messages",
            "/anthropic/v1/messages",
        ):
            target = resolve_target(config, classification, parsed_body, query=query)
            return handle_anthropic_messages(target, parsed_body, raw_body, headers)

        return make_anthropic_error(
            404,
            f"Unsupported Anthropic endpoint: {method} {path}",
            "not_found_error",
        )

    # OpenAI endpoint dispatch
    if upper_method == "GET" and (
        subpath in ("/v1/models", "/api/v1/models", "/models")
        or subpath.startswith(("/v1/models/", "/api/v1/models/", "/models/"))
    ):
        return handle_openai_models(config, classification.provider_hint, subpath)

    try:
        target = resolve_target(config, classification, parsed_body, query=query)
    except KeyError as exc:
        return make_openai_error(404, str(exc), code="provider_not_found")

    return handle_openai_request(target, upper_method, parsed_body, raw_body, headers)


class RouterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], config: RouterConfig) -> None:
        self.router_config = config
        super().__init__(server_address, RouterRequestHandler)


class RouterRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default per-request access logs near credentials
        return

    def _read_body(self) -> bytes:
        transfer_encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in transfer_encoding:
            chunks = []
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                size_str = line.split(b";", 1)[0].strip()
                if not size_str:
                    break
                size = int(size_str, 16)
                if size == 0:
                    # Read trailing CRLF / trailers
                    while True:
                        trailer = self.rfile.readline()
                        if not trailer or trailer in (b"\r\n", b"\n"):
                            break
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)  # CRLF after chunk
            return b"".join(chunks)

        content_length_str = self.headers.get("Content-Length")
        if content_length_str:
            try:
                length = int(content_length_str)
            except ValueError:
                length = 0
            if length > 0:
                return self.rfile.read(length)
        return b""

    def _handle_any(self, method: str) -> None:
        try:
            raw_body = self._read_body()
            header_dict = {k: v for k, v in self.headers.items()}
            result = dispatch_request(
                config=self.server.router_config,  # type: ignore[attr-defined]
                method=method,
                raw_target=self.path,
                headers=header_dict,
                raw_body=raw_body,
            )
            self._send_result(result)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            sys.stderr.write(f"llm-router: internal error handling {method} {self.path}: {exc}\n")
            sys.stderr.flush()
            try:
                err_res = make_openai_error(500, "Internal router error", err_type="server_error")
                self._send_result(err_res)
            except Exception:
                self.close_connection = True

    def _send_result(self, result: HandlerResult) -> None:
        if result.is_stream and result.stream_chunks is not None:
            self.send_response(result.status)
            for k, v in result.headers.items():
                if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            conn_hdr = (self.headers.get("Connection") or "").lower()
            if conn_hdr == "close":
                self.send_header("Connection", "close")
                self.close_connection = True
            else:
                self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()
            for chunk in result.stream_chunks:
                if chunk:
                    header_line = f"{len(chunk):X}\r\n".encode("ascii")
                    self.wfile.write(header_line)
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        body = result.body or b""
        self.send_response(result.status)
        for k, v in result.headers.items():
            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        conn_hdr = (self.headers.get("Connection") or "").lower()
        if conn_hdr == "close":
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self) -> None:
        self._handle_any("GET")

    def do_POST(self) -> None:
        self._handle_any("POST")

    def do_OPTIONS(self) -> None:
        self._handle_any("OPTIONS")

    def do_HEAD(self) -> None:
        self._handle_any("HEAD")

    def do_DELETE(self) -> None:
        self._handle_any("DELETE")

    def do_PUT(self) -> None:
        self._handle_any("PUT")


def create_server(config: RouterConfig, host: Optional[str] = None, port: Optional[int] = None) -> RouterHTTPServer:
    bind_host = host or config.host or "127.0.0.1"
    bind_port = port if port is not None else config.port
    return RouterHTTPServer((bind_host, bind_port), config)
