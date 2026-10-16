"""End-to-end tests for OpenAI-compatible and Anthropic-compatible endpoints."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_router.config import parse_config_sexp
from llm_router.server import create_server


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    recorded_requests: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self) -> None:
        raw = self._read_body()
        self.recorded_requests.append(
            {
                "method": "GET",
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": None,
            }
        )
        if self.path.startswith("/html502"):
            body = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
            self.send_response(502)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return

        payload = {"object": "list", "data": [{"id": "upstream-model", "object": "model"}]}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        # Include a huge Set-Cookie header to verify router strips it and doesn't overflow
        self.send_header("Set-Cookie", "__cf_bm=" + ("x" * 8192) + "; HttpOnly")
        self.send_header("Cf-Ray", "test-ray-123")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        raw = self._read_body()
        parsed = json.loads(raw.decode("utf-8")) if raw else None
        self.recorded_requests.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": parsed,
            }
        )

        if isinstance(parsed, dict) and parsed.get("model") == "trigger-502":
            body = b"<html><head><title>502 Bad Gateway</title></head><body>cloudflare</body></html>"
            self.send_response(502)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return

        if isinstance(parsed, dict) and parsed.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Set-Cookie", "__cf_bm=secretcookie")
            self.send_header("Connection", "close")
            self.end_headers()
            for token in ("Hello", " from ", str(parsed.get("model"))):
                chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": parsed.get("model"),
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        resp_payload = {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "model": parsed.get("model") if isinstance(parsed, dict) else "unknown",
            "received_path": self.path,
            "received_auth": self.headers.get("Authorization"),
            "received_x_api_key": self.headers.get("x-api-key"),
            "received_body": parsed,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Upstream OK ({parsed.get('model') if isinstance(parsed, dict) else ''})",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        data = json.dumps(resp_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Set-Cookie", "__cf_bm=" + ("y" * 8192))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)


class TestRouterEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FakeUpstreamHandler.recorded_requests = []
        cls.upstream_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        cls.upstream_port = cls.upstream_srv.server_address[1]
        cls.upstream_thread = threading.Thread(target=cls.upstream_srv.serve_forever, daemon=True)
        cls.upstream_thread.start()

        sexp = f"""
        (api-proxy
         (disabled #f)
         (port 0)
         (providers
          (openrouter (upstream "http://127.0.0.1:{cls.upstream_port}")
                      (path "/api")
                      (key "sk-or-secret-key")
                      (models
                       (argon (overrides (model . "google/gemini-4-argon")
                                         (reasoning . ((effort . "medium")))))
                       (fanout (overrides (model . "glm-5"))
                               (upstream "http://127.0.0.1:{cls.upstream_port}")
                               (path "/fanout-prefix")
                               (key "sk-fanout-secret"))))
          (kimi (upstream "http://127.0.0.1:{cls.upstream_port}")
                (key "sk-kimi-secret-key")
                (models
                 (k3 (overrides (model . "kimi-k3") (thinking . "max")))))
          (kimi/slash (upstream "http://127.0.0.1:{cls.upstream_port}")
                      (path "/slash-up")
                      (key "sk-slash-secret"))))
        """
        cls.config = parse_config_sexp(sexp)
        cls.router_srv = create_server(cls.config, host="127.0.0.1", port=0)
        cls.router_port = cls.router_srv.server_address[1]
        cls.router_thread = threading.Thread(target=cls.router_srv.serve_forever, daemon=True)
        cls.router_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.router_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.router_srv.shutdown()
        cls.router_srv.server_close()
        cls.upstream_srv.shutdown()
        cls.upstream_srv.server_close()

    def setUp(self) -> None:
        FakeUpstreamHandler.recorded_requests.clear()

    def _request(
        self,
        method: str,
        path: str,
        body: object = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        hdrs = {"Authorization": "Bearer local-hermes"}
        if headers is not None:
            hdrs = headers
        data = None
        if body is not None:
            if isinstance(body, (bytes, bytearray)):
                data = bytes(body)
            else:
                data = json.dumps(body).encode("utf-8")
                hdrs.setdefault("Content-Type", "application/json")
        req = Request(f"{self.base_url}{path}", data=data, headers=hdrs, method=method)
        try:
            with urlopen(req, timeout=10) as resp:
                return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
        except HTTPError as err:
            return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read()

    # -----------------------------------------------------------------------
    # Authentication & Error Gating
    # -----------------------------------------------------------------------

    def test_auth_gating_openai_and_anthropic(self) -> None:
        # Missing bearer -> 401 OpenAI error JSON
        status, _, raw = self._request("GET", "/v1/models", headers={})
        self.assertEqual(status, 401)
        err = json.loads(raw)
        self.assertEqual(err["error"]["type"], "authentication_error")

        # Invalid bearer -> 401
        status, _, _ = self._request("GET", "/openrouter/v1/models", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(status, 401)

        # Missing bearer on Anthropic endpoint -> 401 Anthropic error JSON
        status, _, raw = self._request(
            "POST",
            "/v1/messages",
            body={"model": "claude-sonnet-4-20250514", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 401)
        err = json.loads(raw)
        self.assertEqual(err["type"], "error")
        self.assertEqual(err["error"]["type"], "authentication_error")

        # All 3 client keys (local-hermes, local-pi, local-claude) work
        for token in ("local-hermes", "local-pi", "local-claude"):
            status, _, _ = self._request("GET", "/v1/models", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(status, 200)

        # x-api-key header also authenticates
        status, _, _ = self._request("GET", "/v1/models", headers={"x-api-key": "local-claude"})
        self.assertEqual(status, 200)

    def test_unmapped_prefix_and_bad_json(self) -> None:
        status, _, raw = self._request("GET", "/nope/x")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(raw))

        status, _, raw = self._request(
            "POST",
            "/openrouter/v1/chat/completions",
            body=b"not-valid-json",
            headers={"Authorization": "Bearer local-hermes", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

        status, _, raw = self._request(
            "POST",
            "/openrouter/v1/chat/completions",
            body=b"[1, 2, 3]",
            headers={"Authorization": "Bearer local-hermes", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    # -----------------------------------------------------------------------
    # OpenAI Simulated Models & Routing
    # -----------------------------------------------------------------------

    def test_simulated_openai_models_catalog(self) -> None:
        # Unified /v1/models returns configured aliases + mock anthropic models without hitting upstream
        status, hdrs, raw = self._request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertNotIn("set-cookie", hdrs)
        self.assertEqual(len(FakeUpstreamHandler.recorded_requests), 0)
        data = json.loads(raw)
        self.assertEqual(data["object"], "list")
        ids = [m["id"] for m in data["data"]]
        self.assertIn("argon", ids)
        self.assertIn("k3", ids)
        self.assertIn("fanout", ids)
        self.assertIn("claude-sonnet-4-20250514", ids)

        # Provider-scoped /openrouter/v1/models lists openrouter's models first, plus routable aliases
        status, _, raw = self._request("GET", "/openrouter/v1/models")
        self.assertEqual(status, 200)
        or_ids = [m["id"] for m in json.loads(raw)["data"]]
        self.assertEqual(or_ids[0], "argon")
        self.assertIn("k3", or_ids)

        # Single model detail endpoint /v1/models/argon
        status, _, raw = self._request("GET", "/v1/models/argon")
        self.assertEqual(status, 200)
        m_obj = json.loads(raw)
        self.assertEqual(m_obj["id"], "argon")
        self.assertEqual(m_obj["root"], "google/gemini-4-argon")

        # Unknown model detail -> 404
        status, _, _ = self._request("GET", "/v1/models/does-not-exist")
        self.assertEqual(status, 404)

    def test_openai_chat_completions_routing_and_overrides(self) -> None:
        # 1. Unified endpoint /v1/chat/completions with model="argon" -> openrouter
        status, hdrs, raw = self._request(
            "POST",
            "/v1/chat/completions",
            body={
                "model": "argon",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 32,
            },
        )
        self.assertEqual(status, 200)
        self.assertNotIn("set-cookie", hdrs)
        self.assertEqual(hdrs.get("x-routed-provider"), "openrouter")
        self.assertEqual(hdrs.get("x-routed-model"), "google/gemini-4-argon")
        resp = json.loads(raw)
        self.assertEqual(resp["received_path"], "/api/v1/chat/completions")
        self.assertEqual(resp["received_auth"], "Bearer sk-or-secret-key")
        self.assertEqual(resp["received_body"]["model"], "google/gemini-4-argon")
        self.assertEqual(resp["received_body"]["reasoning"], {"effort": "medium"})
        self.assertEqual(resp["received_body"]["max_tokens"], 32)

        # 2. Cross-provider routing: requesting model="k3" on /openrouter/v1/chat/completions
        #    automatically routes to the `kimi` provider!
        status, hdrs, raw = self._request(
            "POST",
            "/openrouter/v1/chat/completions",
            body={
                "model": "k3",
                "messages": [{"role": "user", "content": "Hi Kimi"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs.get("x-routed-provider"), "kimi")
        resp = json.loads(raw)
        self.assertEqual(resp["received_path"], "/v1/chat/completions")
        self.assertEqual(resp["received_auth"], "Bearer sk-kimi-secret-key")
        self.assertEqual(resp["received_body"]["model"], "kimi-k3")
        self.assertEqual(resp["received_body"]["thinking"], "max")

        # 3. Per-model upstream fan-out (`fanout` model under `openrouter`)
        status, _, raw = self._request(
            "POST",
            "/openrouter/v1/chat/completions",
            body={"model": "fanout", "messages": [{"role": "user", "content": "Fanout"}]},
        )
        self.assertEqual(status, 200)
        resp = json.loads(raw)
        self.assertEqual(resp["received_path"], "/fanout-prefix/v1/chat/completions")
        self.assertEqual(resp["received_auth"], "Bearer sk-fanout-secret")
        self.assertEqual(resp["received_body"]["model"], "glm-5")

        # 4. Provider with slash in name (`kimi/slash`) and query string preservation
        status, _, raw = self._request(
            "POST",
            "/kimi/slash/v1/chat/completions?beta=1",
            body={"model": "raw-model", "messages": [{"role": "user", "content": "Slash"}]},
        )
        self.assertEqual(status, 200)
        resp = json.loads(raw)
        self.assertEqual(resp["received_path"], "/slash-up/v1/chat/completions?beta=1")
        self.assertEqual(resp["received_auth"], "Bearer sk-slash-secret")
        self.assertEqual(resp["received_body"]["model"], "raw-model")

    def test_openai_streaming_sse_and_html_502_normalization(self) -> None:
        # Streaming SSE
        status, hdrs, raw = self._request(
            "POST",
            "/v1/chat/completions",
            body={
                "model": "argon",
                "stream": True,
                "messages": [{"role": "user", "content": "Stream test"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", hdrs.get("content-type", ""))
        self.assertNotIn("set-cookie", hdrs)
        text = raw.decode("utf-8")
        self.assertIn("data: [DONE]", text)
        self.assertIn("google/gemini-4-argon", text)

        # Upstream returning HTML 502 is converted to structured OpenAI JSON error
        status, hdrs, raw = self._request(
            "POST",
            "/openrouter/v1/chat/completions",
            body={
                "model": "trigger-502",
                "messages": [{"role": "user", "content": "fail"}],
            },
        )
        self.assertEqual(status, 502)
        self.assertIn("application/json", hdrs.get("content-type", ""))
        err = json.loads(raw)
        self.assertEqual(err["error"]["type"], "upstream_error")
        self.assertEqual(err["error"]["code"], "bad_gateway")

    # -----------------------------------------------------------------------
    # Anthropic-Compatible Mock Endpoints
    # -----------------------------------------------------------------------

    def test_anthropic_messages_non_streaming_and_exact_reply(self) -> None:
        for path in ("/v1/messages", "/anthropic/v1/messages", "/kimi/anthropic/v1/messages"):
            status, hdrs, raw = self._request(
                "POST",
                path,
                body={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 128,
                    "system": "You are a helpful assistant.",
                    "messages": [
                        {"role": "user", "content": "Reply with exactly: ANTHROPIC-MOCK-OK"}
                    ],
                },
                headers={
                    "x-api-key": "local-claude",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            self.assertEqual(status, 200, f"failed on {path}")
            self.assertIn("request-id", hdrs)
            msg = json.loads(raw)
            self.assertEqual(msg["type"], "message")
            self.assertEqual(msg["role"], "assistant")
            self.assertEqual(msg["model"], "claude-sonnet-4-20250514")
            self.assertEqual(msg["stop_reason"], "end_turn")
            self.assertEqual(msg["content"][0]["type"], "text")
            self.assertEqual(msg["content"][0]["text"], "ANTHROPIC-MOCK-OK")
            self.assertIn("input_tokens", msg["usage"])
            self.assertIn("output_tokens", msg["usage"])

    def test_anthropic_messages_streaming_and_thinking(self) -> None:
        status, hdrs, raw = self._request(
            "POST",
            "/v1/messages",
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "stream": True,
                "thinking": {"type": "enabled", "budget_tokens": 512},
                "messages": [
                    {"role": "user", "content": "Reply with exactly: STREAM-OK"}
                ],
            },
            headers={
                "x-api-key": "local-claude",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", hdrs.get("content-type", ""))
        sse_text = raw.decode("utf-8")
        self.assertIn("event: message_start", sse_text)
        self.assertIn("event: content_block_start", sse_text)
        self.assertIn("thinking_delta", sse_text)
        self.assertIn("signature_delta", sse_text)
        self.assertIn("text_delta", sse_text)
        self.assertIn("event: message_delta", sse_text)
        self.assertIn("event: message_stop", sse_text)

        # Reconstruct streamed text deltas
        reconstructed = []
        for line in sse_text.splitlines():
            if line.startswith("data: "):
                obj = json.loads(line[6:])
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        reconstructed.append(delta.get("text", ""))
        self.assertEqual("".join(reconstructed), "STREAM-OK")

    def test_anthropic_tool_use_and_count_tokens_and_models(self) -> None:
        # Tool call round-trip
        status, _, raw = self._request(
            "POST",
            "/v1/messages",
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    }
                ],
                "tool_choice": {"type": "any"},
                "messages": [{"role": "user", "content": "Tokyo weather"}],
            },
            headers={"x-api-key": "local-claude", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        msg = json.loads(raw)
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(msg["content"][0]["type"], "tool_use")
        self.assertEqual(msg["content"][0]["name"], "get_weather")
        tool_id = msg["content"][0]["id"]

        # Follow-up with tool_result
        status, _, raw = self._request(
            "POST",
            "/v1/messages",
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": "Tokyo weather"},
                    {"role": "assistant", "content": msg["content"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tool_id, "content": "Sunny 22C"}
                        ],
                    },
                ],
            },
            headers={"x-api-key": "local-claude", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        msg2 = json.loads(raw)
        self.assertEqual(msg2["stop_reason"], "end_turn")
        self.assertIn("Sunny 22C", msg2["content"][0]["text"])

        # Count tokens
        status, _, raw = self._request(
            "POST",
            "/v1/messages/count_tokens",
            body={
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Count my tokens please"}],
            },
            headers={"x-api-key": "local-claude", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertGreater(json.loads(raw)["input_tokens"], 0)

        # Anthropic models list & detail
        status, _, raw = self._request(
            "GET",
            "/anthropic/v1/models",
            headers={"x-api-key": "local-claude"},
        )
        self.assertEqual(status, 200)
        anth_models = json.loads(raw)
        self.assertTrue(len(anth_models["data"]) >= 3)

        status, _, raw = self._request(
            "GET",
            "/anthropic/v1/models/claude-sonnet-4-20250514",
            headers={"x-api-key": "local-claude"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["id"], "claude-sonnet-4-20250514")

    def test_anthropic_live_upstream_forwarding_when_configured(self) -> None:
        sexp = f"""
        (api-proxy
         (disabled #f)
         (port 0)
         (providers
          (anthropic (upstream "http://127.0.0.1:{self.upstream_port}")
                     (key "sk-ant-real-key")
                     (models
                      (sonnet (overrides (model . "claude-sonnet-4-20250514")))))))
        """
        cfg = parse_config_sexp(sexp)
        self.assertTrue(cfg.has_live_anthropic_provider())
        srv = create_server(cfg, host="127.0.0.1", port=0)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            req = Request(
                f"http://127.0.0.1:{port}/v1/messages",
                data=json.dumps(
                    {
                        "model": "sonnet",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": "Hello real anthropic"}],
                    }
                ).encode("utf-8"),
                headers={
                    "x-api-key": "local-claude",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read())
                self.assertEqual(data["received_path"], "/v1/messages")
                self.assertEqual(data["received_x_api_key"], "sk-ant-real-key")
                self.assertEqual(data["received_body"]["model"], "claude-sonnet-4-20250514")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_openai_mock_endpoints_and_legacy_complete(self) -> None:
        # Requesting model="mock" on /v1/embeddings, /v1/responses, /v1/completions, /v1/chat/completions
        sexp = """
        (api-proxy
         (disabled #f)
         (port 0)
         (providers
          (local-mock (mock #t)
                      (models
                       (mock-gpt (overrides (model . "gpt-4o-mock")))))))
        """
        cfg = parse_config_sexp(sexp)
        srv = create_server(cfg, host="127.0.0.1", port=0)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            for endpoint, payload, expected_obj in (
                (
                    "/v1/chat/completions",
                    {"model": "mock-gpt", "messages": [{"role": "user", "content": "Reply with exactly: MOCK-CHAT-OK"}]},
                    "chat.completion",
                ),
                (
                    "/v1/completions",
                    {"model": "mock-gpt", "prompt": "Reply with exactly: MOCK-CMPL-OK"},
                    "text_completion",
                ),
                (
                    "/v1/responses",
                    {"model": "mock-gpt", "input": "Reply with exactly: MOCK-RESP-OK"},
                    "response",
                ),
                (
                    "/v1/embeddings",
                    {"model": "mock-gpt", "input": "embed me"},
                    "list",
                ),
            ):
                req = Request(
                    f"http://127.0.0.1:{port}{endpoint}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Authorization": "Bearer local-hermes", "Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=10) as resp:
                    self.assertEqual(resp.status, 200)
                    out = json.loads(resp.read())
                    self.assertEqual(out["object"], expected_obj)
        finally:
            srv.shutdown()
            srv.server_close()

        # Anthropic legacy /v1/complete endpoint
        status, _, raw = self._request(
            "POST",
            "/v1/complete",
            body={"model": "claude-2.1", "prompt": "Reply with exactly: LEGACY-OK", "max_tokens_to_sample": 32},
            headers={"x-api-key": "local-claude", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["completion"], "LEGACY-OK")

    def test_kimi_coding_provider_scoped_anthropic_and_pi_models(self) -> None:
        # Verify the exact kimi-coding config from dwi/services/api-proxy.scm works for
        # Claude Code (`POST /kimi-coding/v1/messages` with `fast`, `max`, and `glm` fan-out)
        sexp = f"""
        (api-proxy
         (disabled #f)
         (port 0)
         (providers
          (kimi (upstream "http://127.0.0.1:{self.upstream_port}")
                (key "sk-kimi")
                (models
                 (k3 (overrides (model . "kimi-k3")))))
          (kimi-coding
           (upstream "http://127.0.0.1:{self.upstream_port}")
           (path "/coding")
           (key "sk-kimi-coding")
           (models
            (fast (overrides (model . "kimi-k2-turbo")))
            (max  (overrides (model . "k3") (thinking . "max")))
            (glm  (overrides (model . "glm-5"))
                  (upstream "http://127.0.0.1:{self.upstream_port}")
                  (path "/anthropic")
                  (key "sk-glm-real"))))))
        """
        cfg = parse_config_sexp(sexp)
        srv = create_server(cfg, host="127.0.0.1", port=0)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            # 1. Verify GET /kimi/v1/models includes all 5 models from config/pi/models.json
            req = Request(
                f"http://127.0.0.1:{port}/kimi/v1/models",
                headers={"Authorization": "Bearer local-pi"},
                method="GET",
            )
            with urlopen(req, timeout=10) as resp:
                ids = [m["id"] for m in json.loads(resp.read())["data"]]
                for expected_id in ("k3", "kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k2.6"):
                    self.assertIn(expected_id, ids)

            # 2. POST /kimi-coding/v1/messages with model="fast" -> /coding/v1/messages
            req = Request(
                f"http://127.0.0.1:{port}/kimi-coding/v1/messages",
                data=json.dumps(
                    {"model": "fast", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
                ).encode("utf-8"),
                headers={"Authorization": "Bearer local-claude", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                self.assertEqual(data["received_path"], "/coding/v1/messages")
                self.assertEqual(data["received_auth"], "Bearer sk-kimi-coding")
                self.assertEqual(data["received_x_api_key"], "sk-kimi-coding")
                self.assertEqual(data["received_body"]["model"], "kimi-k2-turbo")

            # 3. POST /kimi-coding/v1/messages with model="glm" -> /anthropic/v1/messages with sk-glm-real
            req = Request(
                f"http://127.0.0.1:{port}/kimi-coding/v1/messages",
                data=json.dumps(
                    {"model": "glm", "max_tokens": 64, "messages": [{"role": "user", "content": "hi glm"}]}
                ).encode("utf-8"),
                headers={"x-api-key": "local-claude", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                self.assertEqual(data["received_path"], "/anthropic/v1/messages")
                self.assertEqual(data["received_auth"], "Bearer sk-glm-real")
                self.assertEqual(data["received_x_api_key"], "sk-glm-real")
                self.assertEqual(data["received_body"]["model"], "glm-5")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
