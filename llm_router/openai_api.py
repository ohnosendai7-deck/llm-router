"""OpenAI-compatible API endpoint simulator and upstream forwarder.

Handles:
  - GET  /v1/models and GET /<provider>/v1/models (and /v1/models/<id>)
  - POST /v1/chat/completions and POST /<provider>/v1/chat/completions
  - POST /v1/completions
  - POST /v1/responses
  - POST /v1/embeddings
  - Arbitrary provider-scoped OpenAI/v1beta subpaths with model override
    resolution, per-model upstream fan-out, SSE streaming, and clean JSON
    error normalization.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .anthropic_api import HandlerResult
from .config import RouterConfig
from .router import ResolvedTarget, build_openai_models_catalog
from .upstream import UpstreamError, request_upstream


def make_openai_error(
    status: int,
    message: str,
    err_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> HandlerResult:
    payload = {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }
    return HandlerResult(
        status=status,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def handle_openai_models(
    config: RouterConfig,
    provider_hint: Optional[str],
    subpath: str,
) -> HandlerResult:
    """Serve OpenAI `/v1/models` and `/v1/models/<model_id>` endpoints."""
    if provider_hint and provider_hint not in ("openai", "anthropic"):
        if config.get_provider(provider_hint) is None:
            return make_openai_error(
                404,
                f"Unknown provider: {provider_hint}",
                code="provider_not_found",
            )

    catalog = build_openai_models_catalog(config, provider_hint=provider_hint)
    clean = subpath.rstrip("/")

    for prefix in ("/v1/models/", "/api/v1/models/", "/models/"):
        if clean.startswith(prefix):
            model_id = clean[len(prefix):]
            for item in catalog:
                if item["id"] == model_id:
                    return HandlerResult(
                        status=200,
                        headers={"Content-Type": "application/json"},
                        body=json.dumps(item).encode("utf-8"),
                    )
            return make_openai_error(
                404,
                f"The model '{model_id}' does not exist",
                code="model_not_found",
                param="model",
            )

    payload = {
        "object": "list",
        "data": catalog,
    }
    return HandlerResult(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def _extract_openai_prompt(parsed_body: Dict[str, Any]) -> str:
    messages = parsed_body.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and isinstance(b.get("text"), str)
                    )
    if isinstance(parsed_body.get("prompt"), str):
        return parsed_body["prompt"]
    if isinstance(parsed_body.get("input"), str):
        return parsed_body["input"]
    return ""


def _synthesize_openai_text(prompt: str, model: str) -> str:
    m = re.search(r"(?:reply|respond|say)\s+(?:with\s+)?exactly\s*:\s*(.+)", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cleaned = prompt.strip()
    if not cleaned:
        return f"Mock response from {model}."
    return f"[Mock {model}] {cleaned}"


def handle_openai_mock_request(
    target: ResolvedTarget,
    parsed_body: Dict[str, Any],
) -> HandlerResult:
    """Serve a simulated OpenAI response when `target.mock` is True."""
    eff_body = target.merged_body or parsed_body
    model = str(eff_body.get("model") or target.effective_model or "mock-model")
    sub = target.subpath.rstrip("/")
    created = int(time.time())
    prompt = _extract_openai_prompt(eff_body)
    reply = _synthesize_openai_text(prompt, model)
    req_hash = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()[:16]

    if sub.endswith("/embeddings"):
        payload = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.01 * ((i % 10) + 1) for i in range(16)],
                }
            ],
            "model": model,
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
        return HandlerResult(
            status=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

    if sub.endswith("/responses"):
        payload = {
            "id": f"resp_{req_hash}",
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{req_hash}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": reply,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        }
        return HandlerResult(
            status=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

    if sub.endswith("/completions") and not sub.endswith("/chat/completions"):
        payload = {
            "id": f"cmpl-{req_hash}",
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "text": reply,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }
        return HandlerResult(
            status=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

    # Default: Chat Completions (`/v1/chat/completions`)
    stream = bool(eff_body.get("stream", False))
    include_usage = bool(
        isinstance(eff_body.get("stream_options"), dict)
        and eff_body["stream_options"].get("include_usage")
    )
    chat_id = f"chatcmpl-{req_hash}"

    if stream:
        def _sse_chunks() -> Iterable[bytes]:
            role_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(role_chunk, separators=(',', ':'))}\n\n".encode("utf-8")

            content_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": reply},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk, separators=(',', ':'))}\n\n".encode("utf-8")

            stop_chunk: Dict[str, Any] = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            if include_usage:
                stop_chunk["usage"] = {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                }
            yield f"data: {json.dumps(stop_chunk, separators=(',', ':'))}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return HandlerResult(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            stream_chunks=_sse_chunks(),
        )

    payload = {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }
    return HandlerResult(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def handle_openai_request(
    target: ResolvedTarget,
    method: str,
    parsed_body: Optional[Dict[str, Any]],
    raw_body: bytes,
    client_headers: Mapping[str, str],
) -> HandlerResult:
    """Forward an OpenAI-compatible request to its resolved upstream (or mock)."""
    if target.mock:
        return handle_openai_mock_request(target, parsed_body or {})

    if target.merged_body is not None:
        body_to_send: Optional[bytes] = json.dumps(target.merged_body).encode("utf-8")
    elif raw_body:
        body_to_send = raw_body
    else:
        body_to_send = None

    stream = bool(isinstance(target.merged_body, dict) and target.merged_body.get("stream", False))

    try:
        up_resp = request_upstream(
            upstream=target.upstream,
            method=method,
            url_path=target.upstream_url_path,
            api_key=target.api_key,
            protocol=target.protocol,
            body_bytes=body_to_send,
            client_headers=client_headers,
            stream=stream,
        )
    except UpstreamError as exc:
        return make_openai_error(
            exc.status_code,
            exc.message,
            err_type="upstream_error",
            code="bad_gateway",
        )

    resp_headers = dict(up_resp.headers)
    resp_headers["X-Routed-Provider"] = target.provider_name
    if target.effective_model:
        resp_headers["X-Routed-Model"] = target.effective_model

    if up_resp.is_stream:
        return HandlerResult(
            status=up_resp.status,
            headers=resp_headers,
            stream_chunks=up_resp.iter_chunks(),
        )

    # If upstream returned a 5xx HTML page (e.g. Cloudflare 502 HTML), wrap it in JSON
    ctype = resp_headers.get("Content-Type", resp_headers.get("content-type", "")).lower()
    if up_resp.status >= 500 and ("text/html" in ctype or up_resp.body.lstrip().startswith(b"<")):
        return make_openai_error(
            up_resp.status,
            f"Upstream provider '{target.provider_name}' ({target.upstream}) returned HTTP {up_resp.status}",
            err_type="upstream_error",
            code="bad_gateway",
        )

    if "Content-Type" not in resp_headers and "content-type" not in resp_headers:
        resp_headers["Content-Type"] = "application/json"

    return HandlerResult(
        status=up_resp.status,
        headers=resp_headers,
        body=up_resp.body,
    )
