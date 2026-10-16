"""Anthropic-compatible API endpoints and spec-compliant mock engine.

Supports:
  - POST /v1/messages (and /anthropic/v1/messages, /<provider>/v1/messages)
    with non-streaming JSON, SSE event streaming (`message_start`, `ping`,
    `content_block_start`, `content_block_delta`, `content_block_stop`,
    `message_delta`, `message_stop`), extended thinking blocks, and tool_use /
    tool_result multi-turn flows.
  - POST /v1/messages/count_tokens
  - POST /v1/complete (legacy Text Completions API)
  - GET  /anthropic/v1/models and /anthropic/v1/models/<model_id>
  - Live upstream forwarding when a non-mock Anthropic provider is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .router import MOCK_ANTHROPIC_MODELS, ResolvedTarget
from .upstream import UpstreamError, request_upstream


@dataclass
class HandlerResult:
    status: int
    headers: Dict[str, str]
    body: bytes = b""
    stream_chunks: Optional[Iterable[bytes]] = None

    @property
    def is_stream(self) -> bool:
        return self.stream_chunks is not None


def make_anthropic_error(
    status: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> HandlerResult:
    payload = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    return HandlerResult(
        status=status,
        headers={
            "Content-Type": "application/json",
            "request-id": "req_mock_error",
        },
        body=json.dumps(payload).encode("utf-8"),
    )


def _deterministic_id(prefix: str, seed_text: str) -> str:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif btype == "tool_result":
                    sub = _extract_text_from_content(block.get("content", ""))
                    parts.append(f"[tool_result:{sub}]")
        return " ".join(p for p in parts if p)
    return ""


def _has_tool_result(messages: List[Any]) -> Optional[str]:
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict):
        return None
    content = last.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return _extract_text_from_content(block.get("content", "ok"))
    return None


def _estimate_tokens(parsed_body: Dict[str, Any]) -> int:
    raw = json.dumps(parsed_body, ensure_ascii=False)
    return max(1, len(raw) // 4)


def _synthesize_reply_text(prompt_text: str, model: str, system_text: str = "") -> str:
    m = re.search(r"(?:reply|respond|say)\s+(?:with\s+)?exactly\s*:\s*(.+)", prompt_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cleaned = prompt_text.strip()
    if not cleaned:
        return f"Mock response from {model}."
    return f"[Mock {model}] {cleaned}"


def _format_sse_event(event_name: str, data: Dict[str, Any]) -> bytes:
    line = f"event: {event_name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
    return line.encode("utf-8")


def handle_anthropic_models(subpath: str) -> HandlerResult:
    clean = subpath.rstrip("/")
    # Match `/v1/models/<model_id>` or `/models/<model_id>`
    for prefix in ("/v1/models/", "/api/v1/models/", "/models/"):
        if clean.startswith(prefix):
            model_id = clean[len(prefix):]
            for m in MOCK_ANTHROPIC_MODELS:
                if m["id"] == model_id:
                    return HandlerResult(
                        status=200,
                        headers={"Content-Type": "application/json"},
                        body=json.dumps(m).encode("utf-8"),
                    )
            # Synthesize a mock model card for any claude-* model requested by ID
            if model_id.startswith("claude-"):
                card = {
                    "id": model_id,
                    "type": "model",
                    "object": "model",
                    "display_name": f"{model_id} (Mock)",
                    "created": 1747180800,
                    "created_at": "2025-05-14T00:00:00Z",
                    "owned_by": "anthropic-mock",
                }
                return HandlerResult(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(card).encode("utf-8"),
                )
            return make_anthropic_error(404, f"Model '{model_id}' not found", "not_found_error")

    payload = {
        "object": "list",
        "data": MOCK_ANTHROPIC_MODELS,
        "has_more": False,
        "first_id": MOCK_ANTHROPIC_MODELS[0]["id"],
        "last_id": MOCK_ANTHROPIC_MODELS[-1]["id"],
    }
    return HandlerResult(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def handle_anthropic_count_tokens(parsed_body: Optional[Dict[str, Any]]) -> HandlerResult:
    if not isinstance(parsed_body, dict):
        return make_anthropic_error(400, "Request body must be a JSON object")
    messages = parsed_body.get("messages")
    if not isinstance(messages, list) or not messages:
        return make_anthropic_error(400, "messages: field required and must be a non-empty list")
    tokens = _estimate_tokens(parsed_body)
    return HandlerResult(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"input_tokens": tokens}).encode("utf-8"),
    )


def handle_anthropic_complete(parsed_body: Optional[Dict[str, Any]]) -> HandlerResult:
    if not isinstance(parsed_body, dict):
        return make_anthropic_error(400, "Request body must be a JSON object")
    model = str(parsed_body.get("model") or "claude-2.1")
    prompt = str(parsed_body.get("prompt") or "")
    stream = bool(parsed_body.get("stream", False))
    reply = _synthesize_reply_text(prompt, model)
    compl_id = _deterministic_id("compl_mock", f"{model}:{prompt}")

    if stream:
        def _stream() -> Iterable[bytes]:
            yield _format_sse_event(
                "completion",
                {
                    "type": "completion",
                    "id": compl_id,
                    "completion": reply,
                    "stop_reason": "stop_sequence",
                    "model": model,
                },
            )
        return HandlerResult(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
            },
            stream_chunks=_stream(),
        )

    payload = {
        "type": "completion",
        "id": compl_id,
        "completion": reply,
        "stop_reason": "stop_sequence",
        "model": model,
    }
    return HandlerResult(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"),
    )


def handle_anthropic_messages(
    target: ResolvedTarget,
    parsed_body: Optional[Dict[str, Any]],
    raw_body: bytes,
    client_headers: Mapping[str, str],
) -> HandlerResult:
    if not isinstance(parsed_body, dict):
        return make_anthropic_error(400, "Request body must be a JSON object")

    # If a live Anthropic upstream is configured, forward the request to it
    if not target.mock:
        body_to_send = (
            json.dumps(target.merged_body).encode("utf-8")
            if target.merged_body is not None
            else raw_body
        )
        stream = bool(parsed_body.get("stream", False))
        try:
            up_resp = request_upstream(
                upstream=target.upstream,
                method="POST",
                url_path=target.upstream_url_path,
                api_key=target.api_key,
                protocol="anthropic",
                body_bytes=body_to_send,
                client_headers=client_headers,
                stream=stream,
            )
        except UpstreamError as exc:
            return make_anthropic_error(exc.status_code, exc.message, "api_error")

        if up_resp.is_stream:
            return HandlerResult(
                status=up_resp.status,
                headers=up_resp.headers,
                stream_chunks=up_resp.iter_chunks(),
            )
        return HandlerResult(
            status=up_resp.status,
            headers=up_resp.headers,
            body=up_resp.body,
        )

    # Built-in Mock Anthropic Messages handler
    messages = parsed_body.get("messages")
    if not isinstance(messages, list) or not messages:
        return make_anthropic_error(400, "messages: field required and must be a non-empty list")

    model = str((target.merged_body or parsed_body).get("model") or "claude-sonnet-4-20250514")
    stream = bool(parsed_body.get("stream", False))
    system_text = _extract_text_from_content(parsed_body.get("system", ""))
    last_user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_text = _extract_text_from_content(msg.get("content", ""))
            if last_user_text:
                break

    tool_result_text = _has_tool_result(messages)
    tools = parsed_body.get("tools") if isinstance(parsed_body.get("tools"), list) else []
    tool_choice = parsed_body.get("tool_choice") if isinstance(parsed_body.get("tool_choice"), dict) else {}
    thinking_cfg = parsed_body.get("thinking") if isinstance(parsed_body.get("thinking"), dict) else {}
    thinking_enabled = thinking_cfg.get("type") == "enabled"

    msg_id = _deterministic_id("msg_mock", f"{model}:{last_user_text}:{len(messages)}")
    req_id = _deterministic_id("req_mock", msg_id)

    content_blocks: List[Dict[str, Any]] = []
    stop_reason = "end_turn"

    if thinking_enabled:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": f"Analyzing request for model {model}: {last_user_text[:120] or 'empty prompt'}",
                "signature": _deterministic_id("sig_mock", msg_id),
            }
        )

    # Check if we should emit a mock tool_use block
    force_tool = tool_choice.get("type") in ("any", "tool")
    wants_tool = bool(tools) and tool_result_text is None and (
        force_tool or "use_tool" in last_user_text.lower() or "call tool" in last_user_text.lower()
    )

    if wants_tool and tools:
        chosen_tool = tools[0]
        if tool_choice.get("type") == "tool" and tool_choice.get("name"):
            for t in tools:
                if isinstance(t, dict) and t.get("name") == tool_choice["name"]:
                    chosen_tool = t
                    break
        tool_name = str(chosen_tool.get("name", "mock_tool")) if isinstance(chosen_tool, dict) else "mock_tool"
        tool_input = parsed_body.get("mock_tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {"query": last_user_text or "test"}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": _deterministic_id("toolu_mock", f"{msg_id}:{tool_name}"),
                "name": tool_name,
                "input": tool_input,
            }
        )
        stop_reason = "tool_use"
    else:
        if tool_result_text is not None:
            reply_text = f"[Mock {model}] Processed tool result: {tool_result_text}"
        else:
            reply_text = _synthesize_reply_text(last_user_text, model, system_text)
        content_blocks.append(
            {
                "type": "text",
                "text": reply_text,
            }
        )

    input_tokens = _estimate_tokens(parsed_body)
    output_tokens = max(1, sum(len(json.dumps(b)) for b in content_blocks) // 4)

    if stream:
        def _sse_stream() -> Iterable[bytes]:
            yield _format_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 1,
                        },
                    },
                },
            )
            yield _format_sse_event("ping", {"type": "ping"})

            for idx, block in enumerate(content_blocks):
                btype = block["type"]
                if btype == "thinking":
                    yield _format_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )
                    yield _format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                        },
                    )
                    yield _format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "signature_delta", "signature": block["signature"]},
                        },
                    )
                    yield _format_sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
                elif btype == "tool_use":
                    yield _format_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {},
                            },
                        },
                    )
                    yield _format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(block["input"]),
                            },
                        },
                    )
                    yield _format_sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
                else:
                    text = block["text"]
                    yield _format_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    # Split text into two chunks for realistic streaming
                    mid = max(1, len(text) // 2)
                    for part in (text[:mid], text[mid:]):
                        if part:
                            yield _format_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "text_delta", "text": part},
                                },
                            )
                    yield _format_sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )

            yield _format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            yield _format_sse_event("message_stop", {"type": "message_stop"})

        return HandlerResult(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "request-id": req_id,
                "anthropic-organization-id": "org_mock",
            },
            stream_chunks=_sse_stream(),
        )

    payload = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
        },
    }
    return HandlerResult(
        status=200,
        headers={
            "Content-Type": "application/json",
            "request-id": req_id,
            "anthropic-organization-id": "org_mock",
        },
        body=json.dumps(payload).encode("utf-8"),
    )
