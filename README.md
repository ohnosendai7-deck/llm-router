# llm-router (`api-proxy`)

A self-contained, zero-external-dependency OpenAI-compatible and Anthropic-compatible LLM API router and key-injecting local gateway for Guix System (`(dwi services api-proxy)`).

## Why this replaces the legacy nginx + Guile rewriter proxy

Previously, `api-proxy` rendered an `nginx.conf` (`127.0.0.1:8402`) paired with a fork-per-connection Guile JSON rewriter (`127.0.0.1:8403`) and a second nginx TLS relay server (`127.0.0.1:8404`). Because it acted as a raw HTTP reverse proxy rather than simulating an OpenAI/Anthropic API server:

1. **`GET /v1/models` leaked upstream catalogs:** Calling `GET /openrouter/v1/models` forwarded directly to OpenRouter and returned 460+ raw upstream models (~800 KB JSON + Cloudflare cookies) instead of the configured models (`argon`, `k3`), and omitted managed model aliases completely.
2. **No cross-provider model routing:** A client pointed at `/openrouter/v1` could not switch to `k3` (configured under `kimi`) without changing its `base_url`; `rewriter.scm` only checked the URL's single provider block and passed unknown aliases straight to the wrong upstream.
3. **No Anthropic endpoint support or mock:** Anthropic endpoints (`/v1/messages`, `/v1/messages/count_tokens`) were not simulated and failed with 404 when no Anthropic upstream was configured.
4. **Transport fragility:** Raw header forwarding and static DNS resolution caused Cloudflare IPv6 `ENETUNREACH` blackholes, 4 KB `proxy_buffer_size` header overflows, and HTML `502 Bad Gateway` responses instead of structured OpenAI/Anthropic JSON error payloads.

## Features

- **Simulated OpenAI-Compatible API** (`/v1/...`, `/openai/v1/...`, and provider-scoped `/<provider>/v1/...`):
  - `GET /v1/models` and `GET /v1/models/<id>`: synthesizes a clean OpenAI model list from all configured providers and model aliases (`argon`, `k3`, etc.).
  - `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`: resolves the requested `"model"` across all configured providers, merges per-model `(overrides ...)` into the top-level request JSON, injects the target provider's (or model's own) upstream API key, and forwards over IPv4-preferred HTTPS with unbuffered SSE streaming.
- **Simulated / Mocked Anthropic-Compatible API** (`/v1/messages`, `/v1/messages/count_tokens`, `/v1/complete`, `/anthropic/v1/...`, `/<provider>/v1/messages`):
  - When no live Anthropic provider is configured (or when `(mock #t)` / placeholder credentials are set), serves a full spec-compliant mock Anthropic Messages API supporting non-streaming JSON, SSE event streaming (`message_start`, `ping`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`), extended `thinking` blocks, `tool_use` / `tool_result` multi-turn flows, and token counting (`/v1/messages/count_tokens`).
  - When a live Anthropic upstream is configured with a real key, forwards requests with `x-api-key` and `anthropic-version` headers.
- **Drop-in `/root/.config/api-proxy/config.scm` Compatibility**:
  - Parses the existing Guix `(api-proxy ...)` S-expression format natively (including `disabled`, `port`, `providers`, `upstream`, `path`, `key`, `models`, `overrides` with booleans `#t`/`#f`, `'null`, vectors `#(...)`, and dotted-pair alists), as well as JSON configs.
  - Validates per-client dummy bearers (`local-hermes`, `local-pi`, `local-claude`) via `Authorization: Bearer ...` or `x-api-key: ...`.

## Usage

```bash
llm-router --config /root/.config/api-proxy/config.scm
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
