"""Routing, authentication, override merging, and model catalog logic."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import ModelConfig, ProviderConfig, RouterConfig


MOCK_ANTHROPIC_MODELS: List[Dict[str, Any]] = [
    {
        "id": "claude-sonnet-4-20250514",
        "type": "model",
        "object": "model",
        "display_name": "Claude Sonnet 4 (Mock)",
        "created": 1747180800,
        "created_at": "2025-05-14T00:00:00Z",
        "owned_by": "anthropic-mock",
        "context_length": 200000,
    },
    {
        "id": "claude-opus-4-20250514",
        "type": "model",
        "object": "model",
        "display_name": "Claude Opus 4 (Mock)",
        "created": 1747180800,
        "created_at": "2025-05-14T00:00:00Z",
        "owned_by": "anthropic-mock",
        "context_length": 200000,
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "type": "model",
        "object": "model",
        "display_name": "Claude Haiku 4.5 (Mock)",
        "created": 1759276800,
        "created_at": "2025-10-01T00:00:00Z",
        "owned_by": "anthropic-mock",
        "context_length": 200000,
    },
    {
        "id": "claude-3-7-sonnet-20250219",
        "type": "model",
        "object": "model",
        "display_name": "Claude 3.7 Sonnet (Mock)",
        "created": 1739923200,
        "created_at": "2025-02-19T00:00:00Z",
        "owned_by": "anthropic-mock",
        "context_length": 200000,
    },
]


def json_merge(parsed: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `parsed` with `overrides` merged over its top level.

    Override keys replace same-named keys; all other keys are preserved.
    Matches the exact semantics of `(dwi services api-proxy rewriter)`'s
    `json-merge`.
    """
    if not overrides:
        return dict(parsed)
    merged: Dict[str, Any] = dict(overrides)
    for k, v in parsed.items():
        if k not in overrides:
            merged[k] = v
    return merged


def authenticate_client(headers: Mapping[str, str], config: RouterConfig) -> Optional[str]:
    """Validate client bearer or x-api-key header against `config.client_keys`.

    Returns the authenticated client name (e.g. 'hermes', 'pi', 'claude'),
    or None if unauthorized.
    """
    # Case-insensitive header lookup
    lower_headers = {k.lower(): v for k, v in headers.items()}

    auth_header = lower_headers.get("authorization", "").strip()
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token in config.client_keys:
                return config.client_keys[token]
        elif auth_header in config.client_keys:
            return config.client_keys[auth_header]

    for hdr_name in ("x-api-key", "api-key"):
        token = lower_headers.get(hdr_name, "").strip()
        if token and token in config.client_keys:
            return config.client_keys[token]

    return None


@dataclass
class RouteClassification:
    kind: str  # "health", "openai", "anthropic", "not_found"
    provider_hint: Optional[str]
    subpath: str
    protocol: str  # "openai" or "anthropic"


def classify_request_path(path: str, config: RouterConfig) -> RouteClassification:
    """Classify an incoming request URL path into protocol, provider hint, and API subpath."""
    clean_path = path or "/"
    if not clean_path.startswith("/"):
        clean_path = "/" + clean_path

    if clean_path in ("/health", "/v1/health", "/status"):
        return RouteClassification("health", None, clean_path, "openai")

    # 1. Check longest-prefix match against configured providers
    sorted_providers = sorted(config.providers, key=lambda p: len(p.name), reverse=True)
    for prov in sorted_providers:
        prefix = f"/{prov.name}"
        if clean_path == prefix or clean_path.startswith(prefix + "/"):
            rest = clean_path[len(prefix):] or "/"
            if not rest.startswith("/"):
                rest = "/" + rest
            if _is_anthropic_subpath(rest) or prov.protocol == "anthropic":
                return RouteClassification("anthropic", prov.name, rest, "anthropic")
            return RouteClassification("openai", prov.name, rest, "openai")

    # 2. Check built-in protocol prefixes (/anthropic/... and /openai/...)
    if clean_path == "/anthropic" or clean_path.startswith("/anthropic/"):
        rest = clean_path[len("/anthropic"):] or "/"
        return RouteClassification("anthropic", "anthropic", rest, "anthropic")

    if clean_path == "/openai" or clean_path.startswith("/openai/"):
        rest = clean_path[len("/openai"):] or "/"
        if _is_anthropic_subpath(rest):
            return RouteClassification("anthropic", None, rest, "anthropic")
        return RouteClassification("openai", None, rest, "openai")

    # 3. Check unified root endpoints (/v1/..., /api/v1/..., /v1beta/..., etc.)
    if _is_anthropic_subpath(clean_path):
        return RouteClassification("anthropic", None, clean_path, "anthropic")

    if _is_openai_root_subpath(clean_path):
        return RouteClassification("openai", None, clean_path, "openai")

    return RouteClassification("not_found", None, clean_path, "openai")


def _is_anthropic_subpath(subpath: str) -> bool:
    p = subpath.rstrip("/")
    if p.startswith("/anthropic/"):
        return True
    return p in (
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/complete",
        "/messages",
        "/messages/count_tokens",
        "/complete",
        "/api/v1/messages",
        "/api/v1/messages/count_tokens",
    )


def _is_openai_root_subpath(subpath: str) -> bool:
    p = subpath.rstrip("/")
    if p in (
        "/v1/models",
        "/models",
        "/api/v1/models",
        "/v1/chat/completions",
        "/chat/completions",
        "/api/v1/chat/completions",
        "/v1/completions",
        "/completions",
        "/api/v1/completions",
        "/v1/responses",
        "/responses",
        "/api/v1/responses",
        "/v1/embeddings",
        "/embeddings",
        "/api/v1/embeddings",
    ):
        return True
    if (
        p.startswith("/v1/")
        or p.startswith("/api/v1/")
        or p.startswith("/v1beta/")
        or p.startswith("/models/")
    ):
        return True
    return False


@dataclass
class ResolvedTarget:
    provider_name: str
    requested_model: Optional[str]
    effective_model: Optional[str]
    upstream: str
    path_prefix: str
    subpath: str
    query: str
    api_key: str
    protocol: str
    mock: bool
    merged_body: Optional[Dict[str, Any]]

    @property
    def upstream_url_path(self) -> str:
        prefix = (self.path_prefix or "").rstrip("/")
        sub = self.subpath if self.subpath.startswith("/") else "/" + self.subpath
        # Avoid duplicating `/anthropic` if both provider path_prefix and subpath have it
        if prefix.endswith("/anthropic") and sub.startswith("/anthropic/"):
            sub = sub[len("/anthropic"):]
        full = f"{prefix}{sub}" if prefix else sub
        if self.query:
            return f"{full}?{self.query}"
        return full


def resolve_target(
    config: RouterConfig,
    classification: RouteClassification,
    parsed_body: Optional[Dict[str, Any]],
    query: str = "",
) -> ResolvedTarget:
    """Resolve a request to its target upstream, path prefix, credentials, and merged JSON body."""
    req_model: Optional[str] = None
    if isinstance(parsed_body, dict) and isinstance(parsed_body.get("model"), str):
        req_model = parsed_body["model"]

    # Anthropic endpoint resolution
    if classification.protocol == "anthropic":
        # Check if a specific Anthropic-capable provider was requested and exists
        if classification.provider_hint:
            prov = config.get_provider(classification.provider_hint)
            if (
                prov
                and (prov.protocol == "anthropic" or (prov.path or "").rstrip("/").endswith("/anthropic"))
                and not prov.mock
                and not prov.is_placeholder_key
                and prov.upstream
            ):
                if req_model:
                    m = prov.find_model(req_model)
                    if m is not None:
                        return _build_resolved_from_model(
                            prov, m, req_model, parsed_body, classification.subpath, query, protocol="anthropic"
                        )
                return _build_resolved_from_provider(
                    prov, req_model, parsed_body, classification.subpath, query, protocol="anthropic"
                )
        # Check if any model entry matches req_model on an anthropic-capable provider
        if req_model:
            for prov in config.providers:
                m = prov.find_model(req_model)
                if m and (m.protocol == "anthropic" or prov.protocol == "anthropic" or (m.path or "").endswith("/anthropic")):
                    if not m.mock and not prov.mock and not prov.is_placeholder_key:
                        return _build_resolved_from_model(
                            prov, m, req_model, parsed_body, classification.subpath, query, protocol="anthropic"
                        )
        # Check if a live `anthropic` provider exists
        anth_prov = config.get_provider("anthropic")
        if anth_prov and not anth_prov.mock and not anth_prov.is_placeholder_key and anth_prov.upstream:
            if req_model:
                m = anth_prov.find_model(req_model)
                if m is not None:
                    return _build_resolved_from_model(
                        anth_prov, m, req_model, parsed_body, classification.subpath, query, protocol="anthropic"
                    )
            return _build_resolved_from_provider(
                anth_prov, req_model, parsed_body, classification.subpath, query, protocol="anthropic"
            )
        # Default: serve from the built-in Anthropic mock engine
        return ResolvedTarget(
            provider_name="anthropic",
            requested_model=req_model or "claude-sonnet-4-20250514",
            effective_model=req_model or "claude-sonnet-4-20250514",
            upstream="mock://anthropic",
            path_prefix="",
            subpath=classification.subpath,
            query=query,
            api_key="",
            protocol="anthropic",
            mock=True,
            merged_body=dict(parsed_body) if isinstance(parsed_body, dict) else None,
        )

    # OpenAI endpoint resolution
    # 1. Explicit provider prefix in the URL (e.g. /openrouter/v1/... or /kimi/v1/...)
    if classification.provider_hint:
        prov = config.get_provider(classification.provider_hint)
        if prov is None:
            raise KeyError(f"Unknown provider: {classification.provider_hint}")
        if req_model:
            # First check if req_model is configured on this provider
            m = prov.find_model(req_model)
            if m is not None:
                return _build_resolved_from_model(
                    prov, m, req_model, parsed_body, classification.subpath, query
                )
            # Cross-provider alias resolution: if a client pointed at /openrouter/v1
            # requests `"model": "k3"` (which is defined on `kimi`), route to `kimi`!
            for other_prov in config.providers:
                if other_prov.name == prov.name:
                    continue
                other_m = other_prov.find_model(req_model)
                if other_m is not None:
                    return _build_resolved_from_model(
                        other_prov, other_m, req_model, parsed_body, classification.subpath, query
                    )
        # Unaliased model or bodiless request: forward directly to `prov`
        return _build_resolved_from_provider(
            prov, req_model, parsed_body, classification.subpath, query
        )

    # 2. Unified root OpenAI endpoint (/v1/chat/completions, /v1/embeddings, etc.)
    if req_model:
        # 2a. Exact match against any configured model alias across all providers
        for prov in config.providers:
            m = prov.find_model(req_model)
            if m is not None:
                return _build_resolved_from_model(
                    prov, m, req_model, parsed_body, classification.subpath, query
                )

        # 2b. Check `<provider>/<model_rest>` prefix (longest provider name first)
        sorted_providers = sorted(config.providers, key=lambda p: len(p.name), reverse=True)
        for prov in sorted_providers:
            prefix = f"{prov.name}/"
            if req_model.startswith(prefix):
                rest_model = req_model[len(prefix):]
                m = prov.find_model(rest_model)
                body_copy = dict(parsed_body) if isinstance(parsed_body, dict) else {}
                body_copy["model"] = rest_model
                if m is not None:
                    return _build_resolved_from_model(
                        prov, m, rest_model, body_copy, classification.subpath, query
                    )
                return _build_resolved_from_provider(
                    prov, rest_model, body_copy, classification.subpath, query
                )

        # 2c. Check if req_model matches the target_model of any configured alias
        for prov in config.providers:
            for m in prov.models:
                if m.target_model == req_model:
                    return _build_resolved_from_model(
                        prov, m, req_model, parsed_body, classification.subpath, query
                    )

        # 2d. Heuristic routing by model family prefix
        if req_model.startswith(("kimi-", "moonshot-")):
            for candidate_name in ("kimi", "code", "kimi-coding"):
                cand = config.get_provider(candidate_name)
                if cand and not cand.mock and not cand.is_placeholder_key:
                    return _build_resolved_from_provider(
                        cand, req_model, parsed_body, classification.subpath, query
                    )

        if "/" in req_model:
            or_prov = config.get_provider("openrouter")
            if or_prov and not or_prov.mock and not or_prov.is_placeholder_key:
                return _build_resolved_from_provider(
                    or_prov, req_model, parsed_body, classification.subpath, query
                )

        if req_model.startswith("claude-") or req_model == "mock":
            # If no OpenRouter provider is active, serve mock response
            or_prov = config.get_provider("openrouter")
            if not or_prov or or_prov.mock or or_prov.is_placeholder_key:
                return ResolvedTarget(
                    provider_name="anthropic",
                    requested_model=req_model,
                    effective_model=req_model,
                    upstream="mock://openai",
                    path_prefix="",
                    subpath=classification.subpath,
                    query=query,
                    api_key="",
                    protocol="openai",
                    mock=True,
                    merged_body=dict(parsed_body) if isinstance(parsed_body, dict) else None,
                )

    # 2e. Fallback to the first active OpenAI provider
    for preferred in ("openrouter", "kimi", "code"):
        cand = config.get_provider(preferred)
        if cand and not cand.mock and not cand.is_placeholder_key:
            return _build_resolved_from_provider(
                cand, req_model, parsed_body, classification.subpath, query
            )
    for prov in config.providers:
        if not prov.mock and not prov.is_placeholder_key:
            return _build_resolved_from_provider(
                prov, req_model, parsed_body, classification.subpath, query
            )
    if config.providers:
        return _build_resolved_from_provider(
            config.providers[0], req_model, parsed_body, classification.subpath, query
        )

    # No providers at all -> mock
    return ResolvedTarget(
        provider_name="mock",
        requested_model=req_model or "mock-model",
        effective_model=req_model or "mock-model",
        upstream="mock://openai",
        path_prefix="",
        subpath=classification.subpath,
        query=query,
        api_key="",
        protocol="openai",
        mock=True,
        merged_body=dict(parsed_body) if isinstance(parsed_body, dict) else None,
    )


def _build_resolved_from_model(
    prov: ProviderConfig,
    m: ModelConfig,
    req_model: Optional[str],
    parsed_body: Optional[Dict[str, Any]],
    subpath: str,
    query: str,
    protocol: Optional[str] = None,
) -> ResolvedTarget:
    merged = json_merge(parsed_body, m.overrides) if isinstance(parsed_body, dict) else None
    eff_model = str(merged.get("model", m.target_model)) if isinstance(merged, dict) else m.target_model
    upstream = m.upstream or prov.upstream
    key = m.key or prov.key
    path_prefix = m.path if m.path is not None else prov.path
    is_mock = m.mock or prov.mock
    proto = protocol or m.protocol or prov.protocol or "openai"
    return ResolvedTarget(
        provider_name=prov.name,
        requested_model=req_model or m.name,
        effective_model=eff_model,
        upstream=upstream,
        path_prefix=path_prefix,
        subpath=subpath,
        query=query,
        api_key=key,
        protocol=proto,
        mock=is_mock,
        merged_body=merged,
    )


def _build_resolved_from_provider(
    prov: ProviderConfig,
    req_model: Optional[str],
    parsed_body: Optional[Dict[str, Any]],
    subpath: str,
    query: str,
    protocol: Optional[str] = None,
) -> ResolvedTarget:
    merged = dict(parsed_body) if isinstance(parsed_body, dict) else None
    return ResolvedTarget(
        provider_name=prov.name,
        requested_model=req_model,
        effective_model=req_model,
        upstream=prov.upstream,
        path_prefix=prov.path,
        subpath=subpath,
        query=query,
        api_key=prov.key,
        protocol=protocol or prov.protocol or "openai",
        mock=prov.mock,
        merged_body=merged,
    )


def build_openai_models_catalog(
    config: RouterConfig,
    provider_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the simulated OpenAI `/v1/models` list from configured providers and models.

    Includes:
      1. All configured model aliases on the target provider (or all providers),
         plus cross-provider model aliases so clients pointed at any provider
         endpoint can discover all routable models.
      2. Standard provider models when a provider has no explicit `(models ...)` block.
      3. Mock Anthropic models on unified `/v1/models` or `/anthropic/v1/models`.
    """
    seen_ids: set[str] = set()
    models_out: List[Dict[str, Any]] = []
    created_ts = 1760000000

    def add_model_entry(
        model_id: str,
        owned_by: str,
        root: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not model_id or model_id in seen_ids:
            return
        seen_ids.add(model_id)
        entry: Dict[str, Any] = {
            "id": model_id,
            "object": "model",
            "created": created_ts,
            "owned_by": owned_by,
            "root": root or model_id,
            "parent": None,
            "permission": [
                {
                    "id": f"modelperm-{model_id}",
                    "object": "model_permission",
                    "created": created_ts,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
        }
        if extra:
            entry.update(extra)
        models_out.append(entry)

    if provider_hint == "anthropic":
        for m in MOCK_ANTHROPIC_MODELS:
            add_model_entry(m["id"], m["owned_by"], root=m["id"], extra=m)
        return models_out

    # Order providers so `provider_hint` comes first if specified
    ordered_providers: List[ProviderConfig] = []
    if provider_hint:
        hinted = config.get_provider(provider_hint)
        if hinted:
            ordered_providers.append(hinted)
    for p in config.providers:
        if p not in ordered_providers:
            ordered_providers.append(p)

    # 1. Add all explicitly configured `(models ...)` entries
    for prov in ordered_providers:
        for m in prov.models:
            add_model_entry(
                model_id=m.name,
                owned_by=prov.name,
                root=m.target_model,
            )

    # 2. For unmanaged providers (no `models` block), expose standard defaults
    # so `GET /<unmanaged>/v1/models` always has valid model entries even without
    # hitting the network.
    for prov in ordered_providers:
        if not prov.models and not prov.is_placeholder_key:
            if prov.name in ("kimi", "code") or "moonshot" in prov.upstream or "kimi" in prov.upstream:
                for mid in ("k3", "kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "kimi-k2.6"):
                    add_model_entry(mid, prov.name, root="kimi-k3" if mid == "k3" else mid)
            elif prov.name == "openrouter" or "openrouter" in prov.upstream:
                add_model_entry("argon", prov.name, root="google/gemini-4-argon")
                add_model_entry("google/gemini-4-argon", prov.name)
            else:
                add_model_entry(f"{prov.name}-default", prov.name)

    # 3. Include mock Anthropic models on the unified catalog
    if provider_hint is None and config.mock_anthropic:
        for m in MOCK_ANTHROPIC_MODELS:
            add_model_entry(m["id"], m["owned_by"], root=m["id"])

    return models_out
