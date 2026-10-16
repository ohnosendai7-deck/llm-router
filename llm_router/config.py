"""Configuration parser and data structures for llm-router.

Supports both the Guix api-proxy s-expression configuration format
(/root/.config/api-proxy/config.scm) and JSON configuration files,
preserving full compatibility with existing provider/model definitions,
per-model overrides, per-model upstream fan-out, and client dummy bearers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_PATH = "/root/.config/api-proxy/config.scm"

DEFAULT_CLIENT_KEYS: Dict[str, str] = {
    "local-hermes": "hermes",
    "local-pi": "pi",
    "local-claude": "claude",
}

SCAFFOLD_CONFIG = """\
;; /root/.config/api-proxy/config.scm — local key-injecting LLM router.
;; One s-expression.  Disabled until you set (disabled #f) and replace
;; the placeholder upstreams/keys with real ones.
(api-proxy
 (disabled #t)
 (port 8402)
 (providers
  (kimi (upstream "api.moonshot.ai")
        (key "***"))
  (anthropic (upstream "api.anthropic.com")
             (key "***"))))
;; Exposes unified OpenAI-compatible endpoints (/v1/models,
;; /v1/chat/completions, /v1/completions, /v1/responses, /v1/embeddings)
;; and Anthropic-compatible endpoints (/v1/messages,
;; /v1/messages/count_tokens, /anthropic/v1/messages), as well as
;; provider-scoped prefixes (/<name>/v1/...).  When no real Anthropic
;; provider is configured, Anthropic endpoints are served by the built-in
;; spec-compliant mock handler.
;;
;; Optional fields per provider:
;;   (path "/anthropic")  — upstream-only path prefix (e.g. "/api" for
;;     OpenRouter, "/coding" for Kimi coding).
;;   (mock #t)            — serve this provider from the local mock engine.
;;   (models (NAME (overrides (model . "real-model") ...)
;;                 [(path "/p")] [(upstream "h") (key "sk-...")])
;;           ...)
;;     — defines routable model aliases: when a request's "model" matches
;;     NAME, its overrides are merged into the top-level JSON and the
;;     request is forwarded to the provider's (or model's own) upstream.
"""


# ---------------------------------------------------------------------------
# S-expression AST types (matching Guile + guile-json conventions)
# ---------------------------------------------------------------------------

class Symbol(str):
    """Scheme symbol."""

    def __repr__(self) -> str:
        return f"Symbol({super().__repr__()})"


@dataclass(frozen=True)
class SexpPair:
    """Scheme dotted pair `(car . cdr)`."""
    car: Any
    cdr: Any


@dataclass(frozen=True)
class SexpVector:
    """Scheme vector `#(...)`, used by guile-json for JSON arrays."""
    items: Tuple[Any, ...]


@dataclass(frozen=True)
class SexpQuote:
    """Scheme quoted datum `'x`."""
    value: Any


class SexpParseError(ValueError):
    """Raised when an s-expression cannot be parsed."""


def tokenize_sexp(text: str) -> List[Tuple[str, Any]]:
    """Tokenize a Scheme s-expression string."""
    tokens: List[Tuple[str, Any]] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN", "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN", ")"))
            i += 1
            continue
        if ch == "'":
            tokens.append(("QUOTE", "'"))
            i += 1
            continue
        if ch == "#" and i + 1 < n and text[i + 1] == "(":
            tokens.append(("VEC_LPAREN", "#("))
            i += 2
            continue
        if ch == '"':
            i += 1
            chars: List[str] = []
            while i < n:
                c = text[i]
                if c == "\\":
                    i += 1
                    if i >= n:
                        raise SexpParseError("Unterminated string escape")
                    esc = text[i]
                    if esc == "n":
                        chars.append("\n")
                    elif esc == "r":
                        chars.append("\r")
                    elif esc == "t":
                        chars.append("\t")
                    elif esc == '"':
                        chars.append('"')
                    elif esc == "\\":
                        chars.append("\\")
                    elif esc == "\n":
                        # Line continuation in Scheme string literal
                        i += 1
                        while i < n and text[i] in (" ", "\t"):
                            i += 1
                        continue
                    else:
                        chars.append(esc)
                    i += 1
                elif c == '"':
                    i += 1
                    break
                else:
                    chars.append(c)
                    i += 1
            else:
                raise SexpParseError("Unterminated string literal")
            tokens.append(("STRING", "".join(chars)))
            continue

        # Atom (symbol, number, boolean, or dot)
        start = i
        while i < n and not text[i].isspace() and text[i] not in '();"':
            i += 1
        raw = text[start:i]
        if raw == ".":
            tokens.append(("DOT", "."))
        elif raw in ("#t", "#true"):
            tokens.append(("BOOL", True))
        elif raw in ("#f", "#false"):
            tokens.append(("BOOL", False))
        else:
            # Try integer or float
            try:
                tokens.append(("NUMBER", int(raw)))
                continue
            except ValueError:
                pass
            try:
                tokens.append(("NUMBER", float(raw)))
                continue
            except ValueError:
                pass
            tokens.append(("SYMBOL", Symbol(raw)))

    return tokens


def parse_sexp(text: str) -> Any:
    """Parse a single top-level Scheme s-expression from text."""
    tokens = tokenize_sexp(text)
    if not tokens:
        raise SexpParseError("Empty s-expression input")

    pos = 0

    def read_expr() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise SexpParseError("Unexpected EOF while reading s-expression")
        tok_type, tok_val = tokens[pos]
        pos += 1

        if tok_type in ("STRING", "NUMBER", "BOOL", "SYMBOL"):
            return tok_val
        if tok_type == "QUOTE":
            return SexpQuote(read_expr())
        if tok_type == "VEC_LPAREN":
            items: List[Any] = []
            while pos < len(tokens) and tokens[pos][0] != "RPAREN":
                items.append(read_expr())
            if pos >= len(tokens):
                raise SexpParseError("Unterminated vector #(...)")
            pos += 1  # consume RPAREN
            return SexpVector(tuple(items))
        if tok_type == "LPAREN":
            items = []
            while pos < len(tokens) and tokens[pos][0] != "RPAREN":
                if tokens[pos][0] == "DOT":
                    if not items:
                        raise SexpParseError("Dot at start of list")
                    pos += 1  # consume DOT
                    cdr_val = read_expr()
                    if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
                        raise SexpParseError("Expected ')' after dotted cdr")
                    pos += 1  # consume RPAREN
                    if len(items) == 1:
                        return SexpPair(items[0], cdr_val)
                    # Improper list `(a b . c)`: fold from right or represent tail
                    # In overrides, `(overrides (k1 . v1) (k2 . v2))` is a proper list
                    # whereas `(k . v)` has len(items) == 1.
                    res: Any = cdr_val
                    for elem in reversed(items):
                        res = SexpPair(elem, res)
                    return res
                items.append(read_expr())
            if pos >= len(tokens):
                raise SexpParseError("Unterminated list (...)")
            pos += 1  # consume RPAREN
            return items
        raise SexpParseError(f"Unexpected token: {tok_type} ({tok_val!r})")

    expr = read_expr()
    return expr


def sexp_to_json_value(val: Any) -> Any:
    """Convert a Scheme s-expression datum to its JSON-compatible Python value.

    Follows guile-json's `scm->json` rules:
      - strings, numbers, booleans -> str, int/float, bool
      - 'null or symbol null -> None
      - SexpVector #(...) -> list (JSON array)
      - list of SexpPairs ((k . v) ...) -> dict (JSON object)
      - SexpQuote 'x -> sexp_to_json_value(x)
      - Symbol -> str
    """
    if val is None or isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, Symbol):
        if str(val) == "null":
            return None
        return str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, SexpQuote):
        if isinstance(val.value, Symbol) and str(val.value) == "null":
            return None
        return sexp_to_json_value(val.value)
    if isinstance(val, SexpVector):
        return [sexp_to_json_value(item) for item in val.items]
    if isinstance(val, SexpPair):
        return {str(val.car): sexp_to_json_value(val.cdr)}
    if isinstance(val, list):
        if not val:
            return {}
        if all(isinstance(item, SexpPair) for item in val):
            return {
                str(item.car): sexp_to_json_value(item.cdr)
                for item in val
            }
        # Support 2-element lists `(k v)` inside alists if written without dot
        if all(
            isinstance(item, list) and len(item) == 2 and isinstance(item[0], (str, Symbol))
            for item in val
        ):
            return {
                str(item[0]): sexp_to_json_value(item[1])
                for item in val
            }
        return [sexp_to_json_value(item) for item in val]
    return val


# ---------------------------------------------------------------------------
# Configuration Data Models
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    upstream: Optional[str] = None
    key: Optional[str] = None
    protocol: Optional[str] = None
    mock: bool = False

    @property
    def target_model(self) -> str:
        """The upstream model name after applying overrides."""
        m = self.overrides.get("model")
        return str(m) if m else self.name


@dataclass
class ProviderConfig:
    name: str
    upstream: str = ""
    path: str = ""
    key: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    protocol: str = "openai"
    mock: bool = False

    @property
    def is_managed(self) -> bool:
        return bool(self.models)

    @property
    def is_placeholder_key(self) -> bool:
        k = (self.key or "").strip()
        return not k or k in ("***", "sk-...", "mock") or k.startswith("«redacted")

    def find_model(self, model_name: str) -> Optional[ModelConfig]:
        for m in self.models:
            if m.name == model_name:
                return m
        return None


@dataclass
class RouterConfig:
    disabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8402
    providers: List[ProviderConfig] = field(default_factory=list)
    client_keys: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLIENT_KEYS))
    mock_anthropic: bool = True

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        for p in self.providers:
            if p.name == name:
                return p
        return None

    def has_live_anthropic_provider(self) -> bool:
        for p in self.providers:
            if p.name == "anthropic" or p.protocol == "anthropic":
                if not p.mock and not p.is_placeholder_key and p.upstream:
                    return True
        return False


def _infer_protocol(name: str, upstream: str, path: str, explicit: Optional[str] = None) -> str:
    if explicit in ("openai", "anthropic"):
        return explicit
    if name == "anthropic" or "anthropic.com" in (upstream or "") or (path or "").rstrip("/").endswith("/anthropic"):
        return "anthropic"
    return "openai"


def _parse_model_sexp(sexp: Any) -> ModelConfig:
    if not isinstance(sexp, list) or not sexp or not isinstance(sexp[0], Symbol):
        raise ValueError(f"api-proxy: bad model form: {sexp!r}")
    name = str(sexp[0])
    overrides: Optional[Dict[str, Any]] = None
    path: Optional[str] = None
    upstream: Optional[str] = None
    key: Optional[str] = None
    protocol: Optional[str] = None
    mock = False

    for field_expr in sexp[1:]:
        if isinstance(field_expr, SexpPair) and str(field_expr.car) == "overrides":
            raw_ov = field_expr.cdr
            overrides = sexp_to_json_value(raw_ov) if isinstance(raw_ov, list) else {}
            continue
        if not isinstance(field_expr, list) or not field_expr:
            raise ValueError(f"api-proxy model: unknown field {field_expr!r}")
        tag = str(field_expr[0])
        rest = field_expr[1:]
        if tag == "overrides":
            overrides = sexp_to_json_value(rest)
            if not isinstance(overrides, dict):
                raise ValueError(f"api-proxy model: overrides must be an alist in {name}")
        elif tag == "path" and len(rest) == 1 and isinstance(rest[0], str):
            path = rest[0]
        elif tag == "upstream" and len(rest) == 1 and isinstance(rest[0], str):
            upstream = rest[0]
        elif tag == "key" and len(rest) == 1 and isinstance(rest[0], str):
            key = rest[0]
        elif tag == "protocol" and len(rest) == 1:
            protocol = str(rest[0])
        elif tag == "mock" and len(rest) == 1 and isinstance(rest[0], bool):
            mock = rest[0]
        else:
            raise ValueError(f"api-proxy model: unknown field {field_expr!r}")

    if overrides is None:
        raise ValueError(f"api-proxy model needs overrides: {name}")
    if (upstream is None) != (key is None):
        raise ValueError(f"api-proxy model: upstream and key come together: {name}")

    return ModelConfig(
        name=name,
        overrides=overrides,
        path=path,
        upstream=upstream,
        key=key,
        protocol=protocol,
        mock=mock,
    )


def _parse_provider_sexp(sexp: Any) -> ProviderConfig:
    if not isinstance(sexp, list) or not sexp or not isinstance(sexp[0], Symbol):
        raise ValueError(f"api-proxy: bad provider form: {sexp!r}")
    name = str(sexp[0])
    upstream: Optional[str] = None
    path = ""
    key: Optional[str] = None
    models: List[ModelConfig] = []
    explicit_protocol: Optional[str] = None
    mock: Optional[bool] = None

    for field_expr in sexp[1:]:
        if not isinstance(field_expr, list) or not field_expr:
            raise ValueError(f"api-proxy provider: unknown field {field_expr!r}")
        tag = str(field_expr[0])
        rest = field_expr[1:]
        if tag == "upstream" and len(rest) == 1 and isinstance(rest[0], str):
            upstream = rest[0]
        elif tag == "path" and len(rest) == 1 and isinstance(rest[0], str):
            path = rest[0]
        elif tag == "key" and len(rest) == 1 and isinstance(rest[0], str):
            key = rest[0]
        elif tag == "models":
            models = [_parse_model_sexp(m) for m in rest]
        elif tag == "protocol" and len(rest) == 1:
            explicit_protocol = str(rest[0])
        elif tag == "mock" and len(rest) == 1 and isinstance(rest[0], bool):
            mock = rest[0]
        else:
            raise ValueError(f"api-proxy provider: unknown field {field_expr!r}")

    if not mock and (upstream is None or key is None):
        raise ValueError(f"api-proxy provider needs upstream and key: {name}")

    protocol = _infer_protocol(name, upstream or "", path, explicit_protocol)
    is_mock = bool(mock) if mock is not None else (
        protocol == "anthropic" and (not key or key.strip() in ("***", "sk-...", "mock"))
    )

    return ProviderConfig(
        name=name,
        upstream=upstream or "",
        path=path,
        key=key or "",
        models=models,
        protocol=protocol,
        mock=is_mock,
    )


def parse_config_sexp(text: str) -> RouterConfig:
    """Parse an `(api-proxy ...)` s-expression string into a RouterConfig."""
    sexp = parse_sexp(text)
    if not isinstance(sexp, list) or not sexp or str(sexp[0]) not in ("api-proxy", "llm-router"):
        raise ValueError(f"api-proxy config: expected (api-proxy ...) form, got {sexp!r}")

    disabled = True
    host = "127.0.0.1"
    port = 8402
    providers: List[ProviderConfig] = []
    client_keys = dict(DEFAULT_CLIENT_KEYS)
    mock_anthropic = True

    for field_expr in sexp[1:]:
        if not isinstance(field_expr, list) or not field_expr:
            raise ValueError(f"api-proxy config: unknown field {field_expr!r}")
        tag = str(field_expr[0])
        rest = field_expr[1:]
        if tag == "disabled" and len(rest) == 1 and isinstance(rest[0], bool):
            disabled = rest[0]
        elif tag == "port" and len(rest) == 1 and isinstance(rest[0], int) and not isinstance(rest[0], bool):
            port = rest[0]
        elif tag == "host" and len(rest) == 1 and isinstance(rest[0], str):
            host = rest[0]
        elif tag == "mock-anthropic" and len(rest) == 1 and isinstance(rest[0], bool):
            mock_anthropic = rest[0]
        elif tag == "clients":
            parsed_clients = sexp_to_json_value(rest)
            if isinstance(parsed_clients, dict):
                client_keys.update({str(k): str(v) for k, v in parsed_clients.items()})
        elif tag == "providers":
            for p_sexp in rest:
                providers.append(_parse_provider_sexp(p_sexp))
        else:
            raise ValueError(f"api-proxy config: unknown field {field_expr!r}")

    if not disabled and not providers:
        raise ValueError("api-proxy config: no providers")

    cfg = RouterConfig(
        disabled=disabled,
        host=host,
        port=port,
        providers=providers,
        client_keys=client_keys,
        mock_anthropic=mock_anthropic,
    )
    return cfg


def parse_config_json(text: str) -> RouterConfig:
    """Parse a JSON configuration string into a RouterConfig."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON config must be an object")

    disabled = bool(data.get("disabled", False))
    host = str(data.get("host", "127.0.0.1"))
    port = int(data.get("port", 8402))
    client_keys = dict(DEFAULT_CLIENT_KEYS)
    if isinstance(data.get("client_keys"), dict):
        client_keys.update({str(k): str(v) for k, v in data["client_keys"].items()})

    providers: List[ProviderConfig] = []
    raw_providers = data.get("providers", [])
    if isinstance(raw_providers, dict):
        items = [{"name": k, **v} for k, v in raw_providers.items()]
    else:
        items = list(raw_providers)

    for p in items:
        name = str(p["name"])
        upstream = str(p.get("upstream", ""))
        path = str(p.get("path", ""))
        key = str(p.get("key", ""))
        mock = bool(p.get("mock", False))
        protocol = _infer_protocol(name, upstream, path, p.get("protocol"))
        models: List[ModelConfig] = []
        raw_models = p.get("models", [])
        if isinstance(raw_models, dict):
            m_items = [{"name": mk, **mv} for mk, mv in raw_models.items()]
        else:
            m_items = list(raw_models)
        for m in m_items:
            models.append(
                ModelConfig(
                    name=str(m["name"]),
                    overrides=dict(m.get("overrides", {})),
                    path=m.get("path"),
                    upstream=m.get("upstream"),
                    key=m.get("key"),
                    protocol=m.get("protocol"),
                    mock=bool(m.get("mock", False)),
                )
            )
        providers.append(
            ProviderConfig(
                name=name,
                upstream=upstream,
                path=path,
                key=key,
                models=models,
                protocol=protocol,
                mock=mock,
            )
        )

    if not disabled and not providers:
        raise ValueError("api-proxy config: no providers")

    return RouterConfig(
        disabled=disabled,
        host=host,
        port=port,
        providers=providers,
        client_keys=client_keys,
        mock_anthropic=bool(data.get("mock_anthropic", True)),
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RouterConfig:
    """Load configuration from an s-expression (.scm) or JSON (.json) file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if p.suffix == ".json" or stripped.startswith("{"):
        return parse_config_json(text)
    return parse_config_sexp(text)


def ensure_scaffold_config(path: str | Path = DEFAULT_CONFIG_PATH) -> bool:
    """If `path` does not exist, write the chmod-600 disabled scaffold config and return True.

    Returns False if the file already existed.
    """
    p = Path(path)
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.chmod(str(p), 0o600)
        os.write(fd, SCAFFOLD_CONFIG.encode("utf-8"))
    finally:
        os.close(fd)
    return True
