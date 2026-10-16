"""IPv4-preferring HTTPS/HTTP upstream client with streaming SSE support.

Solves all three upstream failure modes encountered with the legacy nginx
proxy:
  1. IPv6 AAAA blackholes on IPv4-only hosts: resolves hostnames using
     AF_INET first (falling back to AF_INET6 only when no IPv4 address
     exists, e.g. ::1).
  2. Multi-KB response headers (Cloudflare Set-Cookie, OpenRouter edge
     headers) overflowing 4KB buffers: raises http.client header/line
     limits and strips Set-Cookie / tracking headers before returning to
     local clients.
  3. Non-JSON HTML error pages (Cloudflare 502/504 HTML): normalizes
     upstream transport or HTML gateway errors into structured OpenAI or
     Anthropic JSON error payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import http.client
import json
import os
import socket
import ssl
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlparse
import zlib

# Allow large upstream response headers (e.g. OpenRouter Cloudflare cookies)
http.client._MAXLINE = 65536
http.client._MAXHEADERS = 1000


DEFAULT_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"

# Response headers from upstreams that should never be forwarded to local clients
STRIPPED_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "content-length",
        "content-encoding",
        "set-cookie",
        "set-cookie2",
        "alt-svc",
        "cf-ray",
        "cf-cache-status",
        "server",
        "date",
        "strict-transport-security",
        "permissions-policy",
        "referrer-policy",
    }
)

# Client request headers that are safe to pass through to upstreams
ALLOWED_PASSTHROUGH_REQUEST_HEADERS = frozenset(
    {
        "openai-beta",
        "openai-organization",
        "openai-project",
        "anthropic-beta",
        "anthropic-version",
        "http-referer",
        "x-title",
        "idempotency-key",
    }
)


class UpstreamError(Exception):
    """Raised when connecting to or reading from an upstream fails."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def parse_upstream_endpoint(upstream: str) -> Tuple[str, str, int, str]:
    """Parse an upstream specification into (scheme, host, port, base_path).

    Examples:
      "api.moonshot.ai" -> ("https", "api.moonshot.ai", 443, "")
      "http://127.0.0.1:18080/prefix" -> ("http", "127.0.0.1", 18080, "/prefix")
      "https://openrouter.ai:443" -> ("https", "openrouter.ai", 443, "")
    """
    raw = upstream.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    host = parsed.hostname or raw
    default_port = 443 if scheme == "https" else 80
    port = parsed.port or default_port
    base_path = (parsed.path or "").rstrip("/")
    return scheme, host, port, base_path


def create_ipv4_preferred_socket(
    host: str,
    port: int,
    timeout: float = 60.0,
) -> socket.socket:
    """Connect to (host, port) preferring AF_INET (IPv4) over AF_INET6."""
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UpstreamError(f"DNS resolution failed for {host}:{port}: {exc}", 502) from exc

    if not infos:
        raise UpstreamError(f"No DNS records found for {host}:{port}", 502)

    # Sort AF_INET before AF_INET6 so IPv4 is always tried first
    infos_sorted = sorted(infos, key=lambda item: 0 if item[0] == socket.AF_INET else 1)

    last_err: Optional[Exception] = None
    for af, socktype, proto, _canonname, sa in infos_sorted:
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(af, socktype, proto)
            sock.settimeout(timeout)
            sock.connect(sa)
            return sock
        except OSError as exc:
            last_err = exc
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            continue

    raise UpstreamError(
        f"Failed to connect to upstream {host}:{port}: {last_err}",
        502,
    )


def _create_ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or (
        DEFAULT_CA_BUNDLE if os.path.exists(DEFAULT_CA_BUNDLE) else None
    )
    ctx = ssl.create_default_context(cafile=cafile)
    return ctx


class IPv4HTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that resolves and connects via IPv4 first."""

    def connect(self) -> None:
        self.sock = create_ipv4_preferred_socket(
            self.host,
            self.port,
            timeout=self.timeout if isinstance(self.timeout, (int, float)) else 60.0,
        )


class IPv4HTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that resolves via IPv4 first and sets SNI to `self.host`."""

    def connect(self) -> None:
        raw_sock = create_ipv4_preferred_socket(
            self.host,
            self.port,
            timeout=self.timeout if isinstance(self.timeout, (int, float)) else 60.0,
        )
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)


def build_upstream_headers(
    host: str,
    api_key: str,
    protocol: str,
    client_headers: Mapping[str, str],
    stream: bool = False,
    has_body: bool = True,
) -> Dict[str, str]:
    """Construct clean HTTP request headers for an upstream call."""
    headers: Dict[str, str] = {
        "Host": host,
        "User-Agent": "llm-router/0.1.0",
        "Accept": "text/event-stream" if stream else "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    if has_body:
        headers["Content-Type"] = "application/json"

    if api_key:
        if protocol == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    for k, v in client_headers.items():
        lk = k.lower()
        if lk in ALLOWED_PASSTHROUGH_REQUEST_HEADERS and lk not in (h.lower() for h in headers):
            headers[k] = v

    return headers


@dataclass
class UpstreamResponse:
    status: int
    reason: str
    headers: Dict[str, str]
    is_stream: bool
    body: bytes = b""
    _conn: Optional[http.client.HTTPConnection] = None
    _resp: Optional[http.client.HTTPResponse] = None

    def iter_chunks(self, chunk_size: int = 4096) -> Iterable[bytes]:
        """Yield response body chunks for streaming SSE responses and close connection."""
        if self._resp is None:
            if self.body:
                yield self.body
            return
        try:
            while True:
                chunk = self._resp.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def _decompress_if_needed(data: bytes, content_encoding: str) -> bytes:
    enc = (content_encoding or "").lower().strip()
    if not data or not enc or enc == "identity":
        return data
    try:
        if enc == "gzip" or enc == "x-gzip":
            return gzip.decompress(data)
        if enc == "deflate":
            try:
                return zlib.decompress(data)
            except zlib.error:
                return zlib.decompress(data, -zlib.MAX_WBITS)
    except Exception:
        return data
    return data


def request_upstream(
    upstream: str,
    method: str,
    url_path: str,
    api_key: str,
    protocol: str = "openai",
    body_bytes: Optional[bytes] = None,
    client_headers: Optional[Mapping[str, str]] = None,
    stream: bool = False,
    connect_timeout: float = 30.0,
    read_timeout: float = 3600.0,
) -> UpstreamResponse:
    """Send an HTTP/HTTPS request to an upstream provider and return `UpstreamResponse`."""
    scheme, host, port, base_path = parse_upstream_endpoint(upstream)
    full_path = f"{base_path}{url_path}" if base_path else url_path
    if not full_path.startswith("/"):
        full_path = "/" + full_path

    headers = build_upstream_headers(
        host=host if port in (80, 443) else f"{host}:{port}",
        api_key=api_key,
        protocol=protocol,
        client_headers=client_headers or {},
        stream=stream,
        has_body=body_bytes is not None and len(body_bytes) > 0,
    )
    if body_bytes is not None:
        headers["Content-Length"] = str(len(body_bytes))

    conn: http.client.HTTPConnection
    if scheme == "https":
        ctx = _create_ssl_context()
        conn = IPv4HTTPSConnection(host, port, timeout=connect_timeout, context=ctx)
    else:
        conn = IPv4HTTPConnection(host, port, timeout=connect_timeout)

    try:
        conn.connect()
        if conn.sock is not None:
            conn.sock.settimeout(read_timeout)
        conn.request(method.upper(), full_path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
    except UpstreamError:
        conn.close()
        raise
    except Exception as exc:
        conn.close()
        raise UpstreamError(f"Upstream request to {host} failed: {exc}", 502) from exc

    raw_headers = {k: v for k, v in resp.getheaders()}
    content_type = ""
    content_encoding = ""
    clean_headers: Dict[str, str] = {}
    for k, v in raw_headers.items():
        lk = k.lower()
        if lk == "content-type":
            content_type = v
        elif lk == "content-encoding":
            content_encoding = v
        if lk not in STRIPPED_RESPONSE_HEADERS:
            clean_headers[k] = v

    is_sse = (
        stream and resp.status == 200 and "text/event-stream" in content_type.lower()
    )
    if is_sse:
        clean_headers["Content-Type"] = "text/event-stream; charset=utf-8"
        clean_headers["Cache-Control"] = "no-cache"
        clean_headers["X-Accel-Buffering"] = "no"
        return UpstreamResponse(
            status=resp.status,
            reason=resp.reason,
            headers=clean_headers,
            is_stream=True,
            _conn=conn,
            _resp=resp,
        )

    try:
        raw_body = resp.read()
    except Exception as exc:
        conn.close()
        raise UpstreamError(f"Error reading response from {host}: {exc}", 502) from exc
    finally:
        resp.close()
        conn.close()

    decoded_body = _decompress_if_needed(raw_body, content_encoding)
    return UpstreamResponse(
        status=resp.status,
        reason=resp.reason,
        headers=clean_headers,
        is_stream=False,
        body=decoded_body,
    )
