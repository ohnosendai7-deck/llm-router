"""CLI entrypoint for llm-router (`llm-router` / `api-proxy`)."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import List, Optional

from .config import (
    DEFAULT_CONFIG_PATH,
    ensure_scaffold_config,
    load_config,
)
from .server import create_server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="Local OpenAI-compatible and Anthropic-compatible LLM router and key-injecting gateway.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=os.environ.get("API_PROXY_CONFIG_FILE", DEFAULT_CONFIG_PATH),
        help=f"Path to config file (.scm or .json, default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("API_PROXY_HOST"),
        help="Override bind host (default: from config or 127.0.0.1)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.environ["API_PROXY_PORT"]) if os.environ.get("API_PROXY_PORT") else None,
        help="Override bind port (default: from config or 8402)",
    )
    parser.add_argument(
        "--scaffold-if-missing",
        action="store_true",
        help="Write a chmod-600 disabled scaffold config if the config file does not exist.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the config file and exit (exit 0 if enabled, exit 2 if disabled/scaffolded).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_path = args.config
    if not os.path.exists(config_path):
        if args.scaffold_if_missing:
            ensure_scaffold_config(config_path)
            print(
                f"api-proxy: wrote disabled scaffold config to {config_path}; "
                "edit it and set (disabled #f) to enable",
                file=sys.stdout,
                flush=True,
            )
            return 2
        print(f"llm-router: config file {config_path} does not exist", file=sys.stderr)
        return 1

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"llm-router: failed to load {config_path}: {exc}", file=sys.stderr)
        return 1

    if cfg.disabled:
        print(
            f"api-proxy: (disabled #t) in {config_path}; not starting",
            file=sys.stdout,
            flush=True,
        )
        return 2

    if args.check_config:
        return 0

    srv = create_server(cfg, host=args.host, port=args.port)

    def _handle_sigterm(signum: int, _frame: object) -> None:
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        srv.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
