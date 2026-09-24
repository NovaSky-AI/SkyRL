"""`skycap serve`: run one capture server against one upstream."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from aiohttp import web

from skycap import __version__
from skycap.server import CaptureServer
from skycap.text import TextBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skycap")
    parser.add_argument("--version", action="version", version=f"skycap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run a capture server")
    serve.add_argument("--upstream-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    serve.add_argument(
        "--upstream-api-key-env",
        default="SKYCAP_UPSTREAM_API_KEY",
        help="environment variable holding the upstream API key (default: %(default)s)",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    return parser


def build_server(args: argparse.Namespace) -> CaptureServer:
    backend = TextBackend(args.upstream_url, api_key=os.environ.get(args.upstream_api_key_env))
    return CaptureServer(backend)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "serve":
        web.run_app(build_server(args).app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
