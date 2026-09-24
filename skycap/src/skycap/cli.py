"""`skycap serve`: run one capture server against one upstream."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from aiohttp import web

from skycap import __version__
from skycap.server import Backend, CaptureServer
from skycap.text import TextBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skycap")
    parser.add_argument("--version", action="version", version=f"skycap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run a capture server")
    serve.add_argument(
        "--mode",
        choices=("text", "tokens"),
        default="text",
        help="text: forward to an OpenAI-compatible server; tokens: render here and call a "
        "token-in/token-out engine (default: %(default)s)",
    )
    serve.add_argument(
        "--upstream-url",
        required=True,
        help="text: OpenAI-compatible base URL (http://host:8000/v1); tokens: the engine or router root",
    )
    serve.add_argument(
        "--upstream-api-key-env",
        default="SKYCAP_UPSTREAM_API_KEY",
        help="environment variable holding the upstream API key (default: %(default)s)",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    tokens = serve.add_argument_group("tokens mode")
    tokens.add_argument("--engine", choices=("vllm", "skyrl"), default="vllm", help="engine wire")
    tokens.add_argument("--tokenizer", help="Hugging Face tokenizer name, rendered through `renderers`")
    tokens.add_argument("--model", default=None, help="model name sent to the engine (default: the request's)")
    tokens.add_argument("--max-model-len", type=int, default=None, help="clamp max_tokens to fit this context")
    tokens.add_argument(
        "--sampling-overrides",
        type=json.loads,
        default=None,
        help="JSON sampling params imposed on every call, e.g. '{\"top_k\": 50}'",
    )
    tokens.add_argument(
        "--sampling-mask",
        action="store_true",
        help="ask the engine for each sampled token's support (vLLM: also start it with return_sampling_mask)",
    )
    tokens.add_argument("--logprobs-mode", default="processed_logprobs", help="recorded on every trajectory")
    tokens.add_argument("--renderer-pool-size", type=int, default=8)
    serve.add_argument(
        "--record-dir",
        default=None,
        help="where ended trajectories are written; without it they stay in memory (development only)",
    )
    serve.add_argument(
        "--ttl",
        type=float,
        default=3600.0,
        help="seconds an open trajectory may be idle before it is written as abandoned (default: %(default)s)",
    )
    return parser


def build_backend(args: argparse.Namespace) -> Backend:
    api_key = os.environ.get(args.upstream_api_key_env)
    if args.mode == "text":
        return TextBackend(args.upstream_url, api_key=api_key)
    if not args.tokenizer:
        raise SystemExit("--mode tokens needs --tokenizer")
    from skycap.tokens.backend import TokensBackend
    from skycap.tokens.engine import ENGINES
    from skycap.tokens.renderer import RenderersRenderer

    return TokensBackend(
        args.upstream_url,
        RenderersRenderer(args.tokenizer, size=args.renderer_pool_size),
        engine=ENGINES[args.engine](),
        api_key=api_key,
        model=args.model,
        max_model_len=args.max_model_len,
        sampling_overrides=args.sampling_overrides,
        sampling_mask=args.sampling_mask,
        logprobs_mode=args.logprobs_mode,
    )


def build_server(args: argparse.Namespace) -> CaptureServer:
    backend = build_backend(args)
    return CaptureServer(backend, record_dir=args.record_dir, ttl=args.ttl)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "serve":
        web.run_app(build_server(args).app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
