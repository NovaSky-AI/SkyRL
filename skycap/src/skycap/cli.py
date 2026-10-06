"""`skycap serve`: run one capture server against one upstream."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys

from aiohttp import web

from skycap import __version__
from skycap.exposure import Exposure, load_exposure
from skycap.server import CaptureServer
from skycap.service import build_backend, serve

logger = logging.getLogger(__name__)


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
        help="record each sampled token's support (start vLLM with return_sampling_mask)",
    )
    tokens.add_argument("--logprobs-mode", default="processed_logprobs", help="recorded on every trajectory")
    tokens.add_argument(
        "--use-raw-content",
        action="store_true",
        help="answer with the completion's text as content, reasoning inline and tool calls unparsed, "
        "as vLLM does with no parsers",
    )
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
    serve.add_argument(
        "--path-rule",
        action="append",
        default=[],
        metavar="[NAME=]MODULE:FUNCTION",
        help="a custom path rule finish may name besides `all` and `final`, by NAME or else by its import path; "
        "repeatable",
    )
    serve.add_argument(
        "--expose",
        default=None,
        metavar="external_host|cloudflare|MODULE:CLASS",
        help="also serve the harness routes alone, reachable from outside this network this way (skycap.exposure); "
        "create then returns each trajectory's exposed_base_url",
    )
    serve.add_argument(
        "--expose-kwargs",
        type=json.loads,
        default={},
        help='JSON arguments of the exposure, e.g. \'{"host": "203.0.113.7", "port": 11500}\' for external_host',
    )
    return parser


def build_server(args: argparse.Namespace) -> CaptureServer:
    if args.mode == "tokens" and not args.tokenizer:
        raise SystemExit("--mode tokens needs --tokenizer")
    backend = build_backend(
        args.upstream_url,
        mode=args.mode,
        api_key=os.environ.get(args.upstream_api_key_env),
        tokenizer=args.tokenizer,
        renderer_pool_size=args.renderer_pool_size,
        model=args.model,
        max_model_len=args.max_model_len,
        sampling_overrides=args.sampling_overrides,
        sampling_mask=args.sampling_mask,
        logprobs_mode=args.logprobs_mode,
        use_raw_content=args.use_raw_content,
    )
    rules: dict[str, str] = {}
    for spec in args.path_rule:
        name, _, rule = spec.rpartition("=")
        rules[name or rule] = rule
    return CaptureServer(backend, record_dir=args.record_dir, ttl=args.ttl, path_rules=rules)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "serve":
        if args.expose is None and args.expose_kwargs:
            raise SystemExit("--expose-kwargs is given without --expose")
        if args.expose is None:
            web.run_app(build_server(args).app(), host=args.host, port=args.port)
        else:
            if not isinstance(args.expose_kwargs, dict):
                raise SystemExit(f"--expose-kwargs must be a JSON object, not {args.expose_kwargs!r}")
            try:
                exposure = load_exposure(args.expose, **args.expose_kwargs)
            except ValueError as error:
                raise SystemExit(f"--expose: {error}") from None
            asyncio.run(_serve_exposed(build_server(args), host=args.host, port=args.port, exposure=exposure))
    return 0


async def _serve_exposed(server: CaptureServer, *, host: str, port: int, exposure: Exposure) -> None:
    """``skycap serve --expose``: the server, its exposed harness listener, until SIGINT or SIGTERM."""
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopping.set)

    def ready(url: str, harness_url: str | None) -> None:
        logger.info("skycap serving at %s; harness routes at %s, exposed at %s", url, harness_url, server.exposed_url)

    advertise = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    await serve(
        server, host=host, port=port, advertise_host=advertise, exposure=exposure, stopping=stopping, ready=ready
    )


if __name__ == "__main__":
    sys.exit(main())
