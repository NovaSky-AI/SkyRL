"""`skycap serve`: run one capture server against one upstream."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from aiohttp import web

from skycap import __version__
from skycap.server import CaptureServer
from skycap.service import build_backend


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
    tokens.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help="JSON chat-template options for the renderer, e.g. '{\"enable_thinking\": false}'",
    )
    tokens.add_argument(
        "--processor-kwargs",
        type=json.loads,
        default=None,
        help="JSON options for a multimodal model's image processor; must match the engine's "
        "mm_processor_kwargs, e.g. '{\"max_pixels\": 1003520}'",
    )
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
        chat_template_kwargs=args.chat_template_kwargs,
        processor_kwargs=args.processor_kwargs,
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
        web.run_app(build_server(args).app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
