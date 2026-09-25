"""Runnable mock upstream, for the benchmark.

Run as its own process so the load generator, the proxy, and the upstream do
not share an event loop. Without that separation the measured absolute
throughput says more about Python contention than about the proxy.

    python -m tools.mock_server --port 9101
    python -m tools.mock_server --port 9101 --workers 8

One process is a single event loop, which becomes the bottleneck well before
the proxy does -- a real serving engine batches across many requests and this
does not. ``--workers`` forks N processes over one listening socket so the
upstream stops being what the benchmark measures. They share the session
prefix map through a manager, so the cached-token model stays exact however
the kernel spreads connections.
"""

from __future__ import annotations

import argparse
import multiprocessing
import socket
import sys
from pathlib import Path
from typing import Any

import uvicorn

# One mock upstream, shared by the test fixtures and by this command. It lives
# with the fixtures because that is its main consumer, and this adds the
# `tests` directory to the path rather than the other way round: nothing in the
# installed package should depend on either.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from support.mock_upstream import MockUpstream  # noqa: E402


def _loop_name() -> str:
    try:
        import uvloop  # noqa: F401
    except ImportError:
        return "auto"
    return "uvloop"


def _serve(arguments: argparse.Namespace, sessions: Any, sock: socket.socket | None) -> None:
    upstream = MockUpstream(
        reply=arguments.reply,
        chunks=arguments.chunks,
        chunk_delay=arguments.chunk_delay,
        session_lengths=sessions,
    )
    # Request recording would grow without bound under load and is not needed
    # by the benchmark.
    upstream.record_requests = False
    config = uvicorn.Config(
        upstream,
        host=arguments.host,
        port=arguments.port,
        log_level="error",
        access_log=False,
        loop=_loop_name(),
    )
    server = uvicorn.Server(config)
    server.run(sockets=[sock] if sock is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--reply", default="benchmark reply payload")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-delay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Processes over one socket")
    arguments = parser.parse_args()

    if arguments.workers <= 1:
        _serve(arguments, None, None)
        return

    # Bind once and let the children inherit it, so the kernel spreads accepts
    # across them.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((arguments.host, arguments.port))
    sock.listen(2048)
    sock.set_inheritable(True)

    with multiprocessing.Manager() as manager:
        sessions = manager.dict()
        children = [
            multiprocessing.Process(target=_serve, args=(arguments, sessions, sock), daemon=True)
            for _ in range(arguments.workers)
        ]
        for child in children:
            child.start()
        try:
            for child in children:
                child.join()
        except KeyboardInterrupt:
            for child in children:
                child.terminate()


if __name__ == "__main__":
    main()
