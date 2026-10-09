"""Tie non-streaming inference work to the lifetime of its HTTP caller."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import Request

T = TypeVar("T")


class RequestDisconnectedError(Exception):
    """The HTTP caller disconnected before inference completed."""


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


async def run_until_disconnected(request: Request, operation: Callable[[], Awaitable[T]]) -> T:
    """Run an operation until it finishes or its HTTP caller disconnects."""
    await request.body()
    work = asyncio.ensure_future(operation())
    disconnected = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({work, disconnected}, return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return work.result()
        disconnected.result()
        raise RequestDisconnectedError("HTTP caller disconnected")
    finally:
        for task in (work, disconnected):
            if not task.done():
                task.cancel()
        cleanup = asyncio.gather(work, disconnected, return_exceptions=True)
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
