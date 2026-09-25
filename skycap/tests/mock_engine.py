"""A deterministic token-in/token-out engine speaking vLLM's generate wire.

The completion is a function of the prompt (and the ``seed`` sampling param):
``re<n>`` where ``n`` is the prompt length, followed by ``END``. A prompt
containing ``TOOL`` gets ``CALL:search:{}``; one containing ``BOOM`` gets a 500.

Routed experts have one row per position but the last, and row ``p`` is
filled with ``p % 256``, so a test can check every node got its own rows.
Each sampled token's support is ``[token, token + 1]``.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from aiohttp import web

from tests.fake_renderer import END, decode, encode

LAYERS, TOP_K = 2, 2


def routed_rows(total: int) -> np.ndarray:
    rows = np.arange(total - 1, dtype=np.int64) % 256
    return np.broadcast_to(rows[:, None, None], (total - 1, LAYERS, TOP_K)).astype(np.uint8)


class MockEngine:
    def __init__(self, completion: Any = None) -> None:
        #: ``completion(prompt_ids, sampling) -> ids`` overrides the default scheme.
        self.completion = completion
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.released: list[str] = []
        #: Return routed experts that cover the wrong number of positions.
        self.bad_routing = False

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1024**3)
        app.router.add_post("/inference/v1/generate", self.vllm)
        app.router.add_post("/finish_session", self.finish_session)
        return app

    def _generate(self, body: dict[str, Any]) -> tuple[list[int], list[float]] | None:
        prompt = body["token_ids"]
        sampling = body["sampling_params"]
        if "BOOM" in decode(prompt):
            return None
        if self.completion is not None:
            completion = list(self.completion(prompt, sampling))
        elif "TOOL" in decode(prompt[-40:]):
            completion = [*encode("CALL:search:{}"), END]
        else:
            completion = [*encode(f"re{len(prompt)}{sampling.get('seed', '')}"), END]
        return completion, [-0.01 * (i + 1) for i in range(len(completion))]

    async def vllm(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        generated = self._generate(body)
        if generated is None:
            return web.json_response({"error": "boom"}, status=500)
        completion, logprobs = generated
        buffer = io.BytesIO()
        rows = routed_rows(len(body["token_ids"]) + len(completion))
        np.save(buffer, rows[:3] if self.bad_routing else rows)
        choice = {
            "index": 0,
            "token_ids": completion,
            "finish_reason": "stop",
            "logprobs": {"content": [{"token": "", "logprob": lp} for lp in logprobs]},
            "routed_experts": base64.b64encode(buffer.getvalue()).decode(),
            "sampling_mask": [[t, t + 1] for t in completion],
        }
        return web.json_response({"choices": [choice]})

    async def finish_session(self, request: web.Request) -> web.Response:
        self.released.append(request.query["session_id"])
        return web.json_response({"ok": True})
