"""Demo of Rollout Routing Replay (R3) through SkyRL's Tinker API.

Samples from an MoE model with routing capture on, then runs a forward_backward
that replays the captured routing — printing the train-vs-rollout logprob mismatch
so you can see R3 keep it small.

This uses the raw HTTP API (not the typed Tinker SDK) because the SDK's sample
response type does not surface the extra ``routed_experts`` field that R3 returns.

Usage:
    # Terminal 1: start a SkyRL Tinker server with R3 enabled (MoE + Megatron)
    bash examples/tinker/router_replay/run_tinker_server_megatron.sh

    # Terminal 2
    TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with requests \\
        python examples/tinker/router_replay/router_replay_demo.py
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import requests
import torch
from transformers import AutoTokenizer

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
logger = logging.getLogger("router_replay_demo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class TinkerHTTP:
    """Minimal raw-HTTP client for the SkyRL Tinker endpoints used here."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}/api/v1/{path}", json=body, headers=self.headers, timeout=120)
        r.raise_for_status()
        return r.json()

    def retrieve(self, request_id: str, timeout: float = 600.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.post(
                f"{self.base_url}/api/v1/retrieve_future",
                json={"request_id": request_id},
                headers=self.headers,
                timeout=120,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 425):
                time.sleep(0.5)
                continue
            r.raise_for_status()
        raise TimeoutError(f"future {request_id} not ready within {timeout}s")

    def create_session(self) -> str:
        return self._post("create_session", {"tags": [], "sdk_version": "demo"})["session_id"]

    def create_model(self, session_id: str, base_model: str) -> str:
        resp = self._post(
            "create_model",
            {"session_id": session_id, "base_model": base_model, "lora_config": {"rank": 0}},
        )
        self.retrieve(resp["request_id"])
        return resp["model_id"]

    def save_for_sampler(self, model_id: str, name: str) -> str:
        resp = self._post("save_weights_for_sampler", {"model_id": model_id, "path": name})
        self.retrieve(resp["future_id"])
        return f"tinker://{model_id}/sampler_weights/{name}"

    def sample(self, model_path: str, prompt_tokens: list[int], max_tokens: int, seed: int) -> dict:
        resp = self._post(
            "asample",
            {
                "model_path": model_path,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
                "num_samples": 1,
                "sampling_params": {"max_tokens": max_tokens, "temperature": 1.0, "seed": seed},
            },
        )
        return self.retrieve(resp["future_id"])

    def forward_backward(self, model_id: str, data: list[dict], loss_fn: str) -> dict:
        resp = self._post(
            "forward_backward",
            {"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": loss_fn}},
        )
        return self.retrieve(resp["future_id"])


def build_r3_datum(prompt_tokens, response_tokens, old_logprobs, advantages, routed_experts) -> dict:
    """A forward_backward datum that replays rollout routing (R3).

    ``routed_experts`` is the ``[num_tokens - 1, num_layers, top_k]`` routing the
    sampler returned; it is flattened into a TensorData whose ``shape`` recovers
    the 3-D layout on the server.
    """
    full_sequence = prompt_tokens + response_tokens
    model_input_tokens = full_sequence[:-1]
    routed = torch.tensor(routed_experts, dtype=torch.int64)
    weights = [1.0] * len(response_tokens)
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": model_input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": [int(t) for t in response_tokens]},
            "weights": {"data": weights},
            "logprobs": {"data": [float(lp) for lp in old_logprobs]},
            "advantages": {"data": [float(a) for a in advantages]},
            "routed_experts": {"data": [int(x) for x in routed.flatten().tolist()], "shape": list(routed.shape)},
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    client = TinkerHTTP(args.base_url, args.api_key)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    session_id = client.create_session()
    model_id = client.create_model(session_id, args.model)
    sampler_path = client.save_for_sampler(model_id, "r3_demo")
    logger.info("model_id=%s sampler=%s", model_id, sampler_path)

    prompts = [
        "Compute 12 * 13 and explain briefly.",
        "What is the capital of France?",
        "Write one sentence about mixture-of-experts models.",
        "List three prime numbers.",
    ][: args.num_prompts]

    data = []
    diffs = []
    for i, question in enumerate(prompts):
        chat = [{"role": "user", "content": question}]
        prompt_tokens = list(
            tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=False)
        )
        result = client.sample(sampler_path, prompt_tokens, args.max_tokens, args.seed + i)
        seq = result["sequences"][0]
        response_tokens = list(seq["tokens"])
        old_logprobs = list(seq.get("logprobs") or [0.0] * len(response_tokens))
        routed_experts = seq.get("routed_experts")
        if routed_experts is None:
            raise RuntimeError(
                "sample response has no routed_experts; is the server MoE + Megatron with "
                "moe_enable_routing_replay=true?"
            )
        text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        logger.info("prompt %d -> %r", i, text[:80])
        data.append(
            build_r3_datum(
                prompt_tokens,
                response_tokens,
                old_logprobs,
                advantages=[0.0] * len(response_tokens),
                routed_experts=routed_experts,
            )
        )

    # importance_sampling returns per-token training logprobs in loss_fn_outputs.
    result = client.forward_backward(model_id, data, loss_fn="importance_sampling")
    for i, (datum, output) in enumerate(zip(data, result["loss_fn_outputs"])):
        train_lp = torch.tensor(output["logprobs"]["data"], dtype=torch.float32)
        roll_lp = torch.tensor(datum["loss_fn_inputs"]["logprobs"]["data"], dtype=torch.float32)
        n = min(len(train_lp), len(roll_lp))
        diff = (train_lp[-n:] - roll_lp[-n:]).abs().mean().item()
        diffs.append(diff)
        logger.info("prompt %d: mean |train - rollout| logprob = %.6f", i, diff)

    logger.info("=" * 60)
    logger.info("R3 mean train-vs-rollout logprob mismatch: %.6f", sum(diffs) / len(diffs))
    logger.info("(Compare against a run with moe_enable_routing_replay=false to see the reduction.)")


if __name__ == "__main__":
    main()
