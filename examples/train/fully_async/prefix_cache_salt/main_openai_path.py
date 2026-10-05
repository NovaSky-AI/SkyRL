"""
Fully-async GSM8K multi-turn run whose LLM calls go through the OpenAI-compatible ``/v1/completions`` route
without a caller ``cache_salt``, the way an external agent harness calls SkyRL's inference endpoint. Measures
prefix-cache reuse across weight syncs (https://github.com/NovaSky-AI/SkyRL/issues/2246).

Writes to ``SKYRL_PREFIX_CACHE_REPRO_DIR``:
- ``requests.jsonl``: one row per LLM call with the client's weight version when sent, prompt and cached prompt
  tokens (``usage.prompt_tokens_details``), and whether the trajectory's previous turn ran under an older version.
- ``metrics.jsonl``: the trainer's ``policy/rollout_train_logprobs_abs_diff_*`` metrics per step.

See ``run_prefix_cache_salt_repro.sh``.
"""

import asyncio
import json
import os
import sys

import ray

from examples.train.fully_async.main_fully_async import FullyAsyncPPOExp
from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineOutput
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import validate_cfg
from skyrl.train.utils import initialize_ray


class OpenAIRouteClient:
    """Sends the generator's token-in calls to ``/v1/completions``, dropping the generator's own salt."""

    def __init__(self, client, requests_path: str):
        self._client = client
        self._requests_path = requests_path
        self._last_version = {}

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def generate(self, input_batch, model=None):
        sampling_params = input_batch.get("sampling_params") or {}
        prompts = input_batch["prompt_token_ids"]
        session_ids = input_batch.get("session_ids") or [None] * len(prompts)
        results = await asyncio.gather(
            *[self._complete(prompt, sampling_params, model, sid) for prompt, sid in zip(prompts, session_ids)]
        )
        return InferenceEngineOutput(
            responses=[r["text"] for r in results],
            response_ids=[r["token_ids"] for r in results],
            stop_reasons=[r["finish_reason"] for r in results],
            response_logprobs=[r["logprobs"] for r in results] if sampling_params.get("logprobs") is not None else None,
            prompt_logprobs=None,
            rollout_expert_indices=None,
            rollout_sample_support=None,
        )

    async def _complete(self, prompt_ids, sampling_params, model, session_id):
        version = self._client.weight_version
        body = {k: v for k, v in sampling_params.items() if v is not None}
        body.update(
            model=model or self._client.model_name, prompt=prompt_ids, logprobs=0, return_tokens_as_token_ids=True
        )
        response = await self._client.completion({"json": body, "headers": {}})
        choice = response["choices"][0]
        usage = response["usage"]
        previous = self._last_version.get(session_id)
        self._last_version[session_id] = version
        cache_salt = getattr(self._client, "cache_salt", None)
        record = {
            "weight_version": version,
            "client_salts": cache_salt is not None and cache_salt() is not None,
            "prompt_tokens": usage["prompt_tokens"],
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "turn_after_sync": previous is not None and version > previous,
        }
        with open(self._requests_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return {
            "text": choice["text"],
            "token_ids": [int(token.split(":", 1)[1]) for token in choice["logprobs"]["tokens"]],
            "finish_reason": choice["finish_reason"],
            "logprobs": choice["logprobs"]["token_logprobs"],
        }


class OpenAIRouteExp(FullyAsyncPPOExp):
    def __init__(self, cfg, out_dir: str):
        super().__init__(cfg)
        self.out_dir = out_dir

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        client = OpenAIRouteClient(inference_engine_client, os.path.join(self.out_dir, "requests.jsonl"))
        return super().get_generator(cfg, tokenizer, client)

    def get_trainer(self, cfg, tracker, **kwargs):
        log = tracker.log
        metrics_path = os.path.join(self.out_dir, "metrics.jsonl")

        def log_and_record(data, step, commit=False):
            diffs = {k: float(v) for k, v in data.items() if "rollout_train_logprobs_abs_diff" in k}
            if diffs:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, **diffs}) + "\n")
            return log(data, step, commit=commit)

        tracker.log = log_and_record
        return super().get_trainer(cfg=cfg, tracker=tracker, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig, out_dir: str):
    OpenAIRouteExp(cfg, out_dir).run()


def main() -> None:
    out_dir = os.environ["SKYRL_PREFIX_CACHE_REPRO_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg, out_dir))


if __name__ == "__main__":
    main()
