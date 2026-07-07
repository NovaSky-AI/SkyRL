"""Eval-only entrypoint that generates against the external Fireworks endpoint.

Mirrors ``skyrl.train.entrypoints.main_generate`` but overrides ``get_inference_client`` to build
:class:`~examples.eval.fireworks.fireworks_client.FireworksInferenceClient` — no local inference
engines and no vLLM. Token-in/token-out is preserved (prompts are sent as raw token ids and
Fireworks returns the generated ``token_ids``), so the stock ``SkyRLGymGenerator`` works unchanged.

Configuration conventions (no custom config fields):
  - ``trainer.policy.model.path`` is the served model's HF id (e.g. ``openai/gpt-oss-20b``). In
    eval-only mode it is consumed solely for the tokenizer and dataset tokenization, and it MUST
    be the served model's tokenizer: token ids are consumed raw by the server, so a mismatch
    degrades generations silently instead of erroring.
  - ``generator.inference_engine.served_model_name`` is the Fireworks model id (e.g.
    ``accounts/fireworks/models/gpt-oss-20b``). ``resolve_policy_model_name`` routes it as the
    request ``model``.
  - ``FIREWORKS_AI_API_KEY`` (env) is the API key; ``FIREWORKS_BASE_URL`` (env, optional)
    overrides the server root for self-hosted OpenAI-compatible endpoints (no ``/v1`` suffix).

Requires the ``fireworks`` uv extra; see ``run_eval_fireworks.sh``.
"""

import asyncio
import os
import sys

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg


class FireworksEvalOnlyEntrypoint(EvalOnlyEntrypoint):
    def get_inference_client(self) -> InferenceEngineInterface:
        try:
            from examples.eval.fireworks.fireworks_client import FireworksInferenceClient
        except ImportError as e:
            raise ImportError(
                "The Fireworks eval example requires the fireworks-ai SDK. Install the `fireworks` "
                "extra, e.g. `uv run --isolated --extra fireworks ...`."
            ) from e

        ie_cfg = self.cfg.generator.inference_engine
        assert ie_cfg.served_model_name, (
            "Set generator.inference_engine.served_model_name to the Fireworks model id "
            "(e.g. accounts/fireworks/models/gpt-oss-20b)"
        )
        api_key = os.environ.get("FIREWORKS_AI_API_KEY")
        assert api_key, "Export FIREWORKS_AI_API_KEY (use 'EMPTY' for keyless self-hosted endpoints)"

        # Surface the tokenizer/model pairing at startup: model.path must be the served model's
        # tokenizer since prompts are sent as raw token ids (see module docstring).
        logger.info(
            f"Fireworks eval: served_model_name={ie_cfg.served_model_name}, "
            f"tokenizer (trainer.policy.model.path)={self.cfg.trainer.policy.model.path}"
        )
        return FireworksInferenceClient(
            model_name=ie_cfg.served_model_name,
            tokenizer=self.tokenizer,
            base_url=os.environ.get("FIREWORKS_BASE_URL"),
            api_key=api_key,
        )


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: SkyRLTrainConfig) -> dict:
    exp = FireworksEvalOnlyEntrypoint(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
