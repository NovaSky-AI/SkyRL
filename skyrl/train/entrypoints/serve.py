"""
Inference-only entrypoint.

Spins up the SkyRL HTTP inference servers (vLLM server groups + ``vllm-router``)
from a provided inference configuration and keeps them alive so that client code
and generation configurations can be iterated against a fixed deployment without
re-launching the engines on every run.

Example::

    uv run --isolated --extra fsdp -m skyrl.train.entrypoints.serve \
        trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct \
        trainer.placement.colocate_all=false \
        generator.inference_engine.num_engines=4 \
        generator.inference_engine.tensor_parallel_size=2

All ``generator.inference_engine.*`` knobs (plus ``trainer.policy.model.path``)
are accepted as overrides.
"""

import asyncio
import sys
import time

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils.utils import initialize_ray, validate_inference_engine_cfg


class InferenceOnlyEntrypoint(BasePPOExp):
    """Sets up and serves the inference servers without a trainer.

    Reuses :class:`BasePPOExp` for tokenizer construction and inference-client
    setup (``get_inference_client`` -> ``_get_new_inference_client``), but skips
    all training-only state (datasets, trainer, tracker).
    """

    def get_train_dataset(self):
        """No training dataset is needed for inference-only serving."""
        return None

    def get_eval_dataset(self):
        """No eval dataset is needed for inference-only serving."""
        return None

    def _teardown(self) -> None:
        """Tear down the router and server groups started during setup."""
        logger.info("Tearing down inference servers...")
        if self._inference_router is not None:
            self._inference_router.shutdown()
        for group in (
            (self._server_groups or []) + (self._prefill_server_groups or []) + (self._decode_server_groups or [])
        ):
            group.shutdown()

    def run(self) -> None:
        # Builds the vLLM server groups + router and returns an HTTP client.
        # The server group actor handles / router process are stored on ``self``
        # by ``_get_new_inference_client`` and kept alive while this task runs.
        client = self.get_inference_client()

        logger.info(
            "Inference servers are up.\n"
            f"  proxy_url (data plane):    {client.proxy_url}\n"
            f"  server_urls (control plane): {client.server_urls}\n"
            "Point your client at proxy_url for OpenAI-compatible requests. "
            "Press Ctrl+C to shut down."
        )

        try:
            while True:
                time.sleep(3600)
        # TODO: Currently this is run inside a ray task so a KeyboardInterrupt on
        # driver never reaches here - Ray sends a SIGTERM. We should propagate the
        # interrupt for better shutdown
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down inference servers...")
        finally:
            try:
                asyncio.run(client.teardown())
            finally:
                self._teardown()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # make sure that the serving loop is not run on the head node.
    exp = InferenceOnlyEntrypoint(cfg)
    exp.run()


def _validate_serve_cfg(cfg: SkyRLTrainConfig) -> None:
    """Validate the config for the inference-only serving path."""
    if cfg.trainer.policy.model.path is None:
        raise ValueError("trainer.policy.model.path must be set to the model to serve.")

    ie_cfg = cfg.generator.inference_engine

    if not ie_cfg.run_engines_locally:
        raise ValueError(
            "The serve entrypoint launches engines locally; set "
            "generator.inference_engine.run_engines_locally=true (the default)."
        )

    if ie_cfg.external_proxy_url is not None or ie_cfg.external_server_urls is not None:
        raise ValueError(
            "The serve entrypoint starts its own servers; "
            "generator.inference_engine.external_proxy_url / external_server_urls "
            "must not be set."
        )

    if cfg.trainer.placement.colocate_all:
        raise ValueError(
            "trainer.placement.colocate_all must be false for inference-only "
            "serving. Colocation puts the engines to sleep after startup so they "
            "can share GPUs with training workers, which do not exist here."
        )

    # Shared inference-engine validation (PD, parallelism, executor backend,
    # new inference layer). Resolves `override_existing_update_group="auto"`.
    validate_inference_engine_cfg(cfg)


def main() -> None:
    # Parse CLI args and build typed config
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])

    # validate inference cfg
    _validate_serve_cfg(cfg)

    # init and run
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
