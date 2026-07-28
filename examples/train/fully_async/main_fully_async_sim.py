"""
Entrypoint for fully-async training with a SIMULATED trainer (no trainer GPUs).

Runs the real generation-side fully-async loop (staleness control, buffer fill,
rate limiting, pause/resume) against inference endpoints, but simulates each
training step with a sleep instead of fwd/bwd + weight broadcast. See
``FullyAsyncTrainerSim`` and ``FullyAsyncConfig.simulate_training``.

Point it at already-served endpoints to benchmark a live deployment directly:

    generator.inference_engine.run_engines_locally=false \
    generator.inference_engine.external_proxy_url=http://<proxy-host>:<port> \
    generator.inference_engine.external_server_urls="['http://<server-host>:<port>']"

Usage:
    uv run --isolated --extra fsdp -m examples.train.fully_async.main_fully_async_sim \
        trainer.fully_async.simulate_training=true \
        trainer.fully_async.simulate_training_step_seconds=600 ...
"""

import asyncio
import faulthandler
import signal
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.fully_async_trainer_sim import FullyAsyncTrainerSim
from skyrl.train.utils import initialize_ray


class FullyAsyncSimPPOExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncTrainerSim(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


# max_retries=0: fail once loudly instead of looping, so the crash is easy to read.
@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # make sure that the training loop is not run on the head node.
    exp = FullyAsyncSimPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    assert (
        cfg.trainer.fully_async.simulate_training
    ), "main_fully_async_sim requires trainer.fully_async.simulate_training=true."
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
