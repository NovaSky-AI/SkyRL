"""Arctic RL entrypoint. Launches an ``ArcticRLClient`` (GPU work runs on the
Arctic server) and wires it into SkyRL's trainer/generator.

Invoked via ``trainer.override_entrypoint=integrations.arctic_rl.entrypoint``
on the ``skyrl.train.entrypoints.main_base`` command line. ARL-specific knobs
live under ``trainer.arctic_rl`` (see ``ArcticRLTrainerConfig`` in
``.config``).
"""

import os
import sys
from typing import Any, Optional

import ray
from loguru import logger

from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg

from . import ArcticGenerator, ArcticPPOTrainer
from .config import build_rl_config


class ArcticRLExp(BasePPOExp):

    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        reconnect_config: Optional[ArcticRLClientConfig] = None,
        server_state: Optional[Any] = None,
    ):
        n_samples = cfg.generator.n_samples_per_prompt
        mini_batch_size = cfg.trainer.policy_mini_batch_size * n_samples
        train_batch_size = cfg.trainer.train_batch_size * n_samples
        grad_accum_steps = max(1, train_batch_size // mini_batch_size)
        lr = cfg.trainer.policy.optimizer_config.lr

        # arctic_platform.rl.create_arctic_rl_client takes an optional
        # server_state used by the ray-protocol client to reattach to an
        # already-initialized server actor (driver pre-init pattern, see main()).
        if reconnect_config is not None:
            self.arctic_client = create_arctic_rl_client(reconnect_config, server_state)
        else:
            self.arctic_client = create_arctic_rl_client(build_rl_config(cfg), server_state)

        logger.info(
            f"DeepSpeed config: lr={lr}, grad_accum_steps={grad_accum_steps}, "
            f"mini_batch={mini_batch_size}, train_batch={train_batch_size}"
        )
        logger.info(
            f"ArcticRLClient ready — "
            f"training_job={self.arctic_client.training_job_id}, "
            f"sample_job={self.arctic_client.sampling_job_id}, "
            f"log_prob_job={self.arctic_client.log_prob_job_id}"
        )
        super().__init__(cfg)

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return ArcticGenerator(
            arctic_client=self.arctic_client,
            tokenizer=tokenizer,
            sampling_params=cfg.generator.sampling_params,
        )

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
                    inference_engine_client, generator, colocate_pg):
        return ArcticPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
            arctic_client=self.arctic_client,
        )

    def _setup_trainer(self):
        logger.info("Setting up ArcticRL trainer (GPU work delegated to on-prem server)")
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        # `colocate` from cfg, not client.config (see _ArcticDispatch.__init__).
        arl_cfg = getattr(self.cfg.trainer, "arctic_rl", None)
        cfg_colocate = bool(getattr(arl_cfg, "colocate", False)) if arl_cfg else False

        from .trainer import _ArcticInferenceEngineStub
        ie_stub = _ArcticInferenceEngineStub(client=self.arctic_client, colocate=cfg_colocate)

        tracker = self.get_tracker()
        generator = self.get_generator(self.cfg, self.tokenizer, None)
        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=ie_stub,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )
        trainer.build_models()
        if cfg_colocate:
            trainer.colocate_all = True
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(
    cfg: SkyRLTrainConfig,
    reconnect_config: Optional[ArcticRLClientConfig] = None,
    server_state: Optional[Any] = None,
):
    exp = ArcticRLExp(cfg, reconnect_config=reconnect_config, server_state=server_state)
    exp.run()


def main() -> None:
    """Arctic RL entrypoint. Dispatched here by ``main_base`` when
    ``trainer.override_entrypoint=integrations.arctic_rl.entrypoint`` is set on
    the CLI. Parses with ``ArcticSkyRLConfig`` (core config + ``trainer.arctic_rl``).

    If the user didn't pass ``trainer.arctic_rl=`` overrides, defaults to an
    empty ``ArcticRLTrainerConfig()`` (=Arctic with default settings) so the
    user only needs the one ``override_entrypoint`` flag to opt into Arctic.
    """
    from .config import ArcticRLTrainerConfig, ArcticSkyRLConfig
    cfg = ArcticSkyRLConfig.from_cli_overrides(sys.argv[1:])
    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_cfg(cfg)

    rl_config = build_rl_config(cfg)
    logger.info("Pre-initializing ArcticRL jobs (before ray.init)…")
    pre_client = create_arctic_rl_client(rl_config)
    reconnect_cfg = pre_client.reconnect_config()
    # reconnect_cfg is a minimal schema (no policy flags); for ray comm the
    # client also owns an in-process server actor state that the reconnecting
    # worker needs (http: None). Policy flags read from `cfg` (see
    # _ArcticDispatch.__init__).
    server_state = (
        pre_client.get_server_state() if rl_config.comm_protocol == "ray" else None
    )
    logger.info(
        f"ArcticRL jobs ready — training={pre_client.training_job_id}, "
        f"sample={pre_client.sampling_job_id}, log_prob={pre_client.log_prob_job_id}"
    )

    from skyrl.train.utils.utils import prepare_runtime_environment
    env_vars = prepare_runtime_environment(cfg)
    # Forward ARCTIC_* env vars to Ray workers — moved here from core utils per
    # reviewer feedback (core stays integration-agnostic).
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("ARCTIC_")})
    # Forward all WANDB_* env vars to Ray workers so wandb can authenticate
    # (skyrl's default propagation only forwards WANDB_API_KEY, which is
    # insufficient when WANDB_BASE_URL points to a non-default endpoint).
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("WANDB_")})
    # Forward the SkyRL repo root on Ray workers' PYTHONPATH so they can import
    # ``integrations.arctic_rl.*`` when deserializing the skyrl_entrypoint task.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _existing_pp = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = _repo_root + (os.pathsep + _existing_pp if _existing_pp else "")
    runtime_env = {"env_vars": env_vars}
    # create_arctic_rl_client(ray) above started a Ray cluster; reuse it via
    # ignore_reinit_error and pass runtime_env at TASK granularity (init's
    # runtime_env is ignored on the second call to an already-initialized cluster).
    ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
    ray.get(
        skyrl_entrypoint.options(runtime_env=runtime_env).remote(
            cfg, reconnect_config=reconnect_cfg, server_state=server_state,
        )
    )


if __name__ == "__main__":
    main()
