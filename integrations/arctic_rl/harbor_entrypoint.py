"""``main_base`` entrypoint that composes Arctic RL training with Harbor rollouts.

Everything reused verbatim from upstream:
  - ``ArcticRLExp`` / ``ArcticPPOTrainer`` — training routed to Arctic server.
  - ``HarborGenerator`` / ``HarborTaskDataset`` — Harbor Trials + reward.

Only new piece: ``ArcticInferenceEngineAdapter`` (openai_bridge.py) puts an
OpenAI HTTP shim in front of the same Arctic ``ReplicaPool`` that serves
training rollouts, so weight sync auto-refreshes what Harbor's sandbox agent
sees.

Do NOT ``from __future__ import annotations``: SkyRL's ``build_nested_dataclass``
introspects ``dataclasses.fields(cls)[i].type`` and expects a concrete class,
not a string annotation. With future-annotations, nested configs (e.g.
``generator: HarborGeneratorConfig``) get parsed as raw dicts and then
``SkyRLTrainConfig.__post_init__`` fails with
``AttributeError: 'dict' object has no attribute 'max_input_length'``.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import ray
import yaml
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
from loguru import logger

from examples.train_integrations.harbor.dataset import HarborTaskDataset  # noqa: E402
from examples.train_integrations.harbor.entrypoints.main_harbor import (  # noqa: E402
    HARBOR_DEFAULT_CONFIG,
    HarborGeneratorConfig,
    _deep_merge,
)
from examples.train_integrations.harbor.harbor_generator import (
    HarborGenerator,  # noqa: E402
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import prepare_runtime_environment

from . import ArcticPPOTrainer
from .config import ArcticRLTrainerConfig, ArcticTrainerConfig, build_rl_config
from .entrypoint import ArcticRLExp
from .openai_bridge import ArcticInferenceEngineAdapter
from .trainer import _ArcticInferenceEngineStub


@dataclass
class ArcticHarborSkyRLConfig(SkyRLTrainConfig):
    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)
    trainer: ArcticTrainerConfig = field(default_factory=ArcticTrainerConfig)


class ArcticHarborExp(ArcticRLExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._openai_adapter: Optional[ArcticInferenceEngineAdapter] = None

    def get_train_dataset(self):
        ds = HarborTaskDataset(data_files=self.cfg.data.train_data)
        assert len(ds) >= self.cfg.trainer.train_batch_size, (
            f"HarborTaskDataset size ({len(ds)}) < train_batch_size "
            f"({self.cfg.trainer.train_batch_size})"
        )
        return ds

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return HarborTaskDataset(data_files=self.cfg.data.val_data)
        return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
        inference_engine_client, generator, colocate_pg,
    ):
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
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        arl_cfg = getattr(self.cfg.trainer, "arctic_rl", None)
        cfg_colocate = bool(getattr(arl_cfg, "colocate", False)) if arl_cfg else False

        # HarborGenerator bakes the shim URL into its LiteLLM ``api_base`` at
        # __init__ time, so the shim has to be listening before we build it.
        ie_cfg = self.cfg.generator.inference_engine
        assert ie_cfg.served_model_name is not None, (
            "generator.inference_engine.served_model_name must be set — Harbor "
            "sends model=hosted_vllm/<served_model_name>."
        )
        assert bool(ie_cfg.enable_http_endpoint), (
            "generator.inference_engine.enable_http_endpoint must be true for Harbor."
        )
        default_max_tokens = int(
            getattr(self.cfg.generator.sampling_params, "max_generate_length", 4096) or 4096
        )
        self._openai_adapter = ArcticInferenceEngineAdapter(
            arctic_client=self.arctic_client,
            tokenizer=self.tokenizer,
            model_name=ie_cfg.served_model_name,
            inference_engine_cfg=ie_cfg,
            default_max_tokens=default_max_tokens,
        )
        self._openai_adapter.spin_up_http_endpoint()

        ie_stub = _ArcticInferenceEngineStub(client=self.arctic_client, colocate=cfg_colocate)
        tracker = self.get_tracker()
        generator = self.get_generator(self.cfg, self.tokenizer, self._openai_adapter)
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
    cfg: ArcticHarborSkyRLConfig,
    reconnect_config: Optional[ArcticRLClientConfig] = None,
    server_state: Optional[Any] = None,
):
    exp = ArcticHarborExp(cfg, reconnect_config=reconnect_config, server_state=server_state)
    exp.run()


def main() -> None:
    cfg = ArcticHarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be set — Harbor uses it to truncate responses."
        )

    # ``create_arctic_rl_client`` starts a Ray cluster on first call, so pre-init
    # BEFORE ``ray.init``; ``ignore_reinit_error=True`` reuses that cluster.
    rl_config = build_rl_config(cfg)
    logger.info("Pre-initializing ArcticRL jobs…")
    pre_client = create_arctic_rl_client(rl_config)
    reconnect_cfg = pre_client.reconnect_config()
    server_state = (
        pre_client.get_server_state() if rl_config.comm_protocol == "ray" else None
    )
    logger.info(
        f"ArcticRL ready — training={pre_client.training_job_id}, "
        f"sample={pre_client.sampling_job_id}, log_prob={pre_client.log_prob_job_id}"
    )

    # Ray workers reconstruct the driver task, so they need our imports on
    # PYTHONPATH and the sandbox-provider creds forwarded.
    env_vars = prepare_runtime_environment(cfg)
    for prefix in ("ARCTIC_", "WANDB_", "DAYTONA_", "MODAL_", "E2B_",
                   "RUNLOOP_", "GKE_", "OPENAI_", "HF_"):
        env_vars.update({k: v for k, v in os.environ.items() if k.startswith(prefix)})
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _existing_pp = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = _repo_root + (os.pathsep + _existing_pp if _existing_pp else "")
    runtime_env = {"env_vars": env_vars}

    ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
    ray.get(skyrl_entrypoint.options(runtime_env=runtime_env).remote(
        cfg, reconnect_config=reconnect_cfg, server_state=server_state,
    ))


if __name__ == "__main__":
    main()
