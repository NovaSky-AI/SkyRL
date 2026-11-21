"""``main_harbor`` override entrypoint: routes Harbor recipes through Arctic RL.

User flow (same for any existing Harbor launcher — no other changes needed):

    python -m examples.train_integrations.harbor.entrypoints.main_harbor \\
        trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint \\
        <existing Harbor overrides>

Everything reused verbatim from upstream:
  * ``ArcticRLExp`` / ``ArcticPPOTrainer`` — training routed to Arctic server.
  * ``HarborGenerator`` / ``HarborTaskDataset`` — Harbor Trials + reward.

Only new piece: ``ArcticInferenceEngineAdapter`` (openai_bridge.py) exposes an
OpenAI-compatible HTTP shim in front of the same Arctic ``ReplicaPool`` that
serves training rollouts, so Harbor's LiteLLM agent (inside the sandbox) sees
the up-to-date policy after every weight sync.

Do NOT ``from __future__ import annotations``: SkyRL's
``build_nested_dataclass`` introspects ``dataclasses.fields(cls)[i].type`` and
expects a concrete class, not a string. With future-annotations, nested
configs (e.g. ``generator: HarborGeneratorConfig``) get parsed as raw dicts
and ``SkyRLTrainConfig.__post_init__`` then fails with
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

# Shim bind address env vars. Kept out of the config schema so this recipe
# adds no new SkyRL config fields. Full table lives in the arctic README.
_SHIM_HOST_ENV = "ARCTIC_HARBOR_SHIM_HOST"
_SHIM_PORT_ENV = "ARCTIC_HARBOR_SHIM_PORT"


@dataclass
class ArcticHarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRL config with Harbor's ``harbor_trial_config`` and Arctic RL's
    ``trainer.arctic_rl`` sub-config both slotted in.
    """

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)
    trainer: ArcticTrainerConfig = field(default_factory=ArcticTrainerConfig)


class ArcticHarborExp(ArcticRLExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._openai_adapter: Optional[ArcticInferenceEngineAdapter] = None

    # -- data ---------------------------------------------------------------
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

    # -- generator ---------------------------------------------------------
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        # HarborGenerator looks up ``inference_engine_client.get_endpoint_url()``
        # to bake the LiteLLM ``api_base`` into every Trial's config; we pass
        # our adapter (which exposes the OpenAI shim we just started).
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

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

    # -- assembly ----------------------------------------------------------
    def _setup_trainer(self):
        logger.info(
            "Setting up ArcticHarbor trainer (GPU work delegated to Arctic RL server, "
            "rollouts served over an OpenAI shim for Harbor's LiteLLM agent)"
        )
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        arl_cfg = getattr(self.cfg.trainer, "arctic_rl", None)
        cfg_colocate = bool(getattr(arl_cfg, "colocate", False)) if arl_cfg else False

        ie_cfg = self.cfg.generator.inference_engine
        assert ie_cfg.served_model_name is not None, (
            "generator.inference_engine.served_model_name must be set — Harbor "
            "sends model=hosted_vllm/<served_model_name>."
        )
        default_max_tokens = int(
            getattr(self.cfg.generator.sampling_params, "max_generate_length", 4096)
            or 4096
        )

        shim_host = os.environ.get(_SHIM_HOST_ENV, "0.0.0.0")
        shim_port = int(os.environ.get(_SHIM_PORT_ENV, "8000"))
        chat_template_path = None
        engine_init_kwargs = getattr(ie_cfg, "engine_init_kwargs", None) or {}
        if hasattr(engine_init_kwargs, "get"):
            chat_template_path = engine_init_kwargs.get("chat_template")
        chat_template = None
        if chat_template_path:
            with open(str(chat_template_path)) as f:
                chat_template = f.read()

        # HarborGenerator bakes the shim URL into LiteLLM's ``api_base`` at
        # __init__ time — start the shim first so the URL is real.
        self._openai_adapter = ArcticInferenceEngineAdapter(
            arctic_client=self.arctic_client,
            tokenizer=self.tokenizer,
            model_name=ie_cfg.served_model_name,
            default_max_tokens=default_max_tokens,
            host=shim_host,
            port=shim_port,
            chat_template=chat_template,
        )
        self._openai_adapter.spin_up_http_endpoint()

        # Trainer-side inference-engine handle: same stub upstream uses, so
        # colocated sleep/wake still routes to the Arctic RL server.
        ie_stub = _ArcticInferenceEngineStub(
            client=self.arctic_client, colocate=cfg_colocate
        )
        tracker = self.get_tracker()
        generator = self.get_generator(
            self.cfg, self.tokenizer, self._openai_adapter
        )
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
    exp = ArcticHarborExp(
        cfg, reconnect_config=reconnect_config, server_state=server_state
    )
    exp.run()


def main() -> None:
    # ``main_harbor.py`` already stripped its ``override_entrypoint=`` peek
    # arg before dispatching here, but SkyRL's strict config parser would
    # still reject any leftover ``trainer.override_entrypoint=…`` in argv, so
    # defensively drop it here too.
    argv = [
        a for a in sys.argv[1:] if not a.startswith("trainer.override_entrypoint=")
    ]
    cfg = ArcticHarborSkyRLConfig.from_cli_overrides(argv)
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be set — Harbor uses it to "
            "truncate responses."
        )

    # ``create_arctic_rl_client`` starts a Ray cluster on first call, so
    # pre-init BEFORE ``ray.init``; ``ignore_reinit_error=True`` reuses that
    # cluster on the follow-up ``ray.init``.
    rl_config = build_rl_config(cfg)
    logger.info("Pre-initializing ArcticRL jobs…")
    pre_client = create_arctic_rl_client(rl_config)
    reconnect_cfg = pre_client.reconnect_config()
    server_state = (
        pre_client.get_server_state() if rl_config.comm_protocol == "ray" else None
    )
    logger.info(
        "ArcticRL ready — training=%s, sample=%s, log_prob=%s",
        pre_client.training_job_id,
        pre_client.sampling_job_id,
        pre_client.log_prob_job_id,
    )

    # Ray workers reconstruct the driver task, so they need our imports on
    # PYTHONPATH and the sandbox-provider creds forwarded.
    env_vars = prepare_runtime_environment(cfg)
    for prefix in (
        "ARCTIC_",
        "WANDB_",
        "DAYTONA_",
        "MODAL_",
        "E2B_",
        "RUNLOOP_",
        "GKE_",
        "OPENAI_",
        "HF_",
    ):
        env_vars.update(
            {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        )
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _existing_pp = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = _repo_root + (
        os.pathsep + _existing_pp if _existing_pp else ""
    )
    runtime_env = {"env_vars": env_vars}

    ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
    ray.get(
        skyrl_entrypoint.options(runtime_env=runtime_env).remote(
            cfg, reconnect_config=reconnect_cfg, server_state=server_state
        )
    )


if __name__ == "__main__":
    main()
