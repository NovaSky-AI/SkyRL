"""Entrypoint for ECHO terminal-agent training on SkyRL hooks."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import ray
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import CriticWorker, RefWorker
from skyrl.train.config import GeneratorConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .chat_template import load_chat_template, resolve_chat_template_path
from .dataset import TerminalAgentTaskDataset
from .generator import TerminalAgentGenerator
from .world_modeling.config import EchoSkyRLTrainConfig
from .world_modeling.fsdp_worker import PolicyWorker
from .world_modeling.trainer import EchoPPOTrainer


def _load_config(argv: list[str]) -> "EchoTerminalAgentSkyRLConfig":
    config_path = None
    remaining = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a config path")
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1

    if config_path is None:
        return EchoTerminalAgentSkyRLConfig.from_cli_overrides(remaining)

    base = OmegaConf.load(Path(config_path))
    overrides = OmegaConf.from_cli(remaining)
    merged = OmegaConf.merge(base, overrides)
    return EchoTerminalAgentSkyRLConfig.from_dict_config(merged)


@dataclass
class TerminalAgentGeneratorConfig(GeneratorConfig):
    max_turns: int = 16
    max_context_tokens: int = 25000
    max_tokens_per_generation: int = 2048
    max_terminal_output_chars: int = 6000
    terminal_output_truncation: str = "start_end"
    max_commands_per_turn: int = 1
    command_selection: Literal["first", "last", "all"] = "first"
    parser_name: Literal["xml", "hermes", "qwen35"] = "xml"
    length_penalty_coef: float = 0.0
    length_penalty_threshold: int = 20000
    correct_threshold: float = 0.5
    verifier_timeout: float = 120.0
    thinking_handling: Literal["keep_all", "strip_rollout", "strip_all"] = "keep_all"
    max_total_tokens: Optional[int] = None
    add_format_warn: bool = False
    max_world_model_tokens: Optional[int] = None
    world_loss_target: Optional[Literal["full_observation", "env_only", "warning_only", "warning_plus_env"]] = None
    world_loss_on_format_warn: bool = True
    world_model_only_paths: list[str] = field(default_factory=list)
    agent_max_concurrency: int = 32
    agent_timeout: float = 1800.0
    docker_memory_mb: Optional[int] = None
    docker_cpus: Optional[int] = None
    base_temp_dir: Optional[str] = None
    max_concurrent_builds: int = 32
    max_build_retries: int = 3
    prompt_key: str = "prompt"
    path_key: str = "path"
    task_binary_key: str = "task_binary"
    dataset_num_workers: int = 8
    dataset_max_rows: Optional[int] = None
    eval_dataset_max_rows_per_source: Optional[int] = None


@dataclass
class EchoTerminalAgentSkyRLConfig(EchoSkyRLTrainConfig):
    generator: TerminalAgentGeneratorConfig = field(default_factory=TerminalAgentGeneratorConfig)
    tokenizer_path: Optional[str] = None
    chat_template_path: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        engine_init_kwargs = self.generator.inference_engine.engine_init_kwargs

        if self.chat_template_path is not None:
            self.chat_template_path = str(resolve_chat_template_path(self.chat_template_path))
            configured_chat_template = engine_init_kwargs.get("chat_template")
            if configured_chat_template is not None and configured_chat_template != self.chat_template_path:
                raise ValueError(
                    "chat_template_path and generator.inference_engine.engine_init_kwargs.chat_template "
                    f"must match when both are set, got {self.chat_template_path!r} and "
                    f"{configured_chat_template!r}"
                )
            engine_init_kwargs["chat_template"] = self.chat_template_path

        if self.tokenizer_path is None:
            return

        configured_tokenizer = engine_init_kwargs.get("tokenizer")
        if configured_tokenizer is not None and configured_tokenizer != self.tokenizer_path:
            raise ValueError(
                "tokenizer_path and generator.inference_engine.engine_init_kwargs.tokenizer "
                f"must match when both are set, got {self.tokenizer_path!r} and {configured_tokenizer!r}"
            )
        engine_init_kwargs["tokenizer"] = self.tokenizer_path


class EchoTerminalAgentExp(BasePPOExp):
    def _ensure_chat_template_loaded(self) -> None:
        template_path = getattr(self.cfg, "chat_template_path", None)
        if not template_path or getattr(self, "_echo_chat_template_loaded", False):
            return
        load_chat_template(self.tokenizer, template_path)
        self._echo_chat_template_loaded = True

    def get_tokenizer_path(self) -> str:
        return self.cfg.tokenizer_path or self.cfg.trainer.policy.model.path

    def get_worker_classes(self):
        if self.cfg.trainer.strategy not in ("fsdp", "fsdp2"):
            raise ValueError("ECHO hook implementation currently supports only fsdp/fsdp2, not Megatron.")
        return PolicyWorker, CriticWorker, RefWorker

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        self._ensure_chat_template_loaded()
        return TerminalAgentGenerator(
            generator_cfg=cfg.generator,
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
        return EchoPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_train_dataset(self):
        self._ensure_chat_template_loaded()
        prompts_dataset = TerminalAgentTaskDataset(
            data_files=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            prompt_key=self.cfg.generator.prompt_key,
            path_key=self.cfg.generator.path_key,
            task_binary_key=self.cfg.generator.task_binary_key,
            num_workers=self.cfg.generator.dataset_num_workers,
            max_rows=self.cfg.generator.dataset_max_rows,
            max_rows_per_source=None,
            world_model_only_paths=self.cfg.generator.world_model_only_paths,
        )
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, "
            f"got size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def get_eval_dataset(self):
        self._ensure_chat_template_loaded()
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return TerminalAgentTaskDataset(
                data_files=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                prompt_key=self.cfg.generator.prompt_key,
                path_key=self.cfg.generator.path_key,
                task_binary_key=self.cfg.generator.task_binary_key,
                num_workers=self.cfg.generator.dataset_num_workers,
                max_rows=self.cfg.generator.dataset_max_rows,
                max_rows_per_source=self.cfg.generator.eval_dataset_max_rows_per_source,
                world_model_only_paths=[],
            )
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = EchoTerminalAgentExp(cfg)
    exp.run()


def main() -> None:
    cfg = _load_config(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.generator.step_wise_trajectories:
        raise ValueError("TerminalAgentGenerator requires generator.step_wise_trajectories=false.")
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for terminal-agent training.")
    if cfg.trainer.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("ECHO hook implementation currently supports only fsdp/fsdp2.")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
