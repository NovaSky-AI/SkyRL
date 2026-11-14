"""
Main entrypoint for fully async Harbor training with ThunderAgent routing.
"""

import os
import subprocess
import sys
from pathlib import Path

import ray
import yaml

from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import prepare_runtime_environment

from .main_thunder_agent import FullyAsyncThunderAgentExp
from .training_config import ThunderAgentHarborConfig

HARBOR_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "harbor_trial" / "default.yaml"
REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _default_socket_ifname() -> str:
    target_ip = os.environ.get("RAY_HEAD_IP") or os.environ.get("ROLLOUT_HOST_IP")
    if target_ip:
        try:
            result = subprocess.run(
                ["ip", "route", "get", target_ip],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            result = None
        if result:
            parts = result.stdout.split()
            if "dev" in parts:
                idx = parts.index("dev") + 1
                if idx < len(parts):
                    return parts[idx]
    return "eth0"


class HarborThunderAgentFullyAsyncExp(FullyAsyncThunderAgentExp):
    """Harbor fully-async experiment that routes inference through ThunderAgent."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from .skyrl_integration.generator import ThunderAgentHarborGenerator

        return ThunderAgentHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    @staticmethod
    def _build_harbor_task_dataset(data_files, max_tasks=None):
        from examples.train_integrations.harbor.dataset import HarborTaskDataset

        dataset = HarborTaskDataset(data_files=data_files)
        if len(dataset) == 0:
            raise ValueError(
                f"HarborTaskDataset resolved zero task directories from data_files={data_files!r}. "
                "Each entry must be a task directory or a directory of task subdirectories "
                "containing instruction.md."
            )
        if max_tasks is not None and max_tasks < len(dataset.task_paths):
            dataset.task_paths = dataset.task_paths[:max_tasks]
        return dataset

    def get_train_dataset(self):
        prompts_dataset = self._build_harbor_task_dataset(
            data_files=self.cfg.data.train_data,
            max_tasks=self.cfg.max_train_tasks,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return self._build_harbor_task_dataset(
                data_files=self.cfg.data.val_data,
                max_tasks=self.cfg.max_eval_tasks,
            )
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    from .skyrl_integration.runtime_setup import patch_mini_swe_agent_environment

    patch_mini_swe_agent_environment()
    exp = HarborThunderAgentFullyAsyncExp(cfg)
    exp.run()


def _recipe_runtime_env(cfg) -> dict[str, str]:
    env_vars = prepare_runtime_environment(cfg)
    socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME") or _default_socket_ifname()
    env_vars.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
    env_vars.setdefault("GLOO_SOCKET_IFNAME", os.environ.get("GLOO_SOCKET_IFNAME", socket_ifname))

    for name in (
        "SKYRL_WORKER_NCCL_TIMEOUT_IN_S",
        "RAY_HEAD_IP",
        "ROLLOUT_HOST_IP",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TRITON_CACHE_DIR",
        "SKYRL_INFERENCE_ROUTER_PORT",
        "THUNDER_AGENT_ROUTER_PORT",
        "HARBOR_SHARED_UV_CACHE_ENV_DIR",
        "HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME",
        "HARBOR_MINI_SWE_AGENT_PACKAGE",
    ):
        if name in os.environ:
            env_vars[name] = os.environ[name]

    pythonpath = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = REPO_ROOT if not pythonpath else f"{REPO_ROOT}:{pythonpath}"
    return env_vars


def _initialize_ray_for_recipe(cfg) -> None:
    os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "fatal")
    ray.init(runtime_env={"env_vars": _recipe_runtime_env(cfg)}, log_to_driver=True)
    sync_registries()


def main() -> None:
    cfg = ThunderAgentHarborConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    if HARBOR_DEFAULT_CONFIG.exists():
        with open(HARBOR_DEFAULT_CONFIG) as f:
            defaults = yaml.safe_load(f)
        cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    _initialize_ray_for_recipe(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
