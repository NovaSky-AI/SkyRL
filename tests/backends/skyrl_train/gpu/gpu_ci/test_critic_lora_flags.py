"""The critic's LoRA honours train_attn / train_mlp like the policy's.

Builds an FSDP critic with a narrowed adapter, checks which modules carry LoRA
weights, and runs one forward_backward + optim_step on it. The expected module
sets are written out rather than derived from the implementation's classifier.

Run:
  uv run --isolated --extra dev --extra fsdp pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_critic_lora_flags.py
"""

import pytest
import ray

from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPCriticWorkerBase
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
)

MODEL_NAME = "trl-internal-testing/tiny-Qwen3ForCausalLM"
ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LEAVES = {"gate_proj", "up_proj", "down_proj"}


class _CriticWorkerBase(FSDPCriticWorkerBase):
    def lora_leaf_names(self) -> list[str]:
        """Leaf names of the modules that carry a LoRA adapter."""
        leaves = set()
        for name, _ in self.model.named_parameters():
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part.startswith("lora_") and i > 0:
                    leaves.add(parts[i - 1])
                    break
        return sorted(leaves)


_CriticWorker = ray.remote(num_gpus=1)(_CriticWorkerBase)


def _cfg(train_attn: bool, train_mlp: bool) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.critic_num_gpus_per_node = 1
    cfg.generator.inference_engine.tensor_parallel_size = 1
    lora = cfg.trainer.critic.model.lora
    lora.rank = 8
    lora.alpha = 16
    lora.train_attn = train_attn
    lora.train_mlp = train_mlp
    return cfg


@pytest.mark.parametrize(
    ("train_attn", "train_mlp", "expected"),
    [(True, False, ATTN_LEAVES | {"value_head"}), (False, True, MLP_LEAVES | {"value_head"})],
    ids=["attn_only", "mlp_only"],
)
def test_narrowed_critic_adapts_only_its_group_and_trains(ray_init_fixture, train_attn, train_mlp, expected):
    cfg = _cfg(train_attn, train_mlp)
    validate_cfg(cfg)
    critic = init_worker_with_type("critic", num_gpus_per_node=1, cfg=cfg, worker_cls=_CriticWorker)

    leaves = set(ray.get(critic.async_run_ray_method("pass_through", "lora_leaf_names"))[0])
    assert leaves == expected, f"critic adapted {sorted(leaves)}, expected {sorted(expected)}"

    dp_size = critic.actor_infos[0].rank.dp_size
    results = ray.get(
        critic.async_run_ray_method("mesh", "forward_backward", data=make_dummy_training_batch(batch_size=dp_size))
    )
    ray.get(critic.async_run_ray_method("pass_through", "optim_step"))
    assert all("critic_loss" in r.metrics for r in results)
