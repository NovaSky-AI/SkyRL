"""GPU integration tests for FSDP MoE Rollout Routing Replay (R3).

Run with:
uv run --isolated --extra dev --extra fsdp -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_fsdp_router_replay.py

Mirrors tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay.py's
``test_forward_backward`` but for the FSDP backend: dummy rollout expert indices
(no vLLM engine needed) drive the routing-replay hooks through the full worker
forward/backward path. A tiny on-disk OlMoE keeps it runnable on a single node.
"""

import math
from pathlib import Path

import pytest
import ray
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
)

TINY_NUM_LAYERS = 4
TINY_NUM_EXPERTS = 8
TINY_TOPK = 2


def _save_tiny_olmoe(path: str) -> None:
    config = OlmoeConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_experts=TINY_NUM_EXPERTS,
        num_experts_per_tok=TINY_TOPK,
        vocab_size=256,
        max_position_embeddings=128,
        norm_topk_prob=False,
    )
    OlmoeForCausalLM(config).save_pretrained(path)


def _get_test_actor_config(model_path: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine.tensor_parallel_size = 2
    # R3 requires the vLLM mp backend and both capture + replay flags.
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    cfg.generator.inference_engine.enable_return_routed_experts = True
    cfg.trainer.policy.fsdp_config.moe_enable_routing_replay = True
    return cfg


@pytest.mark.parametrize(
    ("packed", "sequence_parallel_size"),
    [
        pytest.param(True, 1, id="packed"),
        pytest.param(False, 1, id="unpacked"),
        pytest.param(True, 2, id="packed_sp2"),
    ],
)
def test_fsdp_router_replay_forward_backward(
    ray_init_fixture: None, tmp_path: Path, packed: bool, sequence_parallel_size: int
) -> None:
    """forward_backward + optim_step run cleanly with routing replay enabled, and
    the loss is finite and non-zero (the dummy advantages are non-zero). The
    packed_sp2 case additionally covers the Ulysses SP slicing of replay indices."""
    model_path = str(tmp_path / "tiny_olmoe")
    _save_tiny_olmoe(model_path)

    cfg = _get_test_actor_config(model_path)
    cfg.trainer.remove_microbatch_padding = packed
    cfg.trainer.flash_attn = packed  # flash-attn varlen is required for sample packing
    cfg.trainer.policy.sequence_parallel_size = sequence_parallel_size
    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        dp_size = actor_group.actor_infos[0].rank.dp_size
        batch = make_dummy_training_batch(batch_size=dp_size, seq_len=10, num_actions=4)
        bsz, seq_len = batch["sequences"].shape
        rollout_expert_indices = torch.randint(
            0, TINY_NUM_EXPERTS, (bsz, seq_len, TINY_NUM_LAYERS, TINY_TOPK), dtype=torch.int16
        )
        rollout_expert_indices[batch["attention_mask"] == 0] = 0
        batch["rollout_expert_indices"] = rollout_expert_indices

        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))

        for result in results:
            loss = result.metrics["policy_loss"]
            assert not math.isnan(loss), "policy_loss should not be NaN with routing replay"
            assert loss != 0.0, "policy_loss should be non-zero given non-zero advantages"
    finally:
        ray.shutdown()
