"""GPU policy worker loss-scaling regression tests."""

from __future__ import annotations

import math
from typing import Callable, Literal

import pytest
import ray
import torch
from ray.util.placement_group import placement_group

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import ResolvedPlacementGroup, validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    import_worker,
    make_dummy_training_batch,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
MICRO_BATCH_SIZES = (1, 2)
LOSS_FNS = ("cross_entropy", "dual_clip")
POLICY_CALLS: tuple[Literal["forward", "forward_backward"], ...] = ("forward", "forward_backward")
POLICY_WORKER_TYPES = [
    "fsdp",
    pytest.param("megatron", marks=pytest.mark.megatron),
]
MICROBATCH_INVARIANCE_REL_TOL = 1e-3
MICROBATCH_INVARIANCE_ABS_TOL = 1e-2


def _make_policy_cfg(
    policy_worker_type: str,
    loss_fn: str,
    *,
    micro_train_batch_size_per_gpu: int = 1,
    micro_forward_batch_size_per_gpu: int = 1,
) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = policy_worker_type
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.logger = "console"
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
    cfg.trainer.micro_forward_batch_size_per_gpu = micro_forward_batch_size_per_gpu
    cfg.trainer.algorithm.policy_loss_type = loss_fn
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.temperature = 1.0
    cfg.generator.sampling_params.temperature = 1.0

    if policy_worker_type == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1

    validate_cfg(cfg)
    return cfg


def _make_loss_batch(loss_fn: str, *, batch_size: int = 4, seq_len: int = 10, num_actions: int = 4):
    batch = make_dummy_training_batch(batch_size=batch_size, seq_len=seq_len, num_actions=num_actions)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    batch["sequences"] = torch.randint(10, 100, (batch_size, seq_len), generator=generator, dtype=torch.long)
    batch["attention_mask"] = torch.ones((batch_size, seq_len), dtype=torch.long)
    batch["action_log_probs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["base_action_log_probs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["values"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["returns"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch["response_mask"] = torch.ones((batch_size, num_actions), dtype=torch.float32)
    batch["rollout_logprobs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)

    if loss_fn == "cross_entropy":
        batch["advantages"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
        # Loss mask scaling replicates logic on `SFTTrainer` -> micro batch agnostic scaling
        batch["loss_mask"] = torch.ones((batch_size, num_actions), dtype=torch.float32) / (batch_size * num_actions)
    elif loss_fn == "dual_clip":
        batch["advantages"] = -torch.ones((batch_size, num_actions), dtype=torch.float32)
        batch["loss_mask"] = torch.ones((batch_size, num_actions), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported loss_fn {loss_fn}")

    batch.metadata = {"response_length": num_actions}
    return batch


def _init_policy_actor_group(cfg: SkyRLTrainConfig):
    num_gpus_per_node = cfg.trainer.placement.policy_num_gpus_per_node
    raw_pg = placement_group([{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
    pg = ResolvedPlacementGroup(raw_pg)
    worker_cls = import_worker(cfg.trainer.strategy, "policy")
    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=1.0,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path))
    return actor_group, raw_pg


def _cleanup_policy_actor_group(actor_group, raw_pg):
    for actor_info in actor_group.actor_infos:
        ray.kill(actor_info.handle, no_restart=True)
    ray.util.remove_placement_group(raw_pg)


def _run_policy_worker_calls(
    policy_worker_type: str,
    loss_fn: str,
    batch_factory: Callable[[], TrainingInputBatch],
    *,
    micro_batch_size: int,
) -> dict[str, WorkerOutput]:
    cfg = _make_policy_cfg(
        policy_worker_type,
        loss_fn,
        micro_train_batch_size_per_gpu=micro_batch_size,
        micro_forward_batch_size_per_gpu=micro_batch_size,
    )
    actor_group = None
    raw_pg = None
    try:
        actor_group, raw_pg = _init_policy_actor_group(cfg)
        dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
        outputs = {}
        for call in POLICY_CALLS:
            batch = batch_factory()
            if call == "forward":
                outputs[call] = dispatch.forward("policy", batch, loss_fn=loss_fn)
            elif call == "forward_backward":
                outputs[call] = dispatch.forward_backward("policy", batch, loss_fn=loss_fn)
            else:
                raise ValueError(f"Unsupported call {call}")
        return outputs
    finally:
        if actor_group is not None and raw_pg is not None:
            _cleanup_policy_actor_group(actor_group, raw_pg)


def _primary_loss(output: WorkerOutput, loss_fn: str) -> float:
    metrics = output.metrics
    if loss_fn == "cross_entropy":
        assert "loss" in metrics
        return metrics["loss"]

    if "policy_loss" in metrics:
        if "final_loss" in metrics:
            assert math.isclose(metrics["final_loss"], metrics["policy_loss"], rel_tol=2e-4, abs_tol=2e-3)
        return metrics["policy_loss"]

    # FSDP forward(loss_fn="dual_clip") goes through _forward_micro_with_loss
    # and reports the selected policy loss under the generic "loss" key.
    assert "loss" in metrics
    return metrics["loss"]


def _assert_loss_output_shape(output: WorkerOutput, loss_fn: str, batch_size: int = 4, num_actions: int = 4):
    assert math.isfinite(_primary_loss(output, loss_fn))
    assert len(output.loss_fn_outputs) == batch_size
    for item in output.loss_fn_outputs:
        assert "logprobs" in item
        assert len(item["logprobs"]) == num_actions
        if loss_fn == "cross_entropy":
            assert "elementwise_loss" in item
            assert len(item["elementwise_loss"]) == num_actions


@pytest.mark.parametrize("policy_worker_type", POLICY_WORKER_TYPES, ids=["fsdp", "megatron"])
@pytest.mark.parametrize("loss_fn", LOSS_FNS)
def test_policy_loss_is_microbatch_size_invariant(ray_init_fixture, policy_worker_type, loss_fn):
    losses_by_call = {call: {} for call in POLICY_CALLS}
    for micro_batch_size in MICRO_BATCH_SIZES:
        outputs_by_call = _run_policy_worker_calls(
            policy_worker_type,
            loss_fn,
            lambda: _make_loss_batch(loss_fn),
            micro_batch_size=micro_batch_size,
        )
        for call, output in outputs_by_call.items():
            _assert_loss_output_shape(output, loss_fn)
            losses_by_call[call][micro_batch_size] = _primary_loss(output, loss_fn)

    for call in POLICY_CALLS:
        assert losses_by_call[call][1] == pytest.approx(
            losses_by_call[call][2],
            rel=MICROBATCH_INVARIANCE_REL_TOL,
            abs=MICROBATCH_INVARIANCE_ABS_TOL,
        )


@pytest.mark.megatron
@pytest.mark.parametrize("loss_fn", LOSS_FNS)
@pytest.mark.parametrize("micro_batch_size", MICRO_BATCH_SIZES)
def test_policy_loss_fsdp_megatron_consistent(ray_init_fixture, micro_batch_size, loss_fn):
    losses_by_call = {call: {} for call in POLICY_CALLS}
    for policy_worker_type in ("fsdp", "megatron"):
        outputs_by_call = _run_policy_worker_calls(
            policy_worker_type,
            loss_fn,
            lambda: _make_loss_batch(loss_fn),
            micro_batch_size=micro_batch_size,
        )
        for call, output in outputs_by_call.items():
            _assert_loss_output_shape(output, loss_fn)
            losses_by_call[call][policy_worker_type] = _primary_loss(output, loss_fn)

    for call in POLICY_CALLS:
        assert losses_by_call[call]["fsdp"] == pytest.approx(
            losses_by_call[call]["megatron"],
            rel=2e-2,
            abs=2e-1,
        )
