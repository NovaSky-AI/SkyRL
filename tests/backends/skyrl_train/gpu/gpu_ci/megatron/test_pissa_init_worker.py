"""Tests for PiSSA decomposition and tensor-parallel initialization."""

from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.workers.megatron import pissa_init_worker
from skyrl.backends.skyrl_train.workers.megatron.pissa_init_worker import (
    PiSSAInitWorker,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
    MODEL_NAME,
    MOE_MODEL_NAME,
    get_test_actor_config,
    get_test_training_batch,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

pytestmark = pytest.mark.megatron


def _principal(weight, rank):
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


def test_pissa_decomposition_uses_principal_components():
    torch.manual_seed(0)
    w = torch.randn(48, 64)
    rank = 4
    scale = 0.5
    linear_in, linear_out, residual = pissa_init_worker.pissa_decompose(w, rank, scale)

    principal = scale * (linear_out @ linear_in)
    torch.testing.assert_close(principal, _principal(w, rank), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(residual + principal, w, atol=1e-6, rtol=1e-6)


def test_rank_too_large_raises():
    w = torch.randn(8, 12)
    with pytest.raises(ValueError):
        pissa_init_worker.pissa_decompose(w, 9, 1.0)  # 9 > min(8, 12)


@pytest.mark.parametrize("group_rank", [0, 1])
def test_distributed_decomposition_uses_rank_zero_factors(monkeypatch, group_rank):
    torch.manual_seed(3)
    weight = torch.randn(12, 8)
    rank = 4
    expected_in, expected_out, _ = pissa_init_worker.pissa_decompose(weight, rank, 1.0)
    factors = iter((expected_in, expected_out))

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group_rank)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, group, group_src: tensor.copy_(next(factors)),
    )
    if group_rank != 0:
        monkeypatch.setattr(
            pissa_init_worker,
            "_pissa_factors",
            lambda *args: pytest.fail("nonzero rank recomputed the SVD"),
        )

    linear_in, linear_out, residual = pissa_init_worker._synchronized_pissa_decompose(weight, rank, 1.0, object())

    torch.testing.assert_close(linear_in, expected_in)
    torch.testing.assert_close(linear_out, expected_out)
    torch.testing.assert_close(residual + linear_out @ linear_in, weight)


@pytest.mark.parametrize(
    ("rank", "lora_type"),
    [
        (0, "lora"),
        (32, "canonical_lora"),
    ],
)
def test_invalid_pissa_config_raises(rank, lora_type):
    with pytest.raises(ValueError):
        pissa_init_worker.validate_pissa_config(rank, lora_type)


@pytest.mark.parametrize("export_raises", [False, True])
def test_residual_export_restores_adapter_weights(monkeypatch, export_raises):
    class ParallelLinearAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_out = torch.nn.Linear(3, 4, bias=False)

    class LoRALinear(torch.nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.adapter = adapter

    class GroupedExpertLinearAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_out = torch.nn.Linear(3, 4, bias=False)

    monkeypatch.setattr(pissa_init_worker, "LoRALinear", LoRALinear)
    monkeypatch.setattr(pissa_init_worker, "ParallelLinearAdapter", ParallelLinearAdapter)
    monkeypatch.setattr(pissa_init_worker, "GroupedExpertLinearAdapter", GroupedExpertLinearAdapter)

    model = torch.nn.Sequential(LoRALinear(ParallelLinearAdapter()), LoRALinear(GroupedExpertLinearAdapter()))
    weights = [module.adapter.linear_out.weight for module in model]
    originals = [weight.detach().clone() for weight in weights]

    if export_raises:
        with pytest.raises(RuntimeError), pissa_init_worker.zeroed_adapters(model):
            for weight in weights:
                torch.testing.assert_close(weight, torch.zeros_like(weight))
            raise RuntimeError("export failed")
    else:
        with pissa_init_worker.zeroed_adapters(model):
            for weight in weights:
                torch.testing.assert_close(weight, torch.zeros_like(weight))

    for weight, original in zip(weights, originals, strict=True):
        torch.testing.assert_close(weight, original)


@pytest.mark.parametrize("input_is_parallel", [False, True], ids=["column_parallel", "row_parallel"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_pissa_initialization_reshards_parallel_adapters(monkeypatch, input_is_parallel, tp_rank):
    torch.manual_seed(6)
    full_weight = torch.randn(12, 8)
    rank = 4
    tp_size = 2
    base_shard_dim = 1 if input_is_parallel else 0
    linear_in_shard_dim = 1 if input_is_parallel else 0
    base_weight = full_weight.chunk(tp_size, dim=base_shard_dim)[tp_rank].clone()
    linear_in_shape = [rank, full_weight.shape[1]]
    linear_in_shape[linear_in_shard_dim] //= tp_size

    base_linear = SimpleNamespace(weight=torch.nn.Parameter(base_weight))
    adapter = SimpleNamespace(
        input_is_parallel=input_is_parallel,
        dim=rank,
        alpha=rank,
        linear_in=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(linear_in_shape))),
        linear_out=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(full_weight.shape[0] // tp_size, rank))),
    )
    monkeypatch.setattr(pissa_init_worker, "_all_gather", lambda *args: full_weight)
    monkeypatch.setattr(
        pissa_init_worker,
        "_synchronized_pissa_decompose",
        lambda weight, rank, scale, group: pissa_init_worker.pissa_decompose(weight, rank, scale),
    )

    pissa_init_worker._init_one_adapter(base_linear, adapter, tp_size, tp_rank, None)

    linear_in, linear_out, residual = pissa_init_worker.pissa_decompose(full_weight, rank, scale=1.0)
    torch.testing.assert_close(base_linear.weight, residual.chunk(tp_size, dim=base_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_in.weight, linear_in.chunk(tp_size, dim=linear_in_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_out.weight, linear_out.chunk(tp_size, dim=0)[tp_rank])


@pytest.mark.parametrize("input_is_parallel", [False, True], ids=["column_parallel", "row_parallel"])
@pytest.mark.parametrize("single_grouped_weight", [False, True], ids=["separate_weights", "grouped_weight"])
@pytest.mark.parametrize("etp_rank", [0, 1])
def test_pissa_initialization_reshards_grouped_experts(monkeypatch, input_is_parallel, single_grouped_weight, etp_rank):
    torch.manual_seed(9)
    full_weights = torch.randn(3, 12, 8)
    rank = 4
    etp_size = 2
    base_shard_dim = 1 if input_is_parallel else 0
    base_shards = [weight.chunk(etp_size, dim=base_shard_dim)[etp_rank].clone() for weight in full_weights]

    if single_grouped_weight:
        base_linear = SimpleNamespace(
            single_grouped_weight=True,
            weight=torch.nn.Parameter(torch.stack(base_shards)),
        )
    else:
        base_linear = SimpleNamespace(single_grouped_weight=False)
        for expert_idx, base_shard in enumerate(base_shards):
            setattr(base_linear, f"weight{expert_idx}", torch.nn.Parameter(base_shard))

    linear_in_shape = [len(full_weights), rank, full_weights.shape[2]]
    linear_in_shape[2 if input_is_parallel else 1] //= etp_size
    adapter = SimpleNamespace(
        expert_tp_group=object(),
        input_is_parallel=input_is_parallel,
        num_local_experts=len(full_weights),
        dim=rank,
        alpha=rank,
        linear_in=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(linear_in_shape))),
        linear_out=SimpleNamespace(
            weight=torch.nn.Parameter(torch.empty(len(full_weights), full_weights.shape[1] // etp_size, rank))
        ),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: etp_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: etp_rank)
    expected_weights = list(full_weights.unbind())
    gathered_weights = iter(expected_weights)
    monkeypatch.setattr(
        pissa_init_worker,
        "_all_gather",
        lambda local, dim, size, group: next(gathered_weights),
    )
    monkeypatch.setattr(
        pissa_init_worker,
        "_synchronized_pissa_decompose",
        lambda weight, rank, scale, group: pissa_init_worker.pissa_decompose(weight, rank, scale),
    )

    pissa_init_worker._init_grouped_adapter(base_linear, adapter)

    for expert_idx, (base_weight, full_weight) in enumerate(
        zip(
            pissa_init_worker._grouped_base_weights(base_linear, len(expected_weights)),
            expected_weights,
            strict=True,
        )
    ):
        linear_in, linear_out, residual = pissa_init_worker.pissa_decompose(full_weight, rank, scale=1.0)
        torch.testing.assert_close(base_weight, residual.chunk(etp_size, dim=base_shard_dim)[etp_rank])
        linear_in_shard_dim = 1 if input_is_parallel else 0
        torch.testing.assert_close(
            adapter.linear_in.weight[expert_idx],
            linear_in.chunk(etp_size, dim=linear_in_shard_dim)[etp_rank],
        )
        torch.testing.assert_close(
            adapter.linear_out.weight[expert_idx],
            linear_out.chunk(etp_size, dim=0)[etp_rank],
        )


@pytest.mark.parametrize(
    ("model_name", "tp", "pp", "ep", "etp", "num_gpus"),
    [
        pytest.param(MODEL_NAME, 2, 2, 1, None, 4, id="dense"),
        pytest.param(MOE_MODEL_NAME, 4, 1, 8, 1, 8, id="grouped_moe"),
    ],
)
def test_pissa_worker_preserves_base_model_forward(ray_init_fixture, model_name, tp, pp, ep, etp, num_gpus):
    batch = get_test_training_batch(max(4, num_gpus))

    def base_cfg():
        cfg = get_test_actor_config(model_name=model_name)
        cfg.trainer.strategy = "megatron"
        cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
        cfg.trainer.bf16 = False
        if ep > 1:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs["num_layers"] = 1
        return cfg

    def megatron_forward(cfg, worker_cls=None):
        if worker_cls is None:
            actor_group = init_worker_with_type(
                "policy",
                shared_pg=None,
                colocate_all=False,
                num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
                cfg=cfg,
            )
        else:
            actor_group = PPORayActorGroup(
                cfg.trainer,
                num_nodes=1,
                num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
                ray_actor_type=worker_cls,
                num_gpus_per_actor=0.75,
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path))
        all_rank = ray.get(actor_group.async_run_ray_method("mesh", "forward", data=batch))
        output = WorkerOutput.cat(actor_group.actor_infos, all_rank)
        return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")

    pissa_cfg = base_cfg()
    pissa_cfg.trainer.policy.model.lora.rank = 32
    pissa_cfg.trainer.policy.model.lora.alpha = 32
    pissa_cfg.trainer.policy.model.lora.share_expert_adapters = False
    logprobs_pissa = megatron_forward(pissa_cfg, worker_cls=PiSSAInitWorker)

    ray.shutdown()
    ray_init_for_tests()

    logprobs_base = megatron_forward(base_cfg())
    torch.testing.assert_close(logprobs_pissa, logprobs_base, rtol=1e-4, atol=1e-4)
