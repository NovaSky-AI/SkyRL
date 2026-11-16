"""Megatron trains bit-identically with and without microbatch prefetching, under pipeline parallelism.

Microbatches are built lazily either way; prefetching builds and uploads microbatch i + 1 while
microbatch i computes. Token-based batching makes them uneven (row padding), and virtual pipeline
stages make every chunk read each microbatch.

uv run --isolated --extra dev --extra megatron -- pytest tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_microbatch_prefetch_megatron.py
"""

import pytest
import ray

from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"


def _set_prefetch(worker, prefetch: bool) -> None:
    worker.model.prefetch_microbatches = prefetch


def _step(prefetch: bool, max_tokens_per_microbatch: int, vpp_size):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.policy.optimizer_config.lr = 0.0
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.max_tokens_per_microbatch = max_tokens_per_microbatch
    if vpp_size is not None:
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["virtual_pipeline_model_parallel_size"] = vpp_size
    validate_cfg(cfg)
    group = init_worker_with_type("policy", num_gpus_per_node=4, cfg=cfg)
    try:
        dispatch = WorkerDispatch(cfg, policy_actor_group=group)
        ray.get([info.handle.__ray_call__.remote(_set_prefetch, prefetch) for info in group.actor_infos])
        batch = make_dummy_training_batch(batch_size=8, seq_len=64, num_actions=16)
        # Left padding of varied width, so token-based microbatches hold different row counts.
        for row, width in enumerate([0, 20, 8, 30, 0, 12, 40, 4]):
            batch["attention_mask"][row, :width] = 0
        forward = dispatch.forward("policy", batch)
        status = dispatch.forward_backward("policy", batch)
        return forward.loss_fn_outputs, status.metrics, dispatch.optim_step("policy")
    finally:
        for info in group.actor_infos:
            ray.kill(info.handle)


@pytest.mark.megatron
@pytest.mark.parametrize(
    "max_tokens_per_microbatch,vpp_size",
    [(-1, None), (96, None), (96, 2)],
    ids=["sample_batching", "token_batching", "token_batching_vpp2"],
)
def test_prefetching_changes_no_output(ray_init_fixture, max_tokens_per_microbatch, vpp_size):
    with_prefetch = _step(True, max_tokens_per_microbatch, vpp_size)
    without_prefetch = _step(False, max_tokens_per_microbatch, vpp_size)

    assert with_prefetch[0] == without_prefetch[0]
    for key in ("policy_loss", "policy_entropy", "final_loss"):
        assert with_prefetch[1][key] == without_prefetch[1][key], key
    assert with_prefetch[2] == without_prefetch[2]
