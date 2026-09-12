"""Compare asynchronous and synchronous DDP across changing optimizer windows.

Requires four GPUs: two independent DP=2 policy groups, with real Megatron
DDP buffers, distributed optimizers, and SkyRL forward/backward schedules.
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
    get_test_actor_config,
    get_test_training_batch,
)
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type


@pytest.mark.megatron
def test_overlap_matches_synchronous_accumulation(ray_init_fixture):
    groups = []
    for overlap in (False, True):
        cfg = get_test_actor_config()
        cfg.trainer.strategy = "megatron"
        cfg.trainer.placement.colocate_all = False
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.trainer.policy.megatron_config.ddp_config.overlap_grad_reduce = overlap
        cfg.trainer.train_batch_size = 16
        cfg.trainer.policy_mini_batch_size = 16
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.generator.n_samples_per_prompt = 1
        groups.append(init_worker_with_type("policy", num_gpus_per_node=2, cfg=cfg))

    # Two microbatches per rank initially, then fewer, then more. The final
    # window also spans two forward_backward requests before one optimizer step.
    for step, request_sizes in enumerate(((8,), (4,), (12,), (4, 8))):
        for batch_size in request_sizes:
            batch = get_test_training_batch(batch_size)
            batch.metadata["global_step"] = step
            for group in groups:
                ray.get(group.async_run_ray_method("mesh", "forward_backward", batch, loss_fn="cross_entropy"))
        norms = [ray.get(group.async_run_ray_method("pass_through", "optim_step")) for group in groups]
        assert all(norm is not None and norm > 0 for rank_norms in norms for norm in rank_norms)
        torch.testing.assert_close(torch.tensor(norms[0]), torch.tensor(norms[1]), rtol=1e-3, atol=1e-5)

    batch = get_test_training_batch(4)
    logprobs = []
    for group in groups:
        outputs = ray.get(group.async_run_ray_method("mesh", "forward", batch))
        combined = WorkerOutput.cat(group.actor_infos, outputs)
        logprobs.append(loss_fn_outputs_to_tensor(combined.loss_fn_outputs, key="logprobs"))
    torch.testing.assert_close(logprobs[0], logprobs[1], rtol=1e-3, atol=1e-3)
