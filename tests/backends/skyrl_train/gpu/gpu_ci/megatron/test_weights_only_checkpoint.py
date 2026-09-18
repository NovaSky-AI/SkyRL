"""Real Megatron checkpoint roundtrip followed by optimizer master copy-back."""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.train.config import SkyRLLoraConfig
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
    get_test_actor_config,
    get_test_training_batch,
)
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type


@pytest.mark.megatron
@pytest.mark.parametrize("lora", [False, True])
def test_weights_only_load_survives_first_optimizer_step(ray_init_fixture, tmp_path, lora):
    cfg = get_test_actor_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.optimizer_config.lr = 1e-3
    cfg.trainer.train_batch_size = 8
    cfg.trainer.policy_mini_batch_size = 8
    cfg.generator.n_samples_per_prompt = 1
    if lora:
        cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)
    group = init_worker_with_type("policy", num_gpus_per_node=2, cfg=cfg)
    batch = get_test_training_batch(8)

    def train_step():
        ray.get(group.async_run_ray_method("mesh", "forward_backward", batch, loss_fn="cross_entropy"))
        ray.get(group.async_run_ray_method("pass_through", "optim_step"))

    def logprobs():
        outputs = ray.get(group.async_run_ray_method("mesh", "forward", batch))
        combined = WorkerOutput.cat(group.actor_infos, outputs)
        return loss_fn_outputs_to_tensor(combined.loss_fn_outputs, key="logprobs")

    train_step()
    expected = logprobs()
    checkpoint = str(tmp_path / "checkpoint")
    ray.get(group.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint))
    ray.get(group.async_run_ray_method("pass_through", "finalize_pending_saves"))

    train_step()  # Leave live masters at different values from the checkpoint.
    assert not torch.allclose(logprobs(), expected, rtol=1e-5, atol=1e-5)
    ray.get(
        group.async_run_ray_method(
            "pass_through",
            "load_checkpoint",
            ckpt_dir=checkpoint,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
    )
    torch.testing.assert_close(logprobs(), expected, rtol=1e-5, atol=1e-5)

    # Zero LR isolates the optimizer's copy-back from its Adam update: stale
    # masters still overwrite the checkpoint even when learning_rate is zero.
    ray.get(group.async_run_ray_method("pass_through", "set_lr", learning_rate=0.0))
    train_step()
    torch.testing.assert_close(logprobs(), expected, rtol=1e-5, atol=1e-5)
