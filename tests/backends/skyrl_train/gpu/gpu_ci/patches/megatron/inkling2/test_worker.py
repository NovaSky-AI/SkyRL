"""Inkling2 through SkyRL's Megatron policy worker on 2 GPUs (expert and / or tensor parallelism).

The synthetic checkpoint (``synthetic_checkpoint.py``) is loaded by the worker the way a training
run loads the release. ``forward`` logprobs on left-padded sequences must match the reference
model run on the unpadded sequences, packed (``remove_microbatch_padding``) and unpacked, and a
``forward_backward`` + ``optim_step`` (default full activation recompute) must produce finite
metrics. Tensor parallelism runs with sequence parallelism (SkyRL enables it with TP), which
exercises the short convolutions across sequence-slice boundaries.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/inkling2/test_worker.py
"""

import math
import tempfile

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.patches.megatron.inkling2.synthetic_checkpoint import (
    UNPADDED_V,
    write_checkpoint,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

pytestmark = pytest.mark.megatron

NUM_GPUS = 2
NUM_ACTIONS = 15


def _batch(batch_size: int = 4) -> TrainingInputBatch:
    """Random in-vocabulary sequences with left padding and right-padded responses (pad id 0)."""
    g = torch.Generator().manual_seed(0)
    lengths = [37, 35, 41, 33]
    pad_before = [4, 0, 1, 6]
    max_len = max(n + p for n, p in zip(lengths, pad_before)) + 3
    sequences, masks, loss_masks = [], [], []
    for n, p in zip(lengths, pad_before):
        after = max_len - n - p
        sequences.append(
            torch.cat(
                [
                    torch.zeros(p, dtype=torch.long),
                    torch.randint(1, UNPADDED_V, (n,), generator=g),
                    torch.zeros(after, dtype=torch.long),
                ]
            )
        )
        masks.append(torch.cat([torch.zeros(p), torch.ones(n), torch.zeros(after)]).long())
        loss_masks.append(torch.cat([torch.ones(NUM_ACTIONS - after), torch.zeros(after)]))
    loss_mask = torch.stack(loss_masks)
    data = TrainingInputBatch(
        {
            "sequences": torch.stack(sequences),
            "attention_mask": torch.stack(masks),
            "action_log_probs": torch.full((batch_size, NUM_ACTIONS), -2.0),
            "base_action_log_probs": torch.full((batch_size, NUM_ACTIONS), -2.0),
            "rollout_logprobs": torch.full((batch_size, NUM_ACTIONS), -2.0),
            "values": torch.zeros(batch_size, NUM_ACTIONS),
            "returns": torch.zeros(batch_size, NUM_ACTIONS),
            "advantages": torch.full((batch_size, NUM_ACTIONS), 0.5),
            "loss_mask": loss_mask,
            "response_mask": loss_mask,
        }
    )
    data.metadata = {"response_length": NUM_ACTIONS}
    return data


@ray.remote(num_gpus=1)
def _reference_action_logprobs(path: str, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, NUM_ACTIONS]`` logprobs of the response tokens (0 where padded)."""
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.reference import (
        load_reference,
    )

    ref = load_reference(path, [torch.device("cuda", 0)])
    out = torch.zeros(sequences.shape[0], NUM_ACTIONS)
    seq_len = sequences.shape[1]
    for row in range(sequences.shape[0]):
        valid = attention_mask[row].bool()
        positions = valid.nonzero().squeeze(-1)
        lp = ref.token_logprobs(sequences[row][valid], lm_head_dtype=torch.bfloat16)["logprobs"]
        for col, pos in enumerate(range(seq_len - NUM_ACTIONS, seq_len)):
            if valid[pos]:
                k = (positions == pos).nonzero().item()
                out[row, col] = lp[k - 1]
    return out


def _config(path: str, tp: int, ep: int, remove_microbatch_padding: bool) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = path
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.remove_microbatch_padding = remove_microbatch_padding
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.algorithm.use_kl_loss = False
    validate_cfg(cfg)
    return cfg


@pytest.mark.parametrize(
    "tp, ep, remove_microbatch_padding",
    [(1, 2, True), (1, 2, False), (2, 1, True), (2, 2, False)],
    ids=["ep2_packed", "ep2_unpacked", "tp2_packed", "tp2_ep2_unpacked"],
)
def test_worker_forward_and_train_step(ray_init_fixture, tp, ep, remove_microbatch_padding):
    with tempfile.TemporaryDirectory() as path:
        write_checkpoint(path, tokenizer="Qwen/Qwen3-0.6B")
        cfg = _config(path, tp, ep, remove_microbatch_padding)
        batch = _batch()
        actor_group = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=NUM_GPUS, cfg=cfg
        )
        outputs = ray.get(actor_group.async_run_ray_method("mesh", "forward", data=batch))
        megatron = loss_fn_outputs_to_tensor(
            WorkerOutput.cat(actor_group.actor_infos, outputs).loss_fn_outputs, key="logprobs"
        )

        batch.metadata["global_step"] = 0
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        ray.shutdown()
        ray_init_for_tests()

        reference = ray.get(_reference_action_logprobs.remote(path, batch["sequences"], batch["attention_mask"]))

    mask = batch["attention_mask"][:, -NUM_ACTIONS:].bool()
    diff = (megatron[mask] - reference[mask]).abs()
    print(f"tp={tp} ep={ep} packed={remove_microbatch_padding}: |dlp| mean {diff.mean():.5f} max {diff.max():.5f}")
    assert diff.mean() < 0.006
    assert diff.max() < 0.03
    for result in results:
        assert math.isfinite(result.metrics["policy_loss"])
        assert result.metrics["policy_entropy"] > 0
