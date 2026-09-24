"""GPU coverage for bounded sampler-support replay.

Run with::

    uv run --isolated --extra dev --extra megatron pytest -s -vvv \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_sample_support_replay.py
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import SAMPLE_SUPPORT_FIELD
from skyrl.backends.skyrl_train.utils.sample_support_replay import sample_support_scores
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as megatron_worker_module,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "Qwen/Qwen3-0.6B"
SUPPORT_WIDTH = 2


def _sample_support_cfg(tensor_parallel_size: int) -> SkyRLTrainConfig:
    """Build a minimal Megatron config for a forward-only support replay test."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_gpus_per_node = tensor_parallel_size
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tensor_parallel_size
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.generator.inference_engine.enable_return_sample_support_set = True
    cfg.generator.sampling_params.top_k = SUPPORT_WIDTH
    cfg.generator.use_conversation_multi_turn = True
    cfg.trainer.algorithm.enable_sample_support_replay = True
    validate_cfg(cfg)
    return cfg


def _synthetic_support(
    responses: list[list[int]],
    vocab_size: int,
) -> PackedTensor:
    """Build two-token support rows that cross a TP=2 vocabulary partition."""
    segments = []
    for response in responses:
        selected = torch.tensor(response, dtype=torch.long)
        other = (selected + vocab_size // 2) % vocab_size
        other = torch.where(other == selected, (other + 1) % vocab_size, other)
        segments.append(torch.stack((selected, other), dim=-1).to(torch.int32))
    return PackedTensor.from_segments(segments)


def _training_inputs() -> tuple[TrainingInputBatch, TrainingInputBatch]:
    """Make matching full-vocabulary and support-conditioned batches from inline prompts."""
    tokenizer = get_tokenizer(MODEL_NAME)
    prompts = [
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "Name the capital of France."}],
    ]
    prompt_ids = [
        tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=False)
        for prompt in prompts
    ]
    responses = [
        tokenizer.encode(" The answer is 4.", add_special_tokens=False),
        tokenizer.encode(" Let us calculate it carefully.", add_special_tokens=False),
    ]
    for response in responses:
        if tokenizer.eos_token_id is not None and response[-1] != tokenizer.eos_token_id:
            response.append(tokenizer.eos_token_id)

    rewards = [[0.0] * len(response) for response in responses]
    loss_masks = [[1] * len(response) for response in responses]
    sequences, attention_mask, response_mask, _, loss_mask, _, _, _, _ = convert_prompts_responses_to_batch_tensors(
        pad_token_id=tokenizer.pad_token_id,
        prompts=prompt_ids,
        responses=responses,
        rewards=rewards,
        loss_masks=loss_masks,
    )
    fields = {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "loss_mask": loss_mask,
    }
    support = _synthetic_support(responses, tokenizer.vocab_size)
    full_vocabulary = TrainingInputBatch(fields)
    support_conditioned = TrainingInputBatch({**fields, SAMPLE_SUPPORT_FIELD: support})
    for batch in (full_vocabulary, support_conditioned):
        batch.metadata = {"response_length": response_mask.shape[1]}
    return full_vocabulary, support_conditioned


def _forward_logprobs(actor_group, batch) -> torch.Tensor:
    """Run the public policy-forward RPC and collect its action logprobs."""
    results = ray.get(actor_group.async_run_ray_method("mesh", "forward", batch))
    output = WorkerOutput.cat(actor_group.actor_infos, results)
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")


@pytest.mark.megatron
def test_sample_support_replay_changes_megatron_logprobs(ray_init_fixture):
    """A bounded support changes the same model's forward-only action logprobs."""
    cfg = _sample_support_cfg(tensor_parallel_size=1)
    full_vocabulary, support_conditioned = _training_inputs()
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=1,
        cfg=cfg,
    )

    bounded_logprobs = _forward_logprobs(actor_group, support_conditioned)
    ray.get(
        actor_group.async_run_ray_method("pass_through", "set_algorithm_config", enable_sample_support_replay=False)
    )
    full_logprobs = _forward_logprobs(actor_group, full_vocabulary)

    assert torch.isfinite(bounded_logprobs).all()
    assert torch.isfinite(full_logprobs).all()
    assert bounded_logprobs.shape == full_logprobs.shape
    # The support contains the target plus one competing token, so its denominator is a strict
    # subset of the full vocabulary's denominator.
    assert torch.all(bounded_logprobs >= full_logprobs - 1e-5)
    assert (bounded_logprobs - full_logprobs).abs().max().item() > 1e-3


class _SampleSupportTPProbeWorker(MegatronPolicyWorkerBase):
    """Exposes a distributed support-normalization probe without production instrumentation."""

    def probe_tp2_matches_full_vocabulary_reference(self) -> dict:
        """Compare TP=2 support scoring with the equivalent unsharded calculation."""
        import megatron.core.parallel_state as mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size != 2:
            raise ValueError(f"expected TP=2, got TP={tp_size}")

        # [B, S] = [2, 3] supplies six distinct positions while staying small.  Eight IDs per
        # rank is the smallest convenient two-way vocabulary split with unambiguous ownership.
        local_vocab_size = 8
        total_vocab_size = local_vocab_size * tp_size
        device = torch.device("cuda", torch.cuda.current_device())
        local_vocab_ids = torch.arange(
            tp_rank * local_vocab_size,
            (tp_rank + 1) * local_vocab_size,
            dtype=torch.float32,
            device=device,
        )
        row_offsets = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3).unsqueeze(-1)
        local_logits = row_offsets * 0.01 + local_vocab_ids.reshape(1, 1, -1) * 0.1
        # Every selected ID alternates ownership between the two shards (total vocab size of 16);
        sampled_ids = torch.tensor([[1, 10, 2], [11, 3, 12]], dtype=torch.long, device=device)
        # synthetic support IDs - include chosen IDs as well as one additional ID - across the rank boundary
        support_ids = torch.stack(
            (sampled_ids, (sampled_ids + total_vocab_size // 2) % total_vocab_size),
            dim=-1,
        ).to(torch.int32)

        sharded = sample_support_scores(
            local_logits,
            sampled_ids,
            support_ids,
            vocab_start_index=tp_rank * local_vocab_size,
            vocab_end_index=(tp_rank + 1) * local_vocab_size,
            tp_group=tp_group,
        ).logprobs

        shards = [torch.empty_like(local_logits) for _ in range(tp_size)]
        # Get unsharded logits and compare
        torch.distributed.all_gather(shards, local_logits, group=tp_group)
        full_logits = torch.cat(shards, dim=-1)
        unsharded = sample_support_scores(
            full_logits,
            sampled_ids,
            support_ids,
            vocab_start_index=0,
            vocab_end_index=total_vocab_size,
            tp_group=None,
        ).logprobs

        torch.testing.assert_close(sharded, unsharded, rtol=0.0, atol=1e-6)
        return {
            "rank": tp_rank,
            "max_abs_diff": (sharded - unsharded).abs().max().item(),
            "support_spans_shards": bool(
                ((support_ids[..., 0] < local_vocab_size) != (support_ids[..., 1] < local_vocab_size)).all()
            ),
        }


_SampleSupportTPProbeWorkerRemote = ray.remote(num_gpus=1)(_SampleSupportTPProbeWorker)


@pytest.mark.megatron
def test_sample_support_replay_tp2_matches_tp1_normalization(ray_init_fixture):
    """TP=2 vocabulary reductions match the equivalent TP=1 support normalization."""
    cfg = _sample_support_cfg(tensor_parallel_size=2)
    original_worker = megatron_worker_module.PolicyWorker
    megatron_worker_module.PolicyWorker = _SampleSupportTPProbeWorkerRemote
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=2,
            cfg=cfg,
        )
        results = ray.get(
            actor_group.async_run_ray_method("pass_through", "probe_tp2_matches_full_vocabulary_reference")
        )
    finally:
        megatron_worker_module.PolicyWorker = original_worker

    assert {result["rank"] for result in results} == {0, 1}
    assert all(result["support_spans_shards"] for result in results)
    assert max(result["max_abs_diff"] for result in results) <= 1e-6
