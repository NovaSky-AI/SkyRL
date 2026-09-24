"""GPU coverage for score centering on the Megatron backend.

Run with::

    NVTE_FLASH_ATTN=0 uv run --isolated --extra dev --extra megatron pytest -s -vvv \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_score_centering.py
"""

import numpy as np
import pytest
import ray
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_LOGPROBS_FIELD,
    SAMPLE_SUPPORT_LOGPROBS_TORCH_DTYPE,
    SAMPLE_SUPPORT_PADDING,
)
from skyrl.backends.skyrl_train.utils.score_centering_support import (
    fused_label_and_head_logprobs,
    score_head_members,
)
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
HEAD_WIDTH = 4


def _score_centering_cfg(tensor_parallel_size: int, *, remove_microbatch_padding: bool) -> SkyRLTrainConfig:
    """Minimal Megatron config for one `reinforce` + score-centering training step."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_gpus_per_node = tensor_parallel_size
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = remove_microbatch_padding
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tensor_parallel_size
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.algorithm.policy_loss_type = "reinforce"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.score_centering.enabled = True
    cfg.trainer.algorithm.score_centering.top_k = HEAD_WIDTH
    validate_cfg(cfg)
    assert cfg.generator.inference_engine.enable_return_sample_support_set
    assert cfg.generator.inference_engine.enable_return_sample_support_logprobs
    return cfg


def _synthetic_head(responses: list[list[int]], vocab_size: int) -> tuple[PackedTensor, PackedTensor]:
    """A head per response token: the sampled token plus decoys, one short row per trajectory."""
    ids, logprobs = [], []
    for response in responses:
        selected = torch.tensor(response, dtype=torch.long)
        members = torch.stack(
            [
                selected,
                (selected + 11) % vocab_size,
                (selected + vocab_size // 2) % vocab_size,
                (selected + 3) % vocab_size,
            ],
            dim=-1,
        )
        row_logprobs = torch.log(torch.tensor([0.5, 0.25, 0.15, 0.05]).expand(len(response), -1).clone())
        members[0, -1] = SAMPLE_SUPPORT_PADDING
        row_logprobs[0, -1] = float("-inf")
        ids.append(members.to(torch.int32))
        logprobs.append(row_logprobs.to(SAMPLE_SUPPORT_LOGPROBS_TORCH_DTYPE))
    return PackedTensor.from_segments(ids), PackedTensor.from_segments(logprobs)


def _training_batch() -> TrainingInputBatch:
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
    rewards = [[0.0] * len(response) for response in responses]
    loss_masks = [[1] * len(response) for response in responses]
    sequences, attention_mask, response_mask, _, loss_mask, _, _, _, _ = convert_prompts_responses_to_batch_tensors(
        pad_token_id=tokenizer.pad_token_id,
        prompts=prompt_ids,
        responses=responses,
        rewards=rewards,
        loss_masks=loss_masks,
    )
    head_ids, head_logprobs = _synthetic_head(responses, tokenizer.vocab_size)
    num_actions = response_mask.shape[1]
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "action_log_probs": torch.zeros_like(loss_mask, dtype=torch.float32),
            "base_action_log_probs": torch.zeros_like(loss_mask, dtype=torch.float32),
            "rollout_logprobs": torch.full_like(loss_mask, -0.5, dtype=torch.float32),
            "advantages": torch.where(loss_mask.bool(), torch.ones_like(loss_mask, dtype=torch.float32), 0.0),
            "values": torch.zeros_like(loss_mask, dtype=torch.float32),
            "returns": torch.zeros_like(loss_mask, dtype=torch.float32),
            SAMPLE_SUPPORT_FIELD: head_ids,
            SAMPLE_SUPPORT_LOGPROBS_FIELD: head_logprobs,
        }
    )
    batch.metadata = {"response_length": num_actions, "global_step": 0}
    return batch


@pytest.mark.megatron
@pytest.mark.parametrize("tensor_parallel_size,remove_microbatch_padding", [(1, False), (1, True), (2, True)])
def test_score_centering_training_step(ray_init_fixture, tensor_parallel_size, remove_microbatch_padding):
    """One `reinforce` + centering step runs, the head is aligned and the centering metrics are logged."""
    cfg = _score_centering_cfg(tensor_parallel_size, remove_microbatch_padding=remove_microbatch_padding)
    batch = _training_batch()
    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=tensor_parallel_size,
        cfg=cfg,
    )

    results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", batch))
    ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

    metrics = results[0].metrics
    assert np.isfinite(metrics["policy_loss"])
    # Every response token's sampled id was placed in its head row.
    assert metrics["loss_metrics/score_centering_sampled_in_head_frac"] == pytest.approx(1.0)
    assert metrics["loss_metrics/score_centering_term_abs_mean"] > 0.0
    assert 0.0 < metrics["loss_metrics/score_centering_sampler_head_mass"] <= 1.0
    assert 0.0 < metrics["loss_metrics/score_centering_trainer_head_mass"] <= 1.0
    assert np.isfinite(metrics["loss_metrics/score_centering_tail_mass_ratio"])
    # Head members scored with the label's own normalizer keep the tail-mass ratio O(1); a mismatch
    # between the two normalizers blows it up (the first fused smoke run read 85).
    assert metrics["loss_metrics/score_centering_tail_mass_ratio"] < 10.0


class _HeadScoringTPProbeWorker(MegatronPolicyWorkerBase):
    """Compares TP=2 head scoring (gather and fused paths) with the unsharded computation."""

    def probe_tp2_head_scoring_matches_unsharded(self) -> dict:
        import megatron.core.parallel_state as mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size != 2:
            raise ValueError(f"expected TP=2, got TP={tp_size}")

        torch.manual_seed(0)
        local_vocab_size, hidden_size = 8, 6
        total_vocab_size = local_vocab_size * tp_size
        device = torch.device("cuda", torch.cuda.current_device())
        # Shared inputs on every rank; each rank owns its vocabulary slice of the LM head.
        hidden = torch.randn(2, 3, hidden_size, device=device)
        lm_head = torch.randn(total_vocab_size, hidden_size, device=device)
        temperature = 0.8
        full_logits = (hidden @ lm_head.T) / temperature
        local_logits = full_logits[..., tp_rank * local_vocab_size : (tp_rank + 1) * local_vocab_size]
        sampled_ids = torch.tensor([[1, 10, 2], [11, 3, 12]], dtype=torch.long, device=device)
        head_ids = torch.stack(
            (
                sampled_ids,
                (sampled_ids + total_vocab_size // 2) % total_vocab_size,
                (sampled_ids + 3) % total_vocab_size,
            ),
            dim=-1,
        ).to(torch.int32)
        head_ids[0, 0, -1] = SAMPLE_SUPPORT_PADDING
        log_softmax = torch.log_softmax(full_logits, dim=-1)
        sampled_logprobs = log_softmax.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        expected = torch.where(head_ids >= 0, log_softmax.gather(-1, head_ids.clamp(min=0).long()), float("-inf"))

        sharded_gather = score_head_members(
            local_logits,
            sampled_ids,
            head_ids,
            vocab_start_index=tp_rank * local_vocab_size,
            vocab_end_index=(tp_rank + 1) * local_vocab_size,
            tp_group=tp_group,
            sampled_logprobs=sampled_logprobs,
        )
        sharded_fused = score_head_members(
            hidden,
            sampled_ids,
            head_ids,
            vocab_start_index=tp_rank * local_vocab_size,
            vocab_end_index=(tp_rank + 1) * local_vocab_size,
            tp_group=tp_group,
            sampled_logprobs=sampled_logprobs,
            lm_head_weight=lm_head[tp_rank * local_vocab_size : (tp_rank + 1) * local_vocab_size],
            temperature=temperature,
            chunk_size=4,
        )
        finite = head_ids >= 0
        torch.testing.assert_close(sharded_gather[finite], expected[finite], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sharded_fused[finite], expected[finite], rtol=1e-5, atol=1e-5)
        assert torch.isneginf(sharded_gather[~finite]).all() and torch.isneginf(sharded_fused[~finite]).all()

        # The fused label + head op, in bf16 like the real model, against the unsharded reference built
        # with the same numerics (bf16 logits, fp32 softmax).
        hidden_bf16 = hidden.to(torch.bfloat16).requires_grad_(True)
        weight_bf16 = lm_head.to(torch.bfloat16)
        local_weight = (
            weight_bf16[tp_rank * local_vocab_size : (tp_rank + 1) * local_vocab_size].clone().requires_grad_(True)
        )
        label, members = fused_label_and_head_logprobs(
            hidden_bf16,
            local_weight,
            sampled_ids,
            head_ids,
            vocab_start_index=tp_rank * local_vocab_size,
            vocab_end_index=(tp_rank + 1) * local_vocab_size,
            tp_group=tp_group,
            temperature=temperature,
            chunk_size=2,
        )
        (label.sum() + members[finite].sum()).backward()
        ref_logits = torch.matmul(hidden_bf16.detach(), (weight_bf16 / temperature).t()).float()
        ref_log_softmax = torch.log_softmax(ref_logits, dim=-1)
        ref_label = ref_log_softmax.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        ref_members = ref_log_softmax.gather(-1, head_ids.clamp(min=0).long())
        torch.testing.assert_close(label, ref_label, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(members[finite], ref_members[finite], rtol=1e-4, atol=1e-4)
        assert torch.isfinite(hidden_bf16.grad).all() and torch.isfinite(local_weight.grad).all()
        return {
            "rank": tp_rank,
            "max_abs_diff": max(
                (sharded_gather[finite] - expected[finite]).abs().max().item(),
                (sharded_fused[finite] - expected[finite]).abs().max().item(),
                (label - ref_label).abs().max().item(),
                (members[finite] - ref_members[finite]).abs().max().item(),
            ),
        }


_HeadScoringTPProbeWorkerRemote = ray.remote(num_gpus=1)(_HeadScoringTPProbeWorker)


@pytest.mark.megatron
def test_head_scoring_tp2_matches_unsharded(ray_init_fixture):
    """TP=2 head logits reduce to the unsharded full-vocabulary logprobs on both scoring paths."""
    cfg = _score_centering_cfg(tensor_parallel_size=2, remove_microbatch_padding=False)
    original_worker = megatron_worker_module.PolicyWorker
    megatron_worker_module.PolicyWorker = _HeadScoringTPProbeWorkerRemote
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=2,
            cfg=cfg,
        )
        results = ray.get(actor_group.async_run_ray_method("pass_through", "probe_tp2_head_scoring_matches_unsharded"))
    finally:
        megatron_worker_module.PolicyWorker = original_worker

    assert {result["rank"] for result in results} == {0, 1}
    assert max(result["max_abs_diff"] for result in results) <= 1e-4
