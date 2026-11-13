"""Streamed training (store_groups_in_ray) against the unstreamed trainer on the same rollouts.

The same policy workers, at lr=0, train one step of synthetic step-wise rollouts twice: once the
unstreamed way (the trainer's own advantage, normalization and staged forward_backward code) and
once streamed (stored trajectories, the summary plan, node-built shards). Losses and grad norms must
agree to bf16 tolerance.

uv run --isolated --extra dev --extra fsdp -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_streamed_training_parity.py -k fsdp
uv run --isolated --extra dev --extra megatron -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_streamed_training_parity.py -k megatron
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import loss_fn_outputs_to_tensor
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
)
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.backends.skyrl_train.workers.worker_utils import select_forward_inputs
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import (
    RowCollation,
    compute_prompt_boundaries,
    rows_to_training_input,
)
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
)
from skyrl.train.generators.base import StoredRolloutGroup, TrajectoryID
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "Qwen/Qwen3-0.6B"
# Built by megatron_tiny_models/create_qwen3_moe_tiny.py: 2 MoE layers, 8 experts, top-2.
TINY_MOE = os.environ.get("SKYRL_TINY_MOE_PATH", "/tmp/qwen3-moe-tiny-random")
NUM_GPUS = 4


def _outputs(num_groups, n_samples, vocab, seed, routes_shape=None):
    """Step-wise groups: a trajectory has 1-3 rows, its rows share its reward."""
    rng = np.random.default_rng(seed)
    outputs = []
    for g in range(num_groups):
        out = {k: [] for k in ("prompt_token_ids", "response_ids", "loss_masks", "rollout_logprobs", "rewards")}
        out.update(is_last_step=[], trajectory_ids=[], rollout_metrics={})
        if routes_shape is not None:
            out["rollout_expert_indices"] = []
        for j in range(n_samples):
            reward = float(rng.integers(0, 2))
            rows = int(rng.integers(1, 4))
            for r in range(rows):
                prompt = rng.integers(10, vocab, size=int(rng.integers(8, 40))).tolist()
                response = rng.integers(10, vocab, size=int(rng.integers(8, 64))).tolist()
                out["prompt_token_ids"].append(prompt)
                out["response_ids"].append(response)
                out["loss_masks"].append((rng.random(len(response)) > 0.2).astype(int).tolist())
                out["rollout_logprobs"].append((-rng.random(len(response))).tolist())
                out["rewards"].append(reward)
                out["is_last_step"].append(r == rows - 1)
                out["trajectory_ids"].append(TrajectoryID(f"p{g}", j))
                if routes_shape is not None:
                    layers, experts, topk = routes_shape
                    tokens = len(prompt) + len(response) - 1
                    picks = np.stack([rng.permutation(experts)[:topk] for _ in range(tokens * layers)])
                    out["rollout_expert_indices"].append(picks.reshape(tokens, layers, topk).astype(np.uint8))
        outputs.append(out)
    return outputs


def _near_policy(cfg, dispatch, outputs, tokenizer_pad, seed):
    """Replace the rollout logprobs with the policy's own plus small noise, so off-policy ratios stay
    near 1 and every loss trains (random logprobs clip every ratio to zero)."""
    rows = {k: [v for o in outputs for v in o[k]] for k in ("prompt_token_ids", "response_ids", "loss_masks")}
    data = rows_to_training_input(
        rows,
        RowCollation(
            pad_token_id=tokenizer_pad, max_seq_len=cfg.trainer.algorithm.max_seq_len, require_rollout_logprobs=False
        ),
    )
    data.metadata["response_length"] = data["response_mask"].shape[1]
    forward = dispatch.forward("policy", select_forward_inputs(data))
    logprobs = loss_fn_outputs_to_tensor(forward.loss_fn_outputs, key="logprobs")[: data.batch_size]
    rng = np.random.default_rng(seed)
    row = 0
    for o in outputs:
        for j, response in enumerate(o["response_ids"]):
            policy = logprobs[row, logprobs.shape[1] - len(response) :].float().cpu().numpy()
            o["rollout_logprobs"][j] = (policy + 0.05 * rng.standard_normal(len(response))).tolist()
            row += 1


def _cfg(strategy, loss, model=MODEL_NAME, r3=False):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = strategy
    cfg.trainer.policy.model.path = model
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.max_tokens_per_microbatch = 512
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.policy.optimizer_config.lr = 0.0
    cfg.generator.step_wise_trajectories = True
    cfg.generator.n_samples_per_prompt = 4
    algorithm = cfg.trainer.algorithm
    algorithm.policy_loss_type = loss
    algorithm.use_kl_loss = False
    algorithm.loss_reduction = "token_mean"
    algorithm.max_seq_len = 128
    algorithm.off_policy_correction.tis_ratio_type = "token"
    algorithm.off_policy_correction.token_tis_ratio_clip_high = 2.0
    if strategy == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    if r3:
        cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.distributed_executor_backend = "mp"
    validate_cfg(cfg)
    return cfg


def _unstreamed_step(cfg, dispatch, outputs, tokenizer_pad):
    """One step the way RayPPOTrainer trains a mini-batch."""
    rows = {k: [v for o in outputs for v in o[k]] for k in outputs[0] if isinstance(outputs[0][k], list)}
    stand_in = SimpleNamespace(cfg=cfg, all_metrics={}, tokenizer=SimpleNamespace(pad_token_id=tokenizer_pad))
    data = rows_to_training_input(
        {k: v for k, v in rows.items() if k != "rewards"},
        RowCollation(
            pad_token_id=tokenizer_pad, max_seq_len=cfg.trainer.algorithm.max_seq_len, require_rollout_logprobs=True
        ),
    )
    rewards = torch.zeros_like(data["loss_mask"])
    rewards[:, -1] = torch.tensor(rows["rewards"])
    data["rewards"] = rewards
    data["values"] = None
    uids = [o["trajectory_ids"][0].instance_id for o in outputs for _ in o["rewards"]]
    data.metadata.update(uids=uids, is_last_step=rows["is_last_step"], avg_response_length=0.0)
    data.metadata["response_length"] = data["response_mask"].shape[1]
    if not RayPPOTrainer._skip_policy_forward(stand_in, True):
        forward = dispatch.forward("policy", select_forward_inputs(data))
        data["action_log_probs"] = loss_fn_outputs_to_tensor(forward.loss_fn_outputs, key="logprobs")[: data.batch_size]
    data = RayPPOTrainer.compute_advantages_and_returns(stand_in, data)
    data.pop("rewards")
    normalization = MinibatchLossNormalization.from_batch(data)
    data = RayPPOTrainer._normalize_advantages(
        stand_in, data, [(0, data.batch_size)], [normalization], compute_prompt_boundaries(uids)
    )
    (chunks,) = dispatch.stage_data("policy", data, [(0, data.batch_size)])
    status = dispatch.forward_backward_from_staged("policy", chunks, loss_normalization=normalization)
    return status.metrics, dispatch.optim_step("policy")


def _streamed_step(cfg, dispatch, outputs, tokenizer_pad):
    groups = [
        GeneratedOutputGroup(StoredRolloutGroup.from_output(o), o["trajectory_ids"][0].instance_id, 0) for o in outputs
    ]
    trainer = SimpleNamespace(cfg=cfg, tokenizer=SimpleNamespace(pad_token_id=tokenizer_pad), dispatch=dispatch)
    trainer._row_collation = lambda: RayPPOTrainer._row_collation(trainer)
    trainer._skip_policy_forward = lambda has: RayPPOTrainer._skip_policy_forward(trainer, has)
    plan = FullyAsyncRayPPOTrainer._plan_streamed_mini_batch(trainer, groups)
    status = dispatch.forward_backward_rollout_shards(
        "policy", plan.shards, loss_normalization=plan.normalization, compute_old_logprobs=plan.compute_old_logprobs
    )
    return status.metrics, dispatch.optim_step("policy")


CASES = [
    pytest.param("fsdp", "regular", MODEL_NAME, False, id="fsdp-dp4-old-logprobs"),
    pytest.param("fsdp", "rollout_is", MODEL_NAME, False, id="fsdp-dp4-rollout-is"),
    pytest.param("megatron", "regular", MODEL_NAME, False, id="megatron-tp2-dp2-old-logprobs"),
    pytest.param("megatron", "rollout_is", TINY_MOE, True, id="megatron-tp2-dp2-r3-moe"),
]


@pytest.mark.parametrize("strategy, loss, model, r3", CASES)
def test_a_streamed_step_trains_as_the_unstreamed_trainer(ray_init_fixture, strategy, loss, model, r3):
    if r3 and not os.path.isdir(model):
        pytest.skip(f"build the tiny MoE first: {model}")
    cfg = _cfg(strategy, loss, model, r3)
    try:
        group = init_worker_with_type("policy", num_gpus_per_node=NUM_GPUS, cfg=cfg)
        dispatch = WorkerDispatch(cfg, policy_actor_group=group)
        # A row count both DP layouts split evenly, so the reference needs no padding rows.
        seed = next(s for s in range(100) if sum(len(o["rewards"]) for o in _outputs(6, 4, 2000, s)) % NUM_GPUS == 0)
        outputs = _outputs(num_groups=6, n_samples=4, vocab=2000, seed=seed, routes_shape=(2, 8, 2) if r3 else None)
        _near_policy(cfg, dispatch, outputs, tokenizer_pad=0, seed=seed)

        reference, reference_grad_norm = _unstreamed_step(cfg, dispatch, outputs, tokenizer_pad=0)
        streamed, streamed_grad_norm = _streamed_step(cfg, dispatch, outputs, tokenizer_pad=0)

        print(f"reference {reference_grad_norm} {reference}\nstreamed {streamed_grad_norm} {streamed}")
        assert streamed["policy_loss"] == pytest.approx(reference["policy_loss"], rel=2e-2, abs=1e-5)
        assert streamed["policy_entropy"] == pytest.approx(reference["policy_entropy"], rel=1e-2)
        assert reference_grad_norm > 0
        assert streamed_grad_norm == pytest.approx(reference_grad_norm, rel=2e-2)
    finally:
        ray.shutdown()
