"""
Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_vlm_init.py
"""

import pytest
import ray
import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits
from skyrl.train.config import (
    MegatronConfig,
    ModelConfig,
    SkyRLTrainConfig,
)
from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def get_test_actor_config(model_name=MODEL_NAME) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.logger = "console"

    validate_cfg(cfg)

    return cfg


def get_test_training_batch(batch_size=4) -> TrainingInputBatch:
    """
    Returns a VLM training batch with one image per sample.

    Builds a batch of ``batch_size`` sequences with variable amounts of
    left padding
    """
    assert batch_size % 4 == 0, "batch size must be divisible by 4"
    num_repeats = batch_size // 4
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the color of this tiny dot?"},
                {
                    "type": "image",
                    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                },
            ],
        },
        {"role": "assistant", "content": "It appears to be cyan, which is a bright blue color."},
    ]

    processor_output = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_dict=True
    )
    # Processing is fragile and varies across transformers releases; normalize to
    # a flat list of token ids.
    ids = processor_output["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], int):
        ids = [ids]
    sequences = [list(ids[0]) for _ in range(batch_size)]

    pixel_values = [processor_output["pixel_values"]] * batch_size
    image_grid_thw = [processor_output["image_grid_thw"]] * batch_size
    attention_masks = [[1] * len(seq) for seq in sequences]
    num_actions = 15

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    pad_before = [4, 0, 1, 6] * num_repeats
    loss_masks = torch.ones(batch_size, num_actions)

    for i, pb in enumerate(pad_before):
        trunc = len(sequences[i]) - pb
        sequences[i] = [pad_token_id] * pb + sequences[i][:trunc]
        attention_masks[i] = [0] * pb + attention_masks[i][:trunc]

    attention_masks = torch.tensor(attention_masks)
    sequences = torch.tensor(sequences)

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_masks,
            "action_log_probs": torch.tensor([[0.1] * num_actions] * batch_size),
            "base_action_log_probs": torch.tensor([[0.2] * num_actions] * batch_size),
            "rollout_logprobs": torch.tensor([[0.11] * num_actions] * batch_size),
            "values": torch.tensor([[0.1] * num_actions] * batch_size),
            "returns": torch.tensor([[0.1] * num_actions] * batch_size),
            "advantages": torch.tensor([[0.5] * num_actions] * batch_size),
            "loss_mask": loss_masks,
            "response_mask": loss_masks,
            "pixel_values": TensorList(pixel_values),
            "image_grid_thw": TensorList(image_grid_thw),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_type", "tp", "pp", "gpus_per_node"),
    [
        ("policy", 2, 1, 2),
        ("policy", 1, 2, 4),
    ],
    ids=[
        "tp2_pp1_policy",
        "tp1_pp2_policy",
    ],
)
@pytest.mark.megatron
async def test_megatron_vlm_forward(ray_init_fixture, worker_type, tp, pp, gpus_per_node):
    cfg = get_test_actor_config(model_name=MODEL_NAME)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.policy_num_gpus_per_node = gpus_per_node
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    cfg.trainer.remove_microbatch_padding = False
    batch = get_test_training_batch(max(4, gpus_per_node))

    actor_group = init_worker_with_type(
        worker_type,
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        cfg=cfg,
    )

    action_log_probs_refs = actor_group.async_run_ray_method("mesh", "forward", data=batch, loss_fn="cross_entropy")
    all_rank_action_log_probs = ray.get(action_log_probs_refs)
    megatron_output = WorkerOutput.cat(actor_group.actor_infos, all_rank_action_log_probs)
    action_log_probs_megatron = loss_fn_outputs_to_tensor(megatron_output.loss_fn_outputs, key="logprobs")

    num_actions = batch.metadata["response_length"]
    if action_log_probs_megatron.shape[1] < num_actions:
        pad_width = num_actions - action_log_probs_megatron.shape[1]
        action_log_probs_megatron = torch.nn.functional.pad(action_log_probs_megatron, (0, pad_width))

    # Check only the non-padding response tokens (padding positions can be -inf).
    response_mask = batch["attention_mask"][:, -num_actions:].bool()
    action_log_probs_megatron_masked = action_log_probs_megatron[response_mask]

    assert not action_log_probs_megatron_masked.isnan().any()
    assert not action_log_probs_megatron_masked.isinf().any()


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_vlm_sft_hf_parity(ray_init_fixture):
    cfg = get_test_actor_config(model_name=MODEL_NAME)
    cfg.trainer.strategy = "megatron"
    # fp32 so the two forwards are numerically comparable. Neither flash nor cuDNN
    # fused attention supports fp32, so pin TE's unfused backend explicitly;
    # megatron-core asserts the NVTE_* env vars match the configured backend.
    cfg.trainer.bf16 = False
    cfg.trainer.flash_attn = False
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["attention_backend"] = "unfused"
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = None
    cfg.trainer.remove_microbatch_padding = False
    # fp32 on a single GPU: the DDP param/grad buffers and the optimizer's fp32
    # master + AdamW state add ~15GB on top of the 8GB model, overflowing a 22GB
    # L4. This test only forwards, so skip that training state.
    cfg.trainer.policy.inference_only_init = True
    batch = get_test_training_batch(batch_size=4)

    actor_group = init_worker_with_type(
        "policy",
        shared_pg=None,
        colocate_all=False,
        num_gpus_per_node=1,
        cfg=cfg,
    )

    action_log_probs_refs = actor_group.async_run_ray_method("mesh", "forward", data=batch, loss_fn="cross_entropy")
    megatron_output = WorkerOutput.cat(actor_group.actor_infos, ray.get(action_log_probs_refs))
    action_log_probs_megatron = loss_fn_outputs_to_tensor(megatron_output.loss_fn_outputs, key="logprobs")

    num_actions = batch.metadata["response_length"]
    if action_log_probs_megatron.shape[1] < num_actions:
        pad_width = num_actions - action_log_probs_megatron.shape[1]
        action_log_probs_megatron = torch.nn.functional.pad(action_log_probs_megatron, (0, pad_width))

    ray.shutdown()
    ray_init_for_tests()

    @ray.remote(num_gpus=1)
    def run_hf_forward(batch, model_name):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, config=config, trust_remote_code=True, dtype=torch.float32
        )
        model.eval()
        model.to("cuda")
        sequences_fwd = batch["sequences"]
        attention_mask = batch["attention_mask"]
        pixel_values = torch.cat([t for t in batch["pixel_values"].tensors])
        image_grid_thw = torch.cat([t for t in batch["image_grid_thw"].tensors])

        num_actions = batch.metadata["response_length"]

        mm_token_type_ids = torch.zeros_like(sequences_fwd, dtype=torch.int32)
        mm_token_type_ids[sequences_fwd == config.image_token_id] = 1

        sequences_rolled = torch.roll(sequences_fwd, shifts=-1, dims=1)
        sequences_fwd, attention_mask, sequences_rolled = (
            sequences_fwd.to("cuda"),
            attention_mask.to("cuda"),
            sequences_rolled.to("cuda"),
        )
        pixel_values, image_grid_thw, mm_token_type_ids = (
            pixel_values.to("cuda"),
            image_grid_thw.to("cuda"),
            mm_token_type_ids.to("cuda"),
        )

        with torch.no_grad():
            output = model(
                sequences_fwd,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            log_probs = logprobs_from_logits(output["logits"], sequences_rolled)
            action_log_probs = log_probs[:, -num_actions - 1 : -1].to("cpu").detach()

        return attention_mask.to("cpu").detach(), action_log_probs.to("cpu").detach(), num_actions

    attention_mask, action_log_probs_hf, num_actions = ray.get(run_hf_forward.remote(batch, MODEL_NAME))

    # Compare only the non-padding response tokens.
    response_mask = attention_mask[:, -num_actions:].bool()
    megatron_masked = action_log_probs_megatron[response_mask]
    hf_masked = action_log_probs_hf[response_mask]

    max_abs_diff = (megatron_masked - hf_masked).abs().max().item()
    mean_abs_diff = (megatron_masked - hf_masked).abs().mean().item()
    print(
        f"VLM SFT HF parity | response_tokens={int(response_mask.sum().item())} "
        f"max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f}"
    )
    assert max_abs_diff < 5e-1, f"Max diff {max_abs_diff} is too large"
    assert mean_abs_diff < 9e-2, f"Avg diff {mean_abs_diff} is too large"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tp", "cp", "sequence_parallel_size", "remove_microbatch_padding", "gpus_per_node", "expected_substring"),
    [
        # Qwen3VL packs sequences internally, `remove_microbatch_padding` is not supported
        (1, 1, 1, True, 2, "pack sequences inside their own forward"),
        (2, 1, 2, False, 2, "sequence parallelism"),
    ],
    ids=[
        "microbatch_padding",
        "sequence_parallel",
    ],
)
@pytest.mark.megatron
async def test_megatron_vlm_unsupported_parallelism_raises(
    ray_init_fixture, tp, cp, sequence_parallel_size, remove_microbatch_padding, gpus_per_node, expected_substring
):
    cfg = get_test_actor_config(model_name=MODEL_NAME)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.policy_num_gpus_per_node = gpus_per_node
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = None
    cfg.trainer.policy.sequence_parallel_size = sequence_parallel_size
    cfg.trainer.remove_microbatch_padding = remove_microbatch_padding
    batch = get_test_training_batch(max(4, gpus_per_node))

    with pytest.raises(Exception, match=expected_substring):
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        action_log_probs_refs = actor_group.async_run_ray_method("mesh", "forward", data=batch, loss_fn="cross_entropy")
        ray.get(action_log_probs_refs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_type", "tp", "pp", "gpus_per_node"),
    [
        ("policy", 1, 1, 4),
        ("policy", 2, 1, 4),
        ("policy", 1, 2, 4),
        ("policy", 2, 2, 4),
    ],
    ids=[
        "tp1_pp1_dp4",
        "tp2_pp1_dp2",
        "tp1_pp2_dp2",
        "tp2_pp2_dp1",
    ],
)
@pytest.mark.megatron
async def test_vlm_train(ray_init_fixture, worker_type, tp, pp, gpus_per_node):
    """
    Parallelism sweep: build an SFTTrainer for a VLM and run 5 train steps on a
    fixed image batch under each TP/PP/DP combination, asserting the loss
    decreases. With 4 GPUs the combos cover pure DP (DP=4), TP only (DP=2),
    PP only (DP=2, exercises the PP-rank-0-only image dispatch), and the
    TP+PP combo (DP=1).
    """
    batch_size = gpus_per_node * 4
    batch_examples = get_test_training_batch(batch_size=batch_size)

    sft_config = SFTConfig(
        model=ModelConfig(path=MODEL_NAME),
        strategy="megatron",
        num_steps=5,
        placement=SFTPlacementConfig(num_gpus_per_node=gpus_per_node),
        megatron_config=MegatronConfig(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
        ),
    )

    trainer = SFTTrainer(sft_config)
    trainer.setup()

    training_losses = []
    for training_step_i in range(5):
        step_i_outputs = trainer.train_step(batch_examples, training_step_i)
        training_losses.append(step_i_outputs["loss"])

    assert training_losses[0] > training_losses[-1]


def _vlm_cp_training_batch(model_name: str) -> TrainingInputBatch:
    """The packing-parity batch (first 8 samples) with non-trivial PPO inputs.

    The old and rollout logprobs are placeholders: ``_run_vlm_cp_layout`` replaces them with
    the layout's own forward logprobs before training.
    """
    batch = get_packing_parity_batch(model_name)[:8]
    shape = batch["advantages"].shape
    batch["advantages"] = torch.full(shape, 0.5)
    for key in ("action_log_probs", "base_action_log_probs", "rollout_logprobs"):
        batch[key] = torch.full(shape, -1.0)
    batch.metadata["global_step"] = 0
    return batch


def _on_policy(batch: TrainingInputBatch, logprobs: torch.Tensor) -> TrainingInputBatch:
    """Copy of ``batch`` with old and rollout logprobs set to ``logprobs`` (importance ratio 1).

    Fixed old logprobs put some tokens near the PPO clip boundary, where bf16-level logprob
    differences between parallel layouts clip different tokens and move the grad norm by
    several percent. With ratio 1 no token is near the boundary.
    """
    on_policy = TrainingInputBatch({k: v for k, v in batch.items()})
    on_policy.metadata = batch.metadata
    on_policy["action_log_probs"] = logprobs.clone()
    on_policy["rollout_logprobs"] = logprobs.clone()
    return on_policy


def _forward_logprobs(actor_group, batch: TrainingInputBatch) -> torch.Tensor:
    refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
    output = WorkerOutput.cat(actor_group.actor_infos, ray.get(refs))
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs").float()


def _run_vlm_cp_layout(model_name, batch, cp, tp=1):
    """Packed forward logprobs, one forward_backward + optim_step, then two half-batch
    forward_backward calls + optim_step; returns (logprobs, results, grad_norms, split_grad_norms).

    lr is 0, so optim_step only reduces, reports and clears the gradients: both phases see the
    same weights and their grad norms are directly comparable.
    """
    cfg = get_test_actor_config(model_name=model_name)
    cfg.trainer.strategy = "megatron"
    num_gpus = tp * cp
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.remove_microbatch_padding = True
    # KL off: the KL term is a mean over each call's microbatches, so two calls weight it twice and
    # the two-call check below would not be comparing like for like. Both checks cover the policy loss.
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.train_batch_size = len(batch)
    cfg.trainer.policy_mini_batch_size = len(batch)
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = PACKING_MICRO_BATCH
    cfg.trainer.micro_train_batch_size_per_gpu = PACKING_MICRO_BATCH
    cfg.trainer.policy.optimizer_config.lr = 0
    with ray_init():
        actor_group = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=num_gpus, cfg=cfg
        )
        logprobs = _forward_logprobs(actor_group, batch)
        train_batch = _on_policy(batch, logprobs)
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", train_batch))
        grad_norms = ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        # Gradient accumulation over two forward_backward calls in one optimizer step (as a
        # Tinker client may do). Under calculate_per_token_loss each call's gradients are
        # divided by that call's own token count. The halves are the two microbatches of the
        # full-batch call, so both modes should give the full-batch gradient.
        half = len(batch) // 2
        for part in (train_batch.slice(0, half), train_batch.slice(half, len(batch))):
            ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", part))
        split_grad_norms = ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        return logprobs, results, grad_norms, split_grad_norms


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "tp"),
    [
        ("Qwen/Qwen3-VL-2B-Instruct", 1),
        ("Qwen/Qwen3.5-0.8B", 1),
        # TP=2 turns on sequence parallelism: the deepstack/SP split runs on top of the CP split.
        ("Qwen/Qwen3-VL-2B-Instruct", 2),
        ("Qwen/Qwen3.5-0.8B", 2),
    ],
    ids=["qwen3_vl", "qwen3_5_vl", "qwen3_vl_tp2_sp", "qwen3_5_vl_tp2_sp"],
)
@pytest.mark.megatron
async def test_megatron_vlm_cp_vs_no_cp(model_name, tp):
    """VLM context parallelism must match CP=1: per-token logprobs and the gradient.

    The full packed stream goes to every CP rank and Megatron-Bridge's Qwen3VLModel
    computes mRoPE, places image features and applies the CP split itself. Images span
    the 2*CP chunk boundaries of most samples, so a mismatched split shows up as large
    logprob differences. The grad-norm check covers loss scaling: the bridge forces
    calculate_per_token_loss under CP, so CP=2 runs Megatron's per-token mode (DDP sums,
    each call's gradients divided by its token count) while CP=1 runs the default mode.
    The batch is on-policy (old logprobs from the layout's own forward), so PPO clipping
    can't differ between layouts.
    Both layouts use DP=1 (TP*1 vs TP*2 GPUs) so their microbatches are
    identical and only the CP split differs; the bf16 floor still applies (changing
    only microbatch composition moves these logprobs by ~0.034 mean), so the logprob
    bar is the bf16 floor while a wrong split shows up as O(1) differences.
    """
    batch = _vlm_cp_training_batch(model_name)
    logprobs_nocp, results_nocp, grad_norms_nocp, split_nocp = _run_vlm_cp_layout(model_name, batch, cp=1, tp=tp)
    logprobs_cp, results_cp, grad_norms_cp, split_cp = _run_vlm_cp_layout(model_name, batch, cp=2, tp=tp)

    scored = batch["loss_mask"].bool()
    diff = (logprobs_cp - logprobs_nocp).abs()[scored]
    print(f"\n[cp parity] {model_name} tp={tp}: logprob max={diff.max().item():.4f} mean={diff.mean().item():.5f}")
    print(f"[cp parity] grad norms CP1={grad_norms_nocp} CP2={grad_norms_cp}")
    print(f"[cp parity] two-call grad norms CP1={split_nocp} CP2={split_cp}")
    print(
        f"[cp parity] policy_loss: CP1={results_nocp[0].metrics['policy_loss']} CP2={results_cp[0].metrics['policy_loss']}"
    )

    assert torch.isfinite(logprobs_cp[scored]).all()
    # bf16 floor: HF bf16 vs fp32 differs by ~5e-2 mean on these answer tokens.
    assert diff.mean().item() < 5e-2
    assert diff.max().item() < 1.0
    gn_nocp, gn_cp = grad_norms_nocp[0], grad_norms_cp[0]
    assert gn_nocp is not None and gn_nocp > 0 and gn_cp is not None
    # Same 10% band as test_megatron_worker's text CP check: a scaling bug is ~2x off.
    assert abs(gn_cp - gn_nocp) / gn_nocp < 0.1, (gn_cp, gn_nocp)
    # Two calls per optimizer step must add up to the one-call gradient in each layout (same
    # weights, lr=0). Dividing the window once by the summed token count would average the two
    # calls instead, about half the norm.
    for gn, split_gn in ((gn_nocp, split_nocp[0]), (gn_cp, split_cp[0])):
        assert split_gn is not None and abs(split_gn - gn) / gn < 0.02, (split_gn, gn)
