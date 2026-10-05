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
        (2, 1, 2, False, 2, "sequence parallelism"),
    ],
    ids=[
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


# Packed-vs-unpacked parity. Microbatches of 4 samples (plus a trailing single-sample
# microbatch), so every packed row has samples after the first: a sample-boundary leak
# (wrong mRoPE restart, or GDN state / conv carried across samples) can only show up there.
PACKING_MICRO_BATCH = 4
PACKING_PROMPTS = [
    ("Describe this picture in one short sentence.", (56, 56), "A small square of noise."),
    ("What shapes can you see here? Answer briefly.", (112, 84), "Mostly scattered dots with no clear shape."),
    ("Is this image bright or dark?", (224, 168), "It is a mix of bright and dark pixels."),
    ("Count the colors.", (84, 224), "Too many colors to count; it looks like random noise."),
    ("Give a title for this image.", (140, 140), "Static."),
    ("Without any image: what is two plus three?", None, "Two plus three is five."),
    ("What would you call this texture?", (196, 112), "A grainy, television-static texture."),
    ("Summarize the image.", (70, 154), "Random colored noise in a tall rectangle."),
    ("One word for this image?", (168, 56), "Noise."),
]


def get_packing_parity_batch(model_name: str) -> TrainingInputBatch:
    """Variable-length image (and one text-only) samples laid out like an RL batch.

    Each row is prompt + assistant answer, left-padded. ``response_length`` is the
    longest answer and ``loss_mask`` marks each row's answer tokens (right-aligned),
    so only text targets are scored -- image-placeholder targets of random-noise
    images have huge, bf16-sensitive logprobs and are never trained on.
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for prompt, size, answer in PACKING_PROMPTS:
        user_content = [{"type": "text", "text": prompt}]
        images = None
        if size is not None:
            from PIL import Image

            w, h = size
            pixels = torch.randint(0, 256, (h, w, 3), generator=gen, dtype=torch.uint8).numpy()
            images = [Image.fromarray(pixels)]
            user_content = [{"type": "image"}] + user_content
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        out = processor(text=[text], images=images, return_tensors="pt")
        prompt_ids = processor(text=[prompt_text], images=images, return_tensors="pt")["input_ids"][0].tolist()
        ids = out["input_ids"][0].tolist()
        # Score what follows the shared prompt prefix (some templates render the
        # generation prompt slightly differently from a completed turn, e.g. think tags).
        prefix = next((k for k, (a, b) in enumerate(zip(ids, prompt_ids)) if a != b), len(prompt_ids))
        assert 0 < len(ids) - prefix <= len(ids) // 2, (prefix, len(ids))
        rows.append((ids, len(ids) - prefix, out.get("pixel_values"), out.get("image_grid_thw")))

    pixel_dim = next(pv.shape[-1] for _, _, pv, _ in rows if pv is not None)
    max_len = max(len(ids) for ids, _, _, _ in rows)
    num_actions = max(resp_len for _, resp_len, _, _ in rows)
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    sequences, attention_mask, loss_mask, pixel_values, image_grid_thw = [], [], [], [], []
    for ids, resp_len, pv, grid in rows:
        pad = max_len - len(ids)
        sequences.append([pad_token_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
        loss_mask.append([0] * (num_actions - resp_len) + [1] * resp_len)
        # Text-only rows carry empty vision tensors.
        pixel_values.append(pv if pv is not None else torch.zeros(0, pixel_dim))
        image_grid_thw.append(grid if grid is not None else torch.zeros(0, 3, dtype=torch.long))

    batch_size = len(rows)
    loss_mask = torch.tensor(loss_mask, dtype=torch.float)
    zeros = torch.zeros(batch_size, num_actions)
    data = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences),
            "attention_mask": torch.tensor(attention_mask),
            "action_log_probs": zeros,
            "base_action_log_probs": zeros,
            "rollout_logprobs": zeros,
            "values": zeros,
            "returns": zeros,
            "advantages": zeros,
            "loss_mask": loss_mask,
            "response_mask": loss_mask,
            "pixel_values": TensorList(pixel_values),
            "image_grid_thw": TensorList(image_grid_thw),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


def _megatron_vlm_forward_logprobs(
    model_name, batch, tp, remove_microbatch_padding, micro_batch=PACKING_MICRO_BATCH, pp=1
) -> torch.Tensor:
    """Inference forward (the RL old/ref-logprob path): [B, response_length], right-aligned."""
    cfg = get_test_actor_config(model_name=model_name)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.policy_num_gpus_per_node = tp * pp
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    cfg.trainer.micro_forward_batch_size_per_gpu = micro_batch
    cfg.trainer.micro_train_batch_size_per_gpu = micro_batch
    cfg.trainer.remove_microbatch_padding = remove_microbatch_padding
    try:
        actor_group = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=tp * pp, cfg=cfg
        )
        refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
        output = WorkerOutput.cat(actor_group.actor_infos, ray.get(refs))
        return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs").float()
    finally:
        ray.shutdown()
        ray_init_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "tp", "pp"),
    [
        ("Qwen/Qwen3-VL-2B-Instruct", 1, 1),
        ("Qwen/Qwen3-VL-2B-Instruct", 2, 1),
        # PP=2: pixel values reach only the first stage; every stage rebuilds mRoPE positions.
        ("Qwen/Qwen3-VL-2B-Instruct", 1, 2),
        # Qwen3.5 goes through the same Qwen3VLModel but with GatedDeltaNet layers, whose
        # state and conv must reset at every packed sample boundary.
        ("Qwen/Qwen3.5-0.8B", 1, 1),
        ("Qwen/Qwen3.5-0.8B", 2, 1),
        ("Qwen/Qwen3.5-0.8B", 1, 2),
    ],
    ids=["qwen3_vl_tp1", "qwen3_vl_tp2_sp", "qwen3_vl_pp2", "qwen3_5_vl_tp1", "qwen3_5_vl_tp2_sp", "qwen3_5_vl_pp2"],
)
@pytest.mark.megatron
async def test_megatron_vlm_packed_vs_unpacked(ray_init_fixture, model_name, tp, pp):
    """Sample packing must add no error beyond ordinary batching.

    Packed and padded (unpacked) microbatches run different kernels on different
    shapes, so in bf16 they agree only to rounding -- and so does unpacked with
    itself when the microbatch composition changes. The reference is therefore
    each sample run alone (unpacked, one sample per microbatch), and the check is
    that packed microbatches of 4 are no further from it than unpacked ones.
    A sample-boundary leak (mRoPE restart, GatedDeltaNet state/conv carried
    across samples) would also make later slots of a packed microbatch worse
    than slot 0.
    """
    batch = get_packing_parity_batch(model_name)
    num_actions = batch.metadata["response_length"]
    unpacked = _megatron_vlm_forward_logprobs(model_name, batch, tp, remove_microbatch_padding=False, pp=pp)
    packed = _megatron_vlm_forward_logprobs(model_name, batch, tp, remove_microbatch_padding=True, pp=pp)
    assert packed.shape == unpacked.shape == (len(PACKING_PROMPTS), num_actions)

    # Reference: every sample alone. At TP>1 (sequence parallel) a microbatch with no image
    # crashes Megatron-Bridge's split_deepstack_embs (fixed upstream in Megatron-Bridge#3869,
    # not yet merged), so the text-only row is left out of the reference there.
    has_image = torch.tensor([size is not None for _, size, _ in PACKING_PROMPTS])
    ref_rows = torch.arange(len(PACKING_PROMPTS)) if tp == 1 else has_image.nonzero().flatten()
    ref_batch = TrainingInputBatch({k: (None if v is None else v[ref_rows]) for k, v in batch.items()})
    ref_batch.metadata = batch.metadata
    alone = _megatron_vlm_forward_logprobs(
        model_name, ref_batch, tp, remove_microbatch_padding=False, micro_batch=1, pp=pp
    )

    scored = batch["loss_mask"].bool()
    for name, t in (("unpacked", unpacked), ("packed", packed), ("alone", alone)):
        rows_scored = scored[ref_rows] if name == "alone" else scored
        assert torch.isfinite(t[rows_scored]).all(), name
    ref_scored = scored[ref_rows]
    packed_vs_alone = (packed[ref_rows] - alone).abs()
    unpacked_vs_alone = (unpacked[ref_rows] - alone).abs()
    packed_vs_unpacked = (packed - unpacked).abs()

    slots = (torch.arange(len(PACKING_PROMPTS)) % PACKING_MICRO_BATCH)[ref_rows]
    print(f"\n[packing parity] {model_name} tp={tp} pp={pp}  (mean/max abs logprob diff on answer tokens)")
    for name, d, m in (
        ("packed vs alone", packed_vs_alone, ref_scored),
        ("unpacked vs alone", unpacked_vs_alone, ref_scored),
        ("packed vs unpacked", packed_vs_unpacked, scored),
    ):
        print(f"  {name:18s}: mean={d[m].mean().item():.5f} max={d[m].max().item():.4f}")
    for j, i in enumerate(ref_rows.tolist()):
        m = ref_scored[j]
        print(
            f"  sample {i} slot {slots[j].item()}: packed-alone={packed_vs_alone[j][m].mean().item():.5f} "
            f"unpacked-alone={unpacked_vs_alone[j][m].mean().item():.5f}"
        )

    packed_err = packed_vs_alone[ref_scored].mean().item()
    unpacked_err = unpacked_vs_alone[ref_scored].mean().item()
    # Packing must not add error beyond what ordinary batching already shows.
    assert packed_err <= 1.5 * unpacked_err + 5e-3, (packed_err, unpacked_err)
    # Guard against gross breakage (a wrong position or boundary is >> bf16 noise).
    assert packed_vs_unpacked[scored].mean().item() < 5e-2
    # No boundary leak: later slots of a packed microbatch are not worse than slot 0.
    first, later = slots == 0, slots > 0
    first_d, later_d = packed_vs_alone[first][ref_scored[first]], packed_vs_alone[later][ref_scored[later]]
    assert later_d.mean().item() <= 3 * first_d.mean().item() + 1e-2, (later_d.mean(), first_d.mean())
    assert later_d.max().item() <= 3 * first_d.max().item() + 0.25, (later_d.max(), first_d.max())
