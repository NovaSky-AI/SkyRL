"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_vlm_init.py
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

    Builds a batch of ``batch_size`` identical sequences with variable amounts of
    left/right padding, the corresponding ``pixel_values`` / ``image_grid_thw``
    TensorLists, and dummy loss/log-prob fields.
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
    # max seq len a few tokens longer than the longest sequence so we always pad
    max_seq_length = max(len(seq) for seq in sequences) + 7

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    pad_before = [4, 0, 1, 6] * num_repeats
    pad_after = [max_seq_length - len(seq) - pad_before[i] for i, seq in enumerate(sequences)]
    loss_masks = torch.stack(
        [torch.cat([torch.ones(num_actions - pad_after[i]), torch.zeros(pad_after[i])]) for i in range(batch_size)]
    )

    for i, (pb, pa) in enumerate(zip(pad_before, pad_after)):
        sequences[i] = [pad_token_id] * pb + sequences[i] + [pad_token_id] * pa
        attention_masks[i] = [0] * pb + attention_masks[i] + [0] * pa

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
    ("worker_type", "tp", "pp", "cp", "ep", "etp", "gpus_per_node"),
    [
        ("policy", 2, 1, 1, 1, None, 2),
    ],
    ids=[
        "tp2_pp1_policy",
    ],
)
@pytest.mark.megatron
async def test_megatron_vlm_forward(ray_init_fixture, worker_type, tp, pp, cp, ep, etp, gpus_per_node):
    """
    Test that the Megatron VLM forward pass produces finite log-probs that line
    up with a plain HuggingFace forward over the same images and tokens.
    """
    cfg = get_test_actor_config(model_name=MODEL_NAME)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.policy_num_gpus_per_node = gpus_per_node
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
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

    ray.shutdown()
    ray_init_for_tests()

    @ray.remote(num_gpus=1)
    def run_hf_forward(batch, model_name):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, config=config, trust_remote_code=True, dtype=torch.bfloat16
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

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
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

    attention_mask, action_log_probs, num_actions = ray.get(run_hf_forward.remote(batch, MODEL_NAME))

    # Compare only the non-padding response tokens.
    response_mask = attention_mask[:, -num_actions:].bool()
    action_log_probs_megatron_masked = action_log_probs_megatron[response_mask]

    assert not action_log_probs_megatron_masked.isnan().any()
    assert not action_log_probs_megatron_masked.isinf().any()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_type", "tp", "pp", "gpus_per_node"),
    [
        ("policy", 1, 2, 4),
    ],
    ids=[
        "tp1_pp2",
    ],
)
@pytest.mark.megatron
async def test_vlm_train(ray_init_fixture, worker_type, tp, pp, gpus_per_node):
    """
    Full SFT loop: build an SFTTrainer for a VLM, run 5 train steps on a fixed
    image batch, and assert the loss decreases.
    """
    batch_size = gpus_per_node * 4
    batch_examples = get_test_training_batch(batch_size=batch_size)

    sft_config = SFTConfig(
        model=ModelConfig(path=MODEL_NAME),
        strategy="megatron",
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

    print(training_losses)
    assert training_losses[0] > training_losses[-1]
