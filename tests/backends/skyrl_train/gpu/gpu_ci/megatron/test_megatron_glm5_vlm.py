"""GLM-5.3-Flash as a vision-language model on the Megatron backend (``language_model_only=false``).

Uses the 4-layer slice of the real checkpoint (eatang/GLM-5.3-Flash-4layer), which keeps the full
vision tower. The slice ships no ``processor_config.json``; the fixture copies GLM-5.3-Flash's.
~24B params in bf16, so every Megatron run uses TP2/EP4 on 4xH100 and the HF reference shards the
model over the same 4 GPUs after Ray releases them.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s -m h100 tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_glm5_vlm.py
"""

import shutil

import pytest
import ray
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

SLICE = "eatang/GLM-5.3-Flash-4layer"
FULL = "zai-org/GLM-5.3-Flash"
TP, EP, NUM_GPUS = 2, 4, 4
MICRO_BATCH = 4

# (prompt, image size or None, answer). Mixed sizes so the packed stream holds images of
# different token counts; one text-only row so an image-free sample sits in a packed microbatch.
PROMPTS = [
    ("Describe this picture in one sentence.", (224, 224), "It is a square of random colored noise."),
    ("What is the dominant color?", (336, 196), "There is no single dominant color here."),
    ("Name a planet in the solar system.", None, "Mars is the fourth planet from the Sun."),
    ("How many objects are in the image?", (168, 280), "I cannot count any distinct objects."),
    ("Is this a photograph or a drawing?", (252, 252), "It looks like generated noise, not a photo."),
    ("Write the word hello.", None, "hello"),
    ("What shape is shown?", (196, 196), "No clear shape is visible in this image."),
    ("Is the image bright or dark?", (280, 168), "It is of medium brightness overall."),
]


@pytest.fixture(scope="module")
def glm5_vl_slice(tmp_path_factory) -> str:
    """Local copy of the slice with GLM-5.3-Flash's processor config added."""
    local = tmp_path_factory.mktemp("glm5_vl_slice")
    snapshot_download(SLICE, local_dir=local)
    for name in ("processor_config.json", "chat_template.jinja"):
        if not (local / name).exists():
            shutil.copy(hf_hub_download(FULL, name), local / name)
    return str(local)


def _config(model_path: str, language_model_only: bool = False) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.logger = "console"
    # KDA needs packed (thd) sequences.
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.policy.language_model_only = language_model_only
    cfg.trainer.ref.language_model_only = language_model_only
    cfg.generator.inference_engine.language_model_only = language_model_only
    # Policy-only forward: no colocated inference engine, no reference model.
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = TP
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = EP
    # ETP defaults to TP; ETP x EP must divide the world size.
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    # Forward-only: the optimizer state of ~24B params does not fit next to the weights.
    cfg.trainer.policy.inference_only_init = True
    validate_cfg(cfg)
    return cfg


def _build_rows(model_path: str):
    """Per row: (input_ids, answer_len, pixel_values | None, image_grid_thw | None)."""
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for prompt, size, answer in PROMPTS:
        content = [{"type": "text", "text": prompt}]
        images = None
        if size is not None:
            from PIL import Image

            w, h = size
            pixels = torch.randint(0, 256, (h, w, 3), generator=gen, dtype=torch.uint8).numpy()
            images = [Image.fromarray(pixels)]
            content = [{"type": "image"}] + content
        messages = [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        out = processor(text=[text], images=images, return_tensors="pt")
        prompt_ids = processor(text=[prompt_text], images=images, return_tensors="pt")["input_ids"][0].tolist()
        ids = out["input_ids"][0].tolist()
        prefix = next((k for k, (a, b) in enumerate(zip(ids, prompt_ids)) if a != b), len(prompt_ids))
        assert 0 < len(ids) - prefix <= len(ids) // 2, (prefix, len(ids))
        rows.append((ids, len(ids) - prefix, out.get("pixel_values"), out.get("image_grid_thw")))
    return processor, rows


def _batch(processor, rows, keep=None) -> TrainingInputBatch:
    """Left-padded RL-style batch; ``loss_mask`` marks each row's answer tokens (right-aligned)."""
    keep = list(range(len(rows))) if keep is None else keep
    rows = [rows[i] for i in keep]
    ref_pv = next(pv for _, _, pv, _ in rows if pv is not None) if any(r[2] is not None for r in rows) else None
    max_len = max(len(ids) for ids, _, _, _ in rows)
    num_actions = max(n for _, n, _, _ in rows)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    seqs, attn, loss_mask, pvs, grids = [], [], [], [], []
    for ids, n, pv, grid in rows:
        pad = max_len - len(ids)
        seqs.append([pad_id] * pad + ids)
        attn.append([0] * pad + [1] * len(ids))
        loss_mask.append([0] * (num_actions - n) + [1] * n)
        if ref_pv is not None:
            pvs.append(pv if pv is not None else ref_pv.new_zeros((0, *ref_pv.shape[1:])))
            grids.append(grid if grid is not None else torch.zeros(0, 3, dtype=torch.long))
    loss_mask = torch.tensor(loss_mask, dtype=torch.float)
    zeros = torch.zeros(len(rows), num_actions)
    fields = {
        "sequences": torch.tensor(seqs),
        "attention_mask": torch.tensor(attn),
        "action_log_probs": zeros,
        "base_action_log_probs": zeros,
        "rollout_logprobs": zeros,
        "values": zeros,
        "returns": zeros,
        "advantages": zeros,
        "loss_mask": loss_mask,
        "response_mask": loss_mask,
    }
    if ref_pv is not None:
        fields["pixel_values"] = TensorList(pvs)
        fields["image_grid_thw"] = TensorList(grids)
    data = TrainingInputBatch(fields)
    data.metadata = {"response_length": num_actions}
    return data


def _megatron_logprobs(model_path, batch, micro_batch=MICRO_BATCH, language_model_only=False) -> torch.Tensor:
    """[B, response_length] logprobs from the policy forward (old/ref-logprob path)."""
    cfg = _config(model_path, language_model_only=language_model_only)
    cfg.trainer.micro_forward_batch_size_per_gpu = micro_batch
    cfg.trainer.micro_train_batch_size_per_gpu = micro_batch
    with ray_init():
        group = init_worker_with_type("policy", shared_pg=None, colocate_all=False, num_gpus_per_node=NUM_GPUS, cfg=cfg)
        out = WorkerOutput.cat(group.actor_infos, ray.get(group.async_run_ray_method("mesh", "forward", data=batch)))
        return loss_fn_outputs_to_tensor(out.loss_fn_outputs, key="logprobs").float()


def _hf_logprobs(model_path, rows, num_actions) -> torch.Tensor:
    """HF Glm5NextForConditionalGeneration, one sample at a time, right-aligned like the batch."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    out = torch.zeros(len(rows), num_actions)
    with torch.no_grad():
        for i, (ids, n, pv, grid) in enumerate(rows):
            input_ids = torch.tensor([ids], device=model.device)
            kwargs = {}
            if pv is not None:
                kwargs = {"pixel_values": pv.to(model.device), "image_grid_thw": grid.to(model.device)}
            logits = model(input_ids=input_ids, **kwargs).logits.float()
            lp = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])[0]
            out[i, num_actions - n :] = lp[-n:].cpu()
    del model
    torch.cuda.empty_cache()
    return out


@pytest.mark.h100
@pytest.mark.megatron
def test_glm5_vlm_forward_matches_hf(glm5_vl_slice):
    """Megatron Glm5NextVLModel (packed, TP2+SP, EP4) vs HF on image and text-only answers."""
    processor, rows = _build_rows(glm5_vl_slice)
    batch = _batch(processor, rows)
    num_actions = batch.metadata["response_length"]
    megatron = _megatron_logprobs(glm5_vl_slice, batch)
    hf = _hf_logprobs(glm5_vl_slice, rows, num_actions)

    scored = batch["loss_mask"].bool()
    assert torch.isfinite(megatron[scored]).all()
    diff = (megatron - hf).abs()[scored]
    print(f"\n[glm5 vlm vs hf] mean={diff.mean().item():.4f} max={diff.max().item():.4f}")
    # The truncated slice has a spread next-token distribution: bf16 noise alone is ~0.05 mean
    # |dlogprob| (see the glm-5.3-flash rows in test_megatron_models.py). Wrong image placement
    # or a dropped vision weight is far above that.
    assert diff.mean().item() < 0.1


@pytest.mark.h100
@pytest.mark.megatron
def test_glm5_vlm_packed_vs_alone(glm5_vl_slice):
    """Packed microbatches of 4 (mixed image/text, incl. image-free samples at TP2+SP) must match
    each sample run alone. A misplaced image (features of sample k in sample j) or a boundary
    leak would make later slots worse than slot 0."""
    processor, rows = _build_rows(glm5_vl_slice)
    batch = _batch(processor, rows)
    packed = _megatron_logprobs(glm5_vl_slice, batch, micro_batch=MICRO_BATCH)
    alone = _megatron_logprobs(glm5_vl_slice, batch, micro_batch=1)

    scored = batch["loss_mask"].bool()
    diff = (packed - alone).abs()
    slots = torch.arange(len(rows)) % MICRO_BATCH
    for i in range(len(rows)):
        print(f"  sample {i} slot {slots[i].item()}: |packed-alone|={diff[i][scored[i]].mean().item():.5f}")
    assert diff[scored].mean().item() < 2e-2
    first, later = slots == 0, slots > 0
    assert diff[later][scored[later]].mean().item() <= 3 * diff[first][scored[first]].mean().item() + 1e-2


@pytest.mark.h100
@pytest.mark.megatron
def test_glm5_vlm_text_only_matches_language_model_only(glm5_vl_slice):
    """On text-only samples the VL wrapper must match the text-only GPTModel path: the only
    difference is where the sequence-parallel scatter happens. The MoE forward is not
    run-to-run deterministic (same path twice: mean ~0.012, max ~0.055 |dlogprob| on this
    slice), so the bar is that noise level, not bitwise equality."""
    processor, rows = _build_rows(glm5_vl_slice)
    text_rows = [i for i, (_, size, _) in enumerate(PROMPTS) if size is None]
    batch = _batch(processor, rows, keep=text_rows)
    vl = _megatron_logprobs(glm5_vl_slice, batch, micro_batch=len(text_rows))
    text = _megatron_logprobs(glm5_vl_slice, batch, micro_batch=len(text_rows), language_model_only=True)
    scored = batch["loss_mask"].bool()
    diff = (vl - text).abs()[scored]
    print(f"\n[glm5 vl vs lm-only, text rows] mean={diff.mean().item():.4f} max={diff.max().item():.4f}")
    assert diff.mean().item() < 0.03
