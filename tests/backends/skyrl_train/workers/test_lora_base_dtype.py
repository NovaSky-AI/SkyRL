"""Parameter dtypes of an FSDP LoRA policy under ``trainer.policy.model.lora.base_dtype``.

``FSDPPolicyWorkerBase`` builds the policy with ``HFModelWrapper(bf16=...)``; ``base_dtype="bfloat16"`` sets
``bf16=True``. These tests load a tiny model through ``HFModelWrapper`` on CPU and check what is stored in which
dtype, and that one FSDP2 training step with SkyRL's wrapping and mixed-precision policy gives the same LoRA
gradients for a bf16-stored base as for an fp32-stored one.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, Qwen3Config

from skyrl.backends.skyrl_train.distributed.fsdp_utils import apply_fsdp2
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import FSDPConfig


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    # HFModelWrapper is built directly; no Ray actors are needed.
    yield


@pytest.fixture(scope="module")
def tiny_bf16_checkpoint(tmp_path_factory):
    """A tiny Qwen3 causal LM saved in bf16, as released checkpoints usually are."""
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny_qwen3_bf16")
    model.save_pretrained(path)
    return str(path)


def _wrap(path: str, bf16: bool) -> HFModelWrapper:
    return HFModelWrapper(path, bf16=bf16, use_flash_attention_2=False, lora_rank=4, lora_alpha=8)


def _dtypes_of(model: torch.nn.Module) -> tuple[set, set]:
    """(dtypes of the LoRA adapter parameters, dtypes of every other parameter)."""
    params = list(model.named_parameters())
    return {p.dtype for n, p in params if "lora_" in n}, {p.dtype for n, p in params if "lora_" not in n}


def test_bf16_base_keeps_fp32_adapters(tiny_bf16_checkpoint):
    wrapper = _wrap(tiny_bf16_checkpoint, bf16=True)
    lora, base = _dtypes_of(wrapper.model)

    assert base == {torch.bfloat16}
    assert lora == {torch.float32}
    trainable = [name for name, p in wrapper.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)


def test_default_stores_everything_in_fp32(tiny_bf16_checkpoint):
    lora, base = _dtypes_of(_wrap(tiny_bf16_checkpoint, bf16=False).model)

    assert base == {torch.float32}
    assert lora == {torch.float32}


def test_bf16_base_holds_the_checkpoint_values(tiny_bf16_checkpoint):
    """The bf16 checkpoint upcast to fp32 is exact, so the two storages hold the same base values."""
    fp32_base = {n: p for n, p in _wrap(tiny_bf16_checkpoint, bf16=False).model.named_parameters() if "lora_" not in n}
    bf16_base = {n: p for n, p in _wrap(tiny_bf16_checkpoint, bf16=True).model.named_parameters() if "lora_" not in n}

    assert fp32_base.keys() == bf16_base.keys()
    for name, value in bf16_base.items():
        torch.testing.assert_close(value.float(), fp32_base[name], rtol=0, atol=0, msg=name)


def _fsdp2_lora_step(rank: int, world_size: int, init_file: str, checkpoint: str, out_path: str) -> None:
    """One forward, backward and optimizer step of the LoRA model per base dtype, sharded as the FSDP policy is."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        # FSDPStrategy's default: one mesh dim over all ranks (full shard), bf16 compute, fp32 gradient reduction.
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=["fsdp"])
        fsdp_kwargs = {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            ),
            "reshard_after_forward": True,
        }
        generator = torch.Generator().manual_seed(rank)
        input_ids = torch.randint(0, 128, (2, 16), generator=generator)
        results = {}
        for bf16 in (False, True):
            torch.manual_seed(0)  # the same LoRA initialisation for both base dtypes
            model = _wrap(checkpoint, bf16=bf16).model
            apply_fsdp2(model, fsdp_kwargs, FSDPConfig())
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            grads = {name: p.grad.full_tensor().float() for name, p in model.named_parameters() if p.grad is not None}
            torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
            _, base = _dtypes_of(model)
            results[bf16] = {"grads": grads, "base_dtypes": base}
        if rank == 0:
            torch.save(results, out_path)
    finally:
        dist.destroy_process_group()


def test_fsdp2_gradients_match_fp32_base(tiny_bf16_checkpoint, tmp_path):
    """Under FSDP2 mixed precision the bf16-stored base gives the fp32-stored base's LoRA gradients, bit for bit."""
    out_path = str(tmp_path / "results.pt")
    mp.spawn(
        _fsdp2_lora_step,
        args=(2, str(tmp_path / "pg_init"), tiny_bf16_checkpoint, out_path),
        nprocs=2,
    )
    results = torch.load(out_path)
    fp32_grads, bf16_grads = results[False]["grads"], results[True]["grads"]

    assert results[True]["base_dtypes"] == {torch.bfloat16}
    assert fp32_grads and fp32_grads.keys() == bf16_grads.keys()
    assert all("lora_" in name for name in fp32_grads)
    for name, grad in fp32_grads.items():
        torch.testing.assert_close(bf16_grads[name], grad, rtol=0, atol=0, msg=name)
