"""CPU tests for the ML4 integration (FSDP2 + PEFT LoRA).

ML4's model code is not part of SkyRL: set ``ML4_TRAIN_PATH`` to the ML4 model-code repo
(the one containing ``ml4_train/`` and ``tests/fake_ckpt.py``); the tests skip otherwise.
``flash_attn`` (and, per test, ``vllm``'s weight-transfer base) are stubbed when missing
(CPU-only hosts); the stubs cover only what these tests import.
"""

import os
import sys
import types
from importlib.machinery import ModuleSpec

import pytest
import torch

ML4_TRAIN_PATH = os.environ.get("ML4_TRAIN_PATH", "")
if ML4_TRAIN_PATH:
    sys.path[:0] = [ML4_TRAIN_PATH, os.path.join(ML4_TRAIN_PATH, "tests")]
# The tiny ML4 checkpoints are custom (trust_remote_code) code without a frozen spec; spawned
# workers inherit this. test_remote_code_identity*.py set or clear it per test.
os.environ.setdefault("SKYRL_REMOTE_CODE_SPEC", "none")


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, None)
    mod.__dict__.update(attrs)
    return mod


def install_stubs() -> None:
    """Stub ``flash_attn`` if missing. Idempotent; also called from spawned worker processes."""
    try:
        import flash_attn.bert_padding  # noqa: F401
    except ImportError:
        sys.modules["flash_attn"] = _module("flash_attn")
        sys.modules["flash_attn.bert_padding"] = _module("flash_attn.bert_padding", pad_input=None, unpad_input=None)


@pytest.fixture
def vllm_weight_transfer(monkeypatch):
    """``vllm.distributed.weight_transfer.base`` (stubbed for this test if vLLM is missing).

    Scoped to the test: a stub ``vllm`` left in ``sys.modules`` breaks SkyRL's
    "vLLM not installed" import guards elsewhere.
    """
    try:
        import vllm.distributed.weight_transfer.base  # noqa: F401

        return
    except ImportError:
        pass
    from dataclasses import dataclass

    @dataclass
    class ParamMeta:
        name: str
        dtype: torch.dtype
        shape: tuple

    class WeightSource:
        pass

    def materialize_full_tensor(t):
        return t.full_tensor() if hasattr(t, "full_tensor") else t

    for name in ("vllm", "vllm.distributed", "vllm.distributed.weight_transfer"):
        monkeypatch.setitem(sys.modules, name, _module(name))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.weight_transfer.base",
        _module(
            "vllm.distributed.weight_transfer.base",
            ParamMeta=ParamMeta,
            WeightSource=WeightSource,
            materialize_full_tensor=materialize_full_tensor,
        ),
    )
    monkeypatch.delitem(sys.modules, "skyrl.backends.skyrl_train.weight_sync.sources", raising=False)


install_stubs()


@pytest.fixture(scope="session")
def ml4_model_dir(tmp_path_factory):
    """Fake consolidated checkpoint + the HF-style dir ``ml4_train.convert`` writes for it."""
    if not ML4_TRAIN_PATH:
        pytest.skip("ML4_TRAIN_PATH not set")
    from fake_ckpt import write_fake_ckpt
    from ml4_train.convert import convert

    root = tmp_path_factory.mktemp("ml4")
    ckpt, model_dir = str(root / "ckpt"), str(root / "model")
    os.makedirs(ckpt)
    write_fake_ckpt(ckpt)
    convert(ckpt, model_dir)
    return model_dir


LORA_RANK, LORA_ALPHA, SEED = 4, 8, 42


def build_ml4_wrapper(model_dir: str, experts_shared_outer_loras: bool = False):
    """HFModelWrapper as the FSDP worker builds it with ``sharded_load_fn`` set (base on meta)."""
    from ml4_train.lora import LORA_TARGET_REGEX

    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

    torch.manual_seed(SEED)
    return HFModelWrapper(
        model_dir,
        bf16=False,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_REGEX,
        model_config_kwargs={"attn_implementation": "eager"},
        meta_init=True,
        meta_init_adapters=True,
        experts_shared_outer_loras=experts_shared_outer_loras,
    )


def load_ml4_wrapper(model_dir: str, experts_shared_outer_loras: bool = False):
    """Single-process reference: ``build_ml4_wrapper`` + plain ``load_sharded``, adapters kept.

    Returns ``(wrapper, initial adapter tensors)``.
    """
    from ml4_train.loading import load_sharded

    wrapper = build_ml4_wrapper(model_dir, experts_shared_outer_loras)
    lora_init = {n: p.detach().clone() for n, p in wrapper.model.named_parameters() if "lora_" in n}
    load_sharded(wrapper.model, model_dir)  # to_empty() wipes the adapters; put them back
    with torch.no_grad():
        for n, p in wrapper.model.named_parameters():
            if n in lora_init:
                p.copy_(lora_init[n])
    return wrapper, lora_init
