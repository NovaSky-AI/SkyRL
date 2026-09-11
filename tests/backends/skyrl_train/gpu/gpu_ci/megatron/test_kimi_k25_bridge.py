"""GPU integration tests for Kimi K2.5-family (``KimiK25ForConditionalGeneration``) support.

Covers the pieces the logprob-parity row in ``test_megatron_models.py`` cannot:
its LoRA adapter is zero-initialized, so parity is blind to what the exported
adapter actually contains.

- vLLM's ``SupportsLoRA`` protocol gate on the patched wrapper (no checkpoint
  needed, so this one runs anywhere vLLM is installed)
- the ``merge_lora=false`` PEFT export layout vLLM hot-loads: per-expert keys,
  flat 2D tensors, and an adapter that points at the INT4 release

Bridge dispatch and provider construction are not asserted here: the
``kimi-k2.5-2layer-int4-qat`` row in ``test_megatron_models.py`` covers them
harder, by requiring the resulting forward to match what vLLM serves.

The worker test loads the 2-layer slice of the real checkpoint (~41 GB of BF16
masters on one GPU), so it carries ``pytest.mark.h100``. Override the defaults
to run it against a full checkpoint:

- ``SKYRL_TEST_KIMI_MODEL`` -- INT4 release the inference engine serves
- ``SKYRL_TEST_KIMI_BF16``  -- BF16 masters the trainer loads, from
  ``examples/train/megatron/dequantize_compressed_tensors_int4.py``

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s -m h100 tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_kimi_k25_bridge.py
"""

import json
import os
import tempfile

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

# 2-layer slices of moonshotai/Kimi-K2.5: layer 0 dense + layer 1 with all 384
# routed experts, the INT4 release and its dequantized masters.
KIMI_MODEL = os.environ.get("SKYRL_TEST_KIMI_MODEL", "eatang/Kimi-K2.5-2layer")
KIMI_BF16 = os.environ.get("SKYRL_TEST_KIMI_BF16", "eatang/Kimi-K2.5-2layer-BF16")


class _NullInferenceClient:
    """Stands in for the inference engine client in the adapter-export test."""

    async def update_named_weights(self, request):
        return None


@pytest.mark.megatron
def test_kimi_k25_vllm_lora_patch():
    """The patched wrapper must satisfy vLLM's *instance-level* SupportsLoRA
    protocol check (which requires every protocol member, not just the three
    documented attributes)."""
    pytest.importorskip("vllm")
    from typing_extensions import get_protocol_members
    from vllm.model_executor.models.interfaces import SupportsLoRA, supports_lora
    from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration

    from skyrl.backends.skyrl_train.patches.vllm_kimi_k25_lora import (
        apply_kimi_k25_lora_patch,
    )

    apply_kimi_k25_lora_patch()

    missing = [m for m in get_protocol_members(SupportsLoRA) if not hasattr(KimiK25ForConditionalGeneration, m)]
    assert not missing, f"missing SupportsLoRA protocol members: {missing}"
    assert supports_lora(KimiK25ForConditionalGeneration)
    # The worker-side gate checks the *instance*; bypass __init__ to test it.
    instance = object.__new__(KimiK25ForConditionalGeneration)
    assert isinstance(instance, SupportsLoRA), "instance-level protocol check failed"
    assert supports_lora(instance)
    assert not KimiK25ForConditionalGeneration.is_3d_moe_weight  # per-expert LoRA keys


def _kimi_worker_cfg(lora_sync_path: str) -> SkyRLTrainConfig:
    """Production Kimi settings: LoRA + fake-INT4 QAT + language_model_only."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = KIMI_MODEL
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False  # forward-only test, no inference engine
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 1

    cfg.trainer.policy.language_model_only = True
    cfg.trainer.ref.language_model_only = True
    cfg.generator.inference_engine.language_model_only = True
    # MLA + sample packing needs TE's cuDNN fused attention (flash_attn=True
    # would export NVTE_FUSED_ATTN=0; see test_megatron_models MLA handling).
    cfg.trainer.flash_attn = False

    lora = cfg.trainer.policy.model.lora
    lora.rank = 8
    lora.alpha = 16
    lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
    lora.lora_sync_path = lora_sync_path
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False
    # Capacity-normalized expert LoRA, as the Kimi example uses (expert rank =
    # rank/topk; keeps the per-expert PEFT export small for 384 experts).
    cfg.trainer.policy.megatron_config.lora_config.normalize_moe_lora = True

    fq = cfg.trainer.policy.model.fake_int4_qat
    fq.enabled = True
    fq.group_size = 32
    fq.scale_divisor = 7.0  # Kimi QAT convention
    fq.q_min = -7.0
    fq.bf16_base_path = KIMI_BF16

    mcfg = cfg.trainer.policy.megatron_config
    if mcfg.transformer_config_kwargs is None:
        mcfg.transformer_config_kwargs = {}
    # A no-op on the 2-layer slice, but keeps the test runnable against a full
    # 61-layer checkpoint. moe_layer_freq is a per-layer list on
    # DeepSeek-V3-family providers, so it must be truncated alongside.
    mcfg.transformer_config_kwargs["num_layers"] = 2
    mcfg.transformer_config_kwargs["moe_layer_freq"] = [0, 1]

    validate_cfg(cfg)
    return cfg


@pytest.mark.asyncio
@pytest.mark.megatron
@pytest.mark.h100
async def test_kimi_worker_forward_and_lora_export():
    """End-to-end through the real worker: BF16-master load, fake-INT4 forward,
    and the merge_lora=false PEFT adapter export that vLLM hot-loads."""
    from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
        get_test_training_batch,
    )

    lora_sync_path = tempfile.mkdtemp(prefix="skyrl_kimi_lora_sync_")
    cfg = _kimi_worker_cfg(lora_sync_path)
    batch = get_test_training_batch(4)

    # MLA + sample packing needs TE cuDNN fused attention (the gpu_ci conftest
    # globally disables it; re-enable like the other MLA model tests).
    with ray_init(extra_env_vars={"NVTE_FUSED_ATTN": "1"}):
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
        output = WorkerOutput.cat(actor_group.actor_infos, ray.get(refs))
        logprobs = loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")
        assert torch.isfinite(logprobs).all(), "non-finite logprobs from the truncated Kimi forward"

        refs = actor_group.async_run_ray_method(
            "pass_through", "_save_lora_adapters_and_sync", lora_sync_path, _NullInferenceClient()
        )
        ray.get(refs)

    from safetensors.torch import load_file

    adapter_state = load_file(os.path.join(lora_sync_path, "adapter_model.safetensors"))
    with open(os.path.join(lora_sync_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)

    keys = sorted(adapter_state.keys())
    assert keys, "empty adapter export"
    prefix = "base_model.model.language_model.model."
    assert all(k.startswith(prefix) for k in keys), f"unexpected key roots: {[k for k in keys[:5]]}"
    assert all(t.ndim == 2 for t in adapter_state.values()), "vLLM PEFT layout must be flat 2D"

    # Attention-out + dense MLP (layer 0) + shared experts + routed experts (layer 1).
    assert any(".self_attn.o_proj.lora_A.weight" in k for k in keys)
    assert any(".layers.0.mlp." in k for k in keys)
    assert any(".mlp.shared_experts." in k for k in keys)
    # Routed experts export per-expert HF-style keys (experts.<idx>.<proj>), the
    # format vLLM's LoRA loader consumes natively (EP-aware; see vllm/lora/lora_model.py).
    num_experts = 384
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expert_keys = [k for k in keys if ".layers.1.mlp.experts." in k and f".{proj}.lora_A.weight" in k]
        assert (
            len(expert_keys) == num_experts
        ), f"{proj}: expected {num_experts} expert LoRA keys, got {len(expert_keys)}"
    assert any(".layers.1.mlp.experts.0.gate_proj.lora_A.weight" in k for k in keys)

    # The adapter must reference the INT4 release (what vLLM serves), not the BF16 masters.
    assert adapter_config["base_model_name_or_path"] == KIMI_MODEL
    assert adapter_config["r"] == 8
    assert adapter_config["target_modules"], "no target_modules inferred"
