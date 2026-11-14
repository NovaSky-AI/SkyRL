"""Inkling2 LoRA on the Megatron side: adapter placement and merged export.

LoRA is applied as SkyRL's worker does for ``target_modules="all-linear"`` (the standard targets
plus ``Inkling2ModelProvider.lora_all_linear_extra_targets``) to the synthetic checkpoint of
``synthetic_checkpoint.py``, with every adapter weight randomized so the adapters change the
output. Every attention projection and expert linear must carry an adapter, and the weights
exported with the adapters merged (what full-weight sync sends to an inference engine), loaded
into the reference model, must reproduce the LoRA model's logprobs. The adapter exported unmerged
(what ``merge_lora=false`` syncs) must reproduce the merge, in the HF layout (``w13`` rows
interleaved).

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/inkling2/test_lora.py
"""

import json
import os
import shutil
import tempfile

import pytest
import ray
import torch

from tests.backends.skyrl_train.gpu.gpu_ci.patches.megatron.inkling2.synthetic_checkpoint import (
    UNPADDED_V,
    L,
    write_checkpoint,
)

pytestmark = pytest.mark.megatron

STANDARD_TARGETS = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


@ray.remote(num_gpus=1)
def _lora_merge_parity():
    import torch.distributed as dist
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel
    from safetensors import safe_open
    from safetensors.torch import save_file

    import skyrl.backends.skyrl_train.patches.megatron.inkling2.bridge  # noqa: F401
    import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.reference import (
        load_reference,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)

    stats = {}
    with tempfile.TemporaryDirectory() as path, tempfile.TemporaryDirectory() as merged_path:
        write_checkpoint(path)
        bridge = AutoBridge.from_hf_pretrained(path, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.sequence_parallel = False
        provider.gradient_accumulation_fusion = False
        provider.inkling2_attention_backend = "dense"
        provider.attention_backend = "flash"
        provider.finalize()
        lora = LoRA(
            target_modules=STANDARD_TARGETS + list(provider.lora_all_linear_extra_targets),
            dim=8,
            alpha=16,
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
        )

        def apply_lora(model):
            model = lora(model, training=True)
            lora.set_params_to_save(model)
            return model

        provider.register_pre_wrap_hook(apply_lora)
        model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()

        adapted = {n.split(".adapter.")[0] for n, _ in model.named_parameters() if ".adapter." in n}
        expected = {
            f"decoder.layers.{i}.self_attention.{m}"
            for i in range(L)
            for m in ("linear_q", "linear_k", "linear_v", "linear_r", "linear_gate", "linear_sink", "linear_proj")
        } | {f"decoder.layers.{i}.mlp.experts.{m}" for i in range(L) for m in ("linear_fc1", "linear_fc2")}
        stats["missing_adapters"] = sorted(e for e in expected if not any(a.endswith(e) for a in adapted))

        torch.manual_seed(0)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if ".adapter." in n:
                    p.copy_(0.05 * torch.randn_like(p))
        ids = torch.randint(0, UNPADDED_V, (41,), device="cuda")
        with torch.no_grad():
            lora_lp = torch.log_softmax(model(ids.view(1, -1), None, None)[0].float(), -1)

        merged = dict(bridge.export_hf_weights([model], cpu=True, show_progress=False, merge_adapter_weights=True))
        # The unmerged adapter (what merge_lora=false syncs to vLLM) must reproduce the merge:
        # merged - base == alpha / r * B @ A for every adapted HF weight (per expert for the 3D ones).
        # Unmerged, adapted weights are exported under PEFT names (``....base_layer.weight``).
        base = {
            n.replace(".base_layer.", "."): t
            for n, t in bridge.export_hf_weights([model], cpu=True, show_progress=False, merge_adapter_weights=False)
        }
        adapters = dict(bridge.export_adapter_weights([model], cpu=True, show_progress=False))
        stats["adapter_vs_merge"] = 0.0
        for name, lora_a in adapters.items():
            if not name.endswith(".lora_A.weight"):
                continue
            module = name.removesuffix(".lora_A.weight")
            weight = module if module.endswith("_weight") else f"{module}.weight"
            delta = merged[weight].float() - base[weight].float()
            pred = lora.alpha / lora.dim * (adapters[f"{module}.lora_B.weight"].float() @ lora_a.float())
            err = ((delta - pred).abs().max() / delta.abs().max()).item()
            stats["adapter_vs_merge"] = max(stats["adapter_vs_merge"], err)
        stats["num_adapters"] = sum(n.endswith(".lora_A.weight") for n in adapters)
        # Merging LoRA returns FP32 for the adapted weights; store each in the checkpoint's dtype
        # (BF16 for the dequantized experts), as weight sync casts for the inference engine.
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
            dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
        merged = {k: v.to(torch.float32 if dtypes.get(k) == "F32" else torch.bfloat16) for k, v in merged.items()}
        save_file({k: v.contiguous() for k, v in merged.items()}, os.path.join(merged_path, "model.safetensors"))
        with open(os.path.join(merged_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in merged}}, f)
        shutil.copy(os.path.join(path, "config.json"), merged_path)
        ref = load_reference(merged_path, [torch.device("cuda", 0)])
        with torch.no_grad():
            ref_lp = torch.log_softmax(ref.logits(ref.hidden_states(ids), torch.bfloat16), -1)
            base = load_reference(path, [torch.device("cuda", 0)])
            base_lp = torch.log_softmax(base.logits(base.hidden_states(ids), torch.bfloat16), -1)
        tgt = ids[1:, None]
        d = (lora_lp[:-1].gather(-1, tgt) - ref_lp[:-1].gather(-1, tgt)).abs()
        stats["token_mean"], stats["token_max"] = d.mean().item(), d.max().item()
        stats["lora_effect_mean"] = (lora_lp[:-1].gather(-1, tgt) - base_lp[:-1].gather(-1, tgt)).abs().mean().item()
    dist.destroy_process_group()
    return stats


def test_lora_targets_and_merged_export(ray_init_fixture):
    stats = ray.get(_lora_merge_parity.remote())
    print(stats)
    assert stats["missing_adapters"] == []
    # 7 attention projections + 2 expert linears per layer; relative error of the BF16 merge.
    assert stats["num_adapters"] == 9 * L
    assert stats["adapter_vs_merge"] < 2e-2
    # The adapters move the logprobs well past the parity tolerance, so the merge is checked.
    assert stats["lora_effect_mean"] > 0.05
    assert stats["token_mean"] < 0.006
    assert stats["token_max"] < 0.025
