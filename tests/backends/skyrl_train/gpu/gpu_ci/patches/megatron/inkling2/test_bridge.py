"""Inkling2 Megatron-Bridge import / export on a synthetic checkpoint in the release layout.

The checkpoint (``synthetic_checkpoint.py``) has one layer group per expert storage format of the
release (BF16, NVFP4, MXFP8). Loading it through
``AutoBridge`` must give a model whose logprobs match the reference model loaded from the same
directory, and exporting must reproduce every tensor (expert weights dequantized to BF16). The looped variant adds the loop's
controllers and cross-pass queries; with its ``loop_config`` deleted from the config the load must fail (the
checkpoint has tensors no mapping reads) instead of silently building an unlooped model.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/inkling2/test_bridge.py
"""

import json
import os
import tempfile

import pytest
import ray
import torch

from tests.backends.skyrl_train.gpu.gpu_ci.patches.megatron.inkling2.synthetic_checkpoint import (
    UNPADDED_V,
    write_checkpoint,
)

pytestmark = pytest.mark.megatron


@ray.remote(num_gpus=1)
def _bridge_roundtrip(looped: bool = False, drop_loop_config: bool = False):
    import torch.distributed as dist
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel
    from safetensors import safe_open

    import skyrl.backends.skyrl_train.patches.megatron.inkling2.bridge  # noqa: F401
    import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.quant import (
        dequantize_expert_weight,
    )
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.reference import (
        load_reference,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29535")
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)

    stats = {}
    with tempfile.TemporaryDirectory() as path:
        write_checkpoint(path, looped=looped)
        if drop_loop_config:
            config_path = os.path.join(path, "config.json")
            config = json.load(open(config_path))
            del config["text_config"]["loop_config"]
            config["model_type"] = "inkling2"  # a plain checkpoint, with the loop tensors left in
            json.dump(config, open(config_path, "w"))
        bridge = AutoBridge.from_hf_pretrained(path, trust_remote_code=True)
        try:
            provider = bridge.to_megatron_provider(load_weights=True)
            provider.sequence_parallel = False
            provider.gradient_accumulation_fusion = False
            provider.inkling2_attention_backend = "dense"
            provider.attention_backend = "flash"
            provider.finalize()
            model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()
        except ValueError as e:
            dist.destroy_process_group()
            return {"load_error": str(e)}
        ref = load_reference(path, [torch.device("cuda", 0)])

        torch.manual_seed(0)
        ids = torch.randint(0, UNPADDED_V, (41,), device="cuda")
        with torch.no_grad():
            meg = torch.log_softmax(model(ids.view(1, -1), None, None)[0].float(), -1)
            ref_lp = torch.log_softmax(ref.logits(ref.hidden_states(ids), torch.bfloat16), -1)
        tgt = ids[1:, None]
        d = (meg[:-1].gather(-1, tgt) - ref_lp[:-1].gather(-1, tgt)).abs()
        stats["token_mean"], stats["token_max"] = d.mean().item(), d.max().item()

        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
            original = {k: f.get_tensor(k) for k in f.keys()}
        exported = dict(bridge.export_hf_weights([model], cpu=True, show_progress=False))
        expected_names = {k for k in original if not k.endswith(("_scale", "_amax", "_outer_scale_denominator"))}
        stats["missing"] = sorted(expected_names - exported.keys())
        stats["unexpected"] = sorted(exported.keys() - expected_names)
        mismatched = []
        for name in sorted(expected_names & exported.keys()):
            want = original[name]
            if name.endswith(("w13_weight", "w2_weight")):
                want = dequantize_expert_weight(name, {k: v for k, v in original.items() if k.startswith(name)})
            if not torch.equal(exported[name].float(), want.float()):
                mismatched.append(name)
        stats["mismatched"] = mismatched
    dist.destroy_process_group()
    return stats


@pytest.mark.parametrize("looped", [False, True], ids=["unlooped", "looped"])
def test_bridge_import_export(ray_init_fixture, looped):
    stats = ray.get(_bridge_roundtrip.remote(looped))
    print(stats)
    assert "load_error" not in stats, stats
    # A single-site bug (see test_model_vs_reference.py) raises the mean error to >= 0.013 (>= 0.07 in the loop). The
    # bridge's model keeps FP32 params here against the BF16-rounding reference: 24 layer runs drift to ~0.011.
    assert stats["token_mean"] < (0.02 if looped else 0.006)
    assert stats["token_max"] < (0.08 if looped else 0.025)
    assert stats["missing"] == [] and stats["unexpected"] == []
    assert stats["mismatched"] == []


def test_bridge_rejects_unmapped_tensors(ray_init_fixture):
    stats = ray.get(_bridge_roundtrip.remote(looped=True, drop_loop_config=True))
    print(stats)
    assert "no Megatron counterpart" in stats.get("load_error", ""), stats
