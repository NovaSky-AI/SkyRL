"""GPU check for patches/megatron/patch_mhc_recompute_mappings.py: same outputs and grads, less saved memory.

  uv run --isolated --extra megatron python examples/train/deepseek_v4_flash/dev/check_mhc_recompute.py
"""

import torch
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig

from skyrl.backends.skyrl_train.patches.megatron.patch_mhc_recompute_mappings import (
    apply_mhc_recompute_mappings_patch,
)

S, N, C = 4096, 4, 4096


def run(patched: bool):
    torch.manual_seed(0)
    cfg = TransformerConfig(
        num_layers=1, hidden_size=C, num_attention_heads=32, enable_mhc_connections=True, mhc_num_residual_streams=N,
        use_fused_mhc=True, bf16=True, params_dtype=torch.bfloat16,
    )
    module = HyperConnectionModule(cfg, layer_number=1).cuda()
    for p in module.parameters():
        p.data = p.data.float()
    with torch.no_grad():
        module.mapping_proj.weight.normal_(0, 0.02)
        module.bias.normal_(0, 0.1)
    x = torch.randn(S, 1, N * C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    manager = CheckpointWithoutOutputManager()
    agg, h_res, h_post, residual = module._forward_with_checkpoint(x, manager)
    layer_out = agg * 1.5  # stand-in for the wrapped layer, as in megatron-core's mhc recompute
    manager.discard_all_outputs_and_register_unified_recompute(layer_out)
    held = torch.cuda.memory_allocated() - before
    loss = layer_out.float().square().mean() + (h_res.float() * 3).sin().sum() + h_post.float().square().sum()
    loss.backward()
    grads = [x.grad] + [p.grad for p in module.parameters()]
    return held, [t.detach().clone() for t in (layer_out, h_res, h_post)], [g.clone() for g in grads]


base_held, base_out, base_grads = run(False)
assert apply_mhc_recompute_mappings_patch()
new_held, new_out, new_grads = run(True)
print(f"held after forward: {base_held / 2**20:.0f} MiB -> {new_held / 2**20:.0f} MiB (x is {S * N * C * 2 / 2**20:.0f} MiB bf16)")
for name, a, b in zip(["agg", "h_res", "h_post"], base_out, new_out):
    print(f"{name}: max |diff| {(a.float() - b.float()).abs().max().item():.3g}")
for i, (a, b) in enumerate(zip(base_grads, new_grads)):
    print(f"grad {i}: max |diff| {(a.float() - b.float()).abs().max().item():.3g} (max |g| {a.float().abs().max().item():.3g})")
