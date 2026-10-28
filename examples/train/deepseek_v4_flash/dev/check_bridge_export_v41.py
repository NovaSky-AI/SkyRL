"""Build a DeepSeek-V4.1 slice through skyrl's DeepSeekV41Bridge on one GPU, then list what the bridge's export
yields versus the HF checkpoint's tensor names (what weight sync would send to vLLM).

Usage: torchrun --nproc-per-node 1 check_bridge_export_v41.py <slice_dir>
"""

import json
import os
import sys

import torch


def main():
    slice_dir = sys.argv[1]
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(0)
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state

    import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401  registers the V4.1 bridge

    parallel_state.initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(42)
    bridge = AutoBridge.from_hf_pretrained(slice_dir, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.mtp_num_layers = None
    provider.moe_token_dispatcher_type = "alltoall"  # as SkyRL configures it (HybridEP is not installed)
    provider.moe_shared_expert_overlap = False
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)

    with open(os.path.join(slice_dir, "model.safetensors.index.json")) as f:
        hf_names = set(json.load(f)["weight_map"])
    exported = {}
    for name, tensor in bridge.export_hf_weights(model, show_progress=False):
        exported[name] = (tuple(tensor.shape), tensor.dtype)
    missing = sorted(n for n in hf_names - set(exported) if ".engram.embed." not in n)
    extra = sorted(set(exported) - hf_names)
    print(f"exported {len(exported)} tensors; hf has {len(hf_names)}")
    print("hf tensors never exported (excluding engram tables):", missing[:40], len(missing))
    print("exported names not in hf:", extra[:20], len(extra))
    sinks = {n: v for n, v in exported.items() if "attn_sink" in n}
    print("exported sinks:", sinks)
    for module_name, module in model[0].named_modules():
        if hasattr(module, "attn_sink") and module.attn_sink is not None:
            print(module_name, "attn_sink[:4] =", module.attn_sink.detach().float()[:4].tolist())
    from safetensors import safe_open

    with open(os.path.join(slice_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    key = "layers.0.attn.attn_sink"
    with safe_open(os.path.join(slice_dir, weight_map[key]), framework="pt") as f:
        print("checkpoint", key, "[:4] =", f.get_tensor(key)[:4].tolist())


if __name__ == "__main__":
    main()
