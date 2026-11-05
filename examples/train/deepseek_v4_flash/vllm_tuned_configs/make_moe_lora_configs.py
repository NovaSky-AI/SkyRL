"""VLLM_TUNED_CONFIG_FOLDER files for the fused MoE-LoRA ops on DeepSeek-V4.1-Flash (384 experts, top-6, rank 32).

The one-shot MoE-LoRA kernel takes only BLOCK_SIZE_M from these configs (it also sets moe_lora_align_block_size's
per-expert padding). Without tuned configs vLLM uses the generic bf16 MoE default (BLOCK_SIZE_M 128 once a step has
more than 512 tokens), but a 640-token decode step routes only ~10 rows to each of 384 experts, so each expert's rows
are padded to 128. Here BLOCK_SIZE_M is the smallest power of two >= 16 that holds the average rows per expert
(tokens * top_k / experts) with 10% headroom, capped at 128; other fields keep vLLM's defaults."""
import json, os, sys
out, gpu = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "NVIDIA_B200"
TOPK, EXPERTS, RANK = 6, 384, 32
def block_m(m):
    avg = m * TOPK / EXPERTS
    for b in (16, 32, 64, 128):
        if avg <= 0.9 * b:
            return b
    return 128
ms = [1, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384, 32768]
for op, k, i in [("fused_moe_lora_w13_shrink", 5120, 384), ("fused_moe_lora_w13_expand", 5120, 384),
                 ("fused_moe_lora_w2_shrink", 5120, 384), ("fused_moe_lora_w2_expand", 5120, 384)]:
    shrink = "shrink" in op
    table = {}
    for slices in ("1", "2"):
        table[slices] = {
            str(m): {str(k): {str(RANK): {str(i): {
                "block_m": block_m(m),
                "block_n": min(64, RANK) if shrink else 64,
                "block_k": 32 if shrink else max(16, min(32, RANK)),
                "num_warps": 4, "num_stages": 3, "group_size_m": 8, "split_k": 1,
            }}}} for m in ms
        }
    with open(os.path.join(out, f"{gpu}_{op.upper()}.json"), "w") as f:
        json.dump({"1": table}, f, indent=1)
print({m: block_m(m) for m in ms})
