"""Score a dumped roundtrip batch with DeepSeek's reference V4.1 implementation (``inference/model.py``).

``test_megatron_models.py`` writes the batch it compared (token ids, attention / response masks, vLLM and
Megatron logprobs) when ``SKYRL_TEST_DUMP_LOGPROBS`` is set. This recomputes the logprobs with the reference
model on the same slice and prints the pairwise gaps, which tells whether Megatron or vLLM is the one off.

The reference runs as released except: BF16 weights (``dtype=bf16``, no expert quantization), no vision, no
DSpark MTP, and full-sequence logits.

Usage (one 8-GPU node):
  torchrun --nproc-per-node 8 ref_logprobs_v41.py <slice_dir> <dump.pt> [--ref-dir <checkpoint>/inference]
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

# Dims the reference shards over model parallel ranks (from inference/convert.py).
SHARD_DIM = {"embed": 0, "wq_b": 0, "wo_a": 0, "wo_b": 1, "head": 0, "attn_sink": 0, "weights_proj": 0}


def model_args(ModelArgs, tc):
    rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=4096,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=tc["vocab_size"],
        dim=tc["hidden_size"],
        moe_inter_dim=tc["moe_intermediate_size"],
        n_layers=tc["num_hidden_layers"],
        n_mtp_layers=0,
        n_heads=tc["num_attention_heads"],
        n_routed_experts=tc["n_routed_experts"],
        n_shared_experts=tc["n_shared_experts"],
        n_activated_experts=tc["num_experts_per_tok"],
        score_func=tc["scoring_func"],
        route_scale=tc["routed_scaling_factor"],
        swiglu_limit=tc["swiglu_limit"],
        q_lora_rank=tc["q_lora_rank"],
        head_dim=tc["head_dim"],
        rope_head_dim=tc["qk_rope_head_dim"],
        norm_eps=tc["rms_norm_eps"],
        o_groups=tc["o_groups"],
        o_lora_rank=tc["o_lora_rank"],
        window_size=tc["sliding_window"],
        compress_ratios=tuple(tc["compress_ratios"][: tc["num_hidden_layers"]]),
        kv_source_layers=tuple(tc["kv_source_layer_ids"]),
        index_source_layers=tuple(tc["index_source_layer_ids"]),
        compress_rope_theta=tc["compress_rope_theta"],
        original_seq_len=rope["original_max_position_embeddings"],
        rope_theta=tc["rope_theta"],
        rope_factor=rope["factor"],
        beta_fast=rope.get("beta_fast", 32),
        beta_slow=rope.get("beta_slow", 1),
        index_n_heads=tc["index_n_heads"],
        index_head_dim=tc["index_head_dim"],
        index_topk=tc["index_topk"],
        candidate_source_layer=tc["candidate_source_layer_id"],
        candidate_topk_blocks=tc["candidate_topk_blocks"],
        candidate_block_size=tc["candidate_block_size"],
        hc_mult=tc["hc_mult"],
        hc_sinkhorn_iters=tc["hc_sinkhorn_iters"],
        hc_eps=tc["hc_eps"],
        engram_layer_ids=tuple(tc["engram_layer_ids"]),
        engram_num_embeddings=tuple(tc["engram_num_embeddings"]),
        engram_max_ngram_size=tc["engram_max_ngram_size"],
        engram_vocab_size=tc["engram_vocab_size"],
        engram_n_heads=tc["engram_n_heads"],
        engram_head_dim=tc["engram_head_dim"],
        engram_pad_id=tc["engram_pad_token_id"],
        engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
        vision_n_layers=0,
        dspark_block_size=0,
    )


def load_shard(model, slice_dir, rank, world, n_experts):
    """inference/convert.py's sharding, read straight from the slice (BF16, so no requantization)."""
    from safetensors import safe_open

    with open(os.path.join(slice_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_file = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, []).append(name)
    local_experts = n_experts // world
    state = {}
    for fname, names in by_file.items():
        with safe_open(os.path.join(slice_dir, fname), framework="pt") as f:
            for name in names:
                if ".experts." in name and "shared_experts" not in name:
                    idx = int(name.split(".experts.")[1].split(".")[0])
                    if not rank * local_experts <= idx < (rank + 1) * local_experts:
                        continue
                    state[name] = f.get_tensor(name)
                elif ".engram.embed." in name:
                    rows = f.get_slice(name).get_shape()[0]
                    part = (rows + world - 1) // world
                    t = f.get_slice(name)[rank * part : min((rank + 1) * part, rows)]
                    if t.size(0) < part:
                        pad = t.new_full((part - t.size(0), t.size(1)), 1 if name.endswith(".scale") else 0)
                        t = torch.cat([t, pad])
                    state[name] = t
                else:
                    t = f.get_tensor(name)
                    key = name.split(".")[-1] if name.split(".")[-1] in SHARD_DIM else name.split(".")[-2]
                    dim = SHARD_DIM.get(key)
                    if dim is not None:
                        size = t.size(dim) // world
                        t = t.narrow(dim, rank * size, size).contiguous()
                    state[name] = t
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:10]
    assert not missing, missing[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slice_dir")
    parser.add_argument("dump")
    parser.add_argument(
        "--ref-dir", default="/mnt/local_storage/avigyabb/models/DeepSeek-V4.1-Flash-1017/inference"
    )
    args = parser.parse_args()

    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_dtype(torch.bfloat16)
    sys.path.insert(0, args.ref_dir)
    from model import ModelArgs, ParallelHead, Transformer
    from transformers import AutoTokenizer

    with open(os.path.join(args.slice_dir, "config.json")) as f:
        tc = json.load(f)["text_config"]
    margs = model_args(ModelArgs, tc)
    tokenizer = AutoTokenizer.from_pretrained(args.slice_dir)
    with torch.device("cuda"):
        model = Transformer(margs, tokenizer)
    load_shard(model, args.slice_dir, rank, world, margs.n_routed_experts)
    model.head.forward = lambda x, full_logits=True: ParallelHead.forward(model.head, x, True)
    torch.set_default_device("cuda")

    dump = torch.load(args.dump)
    seqs, attn = dump["sequences"], dump["attention_mask"].bool()
    resp_mask = dump["response_mask"].bool()
    n_resp = resp_mask.size(1)
    ref = torch.zeros_like(dump["vllm"], dtype=torch.float32)
    for b in range(seqs.size(0)):
        cols = attn[b].nonzero().squeeze(-1)
        tokens = seqs[b, cols].unsqueeze(0).cuda()
        _, logits, _ = model(tokens, 0)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        full = torch.zeros(seqs.size(1), dtype=torch.float32, device="cpu")
        full[cols[1:]] = logp[:-1].gather(-1, tokens[0, 1:, None]).squeeze(-1).cpu()
        ref[b] = full[-n_resp:]
    if rank == 0:
        m = resp_mask
        v, g = dump["vllm"].float(), dump["megatron"].float()
        print(f"tokens compared: {int(m.sum())}")
        print(f"mean logprob   ref {ref[m].mean():.4f}  vllm {v[m].mean():.4f}  megatron {g[m].mean():.4f}")
        print(f"|ref - vllm|      {(ref - v)[m].abs().mean():.4f}")
        print(f"|ref - megatron|  {(ref - g)[m].abs().mean():.4f}")
        print(f"|vllm - megatron| {(v - g)[m].abs().mean():.4f}")
        torch.save({"ref": ref}, args.dump + ".ref.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
