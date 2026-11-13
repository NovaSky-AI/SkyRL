"""Build an N-layer CI slice of Qwen3.8-Flash-Next (qwen4_exp).

    uv run --isolated --with torch --with safetensors --with numpy python examples/train/qwen3_8_flash_next/make_slice.py \
        <src_dir> <dst_dir> --num-layers 4 [--ple-vocab-base 2048]

Keeps the first ``--num-layers`` text layers (4 covers every layer type: 3 GDN + 1 gated
attention with the QSA indexer, and the PLE layer at ``ple_layer_ids=[2]``), the embedding, LM
head, final hyper-connection mixer and the vision tower; drops MTP. Everything kept is copied as
raw safetensors bytes (no dtype round trip).

The real PLE n-gram table is ~51B parameters (~100 GB). With ``--ple-vocab-base`` it is replaced
by a small random table: ``ngram_vocab_size_base`` (and ``split_ngram_parts``) are rewritten in
the config, the ``ngram_heads_vocab_sizes`` / ``ngram_heads_offsets`` buffers are recomputed with
the HF formula, and the rows are N(0, std of the real table). HF, vLLM and the Megatron bridge all
derive the hashing from those, so they agree on the slice. ``--ple-vocab-base 0`` keeps the real
table (symlinked shard files). ``layer_multipliers`` does not depend on the n-gram vocab size and
is copied unchanged.

Published as ``xguihugging/Qwen3.8-Flash-Next-4layer``; republish with e.g. ``hf upload <org>/Qwen3.8-Flash-Next-4layer <dst_dir>``.
For a local copy (``SKYRL_QWEN38_4LAYER_PATH``), keep "Qwen3.8-Flash-Next" in ``dst_dir``: the GPU tests
key the model's settings on that name.
"""

import argparse
import collections
import json
import math
import os
import re
import shutil
import struct

PLE_TABLE_STD = 0.0075  # std of the released table's rows (shards 0 and 64)

layer_re = re.compile(r"model\.language_model\.layers\.(\d+)\.")


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def write_raw(dst_file, copy, src):
    """Copy the listed tensors' raw bytes from their source shards into one safetensors file."""
    header, entries, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for f, ks in copy.items():
        h, base = read_header(os.path.join(src, f))
        for k in ks:
            lo, hi = h[k]["data_offsets"]
            header[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [offset, offset + hi - lo]}
            entries.append((os.path.join(src, f), base + lo, hi - lo))
            offset += hi - lo
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(dst_file, "wb") as w:
        w.write(struct.pack("<Q", len(hb)) + hb)
        for path, start, length in entries:
            with open(path, "rb") as r:
                r.seek(start)
                while length:
                    chunk = r.read(min(length, 1 << 28))
                    w.write(chunk)
                    length -= len(chunk)
    return [k for k in header if k != "__metadata__"]


# HF modeling_qwen4_exp: per-head n-gram vocab sizes are the (global_head + 1)-th prime after
# ngram_vocab_size_base - 1.
def _is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start, count):
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def small_ple_tensors(text_cfg, prefix, ple_layer_index, vocab_base, parts, seed):
    import torch

    ngram_heads = (text_cfg["ngram_size"] - 1) * text_cfg["heads_per_ngram"]
    head_dim = text_cfg["ple_embed_dim"] // ngram_heads
    sizes, offsets, total = [], [], 0
    for head in range(ngram_heads):
        size = _find_nth_prime_after(vocab_base - 1, ple_layer_index * ngram_heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    divisor = text_cfg["make_ngram_vocab_size_divisible_by"]
    padded = math.ceil(total / divisor) * divisor
    gen = torch.Generator().manual_seed(seed + ple_layer_index)
    table = (torch.randn(padded, head_dim, generator=gen) * PLE_TABLE_STD).to(torch.bfloat16)
    # vLLM's loader slices shard i at i * ceil(padded / parts).
    shard_rows = -(-padded // parts)
    out = {
        f"{prefix}ngram_heads_vocab_sizes": torch.tensor(sizes, dtype=torch.long),
        f"{prefix}ngram_heads_offsets": torch.tensor(offsets, dtype=torch.long),
    }
    for i in range(parts):
        out[f"{prefix}ngram_embedding.shard_{i}.weight"] = table[i * shard_rows : (i + 1) * shard_rows].clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--ple-vocab-base", type=int, default=2048, help="0 keeps the released PLE table")
    p.add_argument("--ple-split-parts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    src, dst, n = args.src, args.dst, args.num_layers
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    t = cfg["text_config"]
    small_ple = args.ple_vocab_base > 0

    def keep(k):
        if k.startswith("mtp."):
            return False
        m = layer_re.match(k)
        if m is not None and int(m.group(1)) >= n:
            return False
        if small_ple and (
            "ple_embedding.ngram_embedding" in k
            or k.endswith("ngram_heads_vocab_sizes")
            or k.endswith("ngram_heads_offsets")
        ):
            return False  # regenerated below
        return True

    files = collections.defaultdict(list)
    for k, f in wm.items():
        files[f].append(k)
    new_wm, copy = {}, collections.defaultdict(list)
    for f, ks in files.items():
        kept = [k for k in ks if keep(k)]
        if not kept:
            continue
        if not small_ple and len(kept) == len(ks) and all("ple.ple_embedding.ngram_embedding" in k for k in ks):
            link = os.path.join(dst, f)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(src, f)), link)
            new_wm.update({k: f for k in kept})
        else:
            copy[f].extend(kept)
    out = "model-slice.safetensors"
    new_wm.update({k: out for k in write_raw(os.path.join(dst, out), copy, src)})

    if small_ple:
        from safetensors.torch import save_file

        ple = {}
        # ple_layer_ids are one-indexed text layers; the HF module name uses the zero-based index.
        for ple_index, layer_id in enumerate(t["ple_layer_ids"]):
            if layer_id - 1 < n:
                prefix = f"model.language_model.layers.{layer_id - 1}.ple.ple_embedding."
                ple.update(
                    small_ple_tensors(t, prefix, ple_index, args.ple_vocab_base, args.ple_split_parts, args.seed)
                )
        save_file(ple, os.path.join(dst, "model-ple.safetensors"), metadata={"format": "pt"})
        new_wm.update({k: "model-ple.safetensors" for k in ple})
        t["ngram_vocab_size_base"] = args.ple_vocab_base
        t["split_ngram_parts"] = args.ple_split_parts

    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": dict(sorted(new_wm.items()))}, f, indent=1)
    t["num_hidden_layers"] = n
    t["layer_types"] = t["layer_types"][:n]
    t["ple_layer_ids"] = [i for i in t["ple_layer_ids"] if i <= n]
    t["mtp_num_hidden_layers"] = 0
    t.pop("mtp", None)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for f in os.listdir(src):
        if not f.endswith(".safetensors") and f not in ("config.json", "model.safetensors.index.json"):
            if not f.startswith("."):
                p_ = os.path.join(dst, f)
                if not os.path.lexists(p_):
                    (shutil.copytree if os.path.isdir(os.path.join(src, f)) else shutil.copy)(os.path.join(src, f), p_)
    print(f"{len(new_wm)} tensors -> {dst}")


if __name__ == "__main__":
    main()
