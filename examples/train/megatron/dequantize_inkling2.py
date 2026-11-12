"""Dequantize the Inkling2 release checkpoint (NVFP4 / MXFP8 routed experts) to BF16.

Training does not need this: Megatron and vLLM both dequantize the release's experts as they load.
It builds BF16 copies and the truncated slices the GPU tests use (``--num-layers`` /
``--num-experts``). Only ``model.llm.layers.N.mlp.experts.{w13,w2}_weight`` are quantized in the release; they
are dequantized with ``skyrl.backends.skyrl_train.patches.megatron.inkling2.quant`` and their
scale tensors dropped. Every other tensor (including the FP32 attn-res queries and router bias)
is copied unchanged, as are the tokenizer and preprocessing files; ``chat_template.jinja`` is
written from SkyRL's ``inkling2`` template and ``tokenizer_config.json`` gets an ``eos_token`` (the
release has neither). Tensor names and layouts are
kept (``w13`` stays gate/up row-interleaved), and ``hf_quant_config.json`` is not copied.

``--num-layers N`` writes the first N trunk layers only (a prefix keeps the attn-res block
boundaries and the local/global attention pattern of the full model): ``num_hidden_layers`` and
``local_layer_ids`` are rewritten, the attn-res queries of the dropped layers are removed, and
the MTP draft (which reads hidden states of layers up to 48) is left out.

``--layers 0,5`` instead keeps an arbitrary ordered list of trunk layers, renumbered 0, 1, ...
(with their attn-res queries and local/global type). Like ``--num-experts``, this builds a
different, smaller model for parity tests, e.g. a shallow model that contains a global layer.

``--num-experts E`` keeps only the first E routed experts of every layer (expert weights, router
rows and router bias). The result is a smaller model with the same layer structure, for
fast parity tests and CI slices; it is not a pruned version of the full model.

With ``--verify`` every NVFP4 tensor is re-quantized with its stored scales and must reproduce
the stored codes exactly (``nvfp4_requantization_mismatches``).

Output is one safetensors file per trunk layer plus one for everything else. Example (8-layer
slice, about 200 GB):

    uv run --isolated --extra megatron examples/train/megatron/dequantize_inkling2.py \
        --input /path/to/Inkling-2-Small \
        --output /path/to/inkling2-8layer-bf16 \
        --num-layers 8 --verify
"""

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.patches.megatron.inkling2.configuration import (
    release_config,
)
from skyrl.backends.skyrl_train.patches.megatron.inkling2.quant import (
    dequantize_expert_weight,
    is_expert_scale_tensor,
    nvfp4_requantization_mismatches,
)
from skyrl.train.generators.utils import CUSTOM_CHAT_TEMPLATES

_LAYER_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.")
_ATTN_QUERY_RE = re.compile(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$")
_MLP_QUERY_RE = re.compile(r"^model\.llm\.attn_res\.mlp_queries\.(\d+)$")
_EXPERT_WEIGHT_RE = re.compile(r"\.mlp\.experts\.w(13|2)_weight$")
_COPY_FILES = (
    "README.md",
    "image_preprocessing.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_COPY_DIRS = ("tiktoken",)
EOS_TOKEN = "<|content_model_end_sampling|>"
# Experts dequantized per GPU call; bounds the FP32 intermediates of a 384 x 4608 x 4608 w13.
_EXPERT_CHUNK = 32
# Tensors with a leading routed-expert axis (sliced by ``--num-experts``).
_EXPERT_AXIS_RE = re.compile(r"\.mlp\.(experts\.w(13|2)_weight(_scale|_amax)?|gate\.(weight|bias))$")


def rename(name: str, layer_map: dict):
    """Output name of checkpoint tensor ``name`` when keeping source layers ``layer_map``
    (``{source index: output index}``), or None if it is dropped. Scale tensors are dropped:
    they are consumed with their weight."""
    if is_expert_scale_tensor(name):
        return None
    if m := _LAYER_RE.match(name):
        src = int(m.group(1))
        return None if src not in layer_map else f"model.llm.layers.{layer_map[src]}." + name[m.end() :]
    # attn_queries.i feeds the attention of layer i + 1 (layer 0 reads the embedding directly).
    if m := _ATTN_QUERY_RE.match(name):
        dst = layer_map.get(int(m.group(1)) + 1)
        return None if not dst else f"model.llm.attn_res.attn_queries.{dst - 1}"
    if m := _MLP_QUERY_RE.match(name):
        dst = layer_map.get(int(m.group(1)))
        return None if dst is None else f"model.llm.attn_res.mlp_queries.{dst}"
    return name


def output_file(name: str) -> str:
    if m := _LAYER_RE.match(name):
        return f"model-layer-{int(m.group(1)):02d}.safetensors"
    return "model-other.safetensors"


def dequantize_chunked(name: str, tensors: dict, device: torch.device, verify: bool):
    """Dequantize one expert weight, ``_EXPERT_CHUNK`` experts at a time on ``device``."""
    weight = tensors[name]
    parts, mismatches = [], 0
    for start in range(0, weight.shape[0], _EXPERT_CHUNK):
        sl = slice(start, start + _EXPERT_CHUNK)
        chunk = {name: weight[sl].to(device)}
        for suffix in ("_scale", "_amax"):
            if name + suffix in tensors:
                chunk[name + suffix] = tensors[name + suffix][sl].to(device)
        if name + "_outer_scale_denominator" in tensors:
            chunk[name + "_outer_scale_denominator"] = tensors[name + "_outer_scale_denominator"].to(device)
        dequantized = dequantize_expert_weight(name, chunk)
        if verify and weight.dtype == torch.uint8:
            mismatches += nvfp4_requantization_mismatches(
                dequantized,
                chunk[name],
                chunk[name + "_scale"],
                chunk[name + "_amax"],
                chunk[name + "_outer_scale_denominator"],
            )
        parts.append(dequantized.cpu())
    return torch.cat(parts), mismatches


def convert_file(job):
    """Write one output file. ``sources`` maps every tensor to load (including the scale tensors
    of quantized expert weights, which are consumed and not written) to its input file, and
    ``names`` maps each written tensor to its output name."""
    out_name, sources, names, input_dir, output_dir, verify, num_experts, gpu = job
    device = torch.device("cuda", gpu) if torch.cuda.is_available() else torch.device("cpu")
    by_src = defaultdict(list)
    for tensor, src in sources.items():
        by_src[src].append(tensor)
    tensors = {}
    for src, src_names in by_src.items():
        with safe_open(Path(input_dir) / src, framework="pt") as f:
            for n in src_names:
                tensors[n] = f.get_tensor(n)
                if num_experts is not None and _EXPERT_AXIS_RE.search(n):
                    tensors[n] = tensors[n][:num_experts].contiguous()
    out, report = {}, []
    for n in sorted(sources):
        if is_expert_scale_tensor(n):
            continue
        if not _EXPERT_WEIGHT_RE.search(n):
            out[names[n]] = tensors[n]
            continue
        src_dtype = tensors[n].dtype
        out[names[n]], mismatches = dequantize_chunked(n, tensors, device, verify)
        report.append((n, str(src_dtype), mismatches))
    save_file(out, str(Path(output_dir) / out_name), metadata={"format": "pt"})
    return out_name, {n: t.numel() * t.element_size() for n, t in out.items()}, report


def set_eos_token(model_dir: Path) -> None:
    """Name the end-of-turn token in ``tokenizer_config.json``: the release sets only
    ``eos_token_id`` (``<|content_model_end_sampling|>``) in ``config.json``."""
    path = model_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(path.read_text())
    tokenizer_config.setdefault("eos_token", EOS_TOKEN)
    path.write_text(json.dumps(tokenizer_config, indent=2))


def rewrite_config(config: dict, layers: list, num_experts) -> dict:
    config = json.loads(json.dumps(config))
    text = config["text_config"]
    is_full = layers == list(range(text["num_hidden_layers"]))
    if num_experts is not None:
        text["n_routed_experts"] = num_experts
    if not is_full:
        local = set(text["local_layer_ids"])
        text["num_hidden_layers"] = len(layers)
        text["local_layer_ids"] = [dst for dst, src in enumerate(layers) if src in local]
    if not is_full or num_experts is not None:
        config.pop("mtp_config", None)
    # A pre-release checkpoint's config gets the release's names (the vLLM build reads only those).
    return release_config(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--num-layers", type=int, default=None, help="keep only the first N trunk layers")
    layer_group.add_argument("--layers", type=str, default=None, help="comma-separated source layers to keep, in order")
    parser.add_argument("--num-experts", type=int, default=None, help="keep only the first E routed experts")
    parser.add_argument("--verify", action="store_true", help="re-quantize NVFP4 tensors and require exact codes")
    parser.add_argument("--workers", type=int, default=None, help="parallel files (default: number of GPUs)")
    args = parser.parse_args()

    config = json.loads((args.input / "config.json").read_text())
    total_layers = config["text_config"]["num_hidden_layers"]
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(args.num_layers or total_layers))
    if not layers or len(set(layers)) != len(layers) or not all(0 <= i < total_layers for i in layers):
        raise ValueError(f"layers must be distinct indices in [0, {total_layers})")
    layer_map = {src: dst for dst, src in enumerate(layers)}
    total_experts = config["text_config"]["n_routed_experts"]
    if (
        args.num_experts is not None
        and not config["text_config"]["num_experts_per_tok"] <= args.num_experts <= total_experts
    ):
        raise ValueError(f"--num-experts must be in [num_experts_per_tok, {total_experts}]")

    weight_map = json.loads((args.input / "model.safetensors.index.json").read_text())["weight_map"]
    files, names = defaultdict(dict), defaultdict(dict)
    for name, src in weight_map.items():
        new_name = rename(name, layer_map)
        if new_name is None:
            continue
        out_file = output_file(new_name)
        files[out_file][name] = src
        names[out_file][name] = new_name
        if _EXPERT_WEIGHT_RE.search(name):
            for suffix in ("_scale", "_amax", "_outer_scale_denominator"):
                if name + suffix in weight_map:
                    files[out_file][name + suffix] = weight_map[name + suffix]
    args.output.mkdir(parents=True, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    workers = args.workers or max(num_gpus, 1)
    jobs = [
        (
            out_name,
            sources,
            names[out_name],
            str(args.input),
            str(args.output),
            args.verify,
            args.num_experts,
            i % max(num_gpus, 1),
        )
        for i, (out_name, sources) in enumerate(sorted(files.items()))
    ]

    start = time.time()
    new_map, total_size, failures = {}, 0, []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for out_name, sizes, report in pool.imap_unordered(convert_file, jobs):
            for n, size in sizes.items():
                new_map[n] = out_name
                total_size += size
            failures += [(n, mismatches) for n, _, mismatches in report if mismatches]
            print(f"[{time.time() - start:7.1f}s] wrote {out_name} ({len(sizes)} tensors)", flush=True)

    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(new_map.items()))}
    (args.output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (args.output / "config.json").write_text(json.dumps(rewrite_config(config, layers, args.num_experts), indent=2))
    for name in _COPY_FILES:
        if (args.input / name).exists():
            shutil.copy2(args.input / name, args.output / name)
    # The checkpoint ships no chat template; write the one SkyRL renders Inkling2 chats with.
    (args.output / "chat_template.jinja").write_text(CUSTOM_CHAT_TEMPLATES["inkling2"])
    set_eos_token(args.output)
    for name in _COPY_DIRS:
        if (args.input / name).is_dir():
            shutil.copytree(args.input / name, args.output / name, dirs_exist_ok=True)
    if "mtp_config" in rewrite_config(config, layers, args.num_experts) and (args.input / "mtp.safetensors").exists():
        shutil.copy2(args.input / "mtp.safetensors", args.output / "mtp.safetensors")

    print(f"done in {time.time() - start:.1f}s: {len(new_map)} tensors, {total_size / 1e9:.1f} GB")
    if failures:
        for n, m in failures:
            print(f"VERIFY FAILED: {n}: {m} codes differ")
        raise SystemExit(1)
    if args.verify:
        print("verify: every NVFP4 code reproduced by re-quantization")


if __name__ == "__main__":
    main()
