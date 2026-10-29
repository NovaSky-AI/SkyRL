"""Consolidated Mistral checkpoint -> ML4 training model.

Name map: vLLM's ``_CHECKPOINT_NAME_MAP`` / ``_STACKED_PARAMS`` under the
Pixtral ``language_model.`` prefix
(vllm/models/mistral_large4/nvidia/model.py:197-226), except that MLP
weights are not stacked: checkpoint ``w1``/``w3``/``w2`` of dense layers,
shared experts and each routed expert load into separate ``gate_proj``/
``up_proj``/``down_proj`` modules (vLLM's LoRA module names; routed experts
stay per expert instead of vLLM's fused w13/w2 parameters).

Two checkpoint formats:

* FP8 (``quantization`` block FP8): ``F8_E4M3`` weight + ``qscale_weight``
  [ceil(out/128), ceil(in/128)] FP32.
* ModelOpt MIXED_PRECISION (``quantization_config``):
  - FP8_PB_WO linears: ``F8_E4M3`` weight + ``weight_scale`` FP32
    [ceil(out/128), 1, ceil(in/128), 1];
  - NVFP4 routed experts: ``weight`` U8 [out, in/2] (low nibble = even column),
    ``weight_scale`` F8_E4M3 [out, in/16], ``weight_scale_2`` and
    ``input_scale`` FP32 scalars. These are copied as is (raw bytes) into the
    ``qat.NVFP4Linear`` modules, which decode them on the fly.

FP8 block weights of both formats are dequantized as

    W[i, j] = float(W_fp8[i, j]) * scale[i // 128, j // 128]

computed in fp32, then cast to the target dtype. This is the weight vLLM's
block-FP8 GEMM represents (the scale multiplies). Scales must be powers of two
(ue8m0); then the BF16 result is exact (e4m3 has 3 mantissa bits, bf16 has 7).
"""

import json
import os
import re
from dataclasses import dataclass

import torch

# Checkpoint tensors deliberately not loaded (text-only RL).
SKIP_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
    "mtp_dspark.",
)

_NAME_MAP: list[tuple[str, str]] = [
    (r"^tok_embeddings(?=\.|$)", "language_model.model.embed_tokens"),
    (r"^norm(?=\.|$)", "language_model.model.norm"),
    (r"^output(?=\.|$)", "language_model.lm_head"),
    (r"^lerope\.alpha$", "language_model.model.rotary_emb.lerope_alpha"),
    (r"^layers\.", "language_model.model.layers."),
    (r"\.attention_norm(?=\.|$)", ".input_layernorm"),
    (r"\.ffn_norm(?=\.|$)", ".post_attention_layernorm"),
    (r"\.attention\.indexer\.ssmax_scaling(?=\.|$)", ".self_attn.indexer.ssmax"),
    (r"\.attention\.ssmax_scaling(?=\.|$)", ".self_attn.ssmax"),
    (r"\.attention\.wq(?=\.|$)", ".self_attn.q_proj"),
    (r"\.attention\.wkv_a_with_mqa(?=\.|$)", ".self_attn.kv_a_proj_with_mqa"),
    (r"\.attention\.kv_a_norm(?=\.|$)", ".self_attn.kv_a_layernorm"),
    (r"\.attention\.wkv_b(?=\.|$)", ".self_attn.kv_b_proj"),
    (r"\.attention\.wo(?=\.|$)", ".self_attn.o_proj"),
    (r"\.attention\.indexer(?=\.|$)", ".self_attn.indexer"),
    (r"\.feed_forward(?=\.|$)", ".mlp"),
    (r"\.shared_experts(?=\.|$)", ".mlp.shared_experts"),
    (r"\.gate(?=\.|$)", ".mlp.gate"),
    (r"\.experts(?=\.|$)", ".mlp.experts"),
    (r"\.w1(?=\.|$)", ".gate_proj"),
    (r"\.w3(?=\.|$)", ".up_proj"),
    (r"\.w2(?=\.|$)", ".down_proj"),
]

# (checkpoint segment, fused param segment, shard index): only the indexer's
# [wk; weights_proj] is stacked.
_STACKED: list[tuple[str, str, int]] = [
    (".wk.", ".wk_weights_proj.", 0),
    (".weights_proj.", ".wk_weights_proj.", 1),
]
_SHARD_ORDER = {".wk_weights_proj.": [".wk.", ".weights_proj."]}


def map_name(ckpt_name: str) -> tuple[str, str | None, int]:
    """Checkpoint tensor name -> (module param name, stack key, shard index).

    ``stack key`` is the checkpoint name of shard 0 for stacked params (used
    to find shard row offsets), else None.
    """
    name = ckpt_name
    for pattern, repl in _NAME_MAP:
        name = re.sub(pattern, repl, name)
    for ck, fused, shard in _STACKED:
        if ck in name:
            if fused in _SHARD_ORDER:
                first = ckpt_name.replace(ck, _SHARD_ORDER[fused][0])
                return name.replace(ck, fused), first, shard
            return name.replace(ck, fused), None, 0
    return name, None, 0


def canonical_param_name(name: str) -> str | None:
    """Strip PEFT wrappers; None for adapter-only params (lora_*)."""
    if ".lora_" in name or "lora_A" in name or "lora_B" in name:
        return None
    name = name.removeprefix("base_model.model.")
    return name.replace(".base_layer.", ".")


@dataclass
class _Source:
    ckpt_name: str
    row_offset: int


class CheckpointReader:
    """Lazy reader over consolidated*.safetensors (index or single file)."""

    def __init__(self, ckpt_dir: str) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.dir = ckpt_dir
        idx = os.path.join(ckpt_dir, "consolidated.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            fn = "consolidated.safetensors"
            with safe_open(os.path.join(ckpt_dir, fn), "pt") as f:
                self.weight_map = {k: fn for k in f.keys()}
        self._handles: dict = {}
        self._meta: dict[str, tuple[list[int], str]] = {}

    def _f(self, name: str):
        fn = self.weight_map[name]
        if fn not in self._handles:
            self._handles[fn] = self._safe_open(os.path.join(self.dir, fn), "pt")
        return self._handles[fn]

    def _shape_dtype(self, name: str) -> tuple[list[int], str]:
        if name not in self._meta:
            sl = self._f(name).get_slice(name)
            self._meta[name] = (list(sl.get_shape()), sl.get_dtype())
        return self._meta[name]

    def shape(self, name: str) -> list[int]:
        return list(self._shape_dtype(name)[0])

    def dtype(self, name: str) -> str:
        return self._shape_dtype(name)[1]

    def release(self, fn: str) -> bool:
        """Close (unmap) file ``fn`` and drop its page cache. Pages another
        process still maps stay; a later read of ``fn`` just reads the disk again."""
        self._handles.pop(fn, None)
        return drop_file_page_cache(os.path.join(self.dir, fn))

    def full(self, name: str) -> torch.Tensor:
        return self._f(name).get_tensor(name)

    def rows(self, name: str, r0: int, r1: int) -> torch.Tensor:
        sl = self._f(name).get_slice(name)
        return sl[r0:r1]

    def rows_dequant(self, name: str, r0: int, r1: int, block=(128, 128)) -> torch.Tensor:
        """Rows [r0, r1) as fp32, dequantizing FP8 block weights."""
        w = self.rows(name, r0, r1)
        if w.dtype != torch.float8_e4m3fn:
            return w.float()
        sname = fp8_scale_name(self, name)
        bo, bi = block
        s = self.rows(sname, r0 // bo, -(-r1 // bo)).float().flatten(1)
        m, e = torch.frexp(s)
        if not bool((m == 0.5).all()):
            raise ValueError(f"{sname}: scales are not powers of two (ue8m0)")
        s = s.repeat_interleave(bo, 0)[r0 - (r0 // bo) * bo :][: r1 - r0]
        s = s.repeat_interleave(bi, 1)[:, : w.shape[1]]
        return w.float() * s


def drop_file_page_cache(path: str) -> bool:
    """``posix_fadvise(DONTNEED)`` on ``path``: evicts its clean pages that no
    process maps. False off POSIX or on error."""
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def fp8_scale_name(reader, name: str) -> str:
    """Block scale of FP8 weight ``name``: ``qscale_weight`` (FP8 checkpoint) or
    ``weight_scale`` [ob, 1, ib, 1] (ModelOpt FP8_PB_WO). Checks the block grid.
    """
    base = name[: -len("weight")]
    for suffix in ("qscale_weight", "weight_scale"):
        sname = base + suffix
        if sname in reader.weight_map:
            break
    else:
        raise ValueError(f"FP8 tensor {name} has no qscale_weight / weight_scale")
    O, I = reader.shape(name)
    sh = reader.shape(sname)
    grid = [-(-O // 128), -(-I // 128)]
    ok = (len(sh) == 2 and sh == grid) or (len(sh) == 4 and sh == [grid[0], 1, grid[1], 1])
    if not ok:
        raise ValueError(f"{sname}: shape {sh} does not match 128x128 blocks of {[O, I]}")
    return sname


def _consumed_scale(reader, ck: str) -> bool:
    """True for FP8 block scales (folded into their weight by the dequant)."""
    for suffix in ("qscale_weight", "weight_scale"):
        if ck.endswith("." + suffix):
            base = ck[: -len(suffix)] + "weight"
            if base not in reader.weight_map:
                raise ValueError(f"orphan scale {ck}")
            return suffix == "qscale_weight" or reader.dtype(base) == "F8_E4M3"
    return False


def build_plan(reader: CheckpointReader) -> tuple[dict[str, list[_Source]], list[str]]:
    """param name -> sources (with row offsets); plus skipped ckpt names."""
    plan: dict[str, list[_Source]] = {}
    skipped = []
    for ck in sorted(reader.weight_map):
        if ck.startswith(SKIP_PREFIXES):
            skipped.append(ck)
            continue
        if _consumed_scale(reader, ck):
            continue
        pname, first, shard = map_name(ck)
        off = 0 if shard == 0 else reader.shape(first)[0]
        plan.setdefault(pname, []).append(_Source(ck, off))
    return plan, skipped


def _local_rows(p: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(local tensor to fill, global row offset) for plain or DTensor params."""
    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard
    except ImportError:  # pragma: no cover
        DTensor = ()
    if isinstance(p, DTensor):
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        for pl in p.placements:
            if not (isinstance(pl, Replicate) or (isinstance(pl, Shard) and pl.dim == 0)):
                raise NotImplementedError(f"placement {pl} (only Shard(0)/Replicate)")
        _, goff = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
        return p.to_local(), goff[0]
    return p, 0


def _reads(reader, t: torch.Tensor, srcs: list[_Source]) -> list[tuple[_Source, int, int, int, int]]:
    """(source, a, b, ra, rb) row ranges this rank copies into ``t``: global
    rows [a, b) from source rows [ra, rb)."""
    if t.dim() == 0:
        return [(srcs[0], 0, 0, 0, 0)]
    local, goff = _local_rows(t)
    r0, r1 = goff, goff + local.shape[0]
    out = []
    for s in srcs:
        a, b = max(r0, s.row_offset), min(r1, s.row_offset + reader.shape(s.ckpt_name)[0])
        if a < b:
            out.append((s, a, b, a - s.row_offset, b - s.row_offset))
    return out


@torch.no_grad()
def load_sharded(model: torch.nn.Module, ckpt_dir: str, device=None, drop_page_cache: bool = True) -> dict:
    """Fill every parameter / persistent buffer of ``model`` from ``ckpt_dir``.

    Works on a plain model, a PEFT-wrapped model (lora_* params are left as
    initialized) and on FSDP2-sharded DTensor params (Shard(0)/Replicate):
    each rank reads only the rows of its local shard, via safetensors slices.
    A model still on ``meta`` is materialized with ``to_empty(device)``.
    Raises on missing params, unexpected checkpoint tensors (other than
    ``SKIP_PREFIXES``) and shape mismatches.

    Tensors are loaded file by file; with ``drop_page_cache`` each file is
    unmapped and its page cache dropped right after this rank's last read of
    it, so the cache does not grow to the checkpoint size during the load.
    """
    if any(t.is_meta for t in model.parameters()):
        model.to_empty(device=device or "cpu")
    reader = CheckpointReader(ckpt_dir)
    targets, plan, skipped = check_plan(model, reader)
    block = (128, 128)
    jobs = []
    for pname, t in targets.items():
        reads = _reads(reader, t, plan[pname])
        files = {reader.weight_map[s.ckpt_name] for s, *_ in reads}
        if t.dtype != torch.uint8:
            files |= {reader.weight_map[fp8_scale_name(reader, s.ckpt_name)]
                      for s, *_ in reads if reader.dtype(s.ckpt_name) == "F8_E4M3"}
        jobs.append((min(files, default=""), pname, t, reads, files))
    jobs.sort(key=lambda j: j[0])
    last = {fn: i for i, j in enumerate(jobs) for fn in j[4]}
    for fn in list(reader._handles):
        if fn not in last:
            reader._handles.pop(fn)
    for i, (_, pname, t, reads, files) in enumerate(jobs):
        if t.dim() == 0:  # NVFP4 weight_scale_2 / input_scale (replicated buffers)
            t.copy_(reader.full(reads[0][0].ckpt_name))
        else:
            local, goff = _local_rows(t)
            for s, a, b, ra, rb in reads:
                if t.dtype == torch.uint8:  # NVFP4: packed bytes / E4M3 scale bits, as is
                    w = reader.rows(s.ckpt_name, ra, rb).view(torch.uint8)
                else:
                    w = reader.rows_dequant(s.ckpt_name, ra, rb, block)
                local[a - goff : b - goff].copy_(w.to(local.dtype))
                del w
        if drop_page_cache:
            for fn in files:
                if last[fn] == i:
                    reader.release(fn)
    reader._handles.clear()
    _check_shared_ws2(targets)
    return {"loaded": sorted(targets), "skipped": skipped}


_TENSOR_SUFFIXES = (
    "weight", "bias", "s", "b", "lerope_alpha", "weight_scale", "weight_scale_2", "input_scale",
)


def check_plan(model: torch.nn.Module, reader) -> tuple[dict, dict, list[str]]:
    """Match model tensors to checkpoint tensors from names/dtypes/shapes only.

    Returns (targets, plan, skipped). Raises on missing or unexpected tensors,
    shape mismatches, FP8 weights without a valid block scale, and packed
    (NVFP4) vs unpacked mismatches. ``reader`` needs ``weight_map``,
    ``shape(name)`` and ``dtype(name)``.
    """
    plan, skipped = build_plan(reader)
    targets: dict[str, torch.Tensor] = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        cname = canonical_param_name(name)
        if cname is None:
            continue
        if not name.endswith(tuple(f".{b}" for b in _TENSOR_SUFFIXES)):
            raise ValueError(f"unexpected model tensor {name}")
        targets[cname] = t
    missing = sorted(set(targets) - set(plan))
    unexpected = sorted(set(plan) - set(targets))
    if missing or unexpected:
        raise ValueError(f"missing={missing[:20]} unexpected={unexpected[:20]}")
    for pname, t in targets.items():
        srcs = plan[pname]
        dts = {reader.dtype(s.ckpt_name) for s in srcs}
        if t.dtype == torch.uint8:
            want = {"U8"} if pname.endswith(".weight") else {"F8_E4M3"}
            if dts != want:
                raise ValueError(f"{pname}: packed NVFP4 tensor needs {want}, got {sorted(dts)}")
        elif "U8" in dts:
            raise ValueError(f"{pname}: NVFP4 checkpoint tensor into a non-NVFP4 module")
        for s in srcs:
            if reader.dtype(s.ckpt_name) == "F8_E4M3" and t.dtype != torch.uint8:
                fp8_scale_name(reader, s.ckpt_name)
        if t.dim() == 0:
            shp = reader.shape(srcs[0].ckpt_name)
            if len(srcs) != 1 or shp != [] or dts != {"F32"}:
                raise ValueError(f"{pname}: scalar <- {[s.ckpt_name for s in srcs]} {shp}")
            continue
        nrows_total = sum(reader.shape(s.ckpt_name)[0] for s in srcs)
        if nrows_total != t.shape[0]:
            raise ValueError(f"{pname}: rows {nrows_total} != {t.shape[0]}")
        for s in srcs:
            shp = reader.shape(s.ckpt_name)
            if list(shp[1:]) != list(t.shape[1:]):
                raise ValueError(f"{pname} <- {s.ckpt_name}: shape {shp} vs {list(t.shape)}")
    return targets, plan, skipped


def _check_shared_ws2(targets: dict[str, torch.Tensor]) -> None:
    """vLLM keeps only the gate's (w1) ``weight_scale_2`` for w13; require
    w1 == w3 so that using it for up is exact."""
    for name, t in targets.items():
        if name.endswith(".gate_proj.weight_scale_2"):
            up = targets[name.replace(".gate_proj.", ".up_proj.")]
            if not torch.equal(t, up):
                raise ValueError(f"{name}: w1/w3 weight_scale_2 differ")
