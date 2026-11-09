"""Import a HuggingFace PEFT LoRA adapter into a SkyRL policy.

The adapter directory holds ``adapter_config.json`` and ``adapter_model.safetensors``
as written by ``peft``, SkyRL's own ``lora_adapter`` checkpoint export, or Tinker's
``tinker_cookbook.weights.build_lora_adapter``. Every adapted HF module becomes a
:class:`LoraPiece` carrying the tensors and the scale the adapter applies to them
(``lora_alpha / r`` with PEFT's ``rank_pattern`` / ``alpha_pattern`` / ``use_rslora``
rules), so the backends only have to place tensors.

Kept free of megatron / peft imports so the transforms are unit-testable on CPU.
"""

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from skyrl.backends.skyrl_train.utils.io import io

PEFT_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

# Tinker trains GatedDeltaNet q/k/v as three adapters; HF Qwen3.5 fuses the three
# projections into one ``in_proj_qkv`` linear whose rows are ``[q; k; v]``.
_TINKER_GDN_QKV = ("in_proj_q", "in_proj_k", "in_proj_v")
_TINKER_GDN_QKV_RE = re.compile(r"^(?P<prefix>.*\.linear_attn)\.(?P<proj>in_proj_[qkv])$")

# Enclosing namespaces that differ between a flat text checkpoint and a VL checkpoint's
# language model (``model.layers`` vs ``model.language_model.layers``).
_NAMESPACE_RE = re.compile(r"^(?:model\.)?(?:language_model\.)?(?:model\.)?")

# Tinker's own export names the output projection ``model.unembed_tokens``; HF and PEFT
# call the same (untied) linear ``lm_head``.
_TINKER_LM_HEAD_RE = re.compile(r"^(?:model\.)?(?:language_model\.)?(?:model\.)?unembed_tokens$")


@dataclass
class LoraPiece:
    """One adapted HF module: ``delta_W = scale * lora_b @ lora_a``."""

    module: str
    """HF module path without the ``base_model.model.`` prefix."""
    lora_a: torch.Tensor
    """``(rank, in_features)``."""
    lora_b: torch.Tensor
    """``(out_features, rank)``."""
    scale: float

    @property
    def rank(self) -> int:
        return int(self.lora_b.shape[-1])

    def delta(self) -> torch.Tensor:
        """``scale * B @ A`` in float32, for tests and parity checks."""
        return self.scale * (self.lora_b.float() @ self.lora_a.float())


@dataclass
class PeftLoraAdapter:
    config: Dict[str, Any]
    pieces: Dict[str, LoraPiece] = field(default_factory=dict)
    """Keyed by :attr:`LoraPiece.module`, in file order."""

    @property
    def r(self) -> int:
        return int(self.config["r"])

    @property
    def lora_alpha(self) -> float:
        return float(self.config["lora_alpha"])

    @property
    def base_scale(self) -> float:
        """The adapter-wide ``lora_alpha / r`` that vLLM applies to every module."""
        return self.lora_alpha / self.r

    def max_rank(self) -> int:
        return max((piece.rank for piece in self.pieces.values()), default=self.r)

    def leaf_modules(self) -> List[str]:
        """Sorted unique leaf names, the ``target_modules`` form (``q_proj``, ``lm_head``, ...)."""
        return sorted({piece.module.rsplit(".", 1)[-1] for piece in self.pieces.values()})


# ---------------------------------------------------------------------------
# Reading and parsing
# ---------------------------------------------------------------------------


def read_peft_adapter(path: str) -> PeftLoraAdapter:
    """Load and parse an adapter directory (local or cloud)."""
    from safetensors.torch import load_file

    with io.local_read_files(path, [ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE]) as read_dir:
        config_path = os.path.join(read_dir, ADAPTER_CONFIG_FILE)
        weights_path = os.path.join(read_dir, ADAPTER_WEIGHTS_FILE)
        for file_path in (config_path, weights_path):
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"PEFT adapter at {path} is missing {os.path.basename(file_path)}; expected "
                    f"{ADAPTER_CONFIG_FILE} and {ADAPTER_WEIGHTS_FILE} (a Tinker checkpoint download must "
                    "first be converted with tinker_cookbook.weights.build_lora_adapter)"
                )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        state = load_file(weights_path)
    return parse_peft_adapter(config, state)


def peft_pattern_value(pattern: Optional[Mapping[str, Any]], module: str, default: Any) -> Any:
    """Look ``module`` up in a PEFT ``rank_pattern`` / ``alpha_pattern`` mapping.

    Mirrors ``peft.utils.other.get_pattern_key``: a key is a regex that must match the
    module path's tail (``(.*\\.)?(key)$``); the first matching key in mapping order wins.
    """
    if not pattern:
        return default
    for key, value in pattern.items():
        if re.match(rf"(.*\.)?({key})$", module):
            return value
    return default


def peft_module_rank(config: Mapping[str, Any], module: str) -> int:
    return int(peft_pattern_value(config.get("rank_pattern"), module, config["r"]))


def peft_module_scale(config: Mapping[str, Any], module: str, rank: int) -> float:
    """The scale PEFT applies to ``module``'s ``B @ A``."""
    alpha = float(peft_pattern_value(config.get("alpha_pattern"), module, config["lora_alpha"]))
    if config.get("use_rslora", False):
        return alpha / math.sqrt(rank)
    return alpha / rank


def strip_peft_prefix(name: str) -> str:
    return name[len(PEFT_PREFIX) :] if name.startswith(PEFT_PREFIX) else name


def parse_peft_adapter(config: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> PeftLoraAdapter:
    """Validate an adapter and build its :class:`LoraPiece` set.

    Only plain 2-D LoRA is accepted: no DoRA, no ``modules_to_save`` full weights, no
    stacked 3-D expert tensors (Tinker's converter expands those to per-expert keys).
    """
    peft_type = str(config.get("peft_type", "LORA")).upper()
    if peft_type != "LORA":
        raise ValueError(f"init_adapter_path: peft_type must be LORA, got {peft_type!r}")
    if config.get("use_dora", False):
        raise ValueError("init_adapter_path: DoRA adapters (use_dora=true) are not supported")
    if config.get("modules_to_save"):
        raise ValueError(
            f"init_adapter_path: modules_to_save={config['modules_to_save']} is not supported; the adapter "
            "must contain LoRA tensors only"
        )
    for key in ("r", "lora_alpha"):
        if key not in config:
            raise ValueError(f"init_adapter_path: adapter_config.json is missing {key!r}")

    config = dict(config)
    config.setdefault("rank_pattern", {})
    config.setdefault("alpha_pattern", {})

    lora_a: Dict[str, torch.Tensor] = {}
    lora_b: Dict[str, torch.Tensor] = {}
    order: List[str] = []
    source_names: Dict[str, str] = {}  # module -> name in the file, for rank / alpha pattern lookups
    for name, tensor in state.items():
        module_name = strip_peft_prefix(name)
        if module_name.endswith(LORA_A_SUFFIX):
            source = module_name[: -len(LORA_A_SUFFIX)]
            module = _alias_module(source)
            lora_a[module] = tensor
        elif module_name.endswith(LORA_B_SUFFIX):
            source = module_name[: -len(LORA_B_SUFFIX)]
            module = _alias_module(source)
            lora_b[module] = tensor
        else:
            raise ValueError(f"init_adapter_path: unexpected non-LoRA tensor {name!r} in {ADAPTER_WEIGHTS_FILE}")
        if source_names.setdefault(module, source) != source:
            raise ValueError(f"init_adapter_path: {source_names[module]!r} and {source!r} both name {module!r}")
        if module not in order:
            order.append(module)

    missing = sorted((set(lora_a) ^ set(lora_b)))
    if missing:
        raise ValueError(f"init_adapter_path: modules without both lora_A and lora_B: {missing[:5]}")

    pieces: Dict[str, LoraPiece] = {}
    for module in order:
        a, b = lora_a[module], lora_b[module]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"init_adapter_path: {module} has lora_A {tuple(a.shape)} / lora_B {tuple(b.shape)}; only 2-D "
                "LoRA tensors are supported (stacked expert tensors must be expanded per expert first)"
            )
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"init_adapter_path: {module} lora_A rank {a.shape[0]} != lora_B rank {b.shape[1]}")
        rank = int(a.shape[0])
        source = source_names[module]
        config_rank = peft_module_rank(config, source)
        if rank != config_rank:
            raise ValueError(
                f"init_adapter_path: {module} tensors have rank {rank} but adapter_config.json gives "
                f"{config_rank} (r / rank_pattern); the config must describe the tensors"
            )
        pieces[module] = LoraPiece(module=module, lora_a=a, lora_b=b, scale=peft_module_scale(config, source, rank))
    if not pieces:
        raise ValueError("init_adapter_path: the adapter contains no LoRA tensors")
    targets = config.get("target_modules")
    if isinstance(targets, (list, tuple, set)) and any(_TINKER_LM_HEAD_RE.match(t) for t in targets):
        config["target_modules"] = sorted({"lm_head" if _TINKER_LM_HEAD_RE.match(t) else t for t in targets})
    return PeftLoraAdapter(config=config, pieces=pieces)


def _alias_module(module: str) -> str:
    """HF name of a module the adapter file may spell differently (Tinker's ``unembed_tokens``)."""
    return "lm_head" if _TINKER_LM_HEAD_RE.match(module) else module


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------


def canonical_module_name(name: str) -> str:
    """Drop the PEFT prefix and the enclosing ``model`` / ``language_model`` namespaces.

    Lets an adapter written against ``model.layers.*`` match a model (or bridge) that
    names the same module ``model.language_model.layers.*`` and vice versa.
    """
    return _NAMESPACE_RE.sub("", strip_peft_prefix(name))


def hf_name_variants(module: str) -> List[str]:
    """Spellings of ``module`` to try against a Megatron-Bridge mapping registry."""
    canonical = canonical_module_name(module)
    variants = [
        module,
        canonical,
        f"model.{canonical}",
        f"model.language_model.{canonical}",
        f"language_model.model.{canonical}",
        f"language_model.{canonical}",
    ]
    return list(dict.fromkeys(variants))


def remap_peft_state_dict(state: Mapping[str, torch.Tensor], model_modules: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Re-key adapter tensors onto the model's own module names (FSDP / PEFT path).

    ``model_modules`` are the LoRA-wrapped module paths of the base model, e.g.
    ``model.layers.0.self_attn.q_proj``; keys come back as
    ``base_model.model.<module>.lora_A.weight``. A tensor whose module has no
    counterpart in the model raises, so trained weights are never dropped silently.
    """
    by_canonical: Dict[str, str] = {}
    for module in model_modules:
        canonical = canonical_module_name(module)
        if canonical in by_canonical and by_canonical[canonical] != module:
            raise ValueError(f"ambiguous module names {by_canonical[canonical]!r} and {module!r}")
        by_canonical[canonical] = module

    remapped: Dict[str, torch.Tensor] = {}
    unmatched: List[str] = []
    for name, tensor in state.items():
        module_name = strip_peft_prefix(name)
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
            if module_name.endswith(suffix):
                module = module_name[: -len(suffix)]
                break
        else:
            raise ValueError(f"unexpected non-LoRA tensor {name!r}")
        target = by_canonical.get(canonical_module_name(module))
        if target is None:
            unmatched.append(module)
            continue
        remapped[f"{PEFT_PREFIX}{target}{suffix}"] = tensor
    if unmatched:
        raise ValueError(
            f"init_adapter_path: {len(unmatched)} adapted module(s) have no LoRA layer in the model: "
            f"{sorted(set(unmatched))[:8]}"
        )
    return remapped


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class StackedLora:
    """Several pieces expressed as one adapter of rank :attr:`rank` and scale ``target_scale``.

    ``lora_a`` is ``(rank, in)``; ``lora_bs[i]`` is ``(out_i, rank)`` for ``pieces[i]`` with the
    scale ratio folded in, so ``target_scale * lora_bs[i] @ lora_a == pieces[i].delta()``.
    """

    lora_a: torch.Tensor
    lora_bs: List[torch.Tensor]
    rank: int
    shared_a: bool


def stack_lora_pieces(pieces: Sequence[LoraPiece], target_scale: float, pad_to_multiple: int = 1) -> StackedLora:
    """Combine pieces that share one input into a single (block-diagonal) adapter.

    Pieces with bit-identical ``lora_a`` and equal scales (an adapter exported from a
    fused module) keep that shared A at the original rank. Otherwise the A matrices are
    stacked along the rank dimension and each B occupies its own column block, which
    reproduces every piece exactly: ``B_fused @ A_fused == [B_1 A_1; B_2 A_2; ...]``.
    ``pad_to_multiple`` appends zero rank rows/columns so the rank divides the
    tensor-parallel size; they contribute nothing and receive zero gradient.
    """
    if not pieces:
        raise ValueError("stack_lora_pieces needs at least one piece")
    if target_scale <= 0:
        raise ValueError(f"target_scale must be positive, got {target_scale}")
    in_features = int(pieces[0].lora_a.shape[1])
    for piece in pieces:
        if int(piece.lora_a.shape[1]) != in_features:
            raise ValueError(
                f"{piece.module}: in_features {piece.lora_a.shape[1]} != {in_features} of {pieces[0].module}"
            )

    shared_a = len(pieces) > 1 and all(
        piece.rank == pieces[0].rank and piece.scale == pieces[0].scale and torch.equal(piece.lora_a, pieces[0].lora_a)
        for piece in pieces[1:]
    )
    rank = pieces[0].rank if shared_a else sum(piece.rank for piece in pieces)
    padded = -(-rank // pad_to_multiple) * pad_to_multiple

    lora_a = torch.zeros(padded, in_features, dtype=torch.float32)
    lora_bs: List[torch.Tensor] = []
    offset = 0
    for piece in pieces:
        factor = piece.scale / target_scale
        if shared_a:
            start, end = 0, piece.rank
        else:
            start, end = offset, offset + piece.rank
            offset = end
        lora_a[start:end] = piece.lora_a.float()
        lora_b = torch.zeros(int(piece.lora_b.shape[0]), padded, dtype=torch.float32)
        lora_b[:, start:end] = piece.lora_b.float() * factor
        lora_bs.append(lora_b)
    return StackedLora(lora_a=lora_a, lora_bs=lora_bs, rank=padded, shared_a=shared_a)


def _integral(value: float) -> Any:
    return int(value) if float(value).is_integer() else value


def fold_tinker_gdn_aliases(adapter: PeftLoraAdapter) -> PeftLoraAdapter:
    """Merge Tinker's ``linear_attn.in_proj_q/k/v`` pieces into HF's ``in_proj_qkv``.

    The three adapters become one block-diagonal adapter of rank ``r_q + r_k + r_v`` on
    ``in_proj_qkv`` (rows ``[q; k; v]``, the HF layout), recorded in ``rank_pattern`` /
    ``alpha_pattern`` so the merged scale equals the q adapter's. ``target_modules`` is
    rewritten accordingly. Adapters without these keys pass through unchanged.
    """
    groups: Dict[str, Dict[str, LoraPiece]] = {}
    for module, piece in adapter.pieces.items():
        match = _TINKER_GDN_QKV_RE.match(module)
        if match:
            groups.setdefault(match.group("prefix"), {})[match.group("proj")] = piece
    if not groups:
        return adapter
    for prefix, group in groups.items():
        missing = [proj for proj in _TINKER_GDN_QKV if proj not in group]
        if missing:
            raise ValueError(
                f"init_adapter_path: {prefix} has {sorted(group)} adapters but lacks {missing}; "
                "in_proj_q/k/v must all be present to fold into in_proj_qkv"
            )

    config = dict(adapter.config)
    config["rank_pattern"] = dict(config.get("rank_pattern") or {})
    config["alpha_pattern"] = dict(config.get("alpha_pattern") or {})
    pieces: Dict[str, LoraPiece] = {}
    folded_ranks: set = set()
    for module, piece in adapter.pieces.items():
        match = _TINKER_GDN_QKV_RE.match(module)
        if match is None:
            pieces[module] = piece
            continue
        prefix = match.group("prefix")
        group = groups[prefix]
        if match.group("proj") != _TINKER_GDN_QKV[0]:
            continue  # emitted together with in_proj_q
        fused_module = f"{prefix}.in_proj_qkv"
        if fused_module in adapter.pieces:
            raise ValueError(f"init_adapter_path: {prefix} has both in_proj_qkv and in_proj_q/k/v adapters")
        ordered = [group[proj] for proj in _TINKER_GDN_QKV]
        stacked = stack_lora_pieces(ordered, target_scale=ordered[0].scale)
        pieces[fused_module] = LoraPiece(
            module=fused_module,
            lora_a=stacked.lora_a.to(ordered[0].lora_a.dtype),
            lora_b=torch.cat(stacked.lora_bs, dim=0).to(ordered[0].lora_b.dtype),
            scale=ordered[0].scale,
        )
        folded_ranks.add((stacked.rank, ordered[0].scale))

    if len(folded_ranks) != 1:
        raise ValueError(
            "init_adapter_path: in_proj_q/k/v adapters differ in rank or scale across layers; one "
            f"rank_pattern entry cannot describe them: {sorted(folded_ranks)}"
        )
    (rank, scale), *_ = folded_ranks
    if config.get("use_rslora", False):
        fused_alpha = _integral(scale * math.sqrt(rank))
    else:
        fused_alpha = _integral(scale * rank)
    # Pattern lookups take the first matching key (as in peft), so the in_proj_qkv entries go first:
    # a broader existing key such as ".*" would otherwise shadow them.
    config["rank_pattern"] = {
        "in_proj_qkv": rank,
        **{k: v for k, v in config["rank_pattern"].items() if k not in _TINKER_GDN_QKV + ("in_proj_qkv",)},
    }
    config["alpha_pattern"] = {
        "in_proj_qkv": fused_alpha,
        **{k: v for k, v in config["alpha_pattern"].items() if k not in _TINKER_GDN_QKV + ("in_proj_qkv",)},
    }
    targets = config.get("target_modules")
    if isinstance(targets, (list, tuple, set)):
        kept = [t for t in targets if t not in _TINKER_GDN_QKV]
        if "in_proj_qkv" not in kept:
            kept.append("in_proj_qkv")
        config["target_modules"] = sorted(kept)
    return PeftLoraAdapter(config=config, pieces=pieces)


def normalize_peft_adapter(adapter: PeftLoraAdapter) -> PeftLoraAdapter:
    """Apply every layout fold that turns a converter's output into a plain PEFT adapter."""
    return fold_tinker_gdn_aliases(adapter)


# ---------------------------------------------------------------------------
# vLLM sync
# ---------------------------------------------------------------------------


def fold_peft_patterns_for_vllm(
    state: Mapping[str, torch.Tensor], config: Mapping[str, Any]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Fold per-module scales into ``lora_B`` so vLLM's single ``lora_alpha / r`` is exact.

    vLLM's PEFT loader ignores ``rank_pattern`` / ``alpha_pattern`` and applies
    ``lora_alpha / r`` from ``adapter_config.json`` to every module; a module whose
    PEFT scale differs (``alpha_pattern``, ``rank_pattern``, ``use_rslora``) gets its
    ``lora_B`` multiplied by ``scale / (lora_alpha / r)``. The returned config has the
    patterns cleared and ``use_rslora`` off; tensor ranks are untouched, so vLLM's
    ``max_lora_rank`` must cover the largest one.
    """
    base_scale = float(config["lora_alpha"]) / int(config["r"])
    folded: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        module_name = strip_peft_prefix(name)
        if module_name.endswith(LORA_B_SUFFIX):
            module = module_name[: -len(LORA_B_SUFFIX)]
            scale = peft_module_scale(config, module, int(tensor.shape[-1]))
            if scale != base_scale:
                tensor = (tensor.to(torch.float32) * (scale / base_scale)).to(tensor.dtype)
        folded[name] = tensor
    new_config = dict(config)
    new_config["rank_pattern"] = {}
    new_config["alpha_pattern"] = {}
    new_config["use_rslora"] = False
    return folded, new_config


def peft_state_max_rank(state: Mapping[str, torch.Tensor]) -> int:
    ranks = [int(t.shape[-1]) for name, t in state.items() if strip_peft_prefix(name).endswith(LORA_B_SUFFIX)]
    return max(ranks) if ranks else 0


def check_adapter_matches_config(adapter: PeftLoraAdapter, rank: int, alpha: float) -> None:
    """``lora.rank`` / ``lora.alpha`` must equal the adapter's ``r`` / ``lora_alpha``."""
    if adapter.r != rank or float(adapter.lora_alpha) != float(alpha):
        raise ValueError(
            f"init_adapter_path: the adapter was trained with r={adapter.r}, lora_alpha={_integral(adapter.lora_alpha)} "
            f"but the config sets trainer.policy.model.lora.rank={rank}, alpha={_integral(alpha)}; set them to the "
            "adapter's values"
        )


def with_pieces(adapter: PeftLoraAdapter, pieces: Dict[str, LoraPiece]) -> PeftLoraAdapter:
    return replace(adapter, pieces=pieces)
