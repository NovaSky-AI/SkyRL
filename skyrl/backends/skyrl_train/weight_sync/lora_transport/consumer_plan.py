"""Plan producer-owned rectangles directly into receiver-local LoRA factors."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import prod
from typing import Any, Mapping

import torch

from skyrl.backends.skyrl_train.weight_sync.lora_layout import (
    convert_moe_expert_lora_key,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    get_qkv_lora_head_mapping,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRASourceSlice,
)


@dataclass(frozen=True)
class LoRAConsumerPull:
    source_rank: int
    source_slice: LoRASourceSlice


@dataclass(frozen=True)
class LoRAConsumerCopy:
    pull_index: int
    module_name: str
    factor_index: int
    component: int
    starts: tuple[int, ...]
    stops: tuple[int, ...]


@dataclass(frozen=True)
class LoRAConsumerPlan:
    source_layout_digest: str
    receiver_plan: Any
    pulls: tuple[LoRAConsumerPull, ...]
    copies: tuple[LoRAConsumerCopy, ...]

    @property
    def source_bytes(self) -> int:
        return 4 * sum(
            prod(
                b - a for a, b in zip(pull.source_slice.starts, pull.source_slice.stops)
            )
            for pull in self.pulls
        )


@dataclass(frozen=True)
class _SourceTile:
    source: LoRABridgeSource
    starts: tuple[int, ...]
    stops: tuple[int, ...]
    source_starts: tuple[int, ...]


def _intersect(starts, stops, other_starts, other_stops):
    lower = tuple(max(a, b) for a, b in zip(starts, other_starts, strict=True))
    upper = tuple(min(a, b) for a, b in zip(stops, other_stops, strict=True))
    return (lower, upper) if all(a < b for a, b in zip(lower, upper)) else None


def _get_source_bounds(
    source: LoRABridgeSource, ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    start = [0] * ndim
    if source.tensor_parallel_axis is not None:
        axis = source.tensor_parallel_axis
        start[axis] += source.tensor_parallel_rank * source.shape[axis]
    if source.expert_parallel_axis is not None:
        axis = source.expert_parallel_axis
        tp_multiplier = (
            source.tensor_parallel_size if source.tensor_parallel_axis == axis else 1
        )
        start[axis] += source.expert_parallel_rank * source.shape[axis] * tp_multiplier
    return tuple(start), tuple(a + b for a, b in zip(start, source.shape))


def _build_qkv_output_tiles(
    group: list[LoRABridgeSource], full_shape: tuple[int, ...]
) -> dict[str, tuple[tuple[int, ...], list[_SourceTile]]]:
    first = group[0]
    if (
        first.component != "linear_out"
        or len(first.hf_param_names) != 3
        or len(full_shape) != 2
    ):
        raise NotImplementedError(
            "QKV source splitting requires one 2D LoRA-B source and three outputs"
        )
    head_size, output_indices = get_qkv_lora_head_mapping(dict(first.transform_config))
    if full_shape[0] != sum(len(indices) for indices in output_indices) * head_size:
        raise ValueError(
            "QKV LoRA source shape does not match its Bridge transform config"
        )

    outputs = {}
    for name, indices in zip(first.hf_param_names, output_indices, strict=True):
        if not name.endswith(".lora_B.weight"):
            raise ValueError(
                "QKV source component does not match its canonical factor names"
            )
        canonical_name = convert_moe_expert_lora_key(
            name.removeprefix("base_model.model."), 2
        )
        output_shape = (len(indices) * head_size, full_shape[1])
        tiles = []
        for output_head, source_head in enumerate(indices):
            region_start = (source_head * head_size, 0)
            region_stop = ((source_head + 1) * head_size, full_shape[1])
            output_start = (output_head * head_size, 0)
            for source in group:
                if (
                    source.tensor_parallel_axis is None
                    and source.tensor_parallel_rank != 0
                ):
                    continue
                if (
                    source.expert_parallel_axis is None
                    and source.expert_parallel_rank != 0
                ):
                    continue
                source_start, source_stop = _get_source_bounds(source, 2)
                overlap = _intersect(
                    source_start, source_stop, region_start, region_stop
                )
                if overlap is None:
                    continue
                lower, upper = overlap
                tiles.append(
                    _SourceTile(
                        source,
                        tuple(
                            a + b - c
                            for a, b, c in zip(output_start, lower, region_start)
                        ),
                        tuple(
                            a + b - c
                            for a, b, c in zip(output_start, upper, region_start)
                        ),
                        tuple(a - b for a, b in zip(lower, source_start)),
                    )
                )
        outputs[canonical_name] = (output_shape, tiles)
    return outputs


def _build_source_tiles(
    layout: LoRABridgeSourceLayout,
) -> dict[str, tuple[tuple[int, ...], list[_SourceTile]]]:
    groups: dict[str, list[LoRABridgeSource]] = {}
    owners = set()
    for source in layout.sources:
        owner = (source.source_rank, source.key)
        if owner in owners:
            raise ValueError(
                "One producer key cannot own multiple distinct source shards"
            )
        owners.add(owner)
        groups.setdefault(source.key, []).append(source)
    outputs = {}
    for group in groups.values():
        first = group[0]
        if first.transform not in (
            "identity",
            "replicate",
            "split_gated_mlp",
            "split_qkv",
        ):
            raise NotImplementedError(
                f"Consumer slices do not support Bridge transform {first.transform!r}"
            )
        if len(first.shape) not in (2, 3) or any(
            source.shape != first.shape for source in group
        ):
            raise NotImplementedError(
                "Consumer slices require uniform 2D or 3D source shards"
            )
        if first.transform == "identity" and len(first.hf_param_names) != 1:
            raise ValueError("Identity sources require exactly one output")
        if first.transform == "split_gated_mlp" and (
            len(first.hf_param_names) != 2 or len(first.shape) != 2
        ):
            raise NotImplementedError("Gated source splitting requires two 2D outputs")
        suffix = (
            ".lora_A.weight" if first.component == "linear_in" else ".lora_B.weight"
        )
        if any(not name.endswith(suffix) for name in first.hf_param_names):
            raise ValueError(
                "Source component does not match its canonical factor names"
            )
        full_shape = list(first.shape)
        for axis, size in (
            (first.tensor_parallel_axis, first.tensor_parallel_size),
            (first.expert_parallel_axis, first.expert_parallel_size),
        ):
            if axis is not None:
                if not 0 <= axis < len(full_shape):
                    raise ValueError("Source shard axis is outside its tensor")
                full_shape[axis] *= size
        if first.transform == "split_qkv":
            qkv_outputs = _build_qkv_output_tiles(group, tuple(full_shape))
            if set(qkv_outputs).intersection(outputs):
                raise ValueError("Multiple independent sources own the same HF output")
            outputs.update(qkv_outputs)
            continue
        for output_index, name in enumerate(first.hf_param_names):
            name = convert_moe_expert_lora_key(
                name.removeprefix("base_model.model."), len(first.shape)
            )
            if name in outputs:
                raise ValueError(f"Multiple independent sources own HF output {name!r}")
            output_shape = full_shape.copy()
            output_start = [0] * len(full_shape)
            if first.transform == "split_gated_mlp":
                if full_shape[0] % 2:
                    raise ValueError(
                        "Gated source output must divide into equal halves"
                    )
                output_shape[0] //= 2
                output_start[0] = output_index * output_shape[0]
            output_stop = tuple(a + b for a, b in zip(output_start, output_shape))
            tiles = []
            for source in group:
                if (
                    source.tensor_parallel_axis is None
                    and source.tensor_parallel_rank != 0
                ):
                    continue
                if (
                    source.expert_parallel_axis is None
                    and source.expert_parallel_rank != 0
                ):
                    continue
                start, stop = _get_source_bounds(source, len(full_shape))
                overlap = _intersect(start, stop, output_start, output_stop)
                if overlap is not None:
                    lower, upper = overlap
                    tiles.append(
                        _SourceTile(
                            source,
                            tuple(a - b for a, b in zip(lower, output_start)),
                            tuple(a - b for a, b in zip(upper, output_start)),
                            tuple(a - b for a, b in zip(lower, start)),
                        )
                    )
            outputs[name] = (tuple(output_shape), tiles)
    return outputs


def build_lora_consumer_plan(
    source_layout: LoRABridgeSourceLayout, receiver_plan: Any
) -> LoRAConsumerPlan:
    """Intersect canonical source ownership with only the consumed local regions."""
    if (
        receiver_plan.rank <= 0
        or not receiver_plan.modules
        or not receiver_plan.target_modules
    ):
        raise ValueError(
            "Consumer plans require a positive rank and nonempty targets and modules"
        )
    module_names = [module.module_name for module in receiver_plan.modules]
    if len(module_names) != len(set(module_names)):
        raise ValueError("Consumer plans require unique module names")
    sources = _build_source_tiles(source_layout)
    pulls: list[LoRAConsumerPull] = []
    pull_indices: dict[LoRAConsumerPull, int] = {}
    copies: list[LoRAConsumerCopy] = []
    for module in receiver_plan.modules:
        if module.dtype != "torch.bfloat16":
            raise ValueError("Consumer destinations must be BF16")
        if not module.factor_shapes:
            raise ValueError("Consumer modules require nonempty factors")
        if len(module.source_names) != len(module.factor_shapes):
            raise ValueError("Consumer source names must match runtime factor order")
        for factor_index, shapes in enumerate(module.factor_shapes):
            for component, destination_shape in enumerate(shapes):
                regions = _get_factor_regions(
                    module,
                    factor_index,
                    component,
                    destination_shape,
                    receiver_plan.rank,
                )
                factor_copies = []
                for (
                    source_name,
                    expected_shape,
                    desired_start,
                    desired_stop,
                    destination_start,
                ) in regions:
                    name = source_name + (
                        ".lora_A.weight" if component == 0 else ".lora_B.weight"
                    )
                    if name not in sources:
                        raise ValueError(f"Missing canonical source for {name!r}")
                    source_shape, tiles = sources[name]
                    if source_shape != expected_shape:
                        raise ValueError(
                            f"Source {name!r} has shape {source_shape}, expected {expected_shape}"
                        )
                    for tile in tiles:
                        overlap = _intersect(
                            tile.starts, tile.stops, desired_start, desired_stop
                        )
                        if overlap is None:
                            continue
                        lower, upper = overlap
                        starts = tuple(
                            a + b - c
                            for a, b, c in zip(tile.source_starts, lower, tile.starts)
                        )
                        stops = tuple(
                            a + b - c
                            for a, b, c in zip(tile.source_starts, upper, tile.starts)
                        )
                        selection = LoRASourceSlice(tile.source.key, starts, stops)
                        selection.validate_shape(tile.source.shape)
                        pull = LoRAConsumerPull(tile.source.source_rank, selection)
                        if pull not in pull_indices:
                            pull_indices[pull] = len(pulls)
                            pulls.append(pull)
                        # Per-expert 2D sources populate one expert of the local 3D factor.
                        prefix = len(destination_shape) - len(desired_start)
                        local_start = tuple(
                            a + b - c
                            for a, b, c in zip(
                                destination_start[prefix:], lower, desired_start
                            )
                        )
                        local_stop = tuple(
                            a + b - c
                            for a, b, c in zip(
                                destination_start[prefix:], upper, desired_start
                            )
                        )
                        copy = LoRAConsumerCopy(
                            pull_indices[pull],
                            module.module_name,
                            factor_index,
                            component,
                            destination_start[:prefix] + local_start,
                            tuple(value + 1 for value in destination_start[:prefix])
                            + local_stop,
                        )
                        factor_copies.append(copy)
                _validate_factor_coverage(destination_shape, factor_copies)
                copies.extend(factor_copies)
    return LoRAConsumerPlan(
        source_layout.layout_digest, receiver_plan, tuple(pulls), tuple(copies)
    )


def _get_factor_regions(module, factor, component, shape, rank):
    layout = module.source_layout
    names = module.source_names[factor]
    global_input = module.global_input_size
    global_output = module.global_output_sizes[factor]
    if layout in ("row", "column", "replicated", "merged"):
        if len(names) != 1 or len(shape) != 2:
            raise ValueError("Dense local factors require one 2D source")
        expected = (rank, global_input) if component == 0 else (global_output, rank)
        start = [0, 0]
        if layout == "row" and component == 0:
            start[1] = module.tp_rank * shape[1]
        elif layout in ("column", "merged") and component == 1:
            start[0] = module.output_shard_ids[factor] * shape[0]
        return [
            (
                names[0],
                expected,
                tuple(start),
                tuple(a + b for a, b in zip(start, shape)),
                (0, 0),
            )
        ]
    if layout not in ("moe", "moe_3d"):
        raise NotImplementedError(f"Unsupported consumer source layout {layout!r}")
    expected_factors = 3 if layout == "moe" else 2
    if len(module.factor_shapes) != expected_factors or module.expert_ids != tuple(
        range(len(module.expert_ids))
    ):
        raise NotImplementedError(
            "Consumer MoE slices require complete EP1 gated expert factors"
        )
    if len(shape) != 3 or shape[0] != len(module.expert_ids):
        raise ValueError("MoE factors must preserve the explicit expert axis")
    # The second runtime factor is the down projection in both MoE layouts.
    input_size = global_input
    if factor == 1:
        input_size = module.global_output_sizes[0] // (2 if layout == "moe_3d" else 1)
    per_expert_shape = (rank, input_size) if component == 0 else (global_output, rank)
    start = [0, 0]
    if factor == 1 and component == 0:
        start[1] = module.tp_rank * shape[-1]
    elif factor != 1 and component == 1:
        start[0] = module.tp_rank * shape[-2]
    if layout == "moe":
        if len(names) != len(module.expert_ids):
            raise ValueError("MoE source names must enumerate every expert")
        return [
            (
                name,
                per_expert_shape,
                tuple(start),
                tuple(a + b for a, b in zip(start, shape[1:])),
                (expert, 0, 0),
            )
            for expert, name in enumerate(names)
        ]
    if len(names) != 1:
        raise ValueError("Fused MoE factors require one canonical source")
    expected = (len(module.expert_ids), *per_expert_shape)
    if factor == 0 and component == 1:
        if global_output % 2 or shape[1] % 2:
            raise ValueError("Fused gate/up output dimensions must be even")
        half = shape[1] // 2
        return [
            (
                names[0],
                expected,
                (0, side * (global_output // 2) + module.tp_rank * half, 0),
                (
                    shape[0],
                    side * (global_output // 2) + (module.tp_rank + 1) * half,
                    rank,
                ),
                (0, side * half, 0),
            )
            for side in range(2)
        ]
    return [
        (
            names[0],
            expected,
            (0, *start),
            (shape[0], *(a + b for a, b in zip(start, shape[1:]))),
            (0, 0, 0),
        )
    ]


def _validate_factor_coverage(shape, copies):
    active: dict[int, LoRAConsumerCopy] = {}
    endings: list[tuple[int, int]] = []
    actual = 0
    for index, copy in enumerate(sorted(copies, key=lambda item: item.starts[0])):
        if any(
            a < 0 or b <= a or b > size
            for a, b, size in zip(copy.starts, copy.stops, shape, strict=True)
        ):
            raise ValueError("Consumer source mapping exceeds the local destination")
        while endings and endings[0][0] <= copy.starts[0]:
            _, expired = heappop(endings)
            del active[expired]
        for previous in active.values():
            if _intersect(copy.starts, copy.stops, previous.starts, previous.stops):
                raise ValueError(
                    "Consumer source ownership overlaps in the destination"
                )
        active[index] = copy
        heappush(endings, (copy.stops[0], index))
        actual += prod(b - a for a, b in zip(copy.starts, copy.stops))
    if actual != prod(shape):
        raise ValueError("Consumer sources do not cover the complete local destination")


def assemble_lora_consumer_factors(
    plan: LoRAConsumerPlan,
    pulled: Mapping[LoRAConsumerPull, torch.Tensor],
    device: torch.device | str,
) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Materialize independent unscaled BF16 factors; the vLLM manager scales once."""
    if set(pulled) != set(plan.pulls):
        raise ValueError("Pulled slices must match the complete consumer plan")
    for pull, tensor in pulled.items():
        shape = tuple(
            b - a for a, b in zip(pull.source_slice.starts, pull.source_slice.stops)
        )
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError("Pulled slices must preserve exact FP32 shape and dtype")
    factors = {
        module.module_name: (
            [
                torch.empty(pair[0], dtype=torch.bfloat16, device=device)
                for pair in module.factor_shapes
            ],
            [
                torch.empty(pair[1], dtype=torch.bfloat16, device=device)
                for pair in module.factor_shapes
            ],
        )
        for module in plan.receiver_plan.modules
    }
    for copy in plan.copies:
        tensor = pulled[plan.pulls[copy.pull_index]]
        shape = tuple(b - a for a, b in zip(copy.starts, copy.stops))
        destination = factors[copy.module_name][copy.component][copy.factor_index]
        destination[tuple(slice(a, b) for a, b in zip(copy.starts, copy.stops))].copy_(
            tensor.reshape(shape)
        )
    return factors
