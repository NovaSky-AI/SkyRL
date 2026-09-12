from dataclasses import replace
from itertools import permutations
from math import prod
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="LoRA transport consumer plans target vLLM LoRA layers")
pytestmark = pytest.mark.vllm
from vllm.lora.layers import (  # noqa: E402
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.layers.fused_moe import (  # noqa: E402
    FusedMoE3DWithLoRA,
    FusedMoEWithLoRA,
)
from vllm.lora.lora_model import LoRAModel  # noqa: E402
from vllm.lora.lora_weights import PackedLoRALayerWeights  # noqa: E402
from vllm.lora.peft_helper import PEFTHelper  # noqa: E402

from skyrl.backends.skyrl_train.weight_sync.lora_layout import (  # noqa: E402
    convert_moe_experts_lora_to_vllm,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (  # noqa: E402
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    reconstruct_lora_bridge_tensors,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (  # noqa: E402
    assemble_lora_consumer_factors,
    build_lora_consumer_plan,
)

local_adapter = pytest.importorskip(
    "vllm.lora.local_adapter", reason="Requires the explicit local-factor vLLM fork API"
)
LocalLoRAModulePlan = local_adapter.LocalLoRAModulePlan
LocalLoRAPlan = local_adapter.LocalLoRAPlan


def _values(shape, offset=0):
    return (
        torch.arange(prod(shape), dtype=torch.float32).reshape(shape) + offset
    ) / 127 + 1


def _append_source(
    sources,
    tensors,
    key,
    names,
    tensor,
    component,
    tp_axis=None,
    tp_size=1,
    ep_axis=None,
    ep_size=1,
    transform="identity",
):
    ep_shards = (
        tensor.chunk(ep_size, dim=ep_axis)
        if ep_axis is not None
        else [tensor] * ep_size
    )
    for ep_rank, ep_shard in enumerate(ep_shards):
        tp_shards = (
            ep_shard.chunk(tp_size, dim=tp_axis)
            if tp_axis is not None
            else [ep_shard] * tp_size
        )
        for tp_rank, shard in enumerate(tp_shards):
            source_rank = ep_rank * tp_size + tp_rank
            source = LoRABridgeSource(
                key,
                source_rank,
                tuple(names),
                component,
                transform,
                tuple(shard.shape),
                tp_axis,
                tp_rank,
                tp_size,
                ep_axis,
                ep_rank,
                ep_size,
                (),
            )
            sources.append(source)
            tensors[(key, tp_rank, ep_rank)] = shard.contiguous().clone()


def _layout(sources):
    return LoRABridgeSourceLayout(
        "adapter",
        tuple(
            sorted(
                sources,
                key=lambda source: (
                    source.key,
                    source.expert_parallel_rank,
                    source.tensor_parallel_rank,
                    source.source_rank,
                ),
            )
        ),
    )


def _module(
    layout,
    names,
    shapes,
    tp_rank,
    outputs,
    shard_ids=None,
    name="model.proj",
    experts=(),
):
    return LocalLoRAModulePlan(
        name,
        "oracle-wrapper",
        layout,
        tuple(tuple(group) for group in names),
        tuple(shapes),
        "torch.bfloat16",
        tp_rank,
        8,
        16,
        tuple(outputs),
        tuple(shard_ids or [tp_rank] * len(outputs)),
        tuple(experts),
    )


def _execute(source_layout, tensors, module):
    receiver = LocalLoRAPlan(4, 12, ("proj",), (module,))
    plan = build_lora_consumer_plan(source_layout, receiver)
    by_owner = {
        (source.source_rank, source.key): tensors[
            (source.key, source.tensor_parallel_rank, source.expert_parallel_rank)
        ]
        for source in source_layout.sources
    }
    before = {key: tensor.clone() for key, tensor in by_owner.items()}
    pulled = {}
    for pull in plan.pulls:
        exact = by_owner[pull.source_rank, pull.source_slice.key][
            pull.source_slice.indices
        ]
        pulled[pull] = exact.contiguous().clone()
        assert torch.equal(
            pulled[pull].view(torch.uint8), exact.contiguous().view(torch.uint8)
        )
    assert plan.source_bytes == sum(tensor.numel() * 4 for tensor in pulled.values())
    factors = assemble_lora_consumer_factors(plan, pulled, "cpu")[module.module_name]
    source_storages = {
        tensor.untyped_storage().data_ptr() for tensor in by_owner.values()
    }
    for component in factors:
        for tensor in component:
            assert tensor.dtype == torch.bfloat16
            assert tensor.untyped_storage().data_ptr() not in source_storages
    for key, tensor in by_owner.items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    return plan, factors, pulled


@pytest.mark.parametrize("tp_rank", range(8))
@pytest.mark.parametrize("layout", ["row", "column", "merged", "replicated"])
def test_dense_source_rank_shards_match_real_vllm_slicing(tp_rank, layout):
    sources, tensors = [], {}
    names = (
        ["model.proj"] if layout != "merged" else ["model.gate_proj", "model.up_proj"]
    )
    for index, name in enumerate(names):
        for component, shape, axis in (("A", (4, 16), 0), ("B", (24, 4), 1)):
            _append_source(
                sources,
                tensors,
                name + component,
                [name + f".lora_{component}.weight"],
                _values(shape, index * 1000),
                "linear_in" if component == "A" else "linear_out",
                axis,
                2,
            )
    full = reconstruct_lora_bridge_tensors(sources, tensors)
    a = [full[name + ".lora_A.weight"].to(torch.bfloat16) for name in names]
    b = [full[name + ".lora_B.weight"].to(torch.bfloat16) for name in names]
    if layout == "replicated":
        expected_a, expected_b = a, b
    else:
        cls = {
            "row": RowParallelLinearWithLoRA,
            "column": ColumnParallelLinearWithLoRA,
            "merged": MergedColumnParallelLinearWithLoRA,
        }[layout]
        layer = object.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.tp_rank, layer.tp_size = tp_rank, 8
        layer.input_size, layer.output_size = 2, 3
        layer.n_slices = len(names)
        layer.is_merged_col_linear = False
        layer.output_slices, layer.output_ids = (3, 3), (tp_rank, tp_rank)
        if layout == "merged":
            expected_a, expected_b = layer.slice_lora_a(a), layer.slice_lora_b(b)
        else:
            expected_a, expected_b = (
                [layer.slice_lora_a(a[0])],
                [layer.slice_lora_b(b[0])],
            )
    shapes = tuple(
        (tuple(x.shape), tuple(y.shape)) for x, y in zip(expected_a, expected_b)
    )
    module = _module(
        layout, [(name,) for name in names], shapes, tp_rank, [24] * len(names)
    )
    plan, (actual_a, actual_b), _ = _execute(_layout(sources), tensors, module)
    for actual, expected in zip(actual_a + actual_b, expected_a + expected_b):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    total = sum(tensor.numel() * 4 for tensor in tensors.values())
    assert plan.source_bytes <= total
    if layout != "replicated":
        assert plan.source_bytes < total


@pytest.mark.parametrize("tp_rank", range(8))
def test_native_shared_ep_factors_pull_once_and_match_global_peft_packing(
    monkeypatch, tp_rank
):
    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)
    sources, tensors = [], {}
    root = "model.layers.0.mlp.experts"
    for ep in range(2):
        experts = range(ep * 2, ep * 2 + 2)
        for projection, component, shape in (
            ("gate_up", "A", (4, 16)),
            ("gate_proj", "B", (16, 4)),
            ("up_proj", "B", (16, 4)),
            ("down_proj", "A", (4, 16)),
            ("down_proj", "B", (16, 4)),
        ):
            projections = (
                ("gate_proj", "up_proj") if projection == "gate_up" else (projection,)
            )
            names = [
                f"{root}.{expert}.{name}.lora_{component}.weight"
                for expert in experts
                for name in projections
            ]
            _append_source(
                sources,
                tensors,
                f"ep{ep}.{projection}.{component}",
                names,
                _values(shape, 1000 * ep + 100 * len(sources)),
                "linear_in" if component == "A" else "linear_out",
                transform="replicate",
            )
    full = reconstruct_lora_bridge_tensors(sources, tensors)
    model = LoRAModel.from_lora_tensors(
        1,
        {"base_model.model." + name: tensor for name, tensor in full.items()},
        PEFTHelper(
            r=4, lora_alpha=4, target_modules=["gate_proj", "up_proj", "down_proj"]
        ),
        device="cpu",
        dtype=torch.bfloat16,
    )
    packed = PackedLoRALayerWeights.pack_moe(
        [
            model.loras[f"{root}.{expert}.{projection}"]
            for expert in range(4)
            for projection in ("gate_proj", "down_proj", "up_proj")
        ],
        root,
    )
    layer = object.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.tp_rank, layer.tp_size = tp_rank, 8
    layer.fully_sharded = False
    layer.moe_config = SimpleNamespace(intermediate_size_per_partition=2)
    expected_a = [
        layer._slice_w13_a(packed.lora_a[0]),
        layer._slice_w2_a(packed.lora_a[1]),
        layer._slice_w13_a(packed.lora_a[2]),
    ]
    expected_b = [
        layer._slice_w13_b(packed.lora_b[0]),
        layer._slice_w2_b(packed.lora_b[1]),
        layer._slice_w13_b(packed.lora_b[2]),
    ]
    module = _module(
        "moe",
        [
            [f"{root}.{expert}.{projection}" for expert in range(4)]
            for projection in ("gate_proj", "down_proj", "up_proj")
        ],
        [(tuple(a.shape), tuple(b.shape)) for a, b in zip(expected_a, expected_b)],
        tp_rank,
        (16, 16, 16),
        name=root,
        experts=range(4),
    )
    plan, (actual_a, actual_b), _ = _execute(_layout(sources), tensors, module)
    for actual, expected in zip(actual_a + actual_b, expected_a + expected_b):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not torch.equal(actual[0], actual[2])
    # Two EP-owned A/B sets, shared within each pair of experts; the gate/up A is pulled once.
    assert plan.source_bytes == 2 * 4 * (4 * 16 + 2 * 2 * 4 + 4 * 2 + 16 * 4)
    assert len(plan.pulls) == 10
    scaled = PackedLoRALayerWeights(root, 4, [12] * 3, actual_a, actual_b)
    scaled.optimize()
    scaled.optimize()
    for actual, expected in zip(scaled.lora_b, expected_b):
        torch.testing.assert_close(actual, expected * 3, rtol=0, atol=0)


@pytest.mark.parametrize("tp_rank", range(8))
def test_fused_expert_ep_and_rank_shards_match_flattening_and_vllm_half_slicing(
    tp_rank,
):
    sources, tensors = [], {}
    root = "model.layers.0.mlp.experts"
    for projection, component, shape in (
        ("gate_up_proj", "A", (4, 4, 16)),
        ("gate_up_proj", "B", (4, 32, 4)),
        ("down_proj", "A", (4, 4, 16)),
        ("down_proj", "B", (4, 16, 4)),
    ):
        name = f"{root}.{projection}.lora_{component}.weight"
        _append_source(
            sources,
            tensors,
            name,
            [name],
            _values(shape, 1000 * len(sources)),
            "linear_in" if component == "A" else "linear_out",
            1 if component == "A" else 2,
            2,
            0,
            2,
        )
    canonical = convert_moe_experts_lora_to_vllm(
        reconstruct_lora_bridge_tensors(sources, tensors)
    )
    # Reverse only the legacy PEFT flattening, as the real vLLM manager does.
    a = [
        canonical[f"{name}.lora_A.weight"].reshape(4, 4, 16).to(torch.bfloat16)
        for name in (root + ".base_layer", root)
    ]
    b = [
        canonical[f"{name}.lora_B.weight"]
        .reshape(size, 4, 4)
        .permute(2, 0, 1)
        .to(torch.bfloat16)
        for name, size in ((root + ".base_layer", 32), (root, 16))
    ]
    layer = object.__new__(FusedMoE3DWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.tp_rank, layer.tp_size = tp_rank, 8
    layer.fully_sharded = False
    layer._base_model = "GlmMoeDsaForCausalLM"
    layer.moe_config = SimpleNamespace(intermediate_size_per_partition=2)
    expected_a = [layer._slice_w13_a(a[0]), layer._slice_w2_a(a[1])]
    expected_b = [layer._slice_w13_b(b[0]), layer._slice_w2_b(b[1])]
    module = _module(
        "moe_3d",
        [(root + ".base_layer",), (root,)],
        [(tuple(x.shape), tuple(y.shape)) for x, y in zip(expected_a, expected_b)],
        tp_rank,
        (32, 16),
        name=root,
        experts=range(4),
    )
    plan, (actual_a, actual_b), _ = _execute(_layout(sources), tensors, module)
    for actual, expected in zip(actual_a + actual_b, expected_a + expected_b):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert plan.source_bytes == 4 * 4 * (4 * 16 + 4 * 4 + 4 * 2 + 16 * 4)
    w13_pulls = [
        pull for pull in plan.pulls if "gate_up_proj.lora_B" in pull.source_slice.key
    ]
    assert len(w13_pulls) == 8
    assert {pull.source_slice.starts[1] for pull in w13_pulls} == {
        2 * tp_rank,
        16 + 2 * tp_rank,
    }


def test_gated_bridge_split_and_replicated_mla_sources_select_only_consumed_outputs():
    sources, tensors = [], {}
    _append_source(
        sources,
        tensors,
        "A",
        ["model.gate.lora_A.weight", "model.up.lora_A.weight"],
        _values((4, 16)),
        "linear_in",
        0,
        2,
        transform="replicate",
    )
    _append_source(
        sources,
        tensors,
        "B",
        ["model.gate.lora_B.weight", "model.up.lora_B.weight"],
        _values((32, 4)),
        "linear_out",
        1,
        2,
        transform="split_gated_mlp",
    )
    module = _module(
        "merged", [("model.gate",), ("model.up",)], [((4, 16), (2, 4))] * 2, 6, (16, 16)
    )
    _, (a, b), _ = _execute(_layout(sources), tensors, module)
    full = reconstruct_lora_bridge_tensors(sources, tensors)
    for index, name in enumerate(("model.gate", "model.up")):
        torch.testing.assert_close(
            a[index], full[name + ".lora_A.weight"].to(torch.bfloat16), rtol=0, atol=0
        )
        torch.testing.assert_close(
            b[index],
            full[name + ".lora_B.weight"][12:14].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("tp_rank", range(2))
@pytest.mark.parametrize("output_gate", [False, True])
def test_qkv_bridge_split_matches_reconstructed_merged_factors(tp_rank, output_gate):
    sources, tensors = [], {}
    names = ("model.q_proj", "model.k_proj", "model.v_proj")
    rank, hidden, heads, groups, head_size = 4, 16, 4, 2, 2
    qkv_heads = (2 * heads + 2 * groups) if output_gate else (heads + 2 * groups)
    a = _values((rank, hidden), 1000)
    _append_source(
        sources,
        tensors,
        "qkv.A",
        [name + ".lora_A.weight" for name in names],
        a,
        "linear_in",
        transform="replicate",
    )
    b = _values((qkv_heads * head_size, rank), 2000)
    _append_source(
        sources,
        tensors,
        "qkv.B",
        [name + ".lora_B.weight" for name in names],
        b,
        "linear_out",
        0,
        2,
        transform="split_qkv",
    )
    qkv_config = (
        ("num_attention_heads", heads),
        ("num_query_groups", groups),
        ("kv_channels", head_size),
        ("hidden_size", hidden),
        ("attention_output_gate", output_gate),
    )
    sources = [
        replace(source, transform_config=qkv_config)
        if source.key == "qkv.B"
        else source
        for source in sources
    ]
    full = reconstruct_lora_bridge_tensors(sources, tensors)
    output_sizes = (
        heads * head_size * (2 if output_gate else 1),
        groups * head_size,
        groups * head_size,
    )
    local_b = [full[name + ".lora_B.weight"].chunk(2, dim=0)[tp_rank] for name in names]
    module = _module(
        "merged",
        [(name,) for name in names],
        [((rank, hidden), tuple(part.shape)) for part in local_b],
        tp_rank,
        output_sizes,
        shard_ids=(tp_rank, tp_rank, tp_rank),
        name="model.qkv_proj",
    )
    plan, (actual_a, actual_b), _ = _execute(_layout(sources), tensors, module)
    for actual in actual_a:
        torch.testing.assert_close(actual, a.to(torch.bfloat16), rtol=0, atol=0)
    for actual, expected in zip(actual_b, local_b, strict=True):
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
    assert plan.source_bytes < sum(tensor.numel() * 4 for tensor in tensors.values())


@pytest.mark.parametrize(
    "invalid", ["shape", "missing", "overlap", "transform", "dtype", "tp"]
)
def test_incompatible_source_map_fails_before_assembly(invalid):
    sources, tensors = [], {}
    for component, shape in (("A", (4, 16)), ("B", (24, 4))):
        _append_source(
            sources,
            tensors,
            component,
            [f"model.proj.lora_{component}.weight"],
            _values(shape),
            "linear_in" if component == "A" else "linear_out",
        )
    module = _module("column", [("model.proj",)], [((4, 16), (3, 4))], 1, (24,))
    if invalid == "shape":
        module = replace(module, global_input_size=17)
    elif invalid == "missing":
        sources.pop()
    elif invalid == "overlap":
        sources.append(replace(sources[0], key="duplicate"))
    elif invalid == "transform":
        sources[0] = replace(sources[0], transform="split_gdn_in_proj")
    elif invalid == "dtype":
        module = replace(module, dtype="torch.float32")
    else:
        module = replace(module, output_shard_ids=(9,))
    with pytest.raises((ValueError, NotImplementedError)):
        build_lora_consumer_plan(
            _layout(sources), LocalLoRAPlan(4, 4, ("proj",), (module,))
        )


@pytest.mark.parametrize("invalid", ["missing", "dtype", "shape"])
def test_bad_pulled_payload_is_rejected_before_destination_allocation(
    monkeypatch, invalid
):
    sources, tensors = [], {}
    for component, shape in (("A", (4, 16)), ("B", (24, 4))):
        _append_source(
            sources,
            tensors,
            component,
            [f"model.proj.lora_{component}.weight"],
            _values(shape),
            "linear_in" if component == "A" else "linear_out",
        )
    module = _module("column", [("model.proj",)], [((4, 16), (3, 4))], 1, (24,))
    plan, _, pulled = _execute(_layout(sources), tensors, module)
    key = next(iter(pulled))
    if invalid == "missing":
        pulled.pop(key)
    elif invalid == "dtype":
        pulled[key] = pulled[key].to(torch.bfloat16)
    else:
        pulled[key] = pulled[key].flatten()
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("allocated destination before validation"),
    )
    with pytest.raises(ValueError):
        assemble_lora_consumer_factors(plan, pulled, "cpu")


def test_plan_uses_metadata_without_allocating_or_reconstructing_tensors(monkeypatch):
    sources, tensors = [], {}
    for component, shape in (("A", (4, 16)), ("B", (24, 4))):
        _append_source(
            sources,
            tensors,
            component,
            [f"model.proj.lora_{component}.weight"],
            _values(shape),
            "linear_in" if component == "A" else "linear_out",
            0,
            2,
        )
    layout = _layout(sources)
    module = _module("column", [("model.proj",)], [((4, 16), (3, 4))], 1, (24,))
    for operation in ("empty", "zeros", "cat", "stack"):
        monkeypatch.setattr(
            torch,
            operation,
            lambda *args, **kwargs: pytest.fail("materialized tensor during planning"),
        )
    plan = build_lora_consumer_plan(layout, LocalLoRAPlan(4, 4, ("proj",), (module,)))
    assert plan.source_bytes == 4 * (4 * 16 + 3 * 4)


def test_ambiguous_producer_key_cannot_alias_distinct_tp_shards():
    sources, tensors = [], {}
    for component, shape in (("A", (4, 16)), ("B", (24, 4))):
        _append_source(
            sources,
            tensors,
            component,
            [f"model.proj.lora_{component}.weight"],
            _values(shape),
            "linear_in" if component == "A" else "linear_out",
            0,
            2,
        )
    sources = [replace(source, source_rank=0) for source in sources]
    module = _module("column", [("model.proj",)], [((4, 16), (3, 4))], 1, (24,))
    with pytest.raises(ValueError, match="producer key"):
        build_lora_consumer_plan(
            _layout(sources), LocalLoRAPlan(4, 4, ("proj",), (module,))
        )


@pytest.mark.parametrize("prefix", ["", "base_model.model."])
def test_native_glm_prefix_matches_real_peft_parser(monkeypatch, prefix):
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)
    name = "model.layers.0.self_attn.q_b_proj"
    sources, tensors = [], {}
    for component, shape in (("A", (4, 16)), ("B", (24, 4))):
        _append_source(
            sources,
            tensors,
            component,
            [f"{prefix}{name}.lora_{component}.weight"],
            _values(shape),
            "linear_in" if component == "A" else "linear_out",
        )
    module = _module("column", [(name,)], [((4, 16), (3, 4))], 5, (24,), name=name)
    _, (a, b), _ = _execute(_layout(sources), tensors, module)
    full = reconstruct_lora_bridge_tensors(sources, tensors)
    model = LoRAModel.from_lora_tensors(
        1,
        full,
        PEFTHelper(r=4, lora_alpha=4, target_modules=["q_b_proj"]),
        device="cpu",
        dtype=torch.bfloat16,
        weights_mapper=getattr(GlmMoeDsaForCausalLM, "hf_to_vllm_mapper", None),
    )
    assert set(model.loras) == {module.module_name}
    torch.testing.assert_close(a[0], model.loras[name].lora_a, rtol=0, atol=0)
    torch.testing.assert_close(b[0], model.loras[name].lora_b[15:18], rtol=0, atol=0)


@pytest.mark.parametrize("empty", ["modules", "factors", "targets"])
def test_empty_consumer_plan_cannot_register_an_empty_adapter(empty):
    sources, tensors = [], {}
    for component, shape in (("A", (4, 16)), ("B", (24, 4))):
        _append_source(
            sources,
            tensors,
            component,
            [f"model.proj.lora_{component}.weight"],
            _values(shape),
            "linear_in" if component == "A" else "linear_out",
        )
    module = _module("column", [("model.proj",)], [((4, 16), (3, 4))], 1, (24,))
    receiver = LocalLoRAPlan(4, 4, ("proj",), (module,))
    if empty == "modules":
        receiver = replace(receiver, modules=())
    elif empty == "targets":
        receiver = replace(receiver, target_modules=())
    else:
        receiver = replace(receiver, modules=(replace(module, factor_shapes=()),))
    with pytest.raises(ValueError, match="nonempty"):
        build_lora_consumer_plan(_layout(sources), receiver)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_coverage_sweep_accepts_shuffled_rectangles_and_touching_boundaries(order):
    from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
        LoRAConsumerCopy,
        _validate_factor_coverage,
    )

    rectangles = [((0, 0), (2, 2)), ((2, 0), (4, 2)), ((0, 2), (4, 4))]
    copies = [
        LoRAConsumerCopy(index, "module", 0, 0, *rectangles[index]) for index in order
    ]
    _validate_factor_coverage((4, 4), copies)


@pytest.mark.parametrize(
    "rectangles, error",
    [
        ([((0, 0), (3, 4)), ((2, 0), (3, 4))], "overlaps"),
        ([((0, 0), (3, 4))], "complete"),
        ([((0, 0), (4, 5))], "exceeds"),
        ([((0, 0), (0, 4)), ((0, 0), (4, 4))], "exceeds"),
    ],
)
def test_coverage_sweep_rejects_overlap_even_when_volume_matches_or_a_hole(
    rectangles, error
):
    from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
        LoRAConsumerCopy,
        _validate_factor_coverage,
    )

    copies = [
        LoRAConsumerCopy(index, "module", 0, 0, start, stop)
        for index, (start, stop) in enumerate(rectangles)
    ]
    with pytest.raises(ValueError, match=error):
        _validate_factor_coverage((4, 4), copies)
