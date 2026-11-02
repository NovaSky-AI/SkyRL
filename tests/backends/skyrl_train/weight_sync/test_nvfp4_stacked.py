"""CPU tests for the stacked NVFP4 expert sync (``weight_sync/nvfp4_stacked.py`` + the receiver's pure parts).

The TE kernel is GPU-only, so ``quantize_nvfp4_matrix`` is replaced by a deterministic stand-in whose outputs
depend on the input bytes: equal bytes after stacking/unstacking then mean the stacking itself is lossless.
"""

import random

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import nvfp4, nvfp4_stacked
from skyrl.backends.skyrl_train.weight_sync.nvfp4_stacked import (
    NVFP4_STACKED_PREFIX,
    STACKED_LEAVES,
    Nvfp4StackedExpertSerializer,
    Nvfp4StackedExpertSpecSerializer,
    local_to_global_expert_ids,
    max_stacked_wire_bytes,
    parse_stacked_wire_name,
    unstack_to_per_expert,
)

_SETTINGS = nvfp4.Nvfp4WeightQuantSettings(
    use_4over6=False, e4m3_max=448, err_mode="MAE", row_scaled_activation=True, backward_override=None
)
NUM_LAYERS, NUM_EXPERTS, HIDDEN, INTER = 3, 4, 32, 16


def _fake_quantize(weight, settings):
    rows, cols = weight.shape
    w = weight.float()
    packed = ((w[:, : cols // 2] * 37 + w[:, cols // 2 :] * 11).abs() * 5).to(torch.uint8)
    scale = (w.abs().reshape(rows, cols // 16, 16).amax(-1) + 0.25).clamp(max=400).to(torch.float8_e4m3fn)
    decode = (w.abs().amax() + 1.0).reshape(()).float()
    return packed.contiguous(), scale.contiguous(), decode


@pytest.fixture(autouse=True)
def fake_quantizer(monkeypatch):
    monkeypatch.setattr(nvfp4, "quantize_nvfp4_matrix", _fake_quantize)
    monkeypatch.setattr(nvfp4_stacked, "quantize_nvfp4_matrix", _fake_quantize)


def _stream(seed=0, hp_tail_layer=NUM_LAYERS - 1):
    """A shuffled export-like stream: experts of every layer in random order, plus non-expert tensors."""
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    out = [("model.embed_tokens.weight", torch.randn(8, HIDDEN, generator=gen).to(torch.bfloat16))]
    for layer in range(NUM_LAYERS):
        out.append((f"model.layers.{layer}.self_attn.q_proj.weight", torch.randn(HIDDEN, HIDDEN, generator=gen)))
        experts = []
        for e in range(NUM_EXPERTS):
            base = f"model.layers.{layer}.mlp.experts.{e}"
            experts.append((f"{base}.gate_proj.weight", torch.randn(INTER, HIDDEN, generator=gen)))
            experts.append((f"{base}.up_proj.weight", torch.randn(INTER, HIDDEN, generator=gen)))
            experts.append((f"{base}.down_proj.weight", torch.randn(HIDDEN, INTER, generator=gen)))
        rng.shuffle(experts)
        out.extend(experts)
        out.append((f"model.layers.{layer}.mlp.gate.weight", torch.randn(NUM_EXPERTS, HIDDEN, generator=gen)))
    out.append(("lm_head.weight", torch.randn(8, HIDDEN, generator=gen)))
    return out


def _run(serializer, stream):
    out = [(n, t) for name, tensor in stream for n, t in serializer(name, tensor)]
    serializer.finish()
    return out


def _bits(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _per_expert():
    return _run(nvfp4.Nvfp4ExpertSerializer(_SETTINGS, NUM_LAYERS, 1), _stream())


def _stacked():
    return _run(Nvfp4StackedExpertSerializer(_SETTINGS, NUM_LAYERS, 1, num_experts=NUM_EXPERTS), _stream())


def test_stacked_names_shapes_and_order():
    stacked = _stacked()
    names = [n for n, _ in stacked]
    for layer in range(NUM_LAYERS - 1):  # NVFP4 layers
        prefix = f"model.layers.{layer}.mlp.experts"
        got = [n for n in names if n.startswith(f"{NVFP4_STACKED_PREFIX}{prefix}.")]
        assert got == [f"{NVFP4_STACKED_PREFIX}{prefix}.{leaf}" for leaf in STACKED_LEAVES]
        # The layer goes out after its last expert, before the layer's router.
        assert names.index(got[-1]) < names.index(f"model.layers.{layer}.mlp.gate.weight")
    shapes = {n: (t.dtype, tuple(t.shape)) for n, t in stacked}
    p = f"{NVFP4_STACKED_PREFIX}model.layers.0.mlp.experts"
    assert shapes[f"{p}.w13_weight"] == (torch.uint8, (NUM_EXPERTS, 2 * INTER, HIDDEN // 2))
    assert shapes[f"{p}.w13_weight_scale"] == (torch.float8_e4m3fn, (NUM_EXPERTS, 2 * INTER, HIDDEN // 16))
    assert shapes[f"{p}.w13_weight_scale_2"] == (torch.float32, (NUM_EXPERTS, 2))
    assert shapes[f"{p}.w13_input_scale"] == (torch.float32, (NUM_EXPERTS, 2))
    assert shapes[f"{p}.w2_weight"] == (torch.uint8, (NUM_EXPERTS, HIDDEN, INTER // 2))
    assert shapes[f"{p}.w2_weight_scale"] == (torch.float8_e4m3fn, (NUM_EXPERTS, HIDDEN, INTER // 16))
    assert shapes[f"{p}.w2_weight_scale_2"] == (torch.float32, (NUM_EXPERTS,))
    assert shapes[f"{p}.w2_input_scale"] == (torch.float32, (NUM_EXPERTS,))
    # BF16 tail layer: per-expert BF16 pass-through, not stacked.
    assert not any("layers.2.mlp.experts" in n and n.startswith(NVFP4_STACKED_PREFIX) for n in names)
    assert "model.layers.2.mlp.experts.0.gate_proj.weight" in names


def test_stacked_bytes_equal_per_expert_bytes():
    """Unstacking each stacked layer reproduces the per-expert stream exactly (names, dtypes, shapes, bytes)."""
    per_expert = dict(_per_expert())
    stacked = _stacked()
    rebuilt = {}
    layers = {}
    for name, tensor in stacked:
        parsed = parse_stacked_wire_name(name)
        if parsed is None:
            rebuilt[name] = tensor
        else:
            layers.setdefault(parsed[0], {})[parsed[1]] = tensor
    for layer_prefix, stacks in layers.items():
        rebuilt.update(unstack_to_per_expert(layer_prefix, stacks))
    assert rebuilt.keys() == per_expert.keys()
    for name, ref in per_expert.items():
        got = rebuilt[name]
        assert got.dtype == ref.dtype and tuple(got.shape) == tuple(ref.shape), name
        assert torch.equal(_bits(got), _bits(ref)), name


def test_w13_rows_are_gate_then_up_and_scale2_shared():
    per_expert = dict(_per_expert())
    stacks = {
        parse_stacked_wire_name(n)[1]: t
        for n, t in _stacked()
        if n.startswith(f"{NVFP4_STACKED_PREFIX}model.layers.1.mlp.experts.")
    }
    for e in range(NUM_EXPERTS):
        base = f"model.layers.1.mlp.experts.{e}"
        assert torch.equal(stacks["w13_weight"][e, :INTER], per_expert[f"{base}.gate_proj.weight"])
        assert torch.equal(stacks["w13_weight"][e, INTER:], per_expert[f"{base}.up_proj.weight"])
        assert stacks["w13_weight_scale_2"][e, 0] == stacks["w13_weight_scale_2"][e, 1]
        assert stacks["w13_weight_scale_2"][e, 0] == per_expert[f"{base}.gate_proj.weight_scale_2"]
        assert torch.equal(stacks["w2_weight"][e], per_expert[f"{base}.down_proj.weight"])
    assert torch.equal(stacks["w13_input_scale"], torch.ones(NUM_EXPERTS, 2))
    assert torch.equal(stacks["w2_input_scale"], torch.ones(NUM_EXPERTS))


def test_spec_serializer_declares_the_stacked_stream():
    real = _stacked()
    specs = [(n, nvfp4.TensorSpec(t.dtype, tuple(t.shape))) for n, t in _stream()]
    declared = _run(Nvfp4StackedExpertSpecSerializer(_SETTINGS, NUM_LAYERS, 1, num_experts=NUM_EXPERTS), specs)
    assert [(n, s.dtype, tuple(s.shape)) for n, s in declared] == [(n, t.dtype, tuple(t.shape)) for n, t in real]


def test_metadata_shrinks():
    assert len(_stacked()) < len(_per_expert())
    # Per NVFP4 layer: 8 stacked tensors instead of 12 per expert.
    n_nvfp4_layers = NUM_LAYERS - 1
    assert len(_per_expert()) - len(_stacked()) == n_nvfp4_layers * (12 * NUM_EXPERTS - 8)


def test_incomplete_layer_and_duplicate_expert_raise():
    serializer = Nvfp4StackedExpertSerializer(_SETTINGS, NUM_LAYERS, 0, num_experts=NUM_EXPERTS)
    for name, tensor in _stream()[:5]:
        list(serializer(name, tensor))
    with pytest.raises(RuntimeError, match="unpaired|incomplete MoE layers"):
        serializer.finish()

    serializer = Nvfp4StackedExpertSerializer(_SETTINGS, NUM_LAYERS, 0, num_experts=NUM_EXPERTS)
    down = ("model.layers.0.mlp.experts.1.down_proj.weight", torch.randn(HIDDEN, INTER))
    list(serializer(*down))
    with pytest.raises(ValueError, match="twice"):
        list(serializer(*down))
    with pytest.raises(ValueError, match="out of range"):
        list(serializer("model.layers.0.mlp.experts.9.down_proj.weight", torch.randn(HIDDEN, INTER)))


def test_parse_wire_name():
    assert parse_stacked_wire_name("model.layers.0.mlp.experts.0.gate_proj.weight") is None
    assert parse_stacked_wire_name(f"{NVFP4_STACKED_PREFIX}model.layers.3.mlp.experts.w2_weight_scale_2") == (
        "model.layers.3.mlp.experts",
        "w2_weight_scale_2",
    )
    with pytest.raises(ValueError):
        parse_stacked_wire_name(f"{NVFP4_STACKED_PREFIX}model.layers.3.mlp.experts.bogus")


def test_ep_slicing_math():
    # EP2, linear placement: rank 1 owns global experts 4..7 in local slots 0..3.
    g2l = [-1, -1, -1, -1, 0, 1, 2, 3]
    assert local_to_global_expert_ids(g2l, 4) == [4, 5, 6, 7]
    # Round-robin placement.
    assert local_to_global_expert_ids([0, -1, 1, -1, 2, -1, 3, -1], 4) == [0, 2, 4, 6]
    with pytest.raises(ValueError):
        local_to_global_expert_ids([0, 0, -1, -1], 2)  # two globals in one slot (redundant experts)
    with pytest.raises(ValueError):
        local_to_global_expert_ids([0, -1, -1, -1], 2)  # an unowned local slot
    stack = torch.arange(8 * 3).reshape(8, 3)
    local = stack.index_select(0, torch.tensor(local_to_global_expert_ids(g2l, 4)))
    assert torch.equal(local, stack[4:])


def test_packed_buffer_floor_for_qwen3_30b():
    # 128 experts x [2*768, 2048/2] uint8 = 192 MiB; w2 is half that.
    assert max_stacked_wire_bytes(128, 2048, 768) == 128 * 1536 * 1024
    from skyrl.backends.skyrl_train.weight_sync import nvfp4_stacked as mod

    assert mod.max_stacked_wire_bytes(4, 32, 16) == 4 * 32 * 16


def test_flag_defaults_off():
    from skyrl.train.config.config import MegatronConfig

    assert MegatronConfig().expert_nvfp4_stacked_sync is True


class _FakeLayer(torch.nn.Module):
    def __init__(self, num_local, shapes):
        super().__init__()
        for leaf, (dtype, shape) in shapes.items():
            self.register_parameter(leaf, torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))


def test_copy_stacks_into_writes_the_local_rows():
    from skyrl.backends.skyrl_train.weight_sync.nvfp4_stacked_receive import (
        copy_stacks_into,
    )

    stacks = {
        parse_stacked_wire_name(n)[1]: t
        for n, t in _stacked()
        if n.startswith(f"{NVFP4_STACKED_PREFIX}model.layers.0.mlp.experts.")
    }
    l2g = [2, 3]  # EP2 rank 1 of 4 experts
    shapes = {leaf: (t.dtype, (2, *t.shape[1:])) for leaf, t in stacks.items()}
    shapes["w13_input_scale"] = (torch.float32, (NUM_EXPERTS, 2))  # global scale factors
    shapes["w2_input_scale"] = (torch.float32, (NUM_EXPERTS,))
    layer = _FakeLayer(2, shapes)
    sources = {leaf: ("global" if leaf.endswith("input_scale") else "local") for leaf in STACKED_LEAVES}
    copy_stacks_into(layer, stacks, (torch.tensor(l2g), sources))
    for leaf in STACKED_LEAVES:
        expected = stacks[leaf] if sources[leaf] == "global" else stacks[leaf][2:]
        assert torch.equal(_bits(getattr(layer, leaf).data), _bits(expected)), leaf


def test_receiver_buffers_partial_layers_across_batches(monkeypatch):
    from skyrl.backends.skyrl_train.weight_sync import nvfp4_stacked_receive as recv

    loaded = []
    receiver = recv.Nvfp4StackedReceiver(model=None, fallback_load=None)
    monkeypatch.setattr(receiver, "_load_layer", lambda prefix, stacks: loaded.append((prefix, dict(stacks))))
    stacked = [(n, t) for n, t in _stacked() if n.startswith(NVFP4_STACKED_PREFIX)]
    first, second = stacked[:3], stacked[3:]
    for n, t in first:
        receiver.accept(n, t)
    receiver.end_of_batch()
    held = next(iter(receiver.pending.values()))
    assert all(held[parse_stacked_wire_name(n)[1]].data_ptr() != t.data_ptr() for n, t in first)  # cloned
    for n, t in second:
        receiver.accept(n, t)
    receiver.finish()
    assert [p for p, _ in loaded] == [f"model.layers.{i}.mlp.experts" for i in range(NUM_LAYERS - 1)]
    assert all(set(s) == set(STACKED_LEAVES) for _, s in loaded)

    receiver = recv.Nvfp4StackedReceiver(model=None, fallback_load=None)
    receiver.accept(*stacked[0])
    with pytest.raises(RuntimeError, match="incomplete layers"):
        receiver.finish()
    with pytest.raises(ValueError, match="twice"):
        receiver.accept(*stacked[0])
