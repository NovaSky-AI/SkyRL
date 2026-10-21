"""CPU unit tests for importing a PEFT LoRA adapter (``lora.init_adapter_path``).

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest -s tests/backends/skyrl_train/distributed/test_lora_import.py

The invariant under test everywhere is that the imported adapter computes the same
``delta_W = scale * B @ A`` per HF projection as the adapter file, whether the pieces are
re-keyed (namespace folds), stacked into one block-diagonal adapter (fused Megatron
modules, Tinker's GatedDeltaNet ``in_proj_q/k/v``), or re-scaled for vLLM's uniform
``lora_alpha / r``.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.distributed.lora_import import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    PEFT_PREFIX,
    LoraPiece,
    canonical_module_name,
    check_adapter_matches_config,
    fold_peft_patterns_for_vllm,
    fold_tinker_gdn_aliases,
    hf_name_variants,
    parse_peft_adapter,
    peft_module_scale,
    peft_pattern_value,
    read_peft_adapter,
    remap_peft_state_dict,
    stack_lora_pieces,
)

HIDDEN = 16
R = 4
ALPHA = 8


def _config(**overrides):
    config = {
        "peft_type": "LORA",
        "r": R,
        "lora_alpha": ALPHA,
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }
    config.update(overrides)
    return config


def _lora(module: str, out: int, rank: int = R, seed: int = 0, prefix: str = PEFT_PREFIX, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return {
        f"{prefix}{module}{LORA_A_SUFFIX}": torch.randn(rank, HIDDEN, generator=g).to(dtype),
        f"{prefix}{module}{LORA_B_SUFFIX}": torch.randn(out, rank, generator=g).to(dtype),
    }


def _state(*entries):
    state = {}
    for entry in entries:
        state.update(entry)
    return state


def _delta(state, module, scale):
    a = state[f"{PEFT_PREFIX}{module}{LORA_A_SUFFIX}"].float()
    b = state[f"{PEFT_PREFIX}{module}{LORA_B_SUFFIX}"].float()
    return scale * (b @ a)


# ---------------------------------------------------------------------------
# Parsing and PEFT scale rules
# ---------------------------------------------------------------------------


def test_peft_pattern_value_matches_module_tail_like_peft():
    pattern = {"q_proj": 8, "layers\\.1\\.self_attn\\.k_proj": 16}
    assert peft_pattern_value(pattern, "model.layers.0.self_attn.q_proj", R) == 8
    assert peft_pattern_value(pattern, "model.layers.1.self_attn.k_proj", R) == 16
    assert peft_pattern_value(pattern, "model.layers.0.self_attn.k_proj", R) == R
    # The key must match the whole tail, not a substring of a leaf.
    assert peft_pattern_value({"proj": 8}, "model.layers.0.self_attn.q_proj", R) == R


def test_parse_pieces_carry_pattern_and_rslora_scales():
    config = _config(rank_pattern={"v_proj": 8}, alpha_pattern={"k_proj": 32})
    state = _state(
        _lora("model.layers.0.self_attn.q_proj", 32),
        _lora("model.layers.0.self_attn.k_proj", 32, seed=1),
        _lora("model.layers.0.self_attn.v_proj", 32, rank=8, seed=2),
    )
    adapter = parse_peft_adapter(config, state)
    assert adapter.r == R and adapter.lora_alpha == ALPHA and adapter.base_scale == 2.0
    assert adapter.pieces["model.layers.0.self_attn.q_proj"].scale == ALPHA / R
    assert adapter.pieces["model.layers.0.self_attn.k_proj"].scale == 32 / R
    assert adapter.pieces["model.layers.0.self_attn.v_proj"].scale == ALPHA / 8
    assert adapter.max_rank() == 8
    assert adapter.leaf_modules() == ["k_proj", "q_proj", "v_proj"]

    rs = parse_peft_adapter(_config(use_rslora=True), _state(_lora("m.q_proj", 32)))
    assert rs.pieces["m.q_proj"].scale == pytest.approx(ALPHA / R**0.5)
    assert peft_module_scale(_config(use_rslora=True), "m.q_proj", R) == pytest.approx(ALPHA / R**0.5)


@pytest.mark.parametrize(
    "config, state, message",
    [
        (_config(use_dora=True), _state(_lora("m.q_proj", 32)), "DoRA"),
        (_config(modules_to_save=["lm_head"]), _state(_lora("m.q_proj", 32)), "modules_to_save"),
        (_config(), {f"{PEFT_PREFIX}m.q_proj{LORA_A_SUFFIX}": torch.zeros(R, HIDDEN)}, "both lora_A and lora_B"),
        (_config(), _state(_lora("m.q_proj", 32, rank=8)), "rank 8 but adapter_config.json gives 4"),
        (_config(), {**_lora("m.q_proj", 32), f"{PEFT_PREFIX}m.lm_head.weight": torch.zeros(2, 2)}, "non-LoRA tensor"),
        (
            _config(),
            {
                f"{PEFT_PREFIX}m.experts.w1{LORA_A_SUFFIX}": torch.zeros(2, R, HIDDEN),
                f"{PEFT_PREFIX}m.experts.w1{LORA_B_SUFFIX}": torch.zeros(2, 8, R),
            },
            "only 2-D",
        ),
    ],
)
def test_parse_rejects_unsupported_adapters(config, state, message):
    with pytest.raises(ValueError, match=message):
        parse_peft_adapter(config, state)


@pytest.mark.parametrize("namespace", ["model.", "model.language_model.", ""])
def test_parse_renames_tinker_unembed_tokens_to_lm_head(namespace):
    """Tinker's own export spells the (untied) output projection ``model.unembed_tokens``."""
    config = _config(target_modules=["q_proj", "unembed_tokens"], alpha_pattern={"unembed_tokens": 32})
    state = _state(_lora("model.layers.0.self_attn.q_proj", 32), _lora(f"{namespace}unembed_tokens", 64, seed=1))
    adapter = parse_peft_adapter(config, state)
    assert sorted(adapter.pieces) == ["lm_head", "model.layers.0.self_attn.q_proj"]
    head = adapter.pieces["lm_head"]
    # The pattern written against the file's name still sets the scale.
    assert head.scale == 32 / R
    torch.testing.assert_close(head.delta(), _delta(state, f"{namespace}unembed_tokens", 32 / R))
    assert adapter.config["target_modules"] == ["lm_head", "q_proj"]
    assert adapter.leaf_modules() == ["lm_head", "q_proj"]


def test_parse_rejects_unembed_tokens_next_to_lm_head():
    state = _state(_lora("lm_head", 64), _lora("model.unembed_tokens", 64, seed=1))
    with pytest.raises(ValueError, match="both name 'lm_head'"):
        parse_peft_adapter(_config(), state)


def test_check_adapter_matches_config():
    adapter = parse_peft_adapter(_config(), _state(_lora("m.q_proj", 32)))
    check_adapter_matches_config(adapter, rank=R, alpha=ALPHA)
    with pytest.raises(ValueError, match="r=4, lora_alpha=8"):
        check_adapter_matches_config(adapter, rank=32, alpha=ALPHA)
    with pytest.raises(ValueError, match="alpha=16"):
        check_adapter_matches_config(adapter, rank=R, alpha=16)


def test_read_peft_adapter_round_trips_a_directory(tmp_path):
    state = _state(_lora("model.layers.0.self_attn.q_proj", 32, dtype=torch.bfloat16))
    save_file(state, os.path.join(tmp_path, "adapter_model.safetensors"))
    with open(os.path.join(tmp_path, "adapter_config.json"), "w") as f:
        json.dump(_config(), f)
    adapter = read_peft_adapter(str(tmp_path))
    piece = adapter.pieces["model.layers.0.self_attn.q_proj"]
    assert piece.lora_a.dtype == torch.bfloat16 and piece.rank == R
    os.remove(os.path.join(tmp_path, "adapter_model.safetensors"))
    with pytest.raises(FileNotFoundError, match="build_lora_adapter"):
        read_peft_adapter(str(tmp_path))


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_canonical_module_name_drops_namespaces_and_prefix():
    for name in (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.q_proj",
    ):
        assert canonical_module_name(name) == "layers.0.self_attn.q_proj"
    assert canonical_module_name("base_model.model.lm_head") == "lm_head"
    assert "model.language_model.layers.0.self_attn.q_proj" in hf_name_variants("model.layers.0.self_attn.q_proj")


def test_remap_peft_state_dict_onto_model_namespace():
    state = _state(_lora("model.language_model.layers.0.self_attn.q_proj", 32), _lora("lm_head", 64, seed=3))
    model_modules = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "lm_head"]
    remapped = remap_peft_state_dict(state, model_modules)
    assert set(remapped) == {
        f"{PEFT_PREFIX}model.layers.0.self_attn.q_proj{LORA_A_SUFFIX}",
        f"{PEFT_PREFIX}model.layers.0.self_attn.q_proj{LORA_B_SUFFIX}",
        f"{PEFT_PREFIX}lm_head{LORA_A_SUFFIX}",
        f"{PEFT_PREFIX}lm_head{LORA_B_SUFFIX}",
    }
    with pytest.raises(ValueError, match="no LoRA layer in the model"):
        remap_peft_state_dict(state, ["model.layers.0.self_attn.k_proj"])


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _pieces(*specs):
    pieces = []
    for i, (module, out, rank, scale) in enumerate(specs):
        g = torch.Generator().manual_seed(100 + i)
        pieces.append(
            LoraPiece(
                module=module,
                lora_a=torch.randn(rank, HIDDEN, generator=g),
                lora_b=torch.randn(out, rank, generator=g),
                scale=scale,
            )
        )
    return pieces


@pytest.mark.parametrize("pad", [1, 4, 8])
def test_stack_lora_pieces_is_block_diagonal_and_exact(pad):
    pieces = _pieces(("q", 24, 4, 2.0), ("k", 8, 4, 2.0), ("v", 8, 6, 0.5))
    target_scale = ALPHA / 14  # what the fused module would apply at the summed rank
    stacked = stack_lora_pieces(pieces, target_scale=target_scale, pad_to_multiple=pad)
    assert not stacked.shared_a
    assert stacked.rank == -(-14 // pad) * pad
    assert stacked.lora_a.shape == (stacked.rank, HIDDEN)
    offset = 0
    for piece, lora_b in zip(pieces, stacked.lora_bs):
        assert lora_b.shape == (piece.lora_b.shape[0], stacked.rank)
        torch.testing.assert_close(target_scale * lora_b @ stacked.lora_a, piece.delta())
        # Only this piece's column block is populated.
        mask = torch.ones(stacked.rank, dtype=torch.bool)
        mask[offset : offset + piece.rank] = False
        assert torch.count_nonzero(lora_b[:, mask]) == 0
        offset += piece.rank
    # The padded rank rows and columns are zero.
    assert torch.count_nonzero(stacked.lora_a[14:]) == 0


def test_stack_lora_pieces_keeps_a_shared_a_at_its_rank():
    base = _pieces(("q", 24, 4, 2.0))[0]
    pieces = [
        LoraPiece("q", base.lora_a, base.lora_b, 2.0),
        LoraPiece("k", base.lora_a.clone(), torch.randn(8, 4), 2.0),
        LoraPiece("v", base.lora_a.clone(), torch.randn(8, 4), 2.0),
    ]
    stacked = stack_lora_pieces(pieces, target_scale=1.0, pad_to_multiple=1)
    assert stacked.shared_a and stacked.rank == 4
    torch.testing.assert_close(stacked.lora_a, base.lora_a)
    for piece, lora_b in zip(pieces, stacked.lora_bs):
        torch.testing.assert_close(lora_b @ stacked.lora_a, piece.delta())

    # A different scale on one piece breaks sharing: it needs its own columns.
    pieces[1].scale = 1.0
    assert not stack_lora_pieces(pieces, target_scale=1.0).shared_a


def test_stack_lora_pieces_rejects_mismatched_inputs():
    pieces = _pieces(("q", 24, 4, 2.0))
    bad = LoraPiece("k", torch.randn(4, HIDDEN + 1), torch.randn(8, 4), 2.0)
    with pytest.raises(ValueError, match="in_features"):
        stack_lora_pieces(pieces + [bad], target_scale=1.0)
    with pytest.raises(ValueError, match="at least one"):
        stack_lora_pieces([], target_scale=1.0)


def test_fold_tinker_gdn_aliases_builds_in_proj_qkv_block_adapter():
    prefix = "model.language_model.layers.0.linear_attn"
    q_rows, k_rows, v_rows = 8, 8, 24
    state = _state(
        _lora(f"{prefix}.in_proj_q", q_rows, seed=1),
        _lora(f"{prefix}.in_proj_k", k_rows, seed=2),
        _lora(f"{prefix}.in_proj_v", v_rows, seed=3),
        _lora(f"{prefix}.out_proj", HIDDEN, seed=4),
        _lora("model.language_model.layers.1.self_attn.q_proj", 32, seed=5),
    )
    config = _config(target_modules=["in_proj_q", "in_proj_k", "in_proj_v", "out_proj", "q_proj"])
    adapter = fold_tinker_gdn_aliases(parse_peft_adapter(config, state))

    fused = adapter.pieces[f"{prefix}.in_proj_qkv"]
    assert not any(m.endswith(("in_proj_q", "in_proj_k", "in_proj_v")) for m in adapter.pieces)
    assert fused.rank == 3 * R and fused.lora_b.shape == (q_rows + k_rows + v_rows, 3 * R)
    expected = torch.cat(
        [_delta(state, f"{prefix}.in_proj_{p}", ALPHA / R) for p in ("q", "k", "v")],
        dim=0,
    )
    torch.testing.assert_close(fused.delta(), expected)

    # PEFT would rebuild the same scale from the patterns.
    assert adapter.config["rank_pattern"] == {"in_proj_qkv": 3 * R}
    assert adapter.config["alpha_pattern"] == {"in_proj_qkv": 3 * ALPHA}
    assert peft_module_scale(adapter.config, f"{prefix}.in_proj_qkv", 3 * R) == fused.scale == ALPHA / R
    assert adapter.config["target_modules"] == ["in_proj_qkv", "out_proj", "q_proj"]
    # Unrelated pieces pass through untouched.
    assert adapter.pieces[f"{prefix}.out_proj"].lora_a is state[f"{PEFT_PREFIX}{prefix}.out_proj{LORA_A_SUFFIX}"]
    # Re-folding is a no-op.
    assert fold_tinker_gdn_aliases(adapter) is adapter


@pytest.mark.parametrize("use_rslora", [False, True])
def test_fold_tinker_gdn_aliases_overrides_win_over_broad_patterns(use_rslora):
    """A broad existing pattern key such as ".*" must not shadow the fused in_proj_qkv entries."""
    prefix = "m.layers.0.linear_attn"
    state = _state(*(_lora(f"{prefix}.in_proj_{p}", 8, seed=i) for i, p in enumerate("qkv")))
    state.update(_lora("m.layers.0.self_attn.o_proj", 8, seed=9))
    config = _config(rank_pattern={".*": R}, alpha_pattern={".*": 2 * ALPHA}, use_rslora=use_rslora)
    original = parse_peft_adapter(config, state)
    folded = fold_tinker_gdn_aliases(original)

    fused = folded.pieces[f"{prefix}.in_proj_qkv"]
    assert fused.rank == 3 * R and fused.scale == original.pieces[f"{prefix}.in_proj_q"].scale
    # The returned config must describe the folded tensors: re-parsing them succeeds and gives
    # every module the scale it had before the fold.
    reparsed = parse_peft_adapter(
        folded.config,
        _state(
            *(
                {f"{PEFT_PREFIX}{m}{LORA_A_SUFFIX}": p.lora_a, f"{PEFT_PREFIX}{m}{LORA_B_SUFFIX}": p.lora_b}
                for m, p in folded.pieces.items()
            )
        ),
    )
    assert reparsed.pieces[f"{prefix}.in_proj_qkv"].scale == pytest.approx(fused.scale)
    o_proj = "m.layers.0.self_attn.o_proj"
    assert reparsed.pieces[o_proj].scale == original.pieces[o_proj].scale
    assert list(folded.config["rank_pattern"]) == ["in_proj_qkv", ".*"]


def test_fold_tinker_gdn_aliases_requires_all_three():
    state = _state(_lora("m.layers.0.linear_attn.in_proj_q", 8), _lora("m.layers.0.linear_attn.in_proj_k", 8, seed=1))
    with pytest.raises(ValueError, match=r"lacks \['in_proj_v'\]"):
        fold_tinker_gdn_aliases(parse_peft_adapter(_config(), state))


def test_fold_tinker_gdn_aliases_rejects_a_layer_without_q_next_to_a_complete_one():
    """A layer missing in_proj_q must raise even when another layer folds cleanly."""
    state = _state(
        _lora("m.layers.0.linear_attn.in_proj_q", 8, seed=1),
        _lora("m.layers.0.linear_attn.in_proj_k", 8, seed=2),
        _lora("m.layers.0.linear_attn.in_proj_v", 8, seed=3),
        _lora("m.layers.1.linear_attn.in_proj_k", 8, seed=4),
        _lora("m.layers.1.linear_attn.in_proj_v", 8, seed=5),
    )
    with pytest.raises(ValueError, match=r"m\.layers\.1\.linear_attn .* lacks \['in_proj_q'\]"):
        fold_tinker_gdn_aliases(parse_peft_adapter(_config(), state))


# ---------------------------------------------------------------------------
# vLLM sync
# ---------------------------------------------------------------------------


def test_fold_peft_patterns_for_vllm_makes_uniform_scale_exact():
    config = _config(rank_pattern={"in_proj_qkv": 12}, alpha_pattern={"in_proj_qkv": 24, "k_proj": 2}, use_rslora=False)
    state = _state(
        _lora("m.layers.0.linear_attn.in_proj_qkv", 40, rank=12, seed=1, dtype=torch.bfloat16),
        _lora("m.layers.0.self_attn.k_proj", 8, seed=2, dtype=torch.bfloat16),
        _lora("m.layers.0.self_attn.q_proj", 8, seed=3, dtype=torch.bfloat16),
    )
    adapter = parse_peft_adapter(config, state)
    folded, folded_config = fold_peft_patterns_for_vllm(state, config)
    assert folded_config["rank_pattern"] == {} and folded_config["alpha_pattern"] == {}
    assert not folded_config["use_rslora"]
    uniform = folded_config["lora_alpha"] / folded_config["r"]
    for module, piece in adapter.pieces.items():
        a = folded[f"{PEFT_PREFIX}{module}{LORA_A_SUFFIX}"].float()
        b = folded[f"{PEFT_PREFIX}{module}{LORA_B_SUFFIX}"].float()
        assert b.dtype == torch.float32 and folded[f"{PEFT_PREFIX}{module}{LORA_B_SUFFIX}"].dtype == torch.bfloat16
        torch.testing.assert_close(uniform * (b @ a), piece.delta(), rtol=1e-2, atol=1e-2)
    # Modules already at the uniform scale are passed through by reference.
    key = f"{PEFT_PREFIX}m.layers.0.self_attn.q_proj{LORA_B_SUFFIX}"
    assert folded[key] is state[key]
