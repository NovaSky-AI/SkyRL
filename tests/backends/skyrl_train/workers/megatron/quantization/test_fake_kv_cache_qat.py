"""CPU tests for KV cache QAT: the straight-through key replacement, megatron's key column order, per-row positions and
the config resolution.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/workers/megatron/quantization/test_fake_kv_cache_qat.py

The cache round trip itself runs on the GPU (Blackwell for nvfp4_ds_mla); the GPU test
``tests/backends/skyrl_train/gpu/gpu_ci/test_kv_cache_round_trip.py`` checks it against vLLM's own cache writer and
reader, bit for bit.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.quantization import (
    fake_kv_cache_qat as kvq,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import _apply_kv_cache_qat_cfg


def test_straight_through_forward_is_round_trip_backward_is_identity():
    key = torch.randn(5, 1, 576, requires_grad=True)
    rows = torch.full((5, 576), 0.25)
    out = kvq._CacheRowStraightThrough.apply(key, lambda: rows)
    assert out.shape == key.shape and torch.equal(out, rows.view(key.shape))
    grad = torch.randn_like(out)
    out.backward(grad)
    assert torch.equal(key.grad, grad)


def test_megatron_key_order_deinterleaves_rope_only():
    rows = torch.arange(576.0).repeat(2, 1)
    out = kvq.to_megatron_key_order(rows)
    assert torch.equal(out[:, :512], rows[:, :512])
    assert torch.equal(out[0, 512:544], torch.arange(512.0, 576.0, 2))
    assert torch.equal(out[0, 544:], torch.arange(513.0, 576.0, 2))


def _thd(cu):
    cu = torch.tensor(cu, dtype=torch.int32)
    return SimpleNamespace(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_q_padded=cu)


def test_row_positions_thd_restart_per_sequence():
    pytest.importorskip("megatron.core")
    pos = kvq.row_positions(_thd([0, 4, 10]), 10, 1, 1, 0, torch.device("cpu"))
    assert pos.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("rank,expected", [(0, [0, 1, 6, 7, 0, 3]), (1, [2, 3, 4, 5, 1, 2])])
def test_row_positions_thd_zigzag_cp(rank, expected):
    """CP2: each rank holds a front and a mirrored back chunk of every (padded) sequence."""
    pytest.importorskip("megatron.core")
    pos = kvq.row_positions(_thd([0, 8, 12]), 6, 1, 2, rank, torch.device("cpu"))
    assert pos.tolist() == expected


def test_row_positions_sbhd_zigzag_cp_repeats_over_batch():
    pytest.importorskip("megatron.core")
    pos = kvq.row_positions(None, 4, 2, 2, 1, torch.device("cpu"))
    assert pos.tolist() == [2, 2, 3, 3, 4, 4, 5, 5]


def _cfg(kv_cache_dtype="nvfp4_ds_mla", force_mqa=True):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.kv_cache_qat = True
    kwargs = {"attention_config": {"sparse_mla_force_mqa": force_mqa}}
    if kv_cache_dtype is not None:
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    cfg.generator.inference_engine.engine_init_kwargs = kwargs
    return cfg


@pytest.mark.parametrize("fmt", ["nvfp4_ds_mla", "fp8_ds_mla"])
def test_format_is_copied_from_the_engines(fmt):
    cfg = _cfg(fmt)
    _apply_kv_cache_qat_cfg(cfg)
    assert cfg.trainer.policy.megatron_config.kv_cache_qat_format == fmt


@pytest.mark.parametrize("kv_cache_dtype", [None, "auto", "bfloat16", "fp8"])
def test_unemulated_cache_is_rejected(kv_cache_dtype):
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        _apply_kv_cache_qat_cfg(_cfg(kv_cache_dtype))


def test_force_mqa_is_required():
    with pytest.raises(ValueError, match="sparse_mla_force_mqa"):
        _apply_kv_cache_qat_cfg(_cfg(force_mqa=False))


@pytest.mark.parametrize("role_kwargs", [False, True])
def test_prefill_decode_disaggregation_is_rejected(role_kwargs):
    """PD is rejected as such, not as an unquantized cache, with the engine kwargs in either place PD allows."""
    cfg = _cfg("nvfp4_ds_mla")
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.enable_pd = True
    if role_kwargs:
        ie_cfg.prefill_init_kwargs = dict(ie_cfg.engine_init_kwargs)
        ie_cfg.decode_init_kwargs = dict(ie_cfg.engine_init_kwargs)
        ie_cfg.engine_init_kwargs = {}
    with pytest.raises(ValueError, match="does not support prefill-decode disaggregation"):
        _apply_kv_cache_qat_cfg(cfg)


def test_disagreeing_format_is_rejected():
    cfg = _cfg("nvfp4_ds_mla")
    cfg.trainer.policy.megatron_config.kv_cache_qat_format = "fp8_ds_mla"
    with pytest.raises(ValueError, match="disagrees"):
        _apply_kv_cache_qat_cfg(cfg)


def test_off_leaves_config_alone_and_rejects_a_format():
    cfg = SkyRLTrainConfig()
    _apply_kv_cache_qat_cfg(cfg)
    assert cfg.trainer.policy.megatron_config.kv_cache_qat_format is None
    cfg.trainer.policy.megatron_config.kv_cache_qat_format = "nvfp4_ds_mla"
    with pytest.raises(ValueError, match="is off"):
        _apply_kv_cache_qat_cfg(cfg)


def test_attention_geometry_check():
    good = SimpleNamespace(
        kv_lora_rank=512, qk_pos_emb_head_dim=64, layernorm_zero_centered_gamma=False, rotary_scaling_factor=1.0
    )
    kvq.check_kv_cache_qat_config(good)
    with pytest.raises(ValueError, match="kv_lora_rank"):
        kvq.check_kv_cache_qat_config(SimpleNamespace(**{**vars(good), "kv_lora_rank": 448}))
    with pytest.raises(ValueError, match="rotary_scaling_factor"):
        kvq.check_kv_cache_qat_config(SimpleNamespace(**{**vars(good), "rotary_scaling_factor": 40.0}))


def test_rounding_defaults_to_the_rollouts_cache():
    cfg = _cfg("nvfp4_ds_mla")
    _apply_kv_cache_qat_cfg(cfg)
    rounding = cfg.trainer.policy.megatron_config.kv_cache_qat_rounding
    assert rounding.four_over_six is False and rounding.stochastic_rounding is False


@pytest.mark.parametrize("fmt", ["nvfp4_ds_mla", "fp8_ds_mla"])
@pytest.mark.parametrize("four_over_six", [False, True])
def test_trainer_rounding_variants(fmt, four_over_six):
    """4/6 block scales are NVFP4-only; stochastic rounding applies to both formats."""
    cfg = _cfg(fmt)
    rounding = cfg.trainer.policy.megatron_config.kv_cache_qat_rounding
    rounding.four_over_six, rounding.stochastic_rounding = four_over_six, True
    if four_over_six and fmt == "fp8_ds_mla":
        with pytest.raises(ValueError, match="nvfp4_ds_mla only"):
            _apply_kv_cache_qat_cfg(cfg)
    else:
        _apply_kv_cache_qat_cfg(cfg)
        assert rounding.four_over_six is four_over_six and rounding.stochastic_rounding


def test_off_rejects_rounding():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.megatron_config.kv_cache_qat_rounding.stochastic_rounding = True
    with pytest.raises(ValueError, match="rounding is set but"):
        _apply_kv_cache_qat_cfg(cfg)
