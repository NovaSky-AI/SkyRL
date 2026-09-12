"""CPU tests for the serialized-FP8 KV/attention scale normalization.

vLLM 0.26 corrupts these scales twice on the serialized-FP8 path (boot: the
compressed-tensors KV method copies dummy-load placeholders verbatim; wake:
``init_fp8_kv_scales`` resets only the k/v tensors), and FlashInfer bakes the
float mirrors into captured attention plans. The fixes all reduce to one
primitive; these tests pin its behavior with fake layers so no GPU or vLLM
install is needed.
"""

import types

import torch

from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
    _normalize_layer_fp8_scales,
    normalize_serialized_fp8_kv_scales,
)


def _fake_attention_layer(garbage: float = -0.000745, q_zero: bool = True):
    layer = types.SimpleNamespace()
    layer._k_scale = torch.nn.Parameter(torch.tensor(garbage), requires_grad=False)
    layer._v_scale = torch.nn.Parameter(torch.tensor(garbage), requires_grad=False)
    layer._q_scale = torch.nn.Parameter(torch.tensor(0.0 if q_zero else garbage), requires_grad=False)
    layer._prob_scale = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
    layer._k_scale_cpu = torch.tensor(garbage)
    layer._v_scale_cpu = torch.tensor(garbage)
    layer._k_scale_float = garbage
    layer._v_scale_float = garbage
    layer._q_scale_float = garbage
    return layer


def _assert_normalized(layer):
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale", "_k_scale_cpu", "_v_scale_cpu"):
        assert getattr(layer, name).item() == 1.0, name
    for name in ("_k_scale_float", "_v_scale_float", "_q_scale_float"):
        assert getattr(layer, name) == 1.0, name


def test_normalize_layer_resets_tensors_floats_and_cpu_copies():
    layer = _fake_attention_layer()
    assert _normalize_layer_fp8_scales(layer) == 1
    _assert_normalized(layer)


def test_normalize_layer_survives_grad_requiring_parameters():
    # The V1 field failure: in-place fill on a grad-requiring leaf throws
    # unless the reset goes through .data.
    layer = _fake_attention_layer()
    layer._k_scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=True)
    assert _normalize_layer_fp8_scales(layer) == 1
    assert layer._k_scale.item() == 1.0


def test_normalize_layer_skips_modules_without_scales():
    module = types.SimpleNamespace(weight=torch.zeros(2))
    assert _normalize_layer_fp8_scales(module) == 0


def test_runner_normalize_walks_forward_context():
    attn_a, attn_b = _fake_attention_layer(), _fake_attention_layer(garbage=7.5, q_zero=False)
    gdn = types.SimpleNamespace()  # linear-attention layer: no KV scales
    runner = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"l.0.attn": attn_a, "l.1.linear_attn": gdn, "l.2.attn": attn_b}
        )
    )
    assert normalize_serialized_fp8_kv_scales(runner) == 2
    _assert_normalized(attn_a)
    _assert_normalized(attn_b)


def test_runner_normalize_tolerates_missing_context():
    assert normalize_serialized_fp8_kv_scales(types.SimpleNamespace()) == 0
    assert (
        normalize_serialized_fp8_kv_scales(
            types.SimpleNamespace(compilation_config=types.SimpleNamespace(static_forward_context={}))
        )
        == 0
    )


def test_wake_patch_wraps_once_and_normalizes_after_original():
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        patch_vllm_fp8_kv_scale_completion,
    )

    calls = []

    class FakeRunner:
        def post_kv_cache_wake_up(self):
            calls.append("original")

    layer = _fake_attention_layer()
    assert patch_vllm_fp8_kv_scale_completion(FakeRunner) is True
    # Second install is a no-op (idempotence flag).
    assert patch_vllm_fp8_kv_scale_completion(FakeRunner) is False

    runner = FakeRunner()
    runner.compilation_config = types.SimpleNamespace(static_forward_context={"attn": layer})
    runner.post_kv_cache_wake_up()
    assert calls == ["original"]
    _assert_normalized(layer)
