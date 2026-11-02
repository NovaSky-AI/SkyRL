"""CPU tests for the serialized-FP8 KV/attention scale normalization.

vLLM 0.26 corrupts these scales twice on the serialized-FP8 path (boot: the
compressed-tensors KV method copies dummy-load placeholders verbatim; wake:
``init_fp8_kv_scales`` resets only the k/v tensors; vLLM 0.30 removed that hook,
so the wake fix now wraps ``Worker.wake_up``), and FlashInfer bakes the
float mirrors into captured attention plans. The fixes all reduce to one
primitive; these tests pin its behavior with fake layers so no GPU or vLLM
install is needed.
"""

import types

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers import vllm_compat
from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
    _normalize_layer_fp8_scales,
    normalize_serialized_fp8_kv_scales,
)


@pytest.fixture
def dummy_weight_boot(monkeypatch):
    """Pretend this process loaded the model with ``load_format="dummy"``."""
    monkeypatch.setattr(vllm_compat, "_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS", True)


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


def _patched_wake_worker():
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        patch_vllm_fp8_kv_scale_completion,
    )

    calls = []

    class FakeWorker:
        def wake_up(self, tags=None):
            calls.append(("original", tags))

    assert patch_vllm_fp8_kv_scale_completion(FakeWorker) is True
    # Second install is a no-op (idempotence flag).
    assert patch_vllm_fp8_kv_scale_completion(FakeWorker) is False
    return FakeWorker, calls


def _worker_with_layer(worker_cls, layer):
    worker = worker_cls()
    worker.model_runner = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context={"attn": layer})
    )
    return worker


@pytest.mark.parametrize("tags", [None, ["weights"], ["kv_cache"]])
def test_wake_patch_wraps_once_and_normalizes_after_original(dummy_weight_boot, tags):
    worker_cls, calls = _patched_wake_worker()
    layer = _fake_attention_layer()

    _worker_with_layer(worker_cls, layer).wake_up(tags=tags)
    assert calls == [("original", tags)]
    _assert_normalized(layer)


def test_wake_patch_leaves_calibrated_checkpoint_scales_alone(monkeypatch):
    # This module is the worker extension for *every* SkyRL vLLM engine, so the
    # patch also runs on one serving a real FP8 checkpoint. Its offline-calibrated
    # k/v scales are the only correct values; forcing 1.0 silently drifts FP8-KV
    # generation, so the reset must stay behind the dummy-weight-boot latch.
    monkeypatch.setattr(vllm_compat, "_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS", False)
    worker_cls, calls = _patched_wake_worker()
    layer = _fake_attention_layer(garbage=0.0421, q_zero=False)

    _worker_with_layer(worker_cls, layer).wake_up()
    assert calls == [("original", None)]
    assert layer._k_scale.item() == pytest.approx(0.0421)
    assert layer._k_scale_float == pytest.approx(0.0421)


def test_wake_patch_tolerates_worker_without_runner(dummy_weight_boot):
    worker_cls, calls = _patched_wake_worker()
    worker_cls().wake_up()
    assert calls == [("original", None)]


@pytest.mark.vllm
def test_wake_patch_targets_installed_vllm_worker():
    # Pins the hook to vLLM's real wake path: an upstream rename must fail here
    # rather than silently leave the post-wake reset uninstalled.
    worker_mod = pytest.importorskip("vllm.v1.worker.gpu_worker")
    assert callable(getattr(worker_mod.Worker, "wake_up", None))


def test_normalize_after_wake_is_gated_on_dummy_boot(monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        normalize_fp8_kv_scales_after_wake,
    )

    layer = _fake_attention_layer()
    runner = types.SimpleNamespace(compilation_config=types.SimpleNamespace(static_forward_context={"attn": layer}))
    monkeypatch.setattr(vllm_compat, "_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS", False)
    assert normalize_fp8_kv_scales_after_wake(runner) == 0
    monkeypatch.setattr(vllm_compat, "_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS", True)
    assert normalize_fp8_kv_scales_after_wake(runner) == 1
    _assert_normalized(layer)


@pytest.mark.parametrize(
    ("load_format", "expected"),
    [
        ("dummy", True),
        ("DUMMY", True),
        ("auto", False),
        ("safetensors", False),
        (None, False),
    ],
)
def test_dummy_boot_latch_tracks_load_format(monkeypatch, load_format, expected):
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        booted_without_checkpoint_weights,
        patch_vllm_dummy_weight_boot_detection,
    )

    monkeypatch.setattr(vllm_compat, "_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS", False)
    observed = []

    class FakeLoader:
        def load_model(self, vllm_config, model_config, prefix=""):
            observed.append(booted_without_checkpoint_weights())
            return "model"

    assert patch_vllm_dummy_weight_boot_detection(FakeLoader) is True
    assert patch_vllm_dummy_weight_boot_detection(FakeLoader) is False

    config = types.SimpleNamespace(load_config=types.SimpleNamespace(load_format=load_format))
    # GPUModelRunner calls this by keyword; the positional form must work too.
    assert FakeLoader().load_model(vllm_config=config, model_config=None) == "model"
    assert FakeLoader().load_model(config, None) == "model"
    # The latch is live *inside* load_model, before process_weights_after_loading.
    assert observed == [expected, expected]
    assert booted_without_checkpoint_weights() is expected


def test_dummy_boot_latch_defaults_to_false_on_unknown_signature(monkeypatch):
    # An upstream signature change must degrade to "leave the scales alone",
    # never to a reset that corrupts calibrated ones.
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        booted_without_checkpoint_weights,
        patch_vllm_dummy_weight_boot_detection,
    )

    monkeypatch.setattr(vllm_compat, "_BOOTED_WITHOUT_CHECKPOINT_WEIGHTS", True)

    class FakeLoader:
        def load_model(self, **kwargs):
            return "model"

    assert patch_vllm_dummy_weight_boot_detection(FakeLoader) is True
    FakeLoader().load_model(config=None)
    assert booted_without_checkpoint_weights() is False


def test_dummy_boot_latch_tolerates_missing_loader():
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        patch_vllm_dummy_weight_boot_detection,
    )

    class NoLoadModel:
        pass

    assert patch_vllm_dummy_weight_boot_detection(NoLoadModel) is False
