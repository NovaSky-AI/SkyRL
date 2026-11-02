"""CPU test for ``patch_per_block_fp8_param``.

vLLM's online ``fp8_per_block`` quantization passes ``layer.weight`` -- a vLLM parameter subclass
with ``__torch_function__`` -- to the ``torch.compile``d ``per_block_cast_to_fp8``, which dynamo
cannot trace (``RecursionError`` on the first weight sync). A recorder stands in for the compiled
function so the test checks only what the patch controls: both bindings hand it a plain tensor that
shares the parameter's storage, and plain tensors pass through untouched.
"""

import pytest
import torch

pytest.importorskip("vllm")

pytestmark = pytest.mark.vllm


@pytest.fixture
def patched(monkeypatch):
    from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
    from vllm.utils import deep_gemm

    from skyrl.backends.skyrl_train.patches.vllm import patch_per_block_fp8_param

    seen = []

    def recorder(x, *args, **kwargs):
        seen.append(x)
        return "quantized"

    # Set through monkeypatch so the patch's rebinding is undone at teardown.
    monkeypatch.setattr(deep_gemm, "per_block_cast_to_fp8", recorder)
    monkeypatch.setattr(online_fp8, "per_block_cast_to_fp8", recorder)
    monkeypatch.setattr(patch_per_block_fp8_param, "_PATCHED", False)
    assert patch_per_block_fp8_param.apply_per_block_fp8_param_patch()
    return deep_gemm, online_fp8, seen


def _vllm_weight_parameter(monkeypatch):
    # BasevLLMParameter is the base of vLLM's weight parameters and defines the __torch_function__
    # dynamo recurses through. Its constructor records the TP rank/size, so stand in a 1-rank group.
    from vllm.model_executor import parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    return parameter.BasevLLMParameter(data=torch.randn(256, 256), weight_loader=lambda *args: None)


@pytest.mark.parametrize("binding", ["deep_gemm", "online_fp8"])
def test_vllm_parameter_is_unwrapped(patched, binding, monkeypatch):
    deep_gemm, online_fp8, seen = patched
    module = deep_gemm if binding == "deep_gemm" else online_fp8
    weight = _vllm_weight_parameter(monkeypatch)

    assert module.per_block_cast_to_fp8(weight, use_ue8m0=True) == "quantized"
    assert type(seen[-1]) is torch.Tensor
    assert seen[-1].data_ptr() == weight.data_ptr()


def test_plain_tensor_passes_through(patched):
    deep_gemm, _, seen = patched
    x = torch.randn(128, 128)
    deep_gemm.per_block_cast_to_fp8(x)
    assert seen[-1] is x
