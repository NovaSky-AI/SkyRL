"""patch_checkpoint_without_output_frozen_inputs: output-discarding recompute on inputs that do not require grad."""

import pytest
import torch

pytestmark = pytest.mark.megatron


def _discard_and_backward():
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")  # frozen activation, e.g. embeddings under LoRA
    w = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))  # trainable consumer, e.g. a LoRA adapter
    ckpt = CheckpointWithoutOutput()
    y = ckpt.checkpoint(torch.nn.functional.gelu, x)
    z = y @ w
    ckpt.discard_output_and_register_recompute(z)
    z.sum().backward()
    expected = torch.nn.functional.gelu(x).t() @ torch.ones(4, 8, device="cuda")
    return w.grad, expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_unpatched_recompute_fails_on_frozen_inputs():
    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_checkpoint_without_output_frozen_inputs as patch,
    )

    if patch._APPLIED:
        pytest.skip("patch already applied in this process")
    # Plain StopIteration here; inside the training worker's coroutine it surfaces as
    # "RuntimeError: coroutine raised StopIteration".
    with pytest.raises((StopIteration, RuntimeError)) as excinfo:
        _discard_and_backward()
    # Only the frozen-input failure counts, not any other RuntimeError.
    assert isinstance(excinfo.value, StopIteration) or "StopIteration" in str(excinfo.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_patched_recompute_uses_frozen_inputs():
    from skyrl.backends.skyrl_train.patches.megatron.patch_checkpoint_without_output_frozen_inputs import (
        apply_checkpoint_without_output_frozen_inputs_patch,
    )

    assert apply_checkpoint_without_output_frozen_inputs_patch()
    grad, expected = _discard_and_backward()
    torch.testing.assert_close(grad, expected)
