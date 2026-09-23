import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("vllm")
if not torch.cuda.is_available():
    pytest.skip("vLLM LoRA model tests require CUDA", allow_module_level=True)

pytestmark = pytest.mark.vllm


def _load_looped_qwen3_module():
    path = Path(__file__).parents[4] / "skyrl/backends/skyrl_train/models/looped_qwen3.py"
    spec = importlib.util.spec_from_file_location("skyrl_looped_qwen3_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


looped_qwen3 = _load_looped_qwen3_module()


def test_lora_delta_path_does_not_call_base_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLoRALayer:
        output_slices = (3, 2)

        def _apply_lora_to_output(self, inputs: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            assert output.shape == (2, 5)
            return output + inputs.sum(dim=-1, keepdim=True)

    monkeypatch.setattr(looped_qwen3, "BaseLinearLayerWithLoRA", FakeLoRALayer)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    output = looped_qwen3._apply_lora_delta(FakeLoRALayer(), inputs)

    assert output.tolist() == [[3.0] * 5, [7.0] * 5]


def test_lora_delta_path_rejects_unwrapped_linear() -> None:
    with pytest.raises(RuntimeError, match="requires vLLM LoRA wrapping"):
        looped_qwen3._apply_lora_delta(torch.nn.Linear(2, 2), torch.ones(1, 2))
