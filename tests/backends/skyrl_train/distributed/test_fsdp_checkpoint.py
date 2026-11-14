import pytest
import torch

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import (
    _load_model_state_dict,
)


class LinearWithSavedMetadata(torch.nn.Linear):
    """Exercise metadata emitted by state_dict but not consumed by its loader."""

    def __init__(self, suffix):
        super().__init__(2, 2)
        self.suffix = suffix

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[f"{prefix}weight{self.suffix}"] = torch.tensor(0)


@pytest.mark.parametrize(
    "suffix",
    [
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state.bitsandbytes__fp4",
        ".quant_state.bitsandbytes__nf4",
    ],
)
def test_strict_load_allows_bitsandbytes_quantization_metadata(suffix: str) -> None:
    model = LinearWithSavedMetadata(suffix)
    state_dict = model.state_dict()

    _load_model_state_dict(model, state_dict, strict=True)


def test_strict_load_rejects_changed_quantization_metadata() -> None:
    model = LinearWithSavedMetadata(".absmax")
    state_dict = model.state_dict()
    state_dict["weight.absmax"] = torch.tensor(1)

    with pytest.raises(RuntimeError, match="quantization state differs"):
        _load_model_state_dict(model, state_dict, strict=True)


def test_strict_load_rejects_quantization_metadata_for_ordinary_weight() -> None:
    model = torch.nn.Linear(2, 2)
    state_dict = model.state_dict()
    state_dict["weight.absmax"] = torch.tensor(0)

    with pytest.raises(RuntimeError, match="quantization state differs"):
        _load_model_state_dict(model, state_dict, strict=True)


def test_non_strict_load_still_ignores_metadata() -> None:
    model = LinearWithSavedMetadata(".absmax")
    state_dict = model.state_dict()
    state_dict["weight.absmax"] = torch.tensor(1)

    _load_model_state_dict(model, state_dict, strict=False)


@pytest.mark.parametrize(
    "state_dict",
    [
        {"weight": torch.zeros(2, 2)},
        {
            "weight": torch.zeros(2, 2),
            "bias": torch.zeros(2),
            "unexpected": torch.tensor(0),
        },
        {
            "weight": torch.zeros(2, 2),
            "bias": torch.zeros(2),
            "unexpected.absmax": torch.tensor(0),
        },
    ],
)
def test_strict_load_rejects_other_incompatible_keys(state_dict) -> None:
    with pytest.raises(RuntimeError, match="loading model state_dict"):
        _load_model_state_dict(torch.nn.Linear(2, 2), state_dict, strict=True)


def _quantized_linear(bnb, weight, quant_type, double_quant):
    model = bnb.nn.Linear4bit(
        64,
        64,
        bias=False,
        quant_type=quant_type,
        compress_statistics=double_quant,
        quant_storage=torch.bfloat16,
        compute_dtype=torch.bfloat16,
    )
    model.weight.data.copy_(weight)
    return model.to("cpu")


@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("double_quant", [False, True])
def test_bitsandbytes_strict_resume_preserves_outputs(quant_type, double_quant):
    bnb = pytest.importorskip("bitsandbytes", minversion="0.50.2")
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(64, 64, dtype=torch.bfloat16, generator=generator)
    inputs = torch.randn(3, 64, dtype=torch.bfloat16, generator=generator)
    saved = _quantized_linear(bnb, weight, quant_type, double_quant)
    resumed = _quantized_linear(bnb, weight, quant_type, double_quant)
    state_dict = {key: value.clone() for key, value in saved.state_dict().items()}
    expected = saved(inputs)
    # Prove that the load restores bytes, rather than merely retaining a model
    # that was already initialized to the expected weights.
    with torch.no_grad():
        resumed.weight.zero_()

    _load_model_state_dict(resumed, state_dict, strict=True)

    torch.testing.assert_close(resumed(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("changed", ["quant_type", "double_quant", "base_weight"])
def test_bitsandbytes_strict_resume_rejects_incompatible_quantization(changed):
    bnb = pytest.importorskip("bitsandbytes", minversion="0.50.2")
    weight = torch.randn(64, 64, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(17))
    saved = _quantized_linear(bnb, weight, "nf4", True)
    resumed = _quantized_linear(
        bnb,
        weight * 2 if changed == "base_weight" else weight,
        "fp4" if changed == "quant_type" else "nf4",
        changed != "double_quant",
    )

    with pytest.raises(RuntimeError, match="Resume with the same base model and quantization settings"):
        _load_model_state_dict(resumed, saved.state_dict(), strict=True)
