import pytest
import torch

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import (
    _load_model_state_dict,
)


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
    model = torch.nn.Linear(2, 2)
    state_dict = model.state_dict()
    state_dict[f"weight{suffix}"] = torch.tensor(0)

    _load_model_state_dict(model, state_dict, strict=True)


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
