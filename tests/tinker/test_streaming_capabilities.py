import pytest
from pydantic import ValidationError

from skyrl.tinker.api import OptimStepRequest, supports_streaming_accumulation
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.types import AdamParams, OptimStepInput


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scaling_is_rejected_at_both_serialization_boundaries(scale):
    with pytest.raises(ValidationError):
        OptimStepRequest(model_id="test", adam_params={}, gradient_scale=scale)
    with pytest.raises(ValidationError):
        OptimStepInput(adam_params=AdamParams(), gradient_scale=scale)


@pytest.mark.parametrize(
    "backend,colocated,supported",
    [
        ("megatron", False, True),
        ("megatron", True, False),
        ("fsdp", False, False),
        ("jax", False, False),
    ],
)
def test_only_non_colocated_megatron_advertises_streaming(backend, colocated, supported):
    config = EngineConfig(
        base_model="test", backend=backend, backend_config={"trainer.placement.colocate_all": colocated}
    )
    assert supports_streaming_accumulation(config) is supported


def test_unspecified_placement_does_not_advertise_streaming():
    assert not supports_streaming_accumulation(EngineConfig(base_model="test", backend="megatron"))
