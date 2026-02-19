"""
uv run --isolated --extra dev pytest -s tests/train/test_config.py
"""

import pytest
import typing
from typing import List
from dataclasses import dataclass, field
from typing import Annotated, Optional

from omegaconf import OmegaConf, DictConfig

from skyrl.train.config.config import (
    BaseConfig,
    SkyRLTrainConfig,
    build_nested_dataclass,
    _resolve_dataclass_type,
)
from skyrl.train.config.utils import get_legacy_config


# Helper dataclasses for testing
@dataclass
class _SimpleConfig(BaseConfig):
    a: int = 0


@dataclass
class _NestedConfig(BaseConfig):
    b: int = 1
    c: Annotated[_SimpleConfig, "test"] = field(default_factory=_SimpleConfig)
    d: Optional[_SimpleConfig] = None


def test_build_nested_dataclass():
    # not all fields are present
    d = {"b": 4, "c": {"a": 2}}
    cfg = build_nested_dataclass(_NestedConfig, d)
    assert cfg.b == 4
    assert cfg.c.a == 2

    # all fields are present
    d = {"b": 4, "c": {"a": 2}, "d": {"a": 3}}
    cfg = build_nested_dataclass(_NestedConfig, d)
    assert cfg.b == 4
    assert cfg.c.a == 2
    assert cfg.d.a == 3


def test_build_nested_dataclass_full_config():
    d = {"trainer": {"policy": {"model": {"path": "path/to/model"}}}}
    cfg = build_nested_dataclass(SkyRLTrainConfig, d)
    assert cfg.trainer.policy.model.path == "path/to/model"


def test_build_nested_dataclass_invalid_config():
    d = {"path": "path/to/model"}
    with pytest.raises(ValueError):
        build_nested_dataclass(SkyRLTrainConfig, d)


def test_build_config_from_dict_config():
    cfg = OmegaConf.create({"a": 1})
    cfg = _SimpleConfig.from_dict_config(cfg)
    assert cfg.a == 1

    cfg = OmegaConf.create({"b": 1, "c": {"a": 2}})
    cfg = _NestedConfig.from_dict_config(cfg)
    assert cfg.b == 1
    assert cfg.c.a == 2


def test_build_config_from_dict_config_invalid_config():
    cfg = OmegaConf.create({"path": "path/to/model"})
    with pytest.raises(ValueError):
        _SimpleConfig.from_dict_config(cfg)


def test_dtype_resolution():
    assert not _resolve_dataclass_type(typing.Optional[int])
    assert _resolve_dataclass_type(typing.Optional[_SimpleConfig]) is _SimpleConfig
    assert _resolve_dataclass_type(typing.Union[None, _SimpleConfig]) is _SimpleConfig
    assert _resolve_dataclass_type(typing.Annotated[_SimpleConfig, "test"]) is _SimpleConfig


def test_cli_overrides():
    # Basic overrides - str, int and dict fields
    overrides = [
        "trainer.policy.model.path=path/to/model",
        "trainer.seed=123",
        "generator.inference_engine.engine_init_kwargs.field=value",
        "generator.sampling_params.temperature=0.7",
    ]
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    assert cfg.trainer.policy.model.path == "path/to/model"
    assert cfg.trainer.seed == 123
    assert cfg.generator.inference_engine.engine_init_kwargs["field"] == "value"
    assert cfg.generator.sampling_params.temperature == 0.7

    # check that temperature is propagated to algorithm config
    assert cfg.trainer.algorithm.temperature == 0.7


def test_cli_overrides_empty_args():
    cfg = SkyRLTrainConfig.from_cli_overrides([])
    assert cfg.trainer.policy.model.path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.trainer.seed == 42


def test_cli_overrides_plus_prefix_rejected():
    with pytest.raises(ValueError, match="The '\\+' prefix"):
        SkyRLTrainConfig.from_cli_overrides(["+new_field=value"])


def test_cli_overrides_invalid_field():
    with pytest.raises(ValueError, match="Invalid fields"):
        SkyRLTrainConfig.from_cli_overrides(["trainer.nonexistent_field=value"])


def test_temperature_propagation():
    """Test that temperature is copied from generator to algorithm config in __post_init__."""
    cfg = SkyRLTrainConfig.from_cli_overrides(["generator.sampling_params.temperature=0.7"])
    assert cfg.generator.sampling_params.temperature == 0.7
    assert cfg.trainer.algorithm.temperature == 0.7


def test_legacy_config_translation():
    """Test that legacy config format is translated to new format."""
    import warnings

    # Use legacy-style paths (flat under generator instead of nested in inference_engine)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = SkyRLTrainConfig.from_cli_overrides(["generator.backend=sglang"])

        # Should have issued a deprecation warning (filter for DeprecationWarning only)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1
        assert "legacy" in str(deprecation_warnings[0].message).lower()

    # Value should be translated to the new nested location
    assert cfg.generator.inference_engine.backend == "sglang"

    # test with full YAML
    full_legacy_cfg = get_legacy_config()
    # custom override
    full_legacy_cfg.generator.backend = "sglang"

    # convert to CLI overrides
    def traverse_and_convert(cfg: DictConfig, parent_key: str = "") -> List[str]:
        overrides = []
        for key, value in cfg.items():
            if isinstance(value, DictConfig):
                overrides.extend(traverse_and_convert(value, f"{parent_key}.{key}" if parent_key else key))
            else:
                overrides.append(f"{parent_key}.{key}={value}" if parent_key else f"{key}={value}")
        return overrides

    full_legacy_cfg_as_overrides = traverse_and_convert(full_legacy_cfg)

    # should pass without error
    translated_cfg = SkyRLTrainConfig.from_cli_overrides(full_legacy_cfg_as_overrides)
    assert translated_cfg.generator.inference_engine.backend == "sglang"


def test_legacy_config_field_rename():
    """Test that renamed fields are translated correctly."""
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # Old field name: num_inference_engines -> new: num_engines
        cfg = SkyRLTrainConfig.from_cli_overrides(["generator.num_inference_engines=4"])

    assert cfg.generator.inference_engine.num_engines == 4
