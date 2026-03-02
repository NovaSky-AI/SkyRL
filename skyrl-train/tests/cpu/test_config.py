"""
uv run --isolated --extra dev pytest -s tests/cpu/test_config.py
"""

import pytest
from dataclasses import dataclass, field
from skyrl_train.config.config import BaseConfig, build_nested_dataclass, SkyRLConfig, _resolve_dataclass_type

import typing
from typing import Annotated, Optional

from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config


# simple test dataclasses
@dataclass
class TestConfigSimple(BaseConfig):
    a: int = 0


@dataclass
class TestConfigNested(BaseConfig):
    b: int = 1
    c: Annotated[TestConfigSimple, "test"] = field(default_factory=TestConfigSimple)
    d: Optional[TestConfigSimple] = None


def test_build_nested_dataclass():
    # not all fields are present
    d = {"b": 4, "c": {"a": 2}}
    cfg = build_nested_dataclass(TestConfigNested, d)
    assert cfg.b == 4
    assert cfg.c.a == 2

    #  all fields are present
    d = {"b": 4, "c": {"a": 2}, "d": {"a": 3}}
    cfg = build_nested_dataclass(TestConfigNested, d)
    assert cfg.b == 4
    assert cfg.c.a == 2
    assert cfg.d.a == 3


def test_build_nested_dataclass_full_config():
    # test overrides for different fields for SkyRLConfig
    d = {"trainer": {"policy": {"model": {"path": "path/to/model"}}}}
    cfg = build_nested_dataclass(SkyRLConfig, d)
    assert cfg.trainer.policy.model.path == "path/to/model"


def test_build_nested_dataclass_invalid_config():
    # test invalid config
    d = {"path": "path/to/model"}
    with pytest.raises(ValueError):
        build_nested_dataclass(SkyRLConfig, d)


def test_build_config_from_yaml():
    config = get_default_config()
    # custom override
    config.trainer.policy.model.path = "path/to/model"
    config_dict = OmegaConf.to_container(config, resolve=True)
    cfg = build_nested_dataclass(SkyRLConfig, config_dict)

    assert cfg.trainer.policy.model.path == "path/to/model"


def test_build_config_from_dict_config():
    cfg = OmegaConf.create({"a": 1})
    cfg = TestConfigSimple.from_dict_config(cfg)
    assert cfg.a == 1

    cfg = OmegaConf.create({"b": 1, "c": {"a": 2}})
    cfg = TestConfigNested.from_dict_config(cfg)
    assert cfg.b == 1
    assert cfg.c.a == 2


def test_build_skyrl_config_from_dict_config():
    cfg = get_default_config()
    cfg = SkyRLConfig.from_dict_config(cfg)
    assert cfg.trainer.policy.model.path == "Qwen/Qwen2.5-1.5B-Instruct"


def test_build_config_from_dict_config_invalid_config():
    cfg = OmegaConf.create({"path": "path/to/model"})
    with pytest.raises(ValueError):
        TestConfigSimple.from_dict_config(cfg)


def test_dtype_resolution():
    assert not _resolve_dataclass_type(typing.Optional[int])

    assert _resolve_dataclass_type(typing.Optional[TestConfigSimple]) is TestConfigSimple

    assert _resolve_dataclass_type(typing.Union[None, TestConfigSimple]) is TestConfigSimple

    assert _resolve_dataclass_type(typing.Annotated[TestConfigSimple, "test"]) is TestConfigSimple
