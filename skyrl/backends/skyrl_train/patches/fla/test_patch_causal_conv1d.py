from types import SimpleNamespace

from skyrl.backends.skyrl_train.patches.fla.patch_causal_conv1d import (
    _set_static_config,
)


def test_set_static_config_replaces_autotune_search_once() -> None:
    original_configs = [object(), object()]
    autotuner = SimpleNamespace(configs=original_configs)
    kernel = SimpleNamespace(fn=autotuner)
    safe_config = object()

    assert _set_static_config(kernel, safe_config)
    assert autotuner.configs == [safe_config]

    second_config = object()
    assert _set_static_config(kernel, second_config)
    assert autotuner.configs == [safe_config]


def test_set_static_config_rejects_unexpected_wrapper() -> None:
    assert not _set_static_config(SimpleNamespace(), object())
