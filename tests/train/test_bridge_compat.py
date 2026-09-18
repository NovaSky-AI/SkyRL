from unittest.mock import MagicMock

from skyrl.backends.skyrl_train.workers.megatron.bridge_compat import (
    _filter_kwargs,
    load_auto_bridge,
)


def test_filter_kwargs_drops_unknown_without_var_keyword():
    def fn(a, b=1):
        return a, b

    assert _filter_kwargs(fn, {"a": 1, "trust_remote_code": True}) == {"a": 1}


def test_filter_kwargs_keeps_all_with_var_keyword():
    def fn(a, **kwargs):
        return a, kwargs

    assert _filter_kwargs(fn, {"a": 1, "trust_remote_code": True}) == {
        "a": 1,
        "trust_remote_code": True,
    }


def test_load_auto_bridge_falls_back_to_from_hf(monkeypatch):
    class Bridge:
        @classmethod
        def from_hf(cls, source, trust_remote_code=False):
            inst = MagicMock()
            inst.source = source
            inst.trust_remote_code = trust_remote_code
            return inst

    fake_mod = MagicMock()
    fake_mod.AutoBridge = Bridge
    monkeypatch.setitem(__import__("sys").modules, "megatron", MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "megatron.bridge", fake_mod)

    loaded = load_auto_bridge("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    assert loaded.source == "Qwen/Qwen2.5-0.5B-Instruct"
    assert loaded.trust_remote_code is True
