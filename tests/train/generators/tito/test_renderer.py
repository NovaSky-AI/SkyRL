"""Tests for Prime renderer auto-resolution."""

from types import SimpleNamespace

from renderers import DefaultRendererConfig

from skyrl.train.generators.tito.renderer import PrimeRendererAdapter


class _Tokenizer:
    name_or_path = "Qwen/Qwen3-8B"


class _Renderer:
    def __init__(self, name):
        self.config = SimpleNamespace(name=name)


def test_adapter_uses_auto_config_with_full_thinking_retention(monkeypatch):
    captured = {}

    def create_renderer(tokenizer, config, *, chat_template_kwargs=None):
        captured["tokenizer"] = tokenizer
        captured["config"] = config
        captured["chat_template_kwargs"] = chat_template_kwargs
        return _Renderer("qwen3")

    monkeypatch.setattr("renderers.create_renderer", create_renderer)
    adapter = PrimeRendererAdapter(
        _Tokenizer(),
        chat_template_kwargs={"enable_thinking": True},
    )

    assert captured["config"].name == "auto"
    assert captured["config"].thinking_retention == "all"
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}
    assert adapter.renderer_name == "qwen3"


def test_adapter_accepts_explicit_typed_renderer_config(monkeypatch):
    captured = {}
    explicit = DefaultRendererConfig()

    def create_renderer(tokenizer, config, *, chat_template_kwargs=None):
        captured["config"] = config
        return _Renderer("default")

    monkeypatch.setattr("renderers.create_renderer", create_renderer)
    adapter = PrimeRendererAdapter(_Tokenizer(), config=explicit)

    assert captured["config"] is explicit
    assert adapter.renderer_name == "default"
