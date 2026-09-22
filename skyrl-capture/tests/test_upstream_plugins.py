"""Upstream kinds contributed from outside this package.

Both registries are open on purpose. capture ships the protocols clients speak
-- `openai`, `anthropic` -- and the batched `tokens` shape it speaks to an
engine, not one trainer's wire: a wire belonging to one trainer is that
project's to define, because carrying them here would mean tracking every
trainer's HTTP shape.

So what has to hold is that somebody else can add one, on either side, and
have it work in a process that is handed only the module name.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from skyrl_capture.config import Config
from skyrl_capture.tito import upstream as tito
from skyrl_capture.upstream import registry
from skyrl_capture.upstream.plugins import load_modules


@pytest.fixture
def clean_registry():
    """Undo whatever a test registers, so order does not matter."""
    before = set(registry.names()), set(tito.names())
    yield
    for name in set(registry.names()) - before[0]:
        registry._PROTOCOLS.pop(name, None)
    for name in set(tito.names()) - before[1]:
        tito._PROTOCOLS.pop(name, None)


@pytest.fixture
def contributed(tmp_path, monkeypatch):
    """A module on the path that registers an engine when imported."""

    def make(module: str, kind: str) -> str:
        (tmp_path / f"{module}.py").write_text(
            textwrap.dedent(
                f"""
                from skyrl_capture.tito.upstream import TitoProtocol, register

                class Contributed(TitoProtocol):
                    name = "{kind}"

                    def url(self, url):
                        return f"{{url.rstrip('/')}}/v2/infer"

                register(Contributed())
                """
            )
        )
        sys.modules.pop(module, None)
        return module

    monkeypatch.syspath_prepend(str(tmp_path))
    return make


def test_capture_ships_protocols_not_one_trainers_variant():
    """A regression guard with an opinion in it.

    The line is whether the payloads belong to the server that serves them.
    `vllm` qualifies: `/inference/v1/generate` is vLLM's own endpoint, mounted
    by `vllm serve`, and the request and response models are vLLM's.

    `skyrl` does not, and lived here briefly by mistake: SkyRL serves the same
    shape on its own path, which makes it a *fork* of this wire. It belongs in
    that repo, as a subclass overriding `generate_path` — which is all it is.
    Anything that arrives here needing more than a path override, for one
    trainer's benefit, is in the wrong place.
    """
    assert set(registry.names()) == {"openai", "anthropic"}
    assert set(tito.names()) == {"tokens", "vllm"}
    assert not tito.is_registered("skyrl"), "a fork of a wire is not a protocol"


def test_an_outside_class_can_be_registered_and_resolved(clean_registry):
    class MyEngine(tito.TitoProtocol):
        name = "my-engine"

        def url(self, url: str) -> str:
            return f"{url.rstrip('/')}/v2/infer"

    tito.register(MyEngine())

    resolved = tito.get("my-engine")
    assert resolved.url("http://engine:9000") == "http://engine:9000/v2/infer"
    # Inherited from `tokens`, so it builds the same request in every other
    # respect -- a new engine is the differences and nothing else.
    body, headers = resolved.request(
        prompt_token_ids=[1, 2], sampling_params={}, model=None, session_id="s"
    )
    assert body["prompt_token_ids"] == [[1, 2]] and headers == {}


def test_a_text_protocol_is_contributed_the_same_way(tmp_path, monkeypatch, clean_registry):
    """The other registry, reached through the same flag."""
    (tmp_path / "contributed_text.py").write_text(
        textwrap.dedent(
            """
            from skyrl_capture.upstream import BaseTextProtocol, register

            class Contributed(BaseTextProtocol):
                name = "contributed-text"
                environment_map = (("CONTRIB_BASE_URL", "base_url"),)

            register(Contributed())
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("contributed_text", None)

    load_modules(["contributed_text"])
    assert registry.get("contributed-text").client_environment("http://x/v1") == {
        "CONTRIB_BASE_URL": "http://x/v1"
    }


def test_importing_the_module_is_what_registers_it(contributed, clean_registry):
    module = contributed("contributed_upstream", "contributed")
    assert not tito.is_registered("contributed")

    assert load_modules([module]) == (module,)
    assert tito.get("contributed").url("http://e:1") == "http://e:1/v2/infer"


def test_several_modules_at_once(contributed, clean_registry):
    first = contributed("contributed_one", "kind-one")
    second = contributed("contributed_two", "kind-two")

    load_modules([first, second])
    assert tito.is_registered("kind-one") and tito.is_registered("kind-two")


def test_a_module_that_does_not_import_fails_loudly_and_says_where_from():
    """Silence here surfaces much later as a startup refused for an unknown
    type, which is a confusing way to learn that a path has a typo."""
    with pytest.raises(RuntimeError, match="--upstream-module names 'no_such_upstream'"):
        load_modules(["no_such_upstream"])

    with pytest.raises(RuntimeError, match="upstream_modules names 'no_such_upstream'"):
        load_modules(["no_such_upstream"], source="upstream_modules")


def test_the_names_ride_in_the_config(contributed, clean_registry):
    """Which is what gets them into a process that was handed only a config.

    `build_runtime` loads them before anything resolves a name, so an embedder
    that passes a `Config` gets the same registries the CLI would.
    """
    module = contributed("contributed_config", "from-config")
    config = Config().with_overrides(upstream_modules=(module,))

    assert config.upstream_modules == (module,)
    load_modules(config.upstream_modules, source="upstream_modules")
    assert tito.is_registered("from-config")
