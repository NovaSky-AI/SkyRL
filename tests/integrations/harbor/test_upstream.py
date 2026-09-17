"""SkyRL's fork of vLLM's token-in/token-out endpoint.

capture ships the wire as `vllm`, because the payloads are vLLM's own and
`vllm serve` mounts `/inference/v1/generate` itself. All this fork changes is
the path -- `vllm_server_actor.py` serves the same shape at
`/skyrl/v1/generate` so it can return routed expert IDs.

So what is worth testing here is small on purpose: the name, the path, and that
nothing else was overridden. The five ways the wire differs from capture's own
shape are capture's to pin, in `tests/test_upstream_vllm.py`.
"""

from __future__ import annotations


def test_importing_the_module_registers_the_fork(registered):
    """capture resolves a target by name, so the name is the contract."""
    assert registered.name == "skyrl"
    assert registered.mode == "tokens"
    assert registered.requires_tokenizer is True


def test_it_serves_the_same_shape_on_skyrls_path(registered):
    assert registered.generate_path == "/skyrl/v1/generate"
    assert registered.token_url("http://router:8000") == "http://router:8000/skyrl/v1/generate"
    assert registered.token_url("http://router:8000/") == "http://router:8000/skyrl/v1/generate"
    already = "http://router:8000/skyrl/v1/generate"
    assert registered.token_url(already) == already


def test_the_wire_itself_is_inherited_not_reimplemented(registered, inherited):
    """If this starts failing, the fork has grown a second copy of the wire --
    which is what putting it in capture was meant to prevent."""
    assert registered.token_request(
        prompt_token_ids=[1], sampling_params={}, model=None, session_id="s"
    ) == inherited
