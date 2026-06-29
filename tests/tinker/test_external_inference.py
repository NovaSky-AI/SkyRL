"""Unit tests for ExternalInferenceClient sample retry and adapter-load self-healing.

Run with:
  uv run --extra dev --extra jax -- pytest tests/tinker/test_external_inference.py
"""

import asyncio
from pathlib import Path

import httpx
import pytest

from skyrl.tinker.extra.external_inference import (
    ExternalInferenceClient,
    _is_retriable_sample_error,
)


def _status_error(status_code: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://engine/v1/completions")
    response = httpx.Response(status_code, request=request, text=body)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_retriable_on_transport_error():
    """Transient transport failures are retried."""
    assert _is_retriable_sample_error(httpx.ConnectError("boom")) is True


def test_retriable_on_404():
    """A missing model (404) means the adapter is not registered yet, so retry."""
    assert _is_retriable_sample_error(_status_error(404, "model does not exist")) is True


def test_retriable_on_400_model_missing():
    """A 400 whose body reports the model is missing is the lazy-load race, so retry."""
    assert _is_retriable_sample_error(_status_error(400, "The model `x` does not exist.")) is True
    assert _is_retriable_sample_error(_status_error(400, "adapter not found")) is True


def test_not_retriable_on_other_400():
    """A genuine client error such as an over-length request is not retried."""
    assert _is_retriable_sample_error(_status_error(400, "max_tokens exceeds context length")) is False


def test_not_retriable_on_403_or_500():
    """Auth failures and server errors are not retried."""
    assert _is_retriable_sample_error(_status_error(403, "forbidden")) is False
    assert _is_retriable_sample_error(_status_error(500, "internal error")) is False


class _ScriptedClient:
    """Async client stub that returns a scripted sequence of completion responses."""

    def __init__(self, events):
        self._events = list(events)
        self.calls = []

    async def post(self, path, json=None, headers=None):
        self.calls.append(path)
        request = httpx.Request("POST", "http://engine" + path)
        if path == "/load_lora_adapter":
            return httpx.Response(200, request=request)
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        if event >= 400:
            return httpx.Response(event, request=request, text="model does not exist")
        return httpx.Response(event, request=request, json={"choices": []})


def _make_client() -> ExternalInferenceClient:
    client = ExternalInferenceClient.__new__(ExternalInferenceClient)
    client.lora_base_dir = Path("/lora")
    return client


def _run(client, scripted, monkeypatch, base_model=None):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("skyrl.tinker.extra.external_inference.asyncio.sleep", _no_sleep)
    return asyncio.run(client._post_completion(scripted, {}, {}, "model_ckpt", base_model=base_model))


def test_happy_path_single_call(monkeypatch):
    """A successful first response issues exactly one completion request."""
    scripted = _ScriptedClient([200])
    result = _run(_make_client(), scripted, monkeypatch)
    assert result == {"choices": []}
    assert scripted.calls == ["/completions"]


def test_self_heals_then_succeeds(monkeypatch):
    """A 404 triggers a proactive adapter load and the retry then succeeds."""
    scripted = _ScriptedClient([404, 200])
    result = _run(_make_client(), scripted, monkeypatch)
    assert result == {"choices": []}
    assert scripted.calls == ["/completions", "/load_lora_adapter", "/completions"]


def test_raises_after_exhausting_retries(monkeypatch):
    """A persistent 404 raises after the retry budget is exhausted."""
    scripted = _ScriptedClient([404] * 6)
    with pytest.raises(httpx.HTTPStatusError):
        _run(_make_client(), scripted, monkeypatch)
    assert scripted.calls.count("/completions") == 6
    assert scripted.calls.count("/load_lora_adapter") == 5


def test_does_not_retry_server_error(monkeypatch):
    """A 500 surfaces immediately without retrying or loading the adapter."""
    scripted = _ScriptedClient([500, 200])
    with pytest.raises(httpx.HTTPStatusError):
        _run(_make_client(), scripted, monkeypatch)
    assert scripted.calls == ["/completions"]


def test_base_model_never_loads_adapter(monkeypatch):
    """Base-model sampling retries transient errors but never calls the adapter-load endpoint."""
    scripted = _ScriptedClient([404, 200])
    result = _run(_make_client(), scripted, monkeypatch, base_model="base")
    assert result == {"choices": []}
    assert "/load_lora_adapter" not in scripted.calls
