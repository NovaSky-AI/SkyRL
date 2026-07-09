"""CPU tests for the SkyRL-Train backend side of Rollout Routing Replay (R3).

These need the SkyRL-Train backend deps (ray/torch) but no GPU or inference
engine. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_router_replay_backend.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Skip cleanly if the SkyRL-Train backend (ray/vllm) can't be imported.
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

import torch  # noqa: E402

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_sample_batch  # noqa: E402

_build_rollout_expert_indices = skyrl_train_backend._build_rollout_expert_indices

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_build_rollout_expert_indices_left_pads_and_aligns():
    # Two samples of differing length; routing has one fewer entry than the full
    # sequence (the last token has no routing), matching the inference engine.
    full_sequences = [[10, 11, 12, 13], [20, 21]]  # lens 4 and 2
    num_layers, topk = 2, 3
    routing_a = [[[t] * topk for _ in range(num_layers)] for t in range(3)]  # 3 = len-1
    routing_b = [[[100 + t] * topk for _ in range(num_layers)] for t in range(1)]  # 1 = len-1
    max_seq_len = 4

    tensor = _build_rollout_expert_indices(full_sequences, [routing_a, routing_b], max_seq_len)
    assert tensor is not None
    assert tuple(tensor.shape) == (2, max_seq_len, num_layers, topk)

    # Sample A: no left pad; routing at positions [0,1,2]; last position (3) zero.
    assert tensor[0, 0, 0, 0].item() == 0
    assert tensor[0, 2, 1, 2].item() == 2
    assert tensor[0, 3].sum().item() == 0
    # Sample B: left pad of 2; routing lands at position 2; last position (3) zero.
    assert tensor[1, 0].sum().item() == 0
    assert tensor[1, 1].sum().item() == 0
    assert tensor[1, 2, 0, 0].item() == 100
    assert tensor[1, 3].sum().item() == 0


def test_build_rollout_expert_indices_none_when_absent():
    assert _build_rollout_expert_indices([[1, 2]], None, 2) is None
    assert _build_rollout_expert_indices([[1, 2]], [None], 2) is None


def test_build_rollout_expert_indices_downcasts_dtype():
    full_sequences = [[1, 2, 3]]
    small = [[[1, 2]] for _ in range(2)]  # expert ids < 256 -> uint8
    assert _build_rollout_expert_indices(full_sequences, [small], 3).dtype == torch.uint8
    big = [[[300, 301]] for _ in range(2)]  # 256..2**15 -> int16
    assert _build_rollout_expert_indices(full_sequences, [big], 3).dtype == torch.int16


def _sample_input(**kwargs) -> types.SampleInput:
    return types.SampleInput(
        base_model=BASE_MODEL,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=4, seed=0),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
        **kwargs,
    )


class _SpyClient:
    def __init__(self):
        self.payloads = []

    async def sample(self, request_payload):
        self.payloads.append(request_payload)
        return {}

    async def aclose(self):
        pass


@pytest.mark.parametrize("replay_enabled", [True, False])
def test_sample_requests_routed_experts_gated_on_replay(monkeypatch, replay_enabled):
    """The sample body sets return_routed_experts iff R3 is enabled on the backend."""
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)

    spy = _SpyClient()
    fake_self = SimpleNamespace(
        _cfg=None,
        _base_lora_signature=None,
        _model_ids_to_role={},
        _inference_engine_client=spy,
        _router_replay_enabled=lambda: replay_enabled,
        _aggregate_sample_results=lambda prepared_batch, outputs: {},
    )
    sample = skyrl_train_backend.SkyRLTrainBackend._sample_with_remote_client

    batch = prepare_sample_batch({"req": ("", _sample_input())})
    sample(fake_self, batch)

    assert len(spy.payloads) == 1
    assert spy.payloads[0]["json"]["return_routed_experts"] is replay_enabled
