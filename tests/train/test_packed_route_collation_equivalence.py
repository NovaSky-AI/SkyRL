"""Bit-identity of packed R3 route collation against the padded-rectangle path.

The packed buffer plus ``cu_seqlens`` replaces a ``[batch, max_total, layers, topk]``
rectangle that ``align_token_metadata`` immediately compacted back to ragged. What
Megatron receives must not change, so both paths are run end to end here -- collation,
micro-batch padding, layer selection, packing/CP layout, and the TP slice -- and the
per-layer tensors handed to ``RouterReplay.set_replay_data`` compared.

Run with:
  uv run --isolated --extra dev pytest tests/train/test_packed_route_collation_equivalence.py
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    align_token_metadata,
    build_token_metadata_layout,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.utils import replay_utils
from skyrl.backends.skyrl_train.utils.replay_utils import (
    _split_replay_indices,
    make_replay_padding_indices,
    replay_padding_row,
)
from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)

NUM_LAYERS = 3
TOPK = 4
PAD_TOKEN_ID = 0
# Above 2**8 so the compact dtype is int16, matching the production route width.
MIN_EXPERT_ID = 300


# ---------------------------------------------------------------------------
# Reference: the padded [batch, max_total, layers, topk] rectangle
# ---------------------------------------------------------------------------


def _reference_padded_routes(
    routes: list[np.ndarray],
    prompt_lens: list[int],
    response_lens: list[int],
) -> torch.Tensor:
    """The pre-change collation: a left-padded batch-major rectangle."""
    max_total = max(p + r for p, r in zip(prompt_lens, response_lens))
    dtype = max((entry.dtype for entry in routes), key=lambda d: d.itemsize)
    torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
    padded = torch.empty((len(routes), max_total, NUM_LAYERS, TOPK), dtype=torch_dtype)
    padding_row = torch.arange(TOPK, dtype=torch_dtype)
    for sample_index, entry in enumerate(routes):
        left_pad = max_total - (prompt_lens[sample_index] + response_lens[sample_index])
        route_end = left_pad + entry.shape[0]
        padded[sample_index, :left_pad] = padding_row
        padded[sample_index, left_pad:route_end] = torch.from_numpy(entry)
        padded[sample_index, route_end:] = padding_row
    return padded


def _reference_replay_data(
    padded_routes: torch.Tensor,
    attention_mask: torch.Tensor,
    local_layers: list[int],
    *,
    packed: bool,
    tp_size: int,
    tp_rank: int,
) -> list[torch.Tensor]:
    """The pre-change trainer path: gather real tokens out of the padded rectangle."""
    layout = build_token_metadata_layout(
        attention_mask,
        padded_routes.device,
        packed=packed,
        fp8_enabled=False,
    )
    local = padded_routes.index_select(2, torch.tensor(local_layers, dtype=torch.long))
    aligned = align_token_metadata(
        local,
        layout,
        replay_padding_row(TOPK, dtype=padded_routes.dtype),
    )
    if tp_size > 1:
        chunk = aligned.shape[1] // tp_size
        aligned = aligned[:, tp_rank * chunk : (tp_rank + 1) * chunk, :, :]
    return _split_replay_indices(aligned)


# ---------------------------------------------------------------------------
# Fixtures and harness
# ---------------------------------------------------------------------------


@pytest.fixture
def parallel_state(monkeypatch):
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    return mpu


@pytest.fixture
def router_replay(monkeypatch):
    """Capture what ``setup_per_microbatch_replay_forward`` hands Megatron."""
    module = types.ModuleType("megatron.core.transformer.moe.router_replay")

    class RouterReplay:
        global_router_replay_instances = [object() for _ in range(NUM_LAYERS)]
        replay_data: list[torch.Tensor] | None = None

        @classmethod
        def set_replay_data(cls, replay_data):
            cls.replay_data = replay_data

        @classmethod
        def set_global_router_replay_action(cls, action):
            pass

    module.RouterReplay = RouterReplay
    module.RouterReplayAction = SimpleNamespace(REPLAY_FORWARD="replay_forward")
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", module)
    monkeypatch.setattr(
        replay_utils,
        "scatter_router_padding_mask_for_model",
        lambda mask, model, model_config: mask,
    )
    return RouterReplay


def _make_batch(lengths: list[tuple[int, int]], *, captured_shortfall: int = 0, seed: int = 0):
    """Build one batch of trajectories with the given ``(prompt_len, response_len)`` pairs.

    ``captured_shortfall`` leaves that many trailing tokens of the last trajectory without a
    captured route, exercising the dummy-row tail that both paths must fill identically.
    """
    rng = np.random.default_rng(seed)
    prompts, responses, rewards, loss_masks, routes = [], [], [], [], []
    for index, (prompt_len, response_len) in enumerate(lengths):
        prompts.append(list(rng.integers(1, 1000, size=prompt_len)))
        responses.append(list(rng.integers(1, 1000, size=response_len)))
        rewards.append([0.0] * response_len)
        loss_masks.append([1] * response_len)
        captured = prompt_len + response_len
        if index == len(lengths) - 1:
            captured -= captured_shortfall
        routes.append(
            rng.integers(
                MIN_EXPERT_ID,
                MIN_EXPERT_ID + 2000,
                size=(captured, NUM_LAYERS, TOPK),
                dtype=np.int16,
            )
        )
    return prompts, responses, rewards, loss_masks, routes


def _run_both_paths(
    lengths: list[tuple[int, int]],
    *,
    packed: bool,
    tp_size: int,
    local_layers: list[int],
    captured_shortfall: int = 0,
    batch_pad_size: int = 0,
    stage_range: tuple[int, int] = (0, NUM_LAYERS),
    monkeypatch,
    parallel_state,
    router_replay,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    prompts, responses, rewards, loss_masks, routes = _make_batch(lengths, captured_shortfall=captured_shortfall)
    (
        sequences,
        attention_mask,
        response_mask,
        _rewards,
        loss_mask,
        _logprobs,
        packed_routes,
    ) = convert_prompts_responses_to_batch_tensors(
        PAD_TOKEN_ID,
        prompts,
        responses,
        rewards,
        loss_masks,
        rollout_expert_indices=routes,
    )
    router_padding_mask = make_router_padding_mask(attention_mask, [entry.shape[0] for entry in routes])
    padded_routes = _reference_padded_routes(routes, [len(p) for p in prompts], [len(r) for r in responses])

    if batch_pad_size:
        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "loss_mask": loss_mask,
                "rollout_expert_indices": packed_routes,
                "router_padding_mask": router_padding_mask,
            }
        )
        batch.metadata = {"uids": [f"u{index}" for index in range(len(prompts))]}
        batch = pad_training_input_batch(batch, batch_pad_size)
        attention_mask = batch["attention_mask"]
        router_padding_mask = batch["router_padding_mask"]
        packed_routes = batch["rollout_expert_indices"]
        # The old path copied row 0 into each padding row and filled its routes with
        # dummies; reproduce exactly that rectangle for the reference.
        padded_routes = torch.cat(
            [
                padded_routes,
                make_replay_padding_indices((batch_pad_size, *padded_routes.shape[1:]), dtype=padded_routes.dtype),
            ],
            dim=0,
        )

    tp_rank = tp_size - 1
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: tp_rank, raising=False)
    router_replay.global_router_replay_instances = [object() for _ in local_layers]
    monkeypatch.setattr(replay_utils, "_get_current_pp_stage_layer_range", lambda model_config: stage_range)

    layout = build_token_metadata_layout(attention_mask, packed_routes.device, packed=packed, fp8_enabled=False)
    replay_utils.setup_per_microbatch_replay_forward(
        packed_routes,
        router_padding_mask,
        attention_mask,
        model=object(),
        model_config=SimpleNamespace(fp8=None, sequence_parallel=False),
        metadata_layout=layout,
        remove_microbatch_padding=packed,
    )
    new_replay_data = [tensor.clone() for tensor in router_replay.replay_data]

    reference = _reference_replay_data(
        padded_routes,
        attention_mask,
        local_layers,
        packed=packed,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    return new_replay_data, reference


def _assert_bit_identical(new_data: list[torch.Tensor], reference: list[torch.Tensor]) -> None:
    assert len(new_data) == len(reference)
    for slot, (produced, expected) in enumerate(zip(new_data, reference, strict=True)):
        assert produced.dtype == expected.dtype == torch.int32, slot
        assert produced.shape == expected.shape, (slot, produced.shape, expected.shape)
        assert torch.equal(produced, expected), slot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

LENGTH_DISTRIBUTIONS = {
    # No padding at all: the packed and padded layouts coincide.
    "uniform": [(8, 8), (8, 8), (8, 8), (8, 8)],
    "mild_ragged": [(8, 8), (7, 8), (8, 6), (6, 7)],
    # Typical RL: an order of magnitude between the shortest and longest trajectory.
    "typical_rl": [(2, 2), (8, 24), (4, 6), (1, 31)],
    "heavy_tail": [(1, 1), (1, 2), (2, 1), (16, 32)],
}


@pytest.mark.parametrize("distribution", sorted(LENGTH_DISTRIBUTIONS))
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_packed_routes_match_padded_rectangle(
    monkeypatch, parallel_state, router_replay, distribution, packed, tp_size
):
    new_data, reference = _run_both_paths(
        LENGTH_DISTRIBUTIONS[distribution],
        packed=packed,
        tp_size=tp_size,
        local_layers=list(range(NUM_LAYERS)),
        monkeypatch=monkeypatch,
        parallel_state=parallel_state,
        router_replay=router_replay,
    )
    _assert_bit_identical(new_data, reference)


@pytest.mark.parametrize("packed", [False, True])
def test_packed_routes_match_with_uncaptured_suffix(monkeypatch, parallel_state, router_replay, packed):
    """A trajectory whose trailing tokens have no captured route needs identical dummy rows."""
    new_data, reference = _run_both_paths(
        LENGTH_DISTRIBUTIONS["typical_rl"],
        packed=packed,
        tp_size=1,
        local_layers=list(range(NUM_LAYERS)),
        captured_shortfall=3,
        monkeypatch=monkeypatch,
        parallel_state=parallel_state,
        router_replay=router_replay,
    )
    _assert_bit_identical(new_data, reference)


@pytest.mark.parametrize("packed", [False, True])
def test_packed_routes_match_under_batch_padding(monkeypatch, parallel_state, router_replay, packed):
    """``pad_training_input_batch`` rows must land on the same dummy routes."""
    new_data, reference = _run_both_paths(
        LENGTH_DISTRIBUTIONS["mild_ragged"],
        packed=packed,
        tp_size=1,
        local_layers=list(range(NUM_LAYERS)),
        batch_pad_size=2,
        monkeypatch=monkeypatch,
        parallel_state=parallel_state,
        router_replay=router_replay,
    )
    _assert_bit_identical(new_data, reference)


@pytest.mark.parametrize("packed", [False, True])
def test_packed_routes_match_on_a_pipeline_stage_subset(monkeypatch, parallel_state, router_replay, packed):
    """Under PP a stage owns a subset of routers, so layer selection must agree too."""
    # A stage starting at layer 1 with 2 routers owns captured layers 1 and 2, not 0.
    new_data, reference = _run_both_paths(
        LENGTH_DISTRIBUTIONS["typical_rl"],
        packed=packed,
        tp_size=1,
        local_layers=[1, 2],
        stage_range=(1, 2),
        monkeypatch=monkeypatch,
        parallel_state=parallel_state,
        router_replay=router_replay,
    )
    _assert_bit_identical(new_data, reference)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_packed_routes_match_under_context_parallelism(monkeypatch, parallel_state, router_replay, cp_size):
    """CP shards each padded sequence into front/back halves per rank."""
    for cp_rank in range(cp_size):
        monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: cp_size, raising=False)
        monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda rank=cp_rank: rank, raising=False)
        new_data, reference = _run_both_paths(
            LENGTH_DISTRIBUTIONS["mild_ragged"],
            packed=True,
            tp_size=1,
            local_layers=list(range(NUM_LAYERS)),
            monkeypatch=monkeypatch,
            parallel_state=parallel_state,
            router_replay=router_replay,
        )
        _assert_bit_identical(new_data, reference)


@pytest.mark.parametrize("distribution", sorted(LENGTH_DISTRIBUTIONS))
def test_packed_collation_allocates_no_padded_rectangle(distribution):
    """The packed buffer must hold exactly the batch's real tokens."""
    prompts, responses, rewards, loss_masks, routes = _make_batch(LENGTH_DISTRIBUTIONS[distribution])
    *_, packed_routes = convert_prompts_responses_to_batch_tensors(
        PAD_TOKEN_ID,
        prompts,
        responses,
        rewards,
        loss_masks,
        rollout_expert_indices=routes,
    )

    total_real = sum(len(p) + len(r) for p, r in zip(prompts, responses))
    max_total = max(len(p) + len(r) for p, r in zip(prompts, responses))
    assert packed_routes.values.shape == (total_real, NUM_LAYERS, TOPK)
    assert packed_routes.values.numel() <= len(prompts) * max_total * NUM_LAYERS * TOPK
    assert packed_routes.cu_seqlens.tolist()[-1] == total_real
