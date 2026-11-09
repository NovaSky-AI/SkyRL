"""CPU tests for the host-side rank-local microbatch builder and its single upload.

Run with:
  uv run --isolated --extra megatron --extra dev -- pytest -s tests/backends/skyrl_train/distributed/test_rank_microbatch.py
"""

import dataclasses
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.transformer.moe.router_replay import RouterReplay  # noqa: E402

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (  # noqa: E402
    get_packed_seq_align_size,
)
from skyrl.backends.skyrl_train.distributed.megatron.rank_microbatch import (  # noqa: E402
    _UPLOAD_ALIGNMENT,
    HostMicrobatch,
    LossTensors,
    MicrobatchFeatures,
    RankView,
    _map_tensors,
    build_rank_microbatch,
    recover_canonical_layout,
)
from skyrl.backends.skyrl_train.training_batch import TensorList  # noqa: E402
from skyrl.backends.skyrl_train.utils.packed_tensor import (  # noqa: E402
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.train.dataset.preprocess import (  # noqa: E402
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)
from tests.backends.skyrl_train.sample_support_helpers import csr

# CI installs megatron-core only for its megatron-marked job.
pytestmark = pytest.mark.megatron

VOCAB = 64
NUM_LAYERS = 4
TOPK = 2


def _host(*, r3=False, ssr=False, lengths=((3, 5), (7, 2), (1, 9))):
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1, VOCAB, (p,), generator=g).tolist() for p, _ in lengths]
    responses = [torch.randint(1, VOCAB, (r,), generator=g).tolist() for _, r in lengths]
    routes = None
    if r3:
        routes = [
            torch.randint(0, 250, (p + r, NUM_LAYERS, TOPK), generator=g).numpy().astype(np.int16) for p, r in lengths
        ]
    support = None
    if ssr:
        support = []
        for (_, r), response in zip(lengths, responses):
            ids = torch.randint(0, VOCAB, (r, TOPK), generator=g).numpy().astype(np.int32)
            ids[:, 0] = response
            support.append(csr(ids))
    sequences, attention_mask, response_mask, _, loss_mask, _, packed_routes, sample_support = (
        convert_prompts_responses_to_batch_tensors(
            0,
            prompts,
            responses,
            [[0.0] * r for _, r in lengths],
            [[1] * r for _, r in lengths],
            rollout_expert_indices=routes,
            rollout_sample_support=support,
        )
    )
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    return HostMicrobatch(
        sequences=sequences,
        attention_mask=attention_mask,
        position_ids=position_ids,
        num_actions=response_mask.shape[1],
        rollout_expert_indices=packed_routes if r3 else None,
        router_padding_mask=make_router_padding_mask(attention_mask, [len(e) for e in routes]) if r3 else None,
        rollout_sample_support=sample_support if ssr else None,
        loss=LossTensors(loss_mask=loss_mask, response_mask=response_mask),
    )


_FEATURES = {item.name for item in dataclasses.fields(MicrobatchFeatures)}


@pytest.fixture(autouse=True)
def _routers(monkeypatch):
    """One router per captured layer, as model creation registers them."""
    routers = [SimpleNamespace(layer_number=layer + 1) for layer in range(NUM_LAYERS)]
    monkeypatch.setattr(RouterReplay, "global_router_replay_instances", routers)


def _view(**overrides):
    """``(RankView, MicrobatchFeatures)`` for ``build_rank_microbatch``, from one set of overrides."""
    fields = dict(
        tp_size=1,
        tp_rank=0,
        cp_size=1,
        cp_rank=0,
        is_first_stage=True,
        is_last_stage=True,
        packed=True,
        stage_layer_range=(0, NUM_LAYERS),
    )
    fields.update(overrides)
    features = {name: fields.pop(name) for name in list(fields) if name in _FEATURES}
    return RankView(**fields), MicrobatchFeatures(**features)


def test_router_replay_rejects_sub_sequences():
    host = _host(r3=True)
    host.sub_seq_lengths = TensorList([torch.tensor([int(n)]) for n in host.attention_mask.sum(1)])
    with pytest.raises(ValueError, match="multi-subsequence"):
        build_rank_microbatch(host, *_view())


def test_routes_for_a_model_without_moe_layers_raise():
    with pytest.raises(ValueError, match="no MoE layers"):
        build_rank_microbatch(_host(r3=True), *_view(stage_layer_range=None))


@pytest.mark.parametrize("packed", [True, False])
def test_upload_round_trip(packed):
    microbatch = build_rank_microbatch(_host(r3=True), *_view(packed=packed))
    moved = microbatch.to("cpu")

    originals, uploaded = [], []
    _map_tensors(microbatch, originals.append)
    _map_tensors(moved, uploaded.append)
    assert len(originals) == len(uploaded)
    for before, after in zip(originals, uploaded):
        assert after.dtype == before.dtype and after.shape == before.shape
        assert torch.equal(after, before)
        # Every view starts at an aligned offset of the one uploaded buffer.
        assert (after.data_ptr() - after.untyped_storage().data_ptr()) % _UPLOAD_ALIGNMENT == 0
    assert len({t.untyped_storage().data_ptr() for t in uploaded if t.numel()}) == 1

    unique = {id(t): t for t in originals}.values()
    # Compact routes travel in the rollout dtype and are widened only on the device.
    assert moved.replay_routes.dtype == torch.int16
    if packed:
        # One tensor, one copy: the loss's cu_seqlens is the packed params' own.
        assert moved.loss_inputs.cu_seqlens_padded is moved.model_inputs.packed_seq_params.cu_seqlens_q_padded
        assert len(unique) < len(originals)


def test_map_tensors_rebuilds_packed_containers_without_their_pickle_path(monkeypatch):
    """An empty packed field (no routed tokens on this rank) must move like any other."""

    def no_byte_round_trip(self):
        raise AssertionError("moving a packed field must not round-trip its tensors through bytes")

    monkeypatch.setattr(PackedTensor, "__reduce__", no_byte_round_trip)
    empty = PackedTensor(torch.zeros(0, 2, dtype=torch.int16), cu_seqlens_from_lengths([0, 0]))
    seen = []
    moved = _map_tensors({"routes": empty}, lambda tensor: seen.append(tensor) or tensor.clone())["routes"]
    assert type(moved) is PackedTensor
    assert len(seen) == 2
    assert moved.values.shape == (0, 2) and moved.values.dtype == torch.int16
    assert moved.cu_seqlens.tolist() == [0, 0, 0]


def test_recover_canonical_layout_inverts_the_unpacked_model_layout():
    host = _host()
    microbatch = build_rank_microbatch(host, *_view(packed=False))
    model_ids = microbatch.model_inputs.input_ids
    # Shorter rows move from their left padding to column 0. Offset so the model layout's padding holds
    # nonzero values, which must not come back; a trailing dimension, as logits have.
    values = (model_ids + VOCAB).unsqueeze(-1).expand(-1, -1, 2)
    expected = ((host.sequences + VOCAB) * host.attention_mask).unsqueeze(-1).expand(-1, -1, 2)

    assert torch.equal(recover_canonical_layout(values, microbatch.loss_inputs), expected)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_packed_targets_are_this_cp_ranks_next_tokens(cp_size):
    host = _host()
    attention_mask = host.attention_mask.bool()
    for cp_rank in range(cp_size):
        microbatch = build_rank_microbatch(host, *_view(cp_size=cp_size, cp_rank=cp_rank))
        cu = microbatch.loss_inputs.cu_seqlens_padded.tolist()
        expected = []
        for row, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
            # Each position's next real token; a segment's last token and its padding predict nothing.
            segment = torch.zeros(end - start, dtype=host.sequences.dtype)
            tokens = host.sequences[row][attention_mask[row]]
            segment[: tokens.numel() - 1] = tokens[1:]
            chunks = segment.chunk(2 * cp_size)
            expected += [chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]]
        assert torch.equal(microbatch.loss_inputs.packed_targets, torch.cat(expected).unsqueeze(0))


@pytest.mark.parametrize("is_first_stage,is_last_stage", [(True, False), (False, True), (False, False)])
def test_fields_follow_pipeline_placement(is_first_stage, is_last_stage):
    host = _host(r3=True, ssr=True)
    view = _view(is_first_stage=is_first_stage, is_last_stage=is_last_stage, sample_support=True)
    microbatch = build_rank_microbatch(host, *view)
    assert microbatch.replay_routes is not None
    if is_last_stage:
        assert microbatch.loss_inputs is not None and microbatch.loss_inputs.sample_support is not None
    else:
        assert microbatch.loss_inputs is None
    if not is_first_stage:
        # Later stages build activations from the previous stage; ids are passed through untouched.
        assert microbatch.model_inputs.input_ids is host.sequences


@pytest.mark.parametrize("model_scatters", [False, True])
def test_router_padding_mask_is_sequence_parallel_chunked(model_scatters):
    tp_size = 2
    host = _host(r3=True)
    full = build_rank_microbatch(host, *_view(tp_size=tp_size, sequence_parallel=False)).model_inputs.padding_mask
    for tp_rank in range(tp_size):
        mask = build_rank_microbatch(
            host,
            *_view(
                tp_size=tp_size,
                tp_rank=tp_rank,
                sequence_parallel=True,
                model_scatters_padding_mask=model_scatters,
            ),
        ).model_inputs.padding_mask
        expected = full if model_scatters else full.chunk(tp_size, dim=1)[tp_rank]
        assert torch.equal(mask, expected)


def test_fp8_alignment_of_packed_segments():
    host = _host()
    microbatch = build_rank_microbatch(host, *_view(fp8_enabled=True, fp8_recipe="delayed"))
    align = get_packed_seq_align_size(1, 1, fp8_enabled=True, fp8_recipe="delayed")
    padded = [n + (-n % align) for n in host.attention_mask.sum(1).tolist()]
    assert microbatch.loss_inputs.cu_seqlens_padded.tolist() == cu_seqlens_from_lengths(padded).tolist()


def _read_only(tensor):
    # The form a tensor field arrives in under Ray: a view onto a read-only object-store buffer.
    buffer = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
    buffer.setflags(write=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(buffer).view(tensor.dtype).reshape(tensor.shape)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(packed=True),
        dict(packed=False),
        dict(packed=True, tp_size=2, tp_rank=1, sequence_parallel=True),
        dict(packed=True, cp_size=2, cp_rank=1),
        dict(packed=True, fp8_enabled=True, fp8_recipe="delayed"),
        dict(packed=True, is_first_stage=False, is_last_stage=False),
    ],
)
def test_received_tensors_are_never_written(overrides):
    original, received = [], []
    source = _host(r3=True, ssr=True)
    _map_tensors(source, original.append)
    host = _map_tensors(source, _read_only)
    _map_tensors(host, received.append)
    # Every tensor, including those nested in the routes and sample-support containers, was replaced.
    assert len(received) == len(original)
    assert not {t.untyped_storage().data_ptr() for t in received} & {t.untyped_storage().data_ptr() for t in original}
    snapshot = [(t._version, t.clone()) for t in received]

    microbatch = build_rank_microbatch(host, *_view(sample_support=True, **overrides))
    microbatch.to("cpu")

    for tensor, (version, before) in zip(received, snapshot):
        assert tensor._version == version and torch.equal(tensor, before)


@pytest.mark.parametrize("packed,cp_size", [(False, 1), (True, 1), (True, 2)])
def test_sparse_metadata_is_staged_in_the_same_rank_layout_as_targets(packed, cp_size):
    host = _host()
    canonical = torch.zeros_like(host.attention_mask, dtype=torch.bool)
    canonical[:, -host.num_actions - 1 : -1] = host.loss.loss_mask.bool()
    for cp_rank in range(cp_size):
        microbatch = build_rank_microbatch(
            host, *_view(packed=packed, cp_size=cp_size, cp_rank=cp_rank, block_sparse=True)
        )
        if packed:
            expected = []
            cu = microbatch.loss_inputs.cu_seqlens_padded.tolist()
            for row, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
                segment = torch.zeros(end - start, dtype=torch.bool)
                real = canonical[row, host.attention_mask[row].bool()]
                segment[: real.numel()] = real
                chunks = segment.chunk(2 * cp_size)
                expected.extend([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]])
            expected = torch.cat(expected).unsqueeze(0)
            assert microbatch.loss_inputs.active_mask.shape == microbatch.loss_inputs.packed_targets.shape
        else:
            expected = canonical[:, :-1]
        torch.testing.assert_close(microbatch.loss_inputs.active_mask, expected)
        uploaded = microbatch.to("cpu")
        torch.testing.assert_close(uploaded.loss_inputs.active_mask, expected)


def test_packed_mtp_gets_packed_positions_and_the_real_rows_token_mask():
    """The native MTP block re-embeds the positions it is given; the draft loss masks padding and synthetic rows."""
    host = _host()
    host.loss.real_sample_mask = torch.tensor([True, False, True])
    microbatch = build_rank_microbatch(host, *_view(mtp=True))
    plain = build_rank_microbatch(host, *_view())

    assert plain.model_inputs.position_ids is None
    positions = microbatch.model_inputs.position_ids
    packed_ids = microbatch.model_inputs.input_ids
    assert positions.shape == packed_ids.shape
    real = host.attention_mask.bool() & host.loss.real_sample_mask[:, None]
    mask = microbatch.loss_inputs.mtp_packed_mask
    assert mask.shape == packed_ids.shape and mask.dtype == torch.float32
    # Each real row's tokens are marked in order; the synthetic row and the alignment padding are not.
    assert int(mask.sum()) == int(real.sum())
    assert torch.equal(packed_ids[mask.bool()], host.sequences[real])
    assert torch.equal(positions[mask.bool()], host.position_ids[real])


@pytest.mark.parametrize("implicit", [True, False])
def test_dsv4_hybrid_attention_gets_no_explicit_mask(implicit):
    """DSV4's hybrid attention builds its causal mask itself; other unpacked models get TE's padding mask."""
    host = _host()
    model_inputs = build_rank_microbatch(host, *_view(packed=False, implicit_causal_mask=implicit)).model_inputs
    if implicit:
        assert model_inputs.attention_mask is None
    else:
        assert model_inputs.attention_mask is not None and model_inputs.attention_mask.dtype == torch.bool


def test_unpacked_width_honors_the_extra_seq_align():
    """``mlp_chunks_for_training`` (or DSV4's compression ratios) pad the unpacked width further."""
    host = _host()
    default = build_rank_microbatch(host, *_view(packed=False)).model_inputs
    aligned = build_rank_microbatch(host, *_view(packed=False, seq_align=64)).model_inputs
    assert aligned.input_ids.shape[1] % 64 == 0
    assert aligned.input_ids.shape[1] >= default.input_ids.shape[1]
