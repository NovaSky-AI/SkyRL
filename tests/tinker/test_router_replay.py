"""CPU unit tests for Rollout Routing Replay (R3) plumbing through the Tinker path.

Run with:
    uv run --isolated --extra dev --extra jax --extra tinker pytest tests/tinker/test_router_replay.py -v

These cover the non-GPU, dependency-light data-flow: the API/types round-trip of
routed experts and the engine's flat->3D reshape. The SkyRL-Train backend tensor
assembly (which needs ray) is covered in
``tests/tinker/skyrl_train/test_router_replay_backend.py``; the end-to-end replay
against a real MoE model is covered by the GPU test in
``tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay.py``.
"""

import pytest

from skyrl.tinker import api, types
from skyrl.tinker.engine import _reshape_routed_experts, prepare_model_pass_batch


def _make_routed_experts_tensor_data(seq_len: int, num_layers: int, topk: int) -> types.TensorData:
    """A TensorData whose flat data encodes value = t*1000 + layer*10 + k for easy checks."""
    flat = [t * 1000 + layer * 10 + k for t in range(seq_len) for layer in range(num_layers) for k in range(topk)]
    return types.TensorData(data=flat, shape=[seq_len, num_layers, topk])


def test_reshape_routed_experts_roundtrip():
    td = _make_routed_experts_tensor_data(seq_len=3, num_layers=2, topk=4)
    reshaped = _reshape_routed_experts(td)
    assert len(reshaped) == 3
    assert len(reshaped[0]) == 2
    assert len(reshaped[0][0]) == 4
    # Spot check the encoding survives.
    assert reshaped[2][1][3] == 2 * 1000 + 1 * 10 + 3


def test_reshape_routed_experts_none_and_empty():
    assert _reshape_routed_experts(None) is None
    assert _reshape_routed_experts(types.TensorData(data=[])) is None


def test_reshape_routed_experts_rejects_bad_shape():
    with pytest.raises(ValueError, match="3-D shape"):
        _reshape_routed_experts(types.TensorData(data=[1, 2, 3], shape=[3]))
    with pytest.raises(ValueError, match="does not match shape"):
        _reshape_routed_experts(types.TensorData(data=[1, 2, 3], shape=[2, 2, 2]))


def test_tensor_data_shape_optional_and_backward_compatible():
    # No shape -> treated as 1-D, existing behavior preserved.
    td = types.TensorData(data=[1.0, 2.0, 3.0])
    assert td.shape is None
    # api TensorData forwards shape through to_types.
    api_td = api.TensorData(data=[1, 2, 3, 4], shape=[2, 2])
    assert api_td.to_types().shape == [2, 2]


def test_api_datum_to_types_carries_routed_experts():
    datum = api.Datum(
        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[2, 3, 4]),
            "advantages": api.TensorData(data=[0.0, 1.0, 0.0]),
            "logprobs": api.TensorData(data=[-0.1, -0.2, -0.3]),
            "routed_experts": api.TensorData(data=list(range(2 * 2 * 4)), shape=[2, 2, 4]),
        },
    )
    t = datum.to_types()
    assert t.loss_fn_inputs.routed_experts is not None
    assert t.loss_fn_inputs.routed_experts.shape == [2, 2, 4]


def test_api_datum_without_routed_experts_stays_none():
    datum = api.Datum(
        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2])]),
        loss_fn_inputs={"target_tokens": api.TensorData(data=[2, 3])},
    )
    assert datum.to_types().loss_fn_inputs.routed_experts is None


def test_generated_sequence_round_trips_routed_experts():
    seq = types.GeneratedSequence(
        stop_reason="stop", tokens=[1, 2], logprobs=[-0.1, -0.2], routed_experts=[[[0, 1]], [[2, 3]]]
    )
    dumped = types.SampleOutput(sequences=[seq]).model_dump()
    assert dumped["sequences"][0]["routed_experts"] == [[[0, 1]], [[2, 3]]]


def test_prepare_model_pass_batch_extracts_routed_experts():
    routed = _make_routed_experts_tensor_data(seq_len=3, num_layers=2, topk=4)
    datum_with = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]),
            weights=types.TensorData(data=[1.0, 1.0, 1.0]),
            advantages=types.TensorData(data=[0.0, 1.0, 0.0]),
            logprobs=types.TensorData(data=[-0.1, -0.2, -0.3]),
            routed_experts=routed,
        ),
    )
    datum_without = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[5, 6])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[6, 7]),
            weights=types.TensorData(data=[1.0, 1.0]),
            advantages=types.TensorData(data=[0.0, 1.0]),
            logprobs=types.TensorData(data=[-0.1, -0.2]),
        ),
    )
    requests = {
        "req1": ("model1", types.ForwardBackwardInput(data=[datum_with, datum_without], loss_fn="importance_sampling")),
    }
    batch = prepare_model_pass_batch(requests)
    assert len(batch.all_routed_experts) == 2
    assert batch.all_routed_experts[0] is not None
    assert len(batch.all_routed_experts[0]) == 3  # seq_len
    assert batch.all_routed_experts[1] is None
