"""Round-trip tests for proto serialization of retrieve_future results.

Each test serializes a result_data dict the way the API server does and
deserializes it with the installed tinker SDK's own proto deserializers,
so the wire conventions (byte layouts, NaN/sentinel fills) are checked
against the exact code the client runs.
"""

import math

import numpy as np
import pytest
import tinker.types as sdk_types
from tinker import ForwardBackwardOutput, SampleResponse
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.proto.response_conv import deserialize_proto_response

from skyrl.tinker import api, types
from skyrl.tinker.proto_serialization import (
    parse_forward_backward_input,
    parse_forward_backward_metadata,
    parse_forward_backward_request,
    serialize_result,
)


def roundtrip_sample(result_data: dict) -> SampleResponse:
    return deserialize_proto_response(serialize_result(types.RequestType.SAMPLE, result_data), SampleResponse)


def roundtrip_forward_backward(result_data: dict) -> ForwardBackwardOutput:
    return deserialize_proto_response(
        serialize_result(types.RequestType.FORWARD_BACKWARD, result_data), ForwardBackwardOutput
    )


def test_sample_sequences():
    result_data = {
        "sequences": [
            {"stop_reason": "stop", "tokens": [1, 2, 3], "logprobs": [-0.5, -1.25, -2.0]},
            {"stop_reason": "length", "tokens": [7], "logprobs": [-0.125]},
        ]
    }
    response = roundtrip_sample(result_data)
    assert len(response.sequences) == 2
    assert response.sequences[0].stop_reason == "stop"
    assert response.sequences[0].tokens == [1, 2, 3]
    assert response.sequences[0].logprobs == [-0.5, -1.25, -2.0]
    assert response.sequences[1].stop_reason == "length"
    assert response.sequences[1].tokens == [7]
    assert response.prompt_logprobs is None
    assert response.topk_prompt_logprobs is None


def test_sample_prompt_logprobs_none_becomes_nan_wire_none_client():
    result_data = {
        "sequences": [{"stop_reason": "stop", "tokens": [1], "logprobs": [-0.5]}],
        "prompt_logprobs": [None, -0.25, -0.75],
    }
    response = roundtrip_sample(result_data)
    assert response.prompt_logprobs == [None, pytest.approx(-0.25), pytest.approx(-0.75)]
    assert math.isnan(response.prompt_logprobs_np[0])


def test_sample_topk_prompt_logprobs():
    result_data = {
        "sequences": [{"stop_reason": "stop", "tokens": [1], "logprobs": [-0.5]}],
        # First position undefined, second full width, third narrower (ragged).
        "topk_prompt_logprobs": [
            None,
            [(11, -0.1), (12, -0.2)],
            [(13, -0.3)],
        ],
    }
    response = roundtrip_sample(result_data)
    topk = response.topk_prompt_logprobs
    assert topk[0] is None
    assert topk[1] == [(11, pytest.approx(-0.1)), (12, pytest.approx(-0.2))]
    assert topk[2] == [(13, pytest.approx(-0.3))]


def test_sample_topk_all_rows_undefined():
    result_data = {
        "sequences": [{"stop_reason": "stop", "tokens": [1], "logprobs": [-0.5]}],
        "topk_prompt_logprobs": [None, None],
    }
    response = roundtrip_sample(result_data)
    assert response.topk_prompt_logprobs == [None, None]


def test_forward_backward_ragged_datums_and_metrics():
    result_data = {
        "loss_fn_output_type": "scalar",
        "loss_fn_outputs": [
            {
                "elementwise_loss": {"data": [1.0, 2.0, 3.0], "dtype": "float32", "shape": [3]},
                "logprobs": {"data": [-0.1, -0.2, -0.3], "dtype": "float32", "shape": [3]},
            },
            {
                "elementwise_loss": {"data": [4.0], "dtype": "float32", "shape": [1]},
                "logprobs": {"data": [-0.4], "dtype": "float32", "shape": [1]},
            },
        ],
        "metrics": {"loss:sum": 10.5, "clip_fraction": 0.25},
    }
    output = roundtrip_forward_backward(result_data)
    assert output.loss_fn_output_type == "scalar"
    assert output.metrics == {"loss:sum": 10.5, "clip_fraction": 0.25}
    assert len(output.loss_fn_outputs) == 2
    first, second = output.loss_fn_outputs
    np.testing.assert_allclose(first["elementwise_loss"].tolist(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(first["logprobs"].tolist(), [-0.1, -0.2, -0.3], rtol=1e-6)
    assert first["logprobs"].dtype == "float32"
    assert first["logprobs"].shape == [3]
    np.testing.assert_allclose(second["elementwise_loss"].tolist(), [4.0])
    assert second["logprobs"].shape == [1]


def test_forward_backward_empty_outputs():
    result_data = {"loss_fn_output_type": "scalar", "loss_fn_outputs": [], "metrics": {"loss:sum": 0.0}}
    output = roundtrip_forward_backward(result_data)
    assert output.loss_fn_outputs == []
    assert output.metrics == {"loss:sum": 0.0}


def test_forward_backward_field_missing_in_one_datum():
    result_data = {
        "loss_fn_output_type": "scalar",
        "loss_fn_outputs": [
            {"logprobs": {"data": [-0.1, -0.2], "dtype": "float32", "shape": [2]}},
            {},
        ],
        "metrics": {},
    }
    output = roundtrip_forward_backward(result_data)
    assert len(output.loss_fn_outputs) == 2
    np.testing.assert_allclose(output.loss_fn_outputs[0]["logprobs"].tolist(), [-0.1, -0.2], rtol=1e-6)
    assert output.loss_fn_outputs[1]["logprobs"].tolist() == []


def test_forward_uses_forward_backward_wire_type():
    result_data = {
        "loss_fn_output_type": "scalar",
        "loss_fn_outputs": [{"logprobs": {"data": [-0.5], "dtype": "float32", "shape": [1]}}],
        "metrics": {},
    }
    proto_bytes = serialize_result(types.RequestType.FORWARD, result_data)
    output = deserialize_proto_response(proto_bytes, ForwardBackwardOutput)
    np.testing.assert_allclose(output.loss_fn_outputs[0]["logprobs"].tolist(), [-0.5])


def test_external_sample_serializes_as_sample_response():
    """EXTERNAL futures (samples forwarded to external inference engines) must
    serialize as SampleResponse like SAMPLE ones."""
    result_data = {"sequences": [{"stop_reason": "stop", "tokens": [5, 6], "logprobs": [-0.5, -1.0]}]}
    proto_bytes = serialize_result(types.RequestType.EXTERNAL, result_data)
    response = deserialize_proto_response(proto_bytes, SampleResponse)
    assert response.sequences[0].tokens == [5, 6]


def test_unsupported_request_type_raises():
    with pytest.raises(ValueError, match="No proto serialization"):
        serialize_result(types.RequestType.OPTIM_STEP, {})


def encode_sdk_fwd_bwd_request(loss_fn_config=None, forward_only=False) -> bytes:
    """Build a fwd_bwd request with the SDK's own proto encoder."""
    request = sdk_types.ForwardBackwardRequest(
        model_id="model_abc",
        seq_id=1,
        forward_backward_input=sdk_types.ForwardBackwardInput(
            data=[
                sdk_types.Datum(
                    model_input=sdk_types.ModelInput.from_ints([1, 2, 3, 4]),
                    loss_fn_inputs={
                        "target_tokens": sdk_types.TensorData(data=[2, 3, 4, 5], dtype="int64", shape=[4]),
                        "weights": sdk_types.TensorData(data=[1.0, 0.5, 1.0, 0.0], dtype="float32", shape=[4]),
                    },
                )
            ],
            loss_fn="cross_entropy",
            loss_fn_config=loss_fn_config,
        ),
    )
    proto_msg = forward_backward_request_to_proto(request)
    proto_msg.forward_only = forward_only
    return proto_msg.SerializeToString()


def test_parse_forward_backward_request():
    request_dict, forward_only = parse_forward_backward_request(encode_sdk_fwd_bwd_request())
    assert not forward_only

    # The dict must validate against the API's JSON request model.
    request = api.ForwardBackwardRequest.model_validate(request_dict)
    assert request.model_id == "model_abc"
    assert request.seq_id == 1
    assert request.forward_backward_input.loss_fn == "cross_entropy"
    assert request.forward_backward_input.loss_fn_config is None
    (datum,) = request.forward_backward_input.data
    (chunk,) = datum.model_input.chunks
    assert chunk.tokens == [1, 2, 3, 4]
    assert datum.loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5]
    assert datum.loss_fn_inputs["weights"].data == [1.0, 0.5, 1.0, 0.0]

    assert parse_forward_backward_metadata(encode_sdk_fwd_bwd_request()) == (
        "model_abc",
        1,
        False,
    )
    internal = parse_forward_backward_input(encode_sdk_fwd_bwd_request())
    assert internal.data[0].loss_fn_inputs.target_tokens.data == [2, 3, 4, 5]
    assert internal.data[0].loss_fn_inputs.advantages.data == []


def test_parse_forward_backward_request_forward_only_and_config():
    body = encode_sdk_fwd_bwd_request(loss_fn_config={"clip_low_threshold": 0.2}, forward_only=True)
    request_dict, forward_only = parse_forward_backward_request(body)
    assert forward_only
    assert request_dict["forward_backward_input"]["loss_fn_config"] == {"clip_low_threshold": 0.2}


def test_parse_forward_backward_request_garbage_raises():
    with pytest.raises(Exception):
        parse_forward_backward_request(b"\xff\xfe not a proto")
