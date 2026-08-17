"""Proto wire-format conversion for the Tinker SDK's binary paths.

Two directions, both required by tinker SDK >= 0.25.0 (older SDKs negotiate
these via server-side client_config flags and fall back to JSON):

- **Responses**: the SDK retrieves ``sample``, ``forward``, and
  ``forward_backward`` results as protobuf (``Accept: application/x-protobuf``
  on ``retrieve_future``) and rejects JSON for these types. Serializers here
  are the inverse of the SDK's ``tinker/proto/response_conv.py``: tokens as
  little-endian int32 bytes, logprobs as float32 bytes, undefined
  ``prompt_logprobs`` entries as NaN, undefined top-k entries sentinel-filled
  (token_id=0, logprob=-99999.0), and each ``BatchedTensor`` concatenating
  per-datum arrays with int64 byte offsets.

- **Requests**: the SDK submits ``forward_backward`` bodies as protobuf
  (``Content-Type: application/x-protobuf``), with forward-only passes sent to
  the same endpoint via the ``forward_only`` flag instead of ``/api/v1/forward``.
  The parser here is the inverse of the SDK's ``tinker/proto/request_conv.py``.
"""

import base64

import numpy as np
from tinker.proto import tinker_public_pb2 as pb

from skyrl.tinker import types

PROTO_CONTENT_TYPE = "application/x-protobuf"

# Request types whose results the SDK can retrieve in proto form. All other
# results are only ever served as JSON. EXTERNAL is a sample forwarded to
# external/backend inference engines; its result_data is a SampleOutput dump.
PROTO_SERIALIZABLE_REQUEST_TYPES = frozenset(
    {
        types.RequestType.SAMPLE,
        types.RequestType.EXTERNAL,
        types.RequestType.FORWARD,
        types.RequestType.FORWARD_BACKWARD,
    }
)

# Sentinel fill for undefined top-k positions; must match the SDK's
# MASK_LOGPROB in tinker/types/sample_response.py.
_TOPK_MASK_TOKEN_ID = 0
_TOPK_MASK_LOGPROB = -99999.0

_STOP_REASON_TO_PROTO = {
    "stop": pb.STOP_REASON_STOP,
    "length": pb.STOP_REASON_LENGTH,
}

_TENSOR_DTYPE_TO_PROTO = {"float32": pb.DTYPE_FLOAT32, "int64": pb.DTYPE_INT64}
_TENSOR_DTYPE_TO_NUMPY = {"float32": np.float32, "int64": np.int64}

# Request tensors only arrive as the public dtypes (the SDK collapses to
# {float32, int64} on the write path).
_PROTO_DTYPE_TO_NUMPY = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64}


def parse_forward_backward_request(body: bytes) -> tuple[dict, bool]:
    """Parse a proto ForwardBackwardRequest body into the JSON-equivalent
    request dict (ready for the API's ``ForwardBackwardRequest`` model) and
    the ``forward_only`` flag."""
    msg = pb.ForwardBackwardRequest()
    msg.ParseFromString(body)

    data = []
    for datum in msg.data:
        chunks = []
        for chunk in datum.model_input:
            which = chunk.WhichOneof("chunk")
            if which == "encoded_text":
                chunks.append(
                    {
                        "type": "encoded_text",
                        "tokens": np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist(),
                    }
                )
            elif which == "image":
                image_chunk = {
                    "type": "image",
                    "data": base64.b64encode(chunk.image.data).decode(),
                    "format": chunk.image.format,
                }
                if chunk.image.HasField("expected_tokens"):
                    image_chunk["expected_tokens"] = chunk.image.expected_tokens
                chunks.append(image_chunk)
            else:
                raise ValueError(f"Unsupported model input chunk type: {which}")
        loss_fn_inputs = {name: _tensor_from_proto(tensor) for name, tensor in datum.loss_fn_inputs.items()}
        data.append({"model_input": {"chunks": chunks}, "loss_fn_inputs": loss_fn_inputs})

    request_dict = {
        "model_id": msg.model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": msg.loss_fn,
            "loss_fn_config": dict(msg.loss_fn_config) or None,
        },
    }
    return request_dict, msg.forward_only


def _tensor_from_proto(tensor: pb.Tensor) -> dict:
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(tensor.dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported tensor dtype value in request: {tensor.dtype}")
    if tensor.WhichOneof("encoding") != "dense":
        raise ValueError("Sparse loss_fn_input tensors are not supported")
    return {"data": np.frombuffer(tensor.dense, dtype=np_dtype).tolist()}


def serialize_result(request_type: types.RequestType, result_data: dict) -> bytes:
    """Serialize a completed future's result_data dict to proto wire bytes."""
    if request_type in (types.RequestType.SAMPLE, types.RequestType.EXTERNAL):
        return _serialize_sample_output(result_data)
    if request_type in (types.RequestType.FORWARD, types.RequestType.FORWARD_BACKWARD):
        return _serialize_forward_backward_output(result_data)
    raise ValueError(f"No proto serialization for request type: {request_type}")


def _serialize_sample_output(result_data: dict) -> bytes:
    output = types.SampleOutput.model_validate(result_data)
    proto = pb.SampleResponse()

    for seq in output.sequences:
        proto.sequences.append(
            pb.SampledSequence(
                stop_reason=_STOP_REASON_TO_PROTO[seq.stop_reason],
                tokens=np.asarray(seq.tokens, dtype=np.int32).tobytes(),
                logprobs=np.asarray(seq.logprobs, dtype=np.float32).tobytes(),
            )
        )

    if output.prompt_logprobs is not None:
        proto.prompt_logprobs = np.array(
            [np.nan if lp is None else lp for lp in output.prompt_logprobs], dtype=np.float32
        ).tobytes()

    if output.topk_prompt_logprobs is not None:
        rows = output.topk_prompt_logprobs
        # k is not recorded in the result, so recover it from the widest row.
        # With every row undefined, use k=1 so prompt_length stays encoded
        # (the client maps fully-masked rows back to None).
        k = max((len(row) for row in rows if row), default=1)
        token_ids = np.full((len(rows), k), _TOPK_MASK_TOKEN_ID, dtype=np.int32)
        logprobs = np.full((len(rows), k), _TOPK_MASK_LOGPROB, dtype=np.float32)
        for i, row in enumerate(rows):
            for j, (token_id, logprob) in enumerate(row or ()):
                token_ids[i, j] = token_id
                logprobs[i, j] = logprob
        proto.topk_prompt_logprobs.CopyFrom(
            pb.TopkPromptLogprobs(
                prompt_length=len(rows),
                k=k,
                token_ids=token_ids.tobytes(),
                logprobs=logprobs.tobytes(),
            )
        )

    return proto.SerializeToString()


def _serialize_forward_backward_output(result_data: dict) -> bytes:
    output = types.ForwardBackwardOutput.model_validate(result_data)
    proto = pb.ForwardBackwardOutput()
    proto.loss_fn_output_type = output.loss_fn_output_type
    for name, value in output.metrics.items():
        proto.metrics[name] = float(value)

    if not output.loss_fn_outputs:
        return proto.SerializeToString()

    record = proto.loss_fn_outputs.add()
    record.num_datums = len(output.loss_fn_outputs)

    # Union of field names across datums, in first-seen order. Backends emit
    # the same fields for every datum in a request, but a missing field still
    # encodes cleanly as an empty slice (equal adjacent offsets).
    field_names = list(dict.fromkeys(name for datum in output.loss_fn_outputs for name in datum))
    for name in field_names:
        tensors = [datum.get(name) for datum in output.loss_fn_outputs]
        dtypes = {t["dtype"] for t in tensors if t is not None}
        if len(dtypes) > 1:
            raise ValueError(f"Field {name} has mixed dtypes across datums: {dtypes}")
        dtype = dtypes.pop() if dtypes else "float32"

        # The SDK slices one BatchedTensor with a single trailing shape, so
        # datums may only be ragged in the leading dimension.
        trailing_shapes = {tuple(t["shape"][1:]) for t in tensors if t is not None}
        if len(trailing_shapes) > 1:
            raise ValueError(f"Field {name} has mixed trailing shapes across datums: {trailing_shapes}")
        trailing_shape = trailing_shapes.pop() if trailing_shapes else ()

        np_dtype = _TENSOR_DTYPE_TO_NUMPY[dtype]
        arrays = [np.asarray(t["data"] if t is not None else [], dtype=np_dtype).ravel() for t in tensors]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([a.nbytes for a in arrays], out=offsets[1:])

        record.fields[name].CopyFrom(
            pb.BatchedTensor(
                data=b"".join(a.tobytes() for a in arrays),
                offsets=offsets.tobytes(),
                dtype=_TENSOR_DTYPE_TO_PROTO[dtype],
                trailing_shape=trailing_shape,
            )
        )

    return proto.SerializeToString()
