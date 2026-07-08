"""Reproduce a forward_backward queue-drain OOM shape against a real service.

The useful path is `run-forward-backward`: pass a SkyRL/Tinker-compatible service
URL, create a real LoRA training client, submit many pending FORWARD_BACKWARD
futures, and wait for actual service results. `summarize` is only a cheap preview
of the request shape.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

FIRST_REQUEST_SEQUENCE_LENGTH = 67_940
PENDING_FORWARD_BACKWARD_SEQUENCE_LENGTHS = [
    62_447,
    63_391,
    71_081,
    77_336,
    74_195,
    78_711,
    75_702,
    136_037,
    111_936,
    113_038,
    138_748,
    56_355,
    44_316,
    44_056,
    62_013,
    65_291,
    70_238,
    64_862,
    65_780,
    63_765,
    63_505,
    63_961,
    56_661,
    111_829,
    102_350,
    151_421,
    155_322,
    155_379,
    164_724,
    93_120,
    88_181,
    85_304,
    87_773,
    123_228,
    117_805,
    110_221,
    93_816,
    69_542,
    67_955,
    57_070,
    68_347,
    103_065,
    97_487,
]
PACKED_BATCH_CUDA_ALLOCATION_GIB = 38.10
FORWARD_BACKWARD_MAX_REQUEST_COUNT = 1

TEXT_HIDDEN_SIZE = 5_120
VISION_NUM_POSITION_EMBEDDINGS = 2_304

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


@dataclass(frozen=True)
class ForwardBackwardRequest:
    request_id: int
    sequence_lengths: list[int]


def build_repro_requests() -> list[ForwardBackwardRequest]:
    return [
        ForwardBackwardRequest(
            request_id=request_id, sequence_lengths=[sequence_length]
        )
        for request_id, sequence_length in enumerate(
            PENDING_FORWARD_BACKWARD_SEQUENCE_LENGTHS,
            start=1,
        )
    ]


def chunk_by_request_count(
    requests: list[ForwardBackwardRequest], max_request_count: int
) -> list[list[ForwardBackwardRequest]]:
    return [
        requests[start : start + max_request_count]
        for start in range(0, len(requests), max_request_count)
    ]


def summarize_batch(batch: list[ForwardBackwardRequest]) -> tuple[int, int, int, int]:
    lengths = [length for request in batch for length in request.sequence_lengths]
    request_count = len(batch)
    example_count = len(lengths)
    max_sequence_length = max(lengths)
    input_tokens = sum(lengths)
    return request_count, example_count, max_sequence_length, input_tokens


def print_batch(label: str, batch: list[ForwardBackwardRequest]) -> None:
    request_count, example_count, max_sequence_length, input_tokens = summarize_batch(
        batch
    )
    print(
        f"{label}: requests={request_count}, examples={example_count}, "
        f"max_sequence_length={max_sequence_length:,}, prepared_input_tokens={input_tokens:,}"
    )


def summarize_repro() -> None:
    requests = build_repro_requests()

    print(
        "Client shape: many pending FORWARD_BACKWARD requests with uneven sequence lengths."
    )
    print(
        "Optional first single request: "
        f"sequence_length={FIRST_REQUEST_SEQUENCE_LENGTH:,}."
    )
    print_batch("single queue drain before limiting", requests)
    print(
        "failure symptom: "
        "process_batch_requests(forward_backward, n=43) hit CUDA OOM while trying to "
        f"allocate {PACKED_BATCH_CUDA_ALLOCATION_GIB:.2f} GiB"
    )
    print(
        "model config note: text hidden_size is "
        f"{TEXT_HIDDEN_SIZE:,}; {VISION_NUM_POSITION_EMBEDDINGS:,} is "
        "vision_config.num_position_embeddings, not a text activation width."
    )
    print()

    chunks = chunk_by_request_count(requests, FORWARD_BACKWARD_MAX_REQUEST_COUNT)
    print(
        "with forward_backward_max_request_count="
        f"{FORWARD_BACKWARD_MAX_REQUEST_COUNT}: {len(chunks)} backend calls"
    )
    for index, chunk in enumerate(chunks, start=1):
        print_batch(f"chunk {index:02d}", chunk)


def _make_datum(sequence_length: int, token_id: int):
    from tinker import types

    tokens = [token_id] * sequence_length
    target_tokens = tokens[1:] + [token_id]
    weights = [1.0] * sequence_length
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=target_tokens, dtype="int64"),
            "weights": types.TensorData(data=weights, dtype="float32"),
        },
    )


def _result_with_timeout(future, timeout_s: float):
    deadline = time.time() + timeout_s
    while True:
        try:
            return future.result(timeout=30)
        except TimeoutError:
            if time.time() >= deadline:
                raise
            print("future_still_pending=true")


def run_forward_backward(args: argparse.Namespace) -> None:
    import tinker

    service_client = tinker.ServiceClient(
        base_url=args.base_url.rstrip("/") + "/",
        api_key=args.api_key,
    )
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        train_unembed=False,
    )
    print(f"created_training_client base_model={args.base_model} rank={args.rank}")

    if args.run_first_completed_request:
        print(
            "submitting optional first request: "
            f"sequence_length={FIRST_REQUEST_SEQUENCE_LENGTH:,}"
        )
        first_datum = _make_datum(FIRST_REQUEST_SEQUENCE_LENGTH, args.token_id)
        first = training_client.forward_backward([first_datum], "cross_entropy")
        first_result = _result_with_timeout(first, args.future_timeout_s)
        print(f"first_forward_backward metrics={first_result.metrics}")

    pending_lengths = PENDING_FORWARD_BACKWARD_SEQUENCE_LENGTHS[
        : args.max_pending_requests
    ]
    print(
        "submitting pending forward_backward requests: "
        f"requests={len(pending_lengths)} request_size=1"
    )
    futures = []
    for index, sequence_length in enumerate(pending_lengths, start=1):
        datum = _make_datum(sequence_length, args.token_id)
        future = training_client.forward_backward([datum], "cross_entropy")
        futures.append((index, sequence_length, future))

    failures = 0
    for index, sequence_length, future in futures:
        print(
            f"awaiting forward_backward result {index}/{len(futures)} seq_len={sequence_length:,}"
        )
        try:
            result = _result_with_timeout(future, args.future_timeout_s)
        except Exception as exc:
            failures += 1
            print(f"forward_backward {index} failed: {type(exc).__name__}: {exc}")
            continue
        print(f"forward_backward {index} metrics={result.metrics}")

    if failures:
        raise RuntimeError(f"{failures} forward_backward requests failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summarize")

    run_parser = subparsers.add_parser("run-forward-backward")
    add_forward_backward_args(run_parser)
    run_parser.add_argument("--base-url", required=True)

    return parser.parse_args()


def add_forward_backward_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", default="tml-dummy")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--future-timeout-s", type=float, default=7200)
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=len(PENDING_FORWARD_BACKWARD_SEQUENCE_LENGTHS),
    )
    parser.add_argument("--run-first-completed-request", action="store_true")


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        summarize_repro()
        return
    if args.command == "run-forward-backward":
        run_forward_backward(args)
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
