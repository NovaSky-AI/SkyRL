"""Reproduce a forward_backward queue-drain pressure shape against a real service.

The useful path is `run-forward-backward`: pass a SkyRL/Tinker-compatible service
URL, create a real LoRA training client, submit many pending FORWARD_BACKWARD
futures, and wait for actual service results. `summarize` is only a cheap preview
of the request shape and expected backend batching pressure.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

LEADING_BLOCKER_SEQUENCE_LENGTH = 10_000
LONG_SEQUENCE_LENGTH = 35_000
SHORT_SEQUENCE_LENGTHS = [
    3_693,
    5_363,
    4_319,
    3_097,
    3_940,
    3_494,
    3_659,
    3_967,
    5_271,
    3_861,
    4_449,
    3_660,
    4_075,
    4_969,
    5_111,
    4_011,
    3_366,
    2_864,
    4_951,
    5_161,
    2_618,
    5_310,
    4_591,
    2_596,
    4_295,
    4_314,
    3_235,
    3_567,
    4_180,
    4_347,
    4_241,
    2_562,
    3_892,
    4_508,
    3_312,
    3_869,
    3_570,
    3_969,
    5_494,
    3_178,
    4_033,
    4_231,
    4_976,
    2_911,
    5_368,
    2_847,
    4_849,
    3_707,
    4_227,
    5_495,
    4_033,
    3_027,
    4_108,
    5_021,
    3_600,
    5_046,
    4_874,
    3_092,
    5_011,
    4_944,
    3_757,
    3_206,
    5_355,
    5_432,
    4_020,
    2_667,
    4_975,
    5_493,
    4_771,
    5_147,
    3_171,
    3_852,
    3_292,
    5_440,
    2_616,
    3_101,
    3_961,
    5_411,
    5_028,
    3_500,
    3_371,
    2_771,
    3_927,
    3_575,
    3_441,
    4_063,
    3_472,
    3_767,
    4_100,
    4_450,
    4_451,
    3_054,
    4_771,
    4_026,
    3_779,
    3_424,
]
JITTERED_SHORT_SEQUENCE_LENGTHS = [
    length + ((index % 7) - 3) * 137
    for index, length in enumerate(SHORT_SEQUENCE_LENGTHS)
]
PENDING_FORWARD_BACKWARD_SEQUENCE_LENGTHS = [
    *SHORT_SEQUENCE_LENGTHS[:8],
    LONG_SEQUENCE_LENGTH,
    *SHORT_SEQUENCE_LENGTHS[8:],
    *JITTERED_SHORT_SEQUENCE_LENGTHS,
]
FORWARD_BACKWARD_MAX_REQUEST_COUNT = 1
UNBOUNDED_EXPECTED_PADDED_ROWS = len(PENDING_FORWARD_BACKWARD_SEQUENCE_LENGTHS)
UNBOUNDED_EXPECTED_MAX_PACKED_SEQUENCE_LENGTH = LONG_SEQUENCE_LENGTH
UNBOUNDED_EXPECTED_PADDED_SEQUENCE_SLOTS = (
    UNBOUNDED_EXPECTED_PADDED_ROWS * UNBOUNDED_EXPECTED_MAX_PACKED_SEQUENCE_LENGTH
)

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
        f"sequence_length={LEADING_BLOCKER_SEQUENCE_LENGTH:,}."
    )
    print_batch("single queue drain before limiting", requests)
    print(
        "pressure symptom: "
        "process_batch_requests(forward_backward) can coalesce the pending requests "
        "into one large train call"
    )
    print(
        "padding note: sample microbatching can pad all rows in the coalesced "
        "batch to the longest sequence before backend microbatching"
    )
    print(
        "expected unbounded padded batch shape: "
        f"rows={UNBOUNDED_EXPECTED_PADDED_ROWS}, "
        f"sequence_slots={UNBOUNDED_EXPECTED_PADDED_SEQUENCE_SLOTS:,}"
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

    leading_future = None
    if not args.skip_leading_blocker:
        print(
            "submitting leading blocker request: "
            f"sequence_length={LEADING_BLOCKER_SEQUENCE_LENGTH:,}"
        )
        leading_datum = _make_datum(LEADING_BLOCKER_SEQUENCE_LENGTH, args.token_id)
        leading_future = training_client.forward_backward(
            [leading_datum], "cross_entropy"
        )

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

    if leading_future is not None:
        print("awaiting leading blocker result")
        try:
            leading_result = _result_with_timeout(leading_future, args.future_timeout_s)
        except Exception as exc:
            failures += 1
            print(f"leading blocker failed: {type(exc).__name__}: {exc}")
        else:
            print(f"leading_blocker metrics={leading_result.metrics}")

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
    parser.add_argument("--skip-leading-blocker", action="store_true")


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
