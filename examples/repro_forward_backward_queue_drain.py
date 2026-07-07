"""Reproduce the forward_backward queue shape that can create oversized batches.

This script is intentionally synthetic: it does not allocate tensors or require a
GPU. It models a client submitting many pending FORWARD_BACKWARD requests with
uneven sequence lengths, then compares the single-drain behavior with the
request-count limited behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUEST_COUNT = 47
EXAMPLE_COUNT = 63
MAX_SEQUENCE_LENGTH = 147_472
TOTAL_INPUT_TOKENS = 5_153_654
OBSERVED_CUDA_ALLOCATION_GIB = 40.02
FORWARD_BACKWARD_MAX_REQUEST_COUNT = 4


@dataclass(frozen=True)
class ForwardBackwardRequest:
    request_id: int
    sequence_lengths: list[int]


def build_repro_requests() -> list[ForwardBackwardRequest]:
    other_token_budget = TOTAL_INPUT_TOKENS - MAX_SEQUENCE_LENGTH
    other_lengths = [other_token_budget // (EXAMPLE_COUNT - 1)] * (EXAMPLE_COUNT - 1)
    for index in range(other_token_budget % (EXAMPLE_COUNT - 1)):
        other_lengths[index] += 1

    example_lengths = [MAX_SEQUENCE_LENGTH, *other_lengths]
    requests = []
    cursor = 0
    for request_id in range(1, REQUEST_COUNT + 1):
        examples_in_request = 2 if request_id <= EXAMPLE_COUNT - REQUEST_COUNT else 1
        requests.append(
            ForwardBackwardRequest(
                request_id=request_id,
                sequence_lengths=example_lengths[cursor : cursor + examples_in_request],
            )
        )
        cursor += examples_in_request
    return requests


def chunk_by_request_count(
    requests: list[ForwardBackwardRequest], max_request_count: int
) -> list[list[ForwardBackwardRequest]]:
    return [requests[start : start + max_request_count] for start in range(0, len(requests), max_request_count)]


def summarize_batch(batch: list[ForwardBackwardRequest]) -> tuple[int, int, int, int]:
    lengths = [length for request in batch for length in request.sequence_lengths]
    request_count = len(batch)
    example_count = len(lengths)
    max_sequence_length = max(lengths)
    input_tokens = sum(lengths)
    return request_count, example_count, max_sequence_length, input_tokens


def print_batch(label: str, batch: list[ForwardBackwardRequest]) -> None:
    request_count, example_count, max_sequence_length, input_tokens = summarize_batch(batch)
    print(
        f"{label}: requests={request_count}, examples={example_count}, "
        f"max_sequence_length={max_sequence_length:,}, input_tokens={input_tokens:,}"
    )


def main() -> None:
    requests = build_repro_requests()

    print("Client submits pending FORWARD_BACKWARD requests with uneven sequence lengths.")
    print_batch("single queue drain before limiting", requests)
    print(f"observed failure symptom: CUDA OOM while trying to allocate {OBSERVED_CUDA_ALLOCATION_GIB:.2f} GiB")
    print()

    chunks = chunk_by_request_count(requests, FORWARD_BACKWARD_MAX_REQUEST_COUNT)
    print(
        "with forward_backward_max_request_count=" f"{FORWARD_BACKWARD_MAX_REQUEST_COUNT}: {len(chunks)} backend calls"
    )
    for index, chunk in enumerate(chunks, start=1):
        print_batch(f"chunk {index:02d}", chunk)


if __name__ == "__main__":
    main()
