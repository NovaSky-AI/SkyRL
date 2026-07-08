from examples.repro_forward_backward_queue_drain import (
    FORWARD_BACKWARD_MAX_REQUEST_COUNT,
    TEXT_HIDDEN_SIZE,
    VISION_NUM_POSITION_EMBEDDINGS,
    build_repro_requests,
    chunk_by_request_count,
    summarize_batch,
)


def test_repro_uses_many_pending_forward_backward_requests_with_uneven_lengths():
    requests = build_repro_requests()

    request_count, example_count, max_sequence_length, input_tokens = summarize_batch(
        requests
    )

    assert request_count == 43
    assert example_count == 43
    assert max_sequence_length == 164_724
    assert input_tokens == 3_827_364


def test_repro_chunks_each_pending_forward_backward_request_individually():
    requests = build_repro_requests()

    chunks = chunk_by_request_count(requests, FORWARD_BACKWARD_MAX_REQUEST_COUNT)

    assert len(chunks) == 43
    assert all(len(chunk) == 1 for chunk in chunks)
    assert [chunk[0].request_id for chunk in chunks] == list(range(1, 44))


def test_repro_documents_text_width_instead_of_vision_position_count():
    assert TEXT_HIDDEN_SIZE == 5_120
    assert VISION_NUM_POSITION_EMBEDDINGS == 2_304
