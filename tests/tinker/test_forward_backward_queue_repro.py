from examples.repro_forward_backward_queue_drain import (
    FORWARD_BACKWARD_MAX_REQUEST_COUNT,
    TEXT_HIDDEN_SIZE,
    UNBOUNDED_EXPECTED_MAX_PACKED_SEQUENCE_LENGTH,
    UNBOUNDED_EXPECTED_PADDED_ROWS,
    UNBOUNDED_EXPECTED_PADDED_SEQUENCE_SLOTS,
    VISION_NUM_POSITION_EMBEDDINGS,
    build_repro_requests,
    chunk_by_request_count,
    summarize_batch,
    summarize_repro,
)


def test_repro_uses_many_pending_forward_backward_requests_with_uneven_lengths():
    requests = build_repro_requests()

    request_count, example_count, max_sequence_length, input_tokens = summarize_batch(
        requests
    )

    assert request_count == 193
    assert example_count == 193
    assert max_sequence_length == 35_000
    assert input_tokens == 816_247


def test_repro_chunks_each_pending_forward_backward_request_individually():
    requests = build_repro_requests()

    chunks = chunk_by_request_count(requests, FORWARD_BACKWARD_MAX_REQUEST_COUNT)

    assert len(chunks) == 193
    assert all(len(chunk) == 1 for chunk in chunks)
    assert [chunk[0].request_id for chunk in chunks] == list(range(1, 194))


def test_repro_documents_text_width_instead_of_vision_position_count():
    assert TEXT_HIDDEN_SIZE == 5_120
    assert VISION_NUM_POSITION_EMBEDDINGS == 2_304


def test_repro_describes_coalesced_sample_padding_pressure(capsys):
    summarize_repro()

    output = capsys.readouterr().out

    assert "one large train call" in output
    assert "pad all rows in the coalesced batch" in output


def test_repro_documents_worst_unbounded_microbatch_shape():
    assert UNBOUNDED_EXPECTED_PADDED_ROWS == 193
    assert UNBOUNDED_EXPECTED_MAX_PACKED_SEQUENCE_LENGTH == 35_000
    assert UNBOUNDED_EXPECTED_PADDED_SEQUENCE_SLOTS == 6_755_000
