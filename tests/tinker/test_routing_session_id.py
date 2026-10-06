from skyrl.tinker.types import SEQ_ID_BLOCK, make_routing_session_id


def test_requests_of_one_unpickled_client_share_a_routing_key():
    first, later = 7 * SEQ_ID_BLOCK, 7 * SEQ_ID_BLOCK + 41
    assert make_routing_session_id("session", first) == make_routing_session_id("session", later)
    assert make_routing_session_id("session", first) != make_routing_session_id("session", 8 * SEQ_ID_BLOCK)


def test_requests_of_an_original_client_keep_per_request_keys():
    assert make_routing_session_id("session", 3) != make_routing_session_id("session", 4)
    assert make_routing_session_id(None, 3) is None
