import pytest

from skyrl.tinker import server_timing


def stage_values(**updates):
    values = {
        "span_id": "request-1/sampler_weight_sync/10",
        "request_id": "request-1",
        "publication_id": "model-test/sampling-session-2/sequence-3",
        "model_id": "model-test",
        "adapter_generation": 2,
        "stage": "sampler_weight_sync",
        "cold": False,
        "owning_component": "tinker_engine_controller",
        "rank": 0,
        "node": "node-a",
        "start_monotonic_ns": 10,
        "end_monotonic_ns": 40,
        "duration_ns": 30,
        "success": True,
    }
    values.update(updates)
    return values


def test_server_stage_schema_is_versioned():
    schema = server_timing.server_stage_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "1.1.0"
    assert {"request_id", "publication_id", "adapter_generation", "rank", "node"} <= set(schema["required"])


def test_structured_stage_failure_preserves_the_operation_error(monkeypatch):
    logs = []
    ticks = iter((100, 160))
    monkeypatch.setattr(
        server_timing,
        "emit_server_stage",
        lambda record: logs.append(server_timing.SERVER_STAGE_PREFIX + record.model_dump_json()),
    )
    failure = RuntimeError("weight sync failed")

    with server_timing.server_request_context("request-7", "model-test", {"sampling_session_seq_id": 4, "seq_id": 5}):
        with pytest.raises(RuntimeError) as caught:
            with server_timing.record_server_stage(
                "sampler_weight_sync",
                "model-test",
                cold=False,
                clock=lambda: next(ticks),
                node="node-a",
            ):
                raise failure

    assert caught.value is failure
    records = server_timing.collect_server_stage_records(chr(10).join(logs))
    assert len(records) == 1
    assert records[0].publication_id == "model-test/sampling-session-4/sequence-5"
    assert records[0].adapter_generation == 4
    assert records[0].success is False
    assert records[0].error_type == "RuntimeError"
    assert records[0].error == "weight sync failed"


def test_server_stage_derivation_keeps_records_and_uses_rank_max():
    rank_zero = server_timing.ServerStageRecord.model_validate(stage_values())
    rank_one = server_timing.ServerStageRecord.model_validate(
        stage_values(
            span_id="request-1/sampler_weight_sync/12",
            rank=1,
            node="node-b",
            start_monotonic_ns=12,
            end_monotonic_ns=52,
            duration_ns=40,
        )
    )

    table = server_timing.derive_server_stage_table([rank_zero, rank_one])

    assert len(table["records"]) == 2
    assert table["summaries"] == [
        {
            "publication_id": "model-test/sampling-session-2/sequence-3",
            "stage": "sampler_weight_sync",
            "cold": False,
            "aggregation": "rank_max_not_sum",
            "rank_max_duration_ns": 40,
            "rank_durations_ns": {"0": [30], "1": [40]},
        }
    ]
    assert "base_weight_loading" in table["unavailable_stages"]
    assert "adapter_activation" in table["unavailable_stages"]
    assert not any(record["stage"] == "adapter_activation" for record in table["records"])


def test_collector_rejects_duplicate_span_identity():
    row = server_timing.ServerStageRecord.model_validate(stage_values()).model_dump_json()
    lines = [server_timing.SERVER_STAGE_PREFIX + row] * 2
    with pytest.raises(ValueError, match="span_id values must be unique"):
        server_timing.collect_server_stage_records(lines)


def test_timing_emission_failure_does_not_replace_stage_failure(monkeypatch):
    failure = RuntimeError("weight sync failed")
    monkeypatch.setattr(
        server_timing,
        "emit_server_stage",
        lambda _: (_ for _ in ()).throw(OSError("log sink failed")),
    )
    with server_timing.server_request_context("request-8", "model-test", {}):
        with pytest.raises(RuntimeError) as caught:
            with server_timing.record_server_stage(
                "sampler_weight_sync",
                "model-test",
                cold=False,
                clock=iter((1, 2)).__next__,
            ):
                raise failure
    assert caught.value is failure
    assert failure.__notes__ == ["Server timing emission failed: log sink failed"]
