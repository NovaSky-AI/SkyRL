import json

import pytest
from pydantic import ValidationError

from skyrl.tinker import profiling_receipts as receipts

SHA = "a" * 64


def receipt_values(operation="optimizer", generation=2, optimizer_step=1, profiler_mode="none"):
    return {
        "run_id": "run-test",
        "operation_id": f"run-test/{operation}",
        "operation": operation,
        "model": {
            "name": "Qwen/Qwen3-0.6B",
            "revision": "model-rev",
            "config_sha256": SHA,
        },
        "provenance": {
            "repository_url": "https://github.com/NovaSky-AI/SkyRL",
            "commit": "b" * 40,
            "dirty": False,
            "image_uri": "registry.example.com/skyrl:test",
            "image_digest": f"sha256:{SHA}",
        },
        "arm": "baseline_a",
        "transport": {"implementation": "safetensors", "revision": "baseline-v1"},
        "profiler_mode": profiler_mode,
        "classification": receipts.classify_phase(operation, generation, optimizer_step),
        "adapter_generation": generation,
        "client_timing": {
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 150,
            "duration_ns": 50,
        },
        "server_spans": [
            {
                "span_id": "span-r0",
                "request_id": "request-test",
                "operation": operation,
                "adapter_generation": generation,
                "start_monotonic_ns": 10,
                "end_monotonic_ns": 40,
                "rank": 0,
                "node": "node-test-a",
                "owning_component": "trainer",
            }
        ],
        "unattributed_duration_ns": 20,
        "adapter_transfer": {
            "logical_bytes": None,
            "transferred_bytes": None,
            "source_dtype": None,
            "destination_dtype": None,
            "layout_digest": None,
        },
        "outcome": {
            "success": True,
            "error_type": None,
            "error": None,
            "active_generation_after": None,
            "first_sample_generation": None,
        },
        "memory": {
            "gpu_allocated_bytes": None,
            "gpu_reserved_bytes": None,
            "cgroup_current_bytes": None,
            "cgroup_peak_bytes": None,
        },
        "trace": {
            "path": "",
            "size_bytes": None,
            "sha256": None,
            "selected_rank": None,
            "window": None,
        },
    }


def test_generation_zero_warmup_and_recurring_classification_are_distinct():
    generation_zero = receipts.classify_phase("adapter_publication", 0, None)
    warmup = receipts.classify_phase("adapter_publication", 1, 0)
    recurring = receipts.classify_phase("adapter_publication", 2, 1)
    assert (generation_zero.lifecycle, generation_zero.kind) == (
        "cold",
        "generation_zero",
    )
    assert (warmup.lifecycle, warmup.kind) == ("warm", "excluded_warmup")
    assert (recurring.lifecycle, recurring.kind) == ("warm", "recurring")


def test_cold_subphase_is_not_classified_as_generation_zero_publication():
    base_load = receipts.classify_phase("base_weight_loading", 0, None)
    publication = receipts.classify_phase("adapter_publication", 0, None)
    assert base_load.kind == "cold_start"
    assert publication.kind == "generation_zero"


def test_profiler_processing_is_housekeeping_even_around_warmup():
    assert receipts.classify_phase("trace_export_flush", 1, 0).kind == "housekeeping"


def test_recurring_model_work_requires_an_optimizer_step():
    with pytest.raises(ValueError, match="optimizer_step is required"):
        receipts.classify_phase("optimizer", 1, None)


def test_monotonic_client_timing_uses_one_clock_domain():
    ticks = iter((1_000, 1_075))
    with receipts.measure_monotonic_client(lambda: next(ticks)) as measured:
        pass
    assert measured["timing"].model_dump() == {
        "start_monotonic_ns": 1_000,
        "end_monotonic_ns": 1_075,
        "duration_ns": 75,
    }


def test_client_timing_rejects_arithmetic_inconsistent_with_endpoints():
    with pytest.raises(ValidationError, match="does not match"):
        receipts.ClientTiming(start_monotonic_ns=1, end_monotonic_ns=3, duration_ns=3)


def test_distributed_spans_remain_rank_local_in_derived_table():
    values = receipt_values()
    rank_one = dict(
        values["server_spans"][0],
        span_id="span-r1",
        rank=1,
        start_monotonic_ns=12,
        end_monotonic_ns=52,
    )
    values["server_spans"].append(rank_one)
    receipt = receipts.build_phase_receipt(**values)
    table = receipts.derive_receipt_table([receipt], ["raw/receipts.jsonl"])
    attribution = table["rows"][0]["server_attribution"]
    assert attribution["aggregation"] == "not_summed"
    assert attribution["combined_duration_ns"] is None
    assert [span["duration_ns"] for span in attribution["spans"]] == [30, 40]


def test_duplicate_distributed_span_identity_is_rejected():
    values = receipt_values()
    values["server_spans"].append(dict(values["server_spans"][0], rank=1))
    with pytest.raises(ValidationError, match="span_id values must be unique"):
        receipts.build_phase_receipt(**values)


def test_server_span_generation_must_match_operation_generation():
    values = receipt_values()
    values["server_spans"][0]["adapter_generation"] = 1
    with pytest.raises(ValidationError, match="generation does not match"):
        receipts.build_phase_receipt(**values)


def test_profiler_disabled_operation_rejects_trace_artifact():
    values = receipt_values()
    values["trace"] = {
        "path": "traces/rank0_w0.pt.trace.json.gz",
        "size_bytes": 10,
        "sha256": SHA,
        "selected_rank": 0,
        "window": 0,
    }
    with pytest.raises(ValidationError, match="profiler-disabled"):
        receipts.build_phase_receipt(**values)


def test_trace_artifact_metadata_cannot_be_partial():
    values = receipt_values(profiler_mode="trainer")
    values["trace"] = {
        "path": "trace.gz",
        "size_bytes": None,
        "sha256": None,
        "selected_rank": 0,
        "window": 0,
    }
    with pytest.raises(ValidationError, match="must be complete"):
        receipts.build_phase_receipt(**values)


@pytest.mark.parametrize(
    "operation,generation,field",
    [
        ("adapter_publication", 2, "active_generation_after"),
        ("sample", 2, "first_sample_generation"),
    ],
)
def test_generation_visible_operations_require_the_observed_generation(operation, generation, field):
    values = receipt_values(operation=operation, generation=generation)
    values["outcome"][field] = generation
    assert receipts.build_phase_receipt(**values).outcome.success
    values["outcome"][field] = generation - 1
    with pytest.raises(ValidationError, match="generation"):
        receipts.build_phase_receipt(**values)


def test_failed_operation_preserves_typed_primary_error():
    values = receipt_values()
    values["outcome"] = {
        "success": False,
        "error_type": "RuntimeError",
        "error": "optimizer failed",
        "active_generation_after": 1,
        "first_sample_generation": None,
    }
    assert receipts.build_phase_receipt(**values).outcome.error == "optimizer failed"


def test_receipt_schema_is_versioned_and_requires_all_contract_sections():
    schema = receipts.receipt_json_schema()
    assert schema["properties"]["schema_version"]["const"] == receipts.SCHEMA_VERSION
    assert {
        "model",
        "provenance",
        "transport",
        "client_timing",
        "server_spans",
        "adapter_transfer",
        "outcome",
        "memory",
        "trace",
    } <= set(schema["required"])


def test_receipts_and_manifest_are_create_only_and_checksummed(tmp_path):
    raw = tmp_path / "raw" / "receipts.jsonl"
    receipt = receipts.build_phase_receipt(**receipt_values())
    receipts.write_receipts(raw, [receipt])
    with pytest.raises(FileExistsError):
        receipts.write_receipts(raw, [receipt])
    manifest = tmp_path / "SHA256SUMS"
    receipts.write_checksum_manifest(manifest, [raw], tmp_path)
    receipts.verify_checksum_manifest(manifest, tmp_path)
    raw.write_text(raw.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        receipts.verify_checksum_manifest(manifest, tmp_path)


def test_derived_table_names_inputs_and_is_reproducible():
    receipt = receipts.build_phase_receipt(**receipt_values())
    first = receipts.derive_receipt_table([receipt], ["raw/receipts.jsonl"])
    second = receipts.derive_receipt_table([receipt.model_dump()], ["raw/receipts.jsonl"])
    assert first["inputs"] == ["raw/receipts.jsonl"]
    assert first["derivation_version"] == receipts.DERIVATION_VERSION
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def receipt_metadata():
    values = receipt_values()
    return receipts.ReceiptMetadata(
        run_id=values["run_id"],
        model=values["model"],
        provenance=values["provenance"],
        arm=values["arm"],
        transport=values["transport"],
        profiler_mode=values["profiler_mode"],
    )


def test_recorder_writes_ordered_validated_operation_receipts(tmp_path):
    ticks = iter((100, 140, 200, 260))
    path = tmp_path / "receipts.jsonl"
    with receipts.ReceiptRecorder(path, receipt_metadata(), lambda: next(ticks)) as recorder:
        with recorder.record("optimizer", 1, 0):
            pass
        with recorder.record("adapter_publication", 2, 1):
            pass
    rows = [receipts.PhaseReceipt.model_validate_json(line) for line in path.read_text().splitlines()]
    assert [row.operation for row in rows] == ["optimizer", "adapter_publication"]
    assert [row.operation_id for row in rows] == [
        "run-test/0000-optimizer",
        "run-test/0001-adapter_publication",
    ]
    assert rows[0].classification.kind == "excluded_warmup"
    assert rows[1].outcome.active_generation_after == 2
    assert rows[0].unattributed_duration_ns == rows[0].client_timing.duration_ns == 40


def test_recorder_with_partial_server_span_retains_full_unattributed_client_time(tmp_path):
    ticks = iter((100, 150))
    path = tmp_path / "receipts.jsonl"
    with receipts.ReceiptRecorder(path, receipt_metadata(), lambda: next(ticks)) as recorder:
        with recorder.record("optimizer", 2, 1) as observations:
            observations["server_spans"] = receipt_values()["server_spans"]
    row = receipts.PhaseReceipt.model_validate_json(path.read_text())
    assert len(row.server_spans) == 1
    assert row.unattributed_duration_ns == row.client_timing.duration_ns == 50


def test_recorder_retains_operation_error_when_receipt_cleanup_also_fails(tmp_path):
    ticks = iter((100, 90))
    failure = RuntimeError("optimizer failed")
    with receipts.ReceiptRecorder(tmp_path / "receipts.jsonl", receipt_metadata(), lambda: next(ticks)) as recorder:
        with pytest.raises(RuntimeError) as caught:
            with recorder.record("optimizer", 2, 1):
                raise failure
    assert caught.value is failure
    assert "Profiling receipt failed" in failure.__notes__[0]


def test_recorder_output_is_create_only(tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_text("preserved\n")
    with pytest.raises(FileExistsError):
        receipts.ReceiptRecorder(path, receipt_metadata())
    assert path.read_text() == "preserved\n"


def test_receipt_metadata_load_rejects_unknown_fields(tmp_path):
    path = tmp_path / "metadata.json"
    data = receipt_metadata().model_dump(mode="json")
    data["unexpected"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        receipts.load_receipt_metadata(path)
