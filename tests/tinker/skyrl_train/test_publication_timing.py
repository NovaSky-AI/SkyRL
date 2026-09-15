"""Client publication includes cold inference startup; server timers distinguish it."""

import tarfile
from types import SimpleNamespace

from skyrl.backends import utils
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend
from skyrl.tinker import server_timing


def test_first_publication_separates_inference_initialization_from_replacement(tmp_path, monkeypatch):
    clock = [0.0]
    logs = []
    calls = []
    monkeypatch.setattr(utils.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(server_timing.time, "monotonic_ns", lambda: int(clock[0] * 1_000_000_000))
    monkeypatch.setattr(
        server_timing,
        "emit_server_stage",
        lambda record: logs.append(server_timing.SERVER_STAGE_PREFIX + record.model_dump_json()),
    )
    monkeypatch.setattr(utils.logger, "info", logs.append)

    def ensure_inference():
        if not backend._inference_engines_initialized:
            calls.append("initialize")
            clock[0] += 3
            backend._inference_engines_initialized = True

    async def publish(model_id):
        calls.append("publish")
        clock[0] += 5

    backend = SimpleNamespace(
        _validate_model_state=lambda model_id: None,
        _get_role=lambda model_id: "policy",
        _inference_engines_initialized=False,
        _ensure_inference_engines=ensure_inference,
        _sleep_inference_engines=lambda: None,
        _base_lora_signature=None,
        _dispatch=SimpleNamespace(save_weights_for_sampler=publish),
    )
    for step in range(2):
        path = tmp_path / f"step-{step}.tar"
        request_data = {"sampling_session_seq_id": step, "seq_id": step}
        with server_timing.server_request_context(str(step + 1), "model-test", request_data):
            SkyRLTrainBackend.save_sampler_checkpoint(backend, path, "model-test", persist=False)
        with tarfile.open(path) as archive:
            assert archive.getnames() == ["MARKER"]
    assert calls == ["initialize", "publish", "publish"]
    initialization = [line for line in logs if "(timing) sampler_inference_init" in line]
    assert "cold=True" in initialization[0] and "3.000s" in initialization[0]
    assert "cold=False" in initialization[1] and "0.000s" in initialization[1]
    publication = [line for line in logs if "(timing) sampler_weight_sync" in line]
    assert len(publication) == 2 and all("5.000s" in line for line in publication)
    stages = server_timing.collect_server_stage_records(logs)
    assert [(stage.stage, stage.cold) for stage in stages] == [
        ("inference_engine_initialization_aggregate", True),
        ("sampler_weight_sync", True),
        ("sampler_weight_sync", False),
    ]
    assert [stage.request_id for stage in stages] == ["1", "1", "2"]
    assert [stage.adapter_generation for stage in stages] == [0, 0, 1]
    assert stages[0].publication_id == "model-test/sampling-session-0/sequence-0"
    assert [stage.duration_ns for stage in stages] == [
        3_000_000_000,
        5_000_000_000,
        5_000_000_000,
    ]
    assert backend._engines_sleep_level is None
