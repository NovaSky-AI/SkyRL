"""Client publication includes cold inference startup; server timers distinguish it."""

import tarfile
from types import SimpleNamespace

from skyrl.backends import utils
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend


def test_first_publication_separates_inference_initialization_from_replacement(tmp_path, monkeypatch):
    clock = [0.0]
    logs = []
    calls = []
    monkeypatch.setattr(utils.time, "perf_counter", lambda: clock[0])
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
        SkyRLTrainBackend.save_sampler_checkpoint(backend, path, "model-test", persist=False)
        with tarfile.open(path) as archive:
            assert archive.getnames() == ["MARKER"]
    assert calls == ["initialize", "publish", "publish"]
    initialization = [line for line in logs if "(timing) sampler_inference_init" in line]
    assert "cold=True" in initialization[0] and "3.000s" in initialization[0]
    assert "cold=False" in initialization[1] and "0.000s" in initialization[1]
    publication = [line for line in logs if "(timing) sampler_weight_sync" in line]
    assert len(publication) == 2 and all("5.000s" in line for line in publication)
    assert backend._engines_sleep_level is None
