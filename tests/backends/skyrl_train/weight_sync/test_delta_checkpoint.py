import json
import subprocess

import numpy as np
import pytest
import ray
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
    _MANIFEST_NAME,
    _MAX_SAFE_PATH_NAME_LEN,
    CheckpointIndex,
    DeltaCheckpointPublisher,
    DeltaPublishResult,
    LocalCheckpointStore,
    _deltas_dir,
    _safe_path_name,
    _weights_dir,
)
from skyrl.backends.skyrl_train.weight_sync.delta_payload import (
    decompress_bytes,
    uint8_tensor_to_bytes,
)


def _source(tensors):
    """A minimal ``WeightSource``-shaped stream: ``create_delta_files`` only iterates."""
    return list(tensors.items())


def _write_checkpoint(path, tensors):
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    with (path / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"model_type": "qwen2"}, f)


def _load_tensor(checkpoint_dir, name):
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _read_state(receiver_dir):
    with (receiver_dir / ".skyrl_weight_sync" / "state.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _store_tensors(store):
    return dict(CheckpointIndex(store._current_checkpoint_dir()).iter_tensors())


def test_delta_checkpoint_publish_fetch_and_reload_roundtrip(tmp_path):
    base_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=torch.bfloat16).view(4, 4),
        "model.layers.0.mlp.down_proj.weight": torch.arange(8, dtype=torch.bfloat16).view(2, 4),
    }
    changed_name = "model.layers.0.self_attn.q_proj.weight"
    unchanged_name = "model.layers.0.mlp.down_proj.weight"
    updated_tensors = {
        changed_name: base_tensors[changed_name] + torch.tensor(1, dtype=torch.bfloat16),
        unchanged_name: base_tensors[unchanged_name],
    }
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    result = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(result)

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    stats = store.fetch(target_version=update_info["target_version"], sync_dir=update_info["sync_dir"])
    assert stats["apply_s"] >= 0.0
    store.validate_ready(1)

    received = _store_tensors(store)
    assert set(received) == {changed_name, unchanged_name}
    assert torch.equal(received[changed_name], updated_tensors[changed_name])
    assert torch.equal(received[unchanged_name], base_tensors[unchanged_name])
    assert torch.equal(_load_tensor(_weights_dir(receiver_dir), changed_name), updated_tensors[changed_name])
    assert _read_state(receiver_dir)["version"] == 1


def test_delta_checkpoint_payload_stores_xor_patch(tmp_path):
    base_tensors = {"a.weight": torch.arange(16, dtype=torch.bfloat16).view(4, 4)}
    updated = base_tensors["a.weight"].clone()
    updated[1, 2] = updated[1, 2] + torch.tensor(3, dtype=torch.bfloat16)
    updated[3, 0] = updated[3, 0] + torch.tensor(5, dtype=torch.bfloat16)
    base_dir = tmp_path / "base"
    sync_dir = tmp_path / "sync"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(sync_dir),
        publish_staging_dir=str(tmp_path / "staging_dir"),
        publish_num_workers=1,
    )
    results = publisher.create_delta_files(_source({"a.weight": updated}))
    publisher.publish(results)

    with (sync_dir / "delta-00000001" / _MANIFEST_NAME).open(encoding="utf-8") as f:
        manifest = json.load(f)
    record = manifest["tensors"][0]
    with safe_open(sync_dir / "delta-00000001" / record["payload_file"], framework="pt", device="cpu") as f:
        compressed = f.get_tensor(record["payload_key"])

    patch = np.frombuffer(
        decompress_bytes(uint8_tensor_to_bytes(compressed), expected_size=record["uncompressed_num_bytes"]),
        dtype=np.uint8,
    )
    base_bytes = base_tensors["a.weight"].contiguous().view(torch.uint8).numpy().reshape(-1)
    updated_bytes = updated.contiguous().view(torch.uint8).numpy().reshape(-1)

    assert np.count_nonzero(patch) > 0
    assert np.array_equal(np.bitwise_xor(base_bytes, patch), updated_bytes)


@pytest.mark.vllm
def test_delta_checkpoint_vllm_multi_thread_safetensors_iterator_roundtrip(tmp_path):

    base_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=torch.bfloat16).view(4, 4),
        "model.layers.0.mlp.down_proj.weight": torch.arange(8, dtype=torch.bfloat16).view(2, 4),
    }
    updated_tensors = {
        "model.layers.0.self_attn.q_proj.weight": base_tensors["model.layers.0.self_attn.q_proj.weight"]
        + torch.tensor(1, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": base_tensors["model.layers.0.mlp.down_proj.weight"],
    }
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    result = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(result)

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=update_info["target_version"], sync_dir=update_info["sync_dir"])

    received = dict(store.iter_tensors(load_format="vllm_multi_thread_safetensors"))
    assert set(received) == set(updated_tensors)
    for name, expected in updated_tensors.items():
        assert torch.equal(received[name], expected)


def test_delta_checkpoint_publisher_converts_to_base_checkpoint_dtype(tmp_path):
    base_tensors = {"a.weight": torch.arange(8, dtype=torch.float32).view(2, 4)}
    runtime_updated = {"a.weight": (base_tensors["a.weight"] + torch.tensor(1.0)).to(torch.bfloat16)}
    expected_checkpoint_tensor = runtime_updated["a.weight"].to(torch.float32)
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    update_info = publisher.create_delta_files(_source(runtime_updated))
    update_info = publisher.publish(update_info)

    with open(tmp_path / "sync" / "delta-00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["tensors"][0]["dtype"] == "float32"
    assert manifest["tensors"][0]["payload_key"] == "a.weight"
    assert manifest["tensors"][0]["checksum_algorithm"] == "xxh3-128"
    assert manifest["tensors"][0]["uncompressed_num_bytes"] == base_tensors["a.weight"].numel() * 4

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    assert not _weights_dir(receiver_dir).exists()
    store.fetch(target_version=1, sync_dir=update_info["sync_dir"])
    received = _store_tensors(store)["a.weight"]
    assert received.dtype == torch.float32
    assert torch.equal(received, expected_checkpoint_tensor)


def test_delta_checkpoint_non_source_rank_drains_without_publishing(tmp_path, monkeypatch):
    base_tensors = {f"model.layers.{idx}.weight": torch.full((4, 4), idx, dtype=torch.bfloat16) for idx in range(4)}
    updated_tensors = {
        name: tensor + torch.tensor(idx + 1, dtype=torch.bfloat16)
        for idx, (name, tensor) in enumerate(base_tensors.items())
    }
    stream = _source(updated_tensors)
    base_dir = tmp_path / "base"
    sync_dir = tmp_path / "sync"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(sync_dir),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )

    # Simulate a non-source rank (rank != 0): it drains the weight stream but must
    # not compute or upload any deltas.
    monkeypatch.setattr(publisher, "_current_rank", lambda: 1)
    result = publisher.create_delta_files(stream)
    assert isinstance(result, DeltaPublishResult)
    assert result.records == []
    assert result.payload_files == []
    assert publisher.snapshot == {}
    assert publisher.version == 1


def test_delta_checkpoint_replays_multiple_versions_for_late_join(tmp_path):
    base_tensors = {"a.weight": torch.arange(16, dtype=torch.bfloat16).view(4, 4)}
    v1_tensors = {"a.weight": base_tensors["a.weight"] + torch.tensor(1, dtype=torch.bfloat16)}
    v2_tensors = {"a.weight": v1_tensors["a.weight"] + torch.tensor(2, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    v1_result = publisher.create_delta_files(_source(v1_tensors))
    publisher.publish(v1_result)
    first_snapshot = publisher.snapshot["a.weight"]
    first_snapshot_id = id(first_snapshot)
    assert first_snapshot.tobytes() == v1_tensors["a.weight"].contiguous().view(torch.uint8).numpy().tobytes()

    v2_result = publisher.create_delta_files(_source(v2_tensors))
    update_info = publisher.publish(v2_result)
    assert id(publisher.snapshot["a.weight"]) == first_snapshot_id
    assert (
        publisher.snapshot["a.weight"].tobytes()
        == v2_tensors["a.weight"].contiguous().view(torch.uint8).numpy().tobytes()
    )

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=2, sync_dir=update_info["sync_dir"])
    store.validate_ready(2)
    assert torch.equal(
        _store_tensors(store)["a.weight"],
        v2_tensors["a.weight"],
    )


def test_delta_checkpoint_splits_payload_files_by_size(tmp_path):
    base_tensors = {
        "a.weight": torch.zeros(256, dtype=torch.bfloat16),
        "b.weight": torch.ones(256, dtype=torch.bfloat16),
    }
    updated_tensors = {
        "a.weight": torch.arange(256, dtype=torch.bfloat16),
        "b.weight": torch.arange(256, dtype=torch.bfloat16) + torch.tensor(3, dtype=torch.bfloat16),
    }
    base_dir = tmp_path / "base"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
        max_file_size_in_gb=1e-9,
    )
    update_info = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(update_info)

    with open(tmp_path / "sync" / "delta-00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert update_info["target_version"] == 1
    assert len(manifest["payload_files"]) == 2


def test_delta_checkpoint_skips_missing_lm_head_when_checkpoint_ties_embeddings(tmp_path):
    base_tensors = {"model.embed_tokens.weight": torch.zeros((4, 4), dtype=torch.bfloat16)}
    updated_embed = torch.ones((4, 4), dtype=torch.bfloat16)
    updated_lm_head = torch.full((4, 4), 2, dtype=torch.bfloat16)
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    update_info = publisher.create_delta_files(
        [
            ("model.embed_tokens.weight", updated_embed),
            ("lm_head.weight", updated_lm_head),
        ]
    )
    update_info = publisher.publish(update_info)
    with open(tmp_path / "sync" / "delta-00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert [record["name"] for record in manifest["tensors"]] == ["model.embed_tokens.weight"]

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=1, uri=update_info["uri"])

    received = _store_tensors(store)
    assert set(received) == {"model.embed_tokens.weight"}
    assert torch.equal(received["model.embed_tokens.weight"], updated_embed)


def test_local_checkpoint_store_fetch_is_single_writer_with_concurrent_ray_actors(tmp_path):
    base_tensors = {"a.weight": torch.arange(16, dtype=torch.bfloat16).view(4, 4)}
    updated_tensors = {"a.weight": base_tensors["a.weight"] + torch.tensor(1, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    sync_dir = tmp_path / "sync"
    counter_path = tmp_path / "fetch_count.json"
    _write_checkpoint(base_dir, base_tensors)
    counter_path.write_text(json.dumps({"count": 0}), encoding="utf-8")

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir), sync_dir=str(sync_dir), publish_staging_dir=str(tmp_path / "staging_dir")
    )
    update_info = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(update_info)

    class FetchActor:
        def fetch(self, base_model_path, local_checkpoint_dir, target_version, uri, counter_file):
            import json
            import time
            from pathlib import Path

            import skyrl.backends.skyrl_train.weight_sync.delta_checkpoint as delta_checkpoint

            original = delta_checkpoint.fetch_delta_directory
            counter = Path(counter_file)

            def counted_fetch(delta_uri, cache_dir, cloud_download_workers=4):
                with delta_checkpoint.FileLock(Path(f"{counter_file}.lock")):
                    data = json.loads(counter.read_text(encoding="utf-8"))
                    data["count"] += 1
                    counter.write_text(json.dumps(data), encoding="utf-8")
                time.sleep(0.2)
                return original(delta_uri, cache_dir, cloud_download_workers=cloud_download_workers)

            delta_checkpoint.fetch_delta_directory = counted_fetch
            store = delta_checkpoint.LocalCheckpointStore(
                base_model_path=base_model_path,
                local_checkpoint_dir=local_checkpoint_dir,
            )
            stats = store.fetch(target_version=target_version, uri=uri)
            with (Path(local_checkpoint_dir) / ".skyrl_weight_sync" / "state.json").open("r", encoding="utf-8") as f:
                state = json.load(f)
            return {"stats": stats, "state": state}

    # Ray is initialized by the session-scoped autouse ``ray_init`` fixture in
    # tests/backends/skyrl_train/conftest.py; tests must not init/shutdown Ray.
    actors = []
    try:
        actor_cls = ray.remote(num_cpus=1)(FetchActor)
        actors = [actor_cls.remote(), actor_cls.remote()]
        results = ray.get(
            [
                actor.fetch.remote(
                    str(base_dir),
                    str(receiver_dir),
                    update_info["target_version"],
                    update_info["uri"],
                    str(counter_path),
                )
                for actor in actors
            ]
        )

        assert json.loads(counter_path.read_text(encoding="utf-8"))["count"] == 1
        assert all(result["state"]["version"] == 1 for result in results)
        assert (_deltas_dir(receiver_dir) / _safe_path_name(update_info["uri"])).exists()
        received = _store_tensors(
            LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
        )
        assert torch.equal(received["a.weight"], updated_tensors["a.weight"])
    finally:
        for actor in actors:
            ray.kill(actor)


def test_delta_checkpoint_unchanged_publish_advances_version(tmp_path):
    base_tensors = {"a.weight": torch.ones(8, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    update_info = publisher.create_delta_files(_source({"a.weight": base_tensors["a.weight"].clone()}))
    update_info = publisher.publish(update_info)

    # An unchanged publish is not skipped: it still advances the version and
    # writes an (empty) delta.
    assert update_info["target_version"] == 1
    assert publisher.version == 1

    with open(tmp_path / "sync" / "delta-00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["tensors"] == []
    assert manifest["payload_files"] == []

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    assert not _weights_dir(receiver_dir).exists()
    store.fetch(target_version=1, uri=update_info["uri"])
    assert _read_state(receiver_dir)["version"] == 1
    assert not _weights_dir(receiver_dir).exists()
    assert torch.equal(
        _store_tensors(store)["a.weight"],
        base_tensors["a.weight"],
    )


def test_safe_path_name_disambiguates_long_sibling_uris():
    # Sibling delta URIs differ only in their final component, which is exactly what a
    # length cap truncates away. Long sync_dirs must not collapse every version onto one cache directory name.
    prefix = "s3://bucket-with-a-long-name/" + "org_xc6lv84h3d7m9dljcc17esfw2i/" * 4 + "delta_weight_sync/run-name"
    v1 = _safe_path_name(f"{prefix}/delta-00000001")
    v2 = _safe_path_name(f"{prefix}/delta-00000002")

    assert v1 != v2
    assert len(v1) <= _MAX_SAFE_PATH_NAME_LEN and len(v2) <= _MAX_SAFE_PATH_NAME_LEN


def test_delta_checkpoint_gcs_cli_publish_fetch_roundtrip(monkeypatch, tmp_path):
    objects = {}

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "gcloud" else None

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        if cmd and cmd[0] == "cp":
            with open(cmd[-2], "rb") as src, open(cmd[-1], "wb") as dst:
                dst.write(src.read())
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        assert cmd[:3] == ["gcloud", "storage", "cp"]
        src, dst = cmd[3], cmd[4]
        if dst.startswith("gs://"):
            objects[dst] = open(src, "rb").read()
        elif src.startswith("gs://"):
            with open(dst, "wb") as f:
                f.write(objects[src])
        else:
            raise AssertionError(f"unexpected command {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.delta_checkpoint.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.delta_checkpoint.subprocess.run",
        fake_run,
    )

    base_tensors = {"a.weight": torch.zeros(8, dtype=torch.bfloat16)}
    updated_tensors = {"a.weight": torch.arange(8, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    staging_dir = tmp_path / "publish-stage"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir="gs://bucket/sync",
        publish_staging_dir=str(staging_dir),
    )
    update_info = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(update_info)

    assert update_info["uri"] == "gs://bucket/sync/delta-00000001"
    assert "gs://bucket/sync/delta-00000001/manifest.json" in objects
    assert any(key.endswith(".safetensors") for key in objects)
    assert not list(staging_dir.rglob("*.tmp"))
    assert not list(staging_dir.rglob("*.safetensors"))

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=1, uri=update_info["uri"])
    received = _store_tensors(store)
    assert torch.equal(received["a.weight"], updated_tensors["a.weight"])


def test_delta_checkpoint_s3_cli_publish_fetch_roundtrip(monkeypatch, tmp_path):
    objects = {}

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "s5cmd" else None

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        if cmd and cmd[0] == "cp":
            with open(cmd[-2], "rb") as src, open(cmd[-1], "wb") as dst:
                dst.write(src.read())
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        # s5cmd takes `cp <src> <dst>` with no subcommand group, unlike `gcloud storage cp`.
        assert cmd[:2] == ["s5cmd", "cp"]
        src, dst = cmd[2], cmd[3]
        if dst.startswith("s3://"):
            objects[dst] = open(src, "rb").read()
        elif src.startswith("s3://"):
            with open(dst, "wb") as f:
                f.write(objects[src])
        else:
            raise AssertionError(f"unexpected command {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.delta_checkpoint.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.delta_checkpoint.subprocess.run",
        fake_run,
    )

    base_tensors = {"a.weight": torch.zeros(8, dtype=torch.bfloat16)}
    updated_tensors = {"a.weight": torch.arange(8, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    staging_dir = tmp_path / "publish-stage"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir="s3://bucket/sync",
        publish_staging_dir=str(staging_dir),
    )
    update_info = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(update_info)

    assert update_info["uri"] == "s3://bucket/sync/delta-00000001"
    assert "s3://bucket/sync/delta-00000001/manifest.json" in objects
    assert any(key.endswith(".safetensors") for key in objects)
    # Payloads are staged locally then uploaded; nothing should be left behind.
    assert not list(staging_dir.rglob("*.tmp"))
    assert not list(staging_dir.rglob("*.safetensors"))

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=1, uri=update_info["uri"])
    received = _store_tensors(store)
    assert torch.equal(received["a.weight"], updated_tensors["a.weight"])


def test_delta_checkpoint_checksum_failure_marks_write_in_progress(tmp_path):
    base_tensors = {"a.weight": torch.arange(8, dtype=torch.bfloat16)}
    updated_tensors = {"a.weight": base_tensors["a.weight"] + torch.tensor(1, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        publish_staging_dir=str(tmp_path / "staging_dir"),
    )
    update_info = publisher.create_delta_files(_source(updated_tensors))
    update_info = publisher.publish(update_info)
    manifest_path = tmp_path / "sync" / "delta-00000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_checksum = manifest["tensors"][0]["checksum"]
    manifest["tensors"][0]["checksum"] = "0" * 32
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.fetch(target_version=1, uri=update_info["uri"])
    assert _read_state(receiver_dir)["write_in_progress"] is True
    # The bad delta must not stay cached, otherwise a repaired source can never be picked up.
    assert not (_deltas_dir(receiver_dir) / _safe_path_name(update_info["uri"])).exists()

    # Repair the manifest in place and re-fetch the *same* URI: recovery must work without
    # having to republish somewhere else.
    manifest["tensors"][0]["checksum"] = original_checksum
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store.fetch(target_version=1, uri=update_info["uri"])
    assert _read_state(receiver_dir)["write_in_progress"] is False
    assert torch.equal(
        _store_tensors(store)["a.weight"],
        updated_tensors["a.weight"],
    )
