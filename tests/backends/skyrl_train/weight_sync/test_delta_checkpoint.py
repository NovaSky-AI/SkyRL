import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
    DeltaCheckpointPublisher,
    DeltaPublishResult,
    LocalCheckpointStore,
    _weights_dir,
)
from skyrl.backends.skyrl_train.weight_sync.delta_strategy import (
    DeltaInitInfo,
    DeltaWeightTransferSender,
)
from skyrl.backends.skyrl_train.weight_sync.weight_extractor import ExtractorShardInfo


def _chunk_from_tensors(tensors):
    return WeightChunk(
        names=list(tensors.keys()),
        dtypes=[str(t.dtype) for t in tensors.values()],
        shapes=[list(t.shape) for t in tensors.values()],
        tensors=list(tensors.values()),
    )


def _shard_info(rank, world_size):
    return ExtractorShardInfo(
        is_source_rank=True,
        replicate=["dp"],
        split=[],
        mesh_rank=MeshRank(dp=rank, sp=0, tp=0, pp=0, world_size=world_size, dp_size=world_size, pp_size=1),
        replicate_world_size=world_size,
        source_index_in_replicate_world=rank,
        rank=rank,
    )


def _non_source_info(rank=1, world_size=2):
    info = _shard_info(rank=rank, world_size=world_size)
    return ExtractorShardInfo(
        is_source_rank=False,
        replicate=info.replicate,
        split=info.split,
        mesh_rank=info.mesh_rank,
        replicate_world_size=1,
        source_index_in_replicate_world=0,
        rank=rank,
    )


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
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )
    update_info = publisher.publish([_chunk_from_tensors(updated_tensors)])

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    stats = store.fetch(target_version=update_info["target_version"], sync_dir=update_info["sync_dir"])
    assert stats["apply_s"] >= 0.0
    store.validate_ready(1)

    received = dict(store.iter_tensors())
    assert set(received) == {changed_name, unchanged_name}
    assert torch.equal(received[changed_name], updated_tensors[changed_name])
    assert torch.equal(received[unchanged_name], base_tensors[unchanged_name])
    assert torch.equal(_load_tensor(_weights_dir(receiver_dir), changed_name), updated_tensors[changed_name])
    assert _read_state(receiver_dir)["version"] == 1


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
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )
    update_info = publisher.publish([_chunk_from_tensors(runtime_updated)])
    with open(tmp_path / "sync" / "v00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["tensors"][0]["dtype"] == "float32"
    assert manifest["tensors"][0]["payload_key"] == "a.weight"
    assert manifest["tensors"][0]["checksum_algorithm"] == "xxh3-128"
    assert manifest["tensors"][0]["uncompressed_num_bytes"] == base_tensors["a.weight"].numel() * 4

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    assert not _weights_dir(receiver_dir).exists()
    store.fetch(target_version=1, sync_dir=update_info["sync_dir"])
    received = dict(store.iter_tensors())["a.weight"]
    assert received.dtype == torch.float32
    assert torch.equal(received, expected_checkpoint_tensor)


def test_delta_checkpoint_non_source_rank_drains_without_publishing(tmp_path):
    base_tensors = {f"model.layers.{idx}.weight": torch.full((4, 4), idx, dtype=torch.bfloat16) for idx in range(4)}
    updated_tensors = {
        name: tensor + torch.tensor(idx + 1, dtype=torch.bfloat16)
        for idx, (name, tensor) in enumerate(base_tensors.items())
    }
    chunks = [_chunk_from_tensors({name: tensor}) for name, tensor in updated_tensors.items()]
    base_dir = tmp_path / "base"
    sync_dir = tmp_path / "sync"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(sync_dir),
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )

    result = publisher.publish(chunks, _non_source_info())
    assert isinstance(result, DeltaPublishResult)
    assert result.records == []
    assert result.payload_files == []
    assert publisher.snapshot == {}
    assert publisher.version == 1
    assert not (sync_dir / "v00000001").exists()


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
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )
    publisher.publish([_chunk_from_tensors(v1_tensors)])
    first_snapshot = publisher.snapshot["a.weight"]
    first_snapshot_id = id(first_snapshot)
    assert first_snapshot.tobytes() == v1_tensors["a.weight"].contiguous().view(torch.uint8).numpy().tobytes()

    update_info = publisher.publish([_chunk_from_tensors(v2_tensors)])
    assert id(publisher.snapshot["a.weight"]) == first_snapshot_id
    assert (
        publisher.snapshot["a.weight"].tobytes()
        == v2_tensors["a.weight"].contiguous().view(torch.uint8).numpy().tobytes()
    )

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=2, sync_dir=update_info["sync_dir"])
    store.validate_ready(2)
    assert torch.equal(dict(store.iter_tensors())["a.weight"], v2_tensors["a.weight"])


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
        local_checkpoint_dir=str(tmp_path / "publisher"),
        max_file_size_in_gb=1e-9,
    )
    update_info = publisher.publish([_chunk_from_tensors(updated_tensors)])

    with open(tmp_path / "sync" / "v00000001" / "manifest.json", encoding="utf-8") as f:
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
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )
    update_info = publisher.publish(
        [
            WeightChunk(
                names=["model.embed_tokens.weight", "lm_head.weight"],
                dtypes=["torch.bfloat16", "torch.bfloat16"],
                shapes=[list(updated_embed.shape), list(updated_lm_head.shape)],
                tensors=[updated_embed, updated_lm_head],
            )
        ]
    )
    with open(tmp_path / "sync" / "v00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert [record["name"] for record in manifest["tensors"]] == ["model.embed_tokens.weight"]

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    store.fetch(target_version=1, uri=update_info["uri"])

    received = dict(store.iter_tensors())
    assert set(received) == {"model.embed_tokens.weight"}
    assert torch.equal(received["model.embed_tokens.weight"], updated_embed)


def test_delta_checkpoint_unchanged_publish_advances_version(tmp_path):
    base_tensors = {"a.weight": torch.ones(8, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )
    update_info = publisher.publish([_chunk_from_tensors({"a.weight": base_tensors["a.weight"].clone()})])

    assert update_info.get("noop") is not True
    assert update_info["target_version"] == 1
    assert publisher.version == 1

    with open(tmp_path / "sync" / "v00000001" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["tensors"] == []
    assert manifest["payload_files"] == []

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    assert not _weights_dir(receiver_dir).exists()
    store.fetch(target_version=1, uri=update_info["uri"])
    assert _read_state(receiver_dir)["version"] == 1
    assert not _weights_dir(receiver_dir).exists()
    assert torch.equal(dict(store.iter_tensors())["a.weight"], base_tensors["a.weight"])


def test_delta_checkpoint_checksum_failure_marks_write_in_progress(tmp_path):
    base_tensors = {"a.weight": torch.arange(8, dtype=torch.bfloat16)}
    updated_tensors = {"a.weight": base_tensors["a.weight"] + torch.tensor(1, dtype=torch.bfloat16)}
    base_dir = tmp_path / "base"
    receiver_dir = tmp_path / "receiver"
    _write_checkpoint(base_dir, base_tensors)

    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(tmp_path / "sync"),
        local_checkpoint_dir=str(tmp_path / "publisher"),
    )
    update_info = publisher.publish([_chunk_from_tensors(updated_tensors)])
    manifest_path = tmp_path / "sync" / "v00000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensors"][0]["checksum"] = "0" * 32
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = LocalCheckpointStore(base_model_path=str(base_dir), local_checkpoint_dir=str(receiver_dir))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.fetch(target_version=1, uri=update_info["uri"])
    assert _read_state(receiver_dir)["write_in_progress"] is True

    # Re-publish a valid delta by rebuilding from scratch to avoid relying on the corrupted manifest.
    shutil_sync = tmp_path / "sync_valid"
    publisher = DeltaCheckpointPublisher(
        base_model_path=str(base_dir),
        sync_dir=str(shutil_sync),
        local_checkpoint_dir=str(tmp_path / "publisher_valid"),
    )
    valid_info = publisher.publish([_chunk_from_tensors(updated_tensors)])
    store.fetch(target_version=1, uri=valid_info["uri"])
    assert _read_state(receiver_dir)["write_in_progress"] is False
    assert torch.equal(dict(store.iter_tensors())["a.weight"], updated_tensors["a.weight"])


@pytest.mark.asyncio
async def test_delta_sender_seed_sync_skips_chunk_iteration(tmp_path):
    class ExplodingChunks:
        def __iter__(self):
            raise AssertionError("seed sync should not iterate weight chunks")

    class FakeInferenceClient:
        async def fetch_weights(self, **kwargs):
            raise AssertionError("seed sync should not call fetch_weights")

        async def pause_generation(self):
            raise AssertionError("seed sync should not pause generation")

        async def start_weight_update(self, **kwargs):
            raise AssertionError("seed sync should not start weight update")

        async def update_named_weights(self, update_info):
            raise AssertionError("seed sync should not update weights")

        async def finish_weight_update(self):
            raise AssertionError("seed sync should not finish weight update")

        async def resume_generation(self):
            raise AssertionError("seed sync should not resume generation")

    sender = DeltaWeightTransferSender(
        DeltaInitInfo(
            override_existing_receiver=False,
            base_model_path=str(tmp_path / "base"),
            sync_dir=str(tmp_path / "sync"),
            local_checkpoint_dir=str(tmp_path / "receiver"),
            publisher_local_checkpoint_dir=str(tmp_path / "publisher"),
        ),
        FakeInferenceClient(),
    )

    await sender.send_chunks(ExplodingChunks())
