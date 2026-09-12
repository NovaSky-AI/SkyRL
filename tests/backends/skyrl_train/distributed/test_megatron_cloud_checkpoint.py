"""Cloud checkpoint staging with real PyTorch DCP metadata and common-state reads."""

import importlib.util
import io
import pickle
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed.checkpoint as dcp

COMMON_KEY = "common_state/shard_0_1"
CLOUD_PATH = "s3://test-bucket/global_step_10/policy"
COMMON_STATE = {"lr_scheduler": {"step": 10}, "rng": {"seed": 42}}


@pytest.fixture
def strategy_module(monkeypatch):
    """Import the actual strategy while replacing GPU-only Megatron dependencies."""

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            if parent in sys.modules and parent.startswith("megatron"):
                monkeypatch.setattr(sys.modules[parent], leaf, module, raising=False)
        return module

    class DistributedOptimizer:
        load_parameter_state_from_dp_reshardable = Mock()

    stub("megatron")
    stub("megatron.core")
    stub("megatron.core.parallel_state", get_data_parallel_group=lambda **kwargs: None)
    stub(
        "megatron.core.dist_checkpointing",
        load=Mock(),
        ShardedObject=Mock(return_value=SimpleNamespace(unique_key=COMMON_KEY)),
    )
    stub(
        "megatron.core.dist_checkpointing.serialization",
        get_default_load_sharded_strategy=Mock(),
        get_default_save_sharded_strategy=Mock(),
    )
    stub("megatron.core.dist_checkpointing.strategies")
    stub("megatron.core.dist_checkpointing.strategies.async_utils", AsyncCallsQueue=Mock())
    stub(
        "megatron.core.dist_checkpointing.strategies.fully_parallel",
        FullyParallelLoadStrategyWrapper=Mock(),
        FullyParallelSaveStrategyWrapper=Mock(),
    )
    stub("megatron.core.optimizer", DistributedOptimizer=DistributedOptimizer)
    stub("megatron.core.optimizer_param_scheduler", OptimizerParamScheduler=object)
    stub(
        "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils",
        **{
            name: Mock()
            for name in (
                "load_megatron_grads_to_gpu",
                "load_megatron_model_to_gpu",
                "load_megatron_optimizer",
                "offload_megatron_grads_to_cpu",
                "offload_megatron_model_to_cpu",
                "offload_megatron_optimizer",
            )
        },
    )
    stub("skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper", MegatronModelWrapper=object)
    path = Path(__file__).parents[4] / "skyrl/backends/skyrl_train/distributed/megatron/megatron_strategy.py"
    spec = importlib.util.spec_from_file_location("_cloud_checkpoint_test_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cloud_checkpoint(directory, common_shard, *, legacy=False):
    directory.mkdir()
    common = io.BytesIO()
    torch.save([COMMON_STATE], common)
    # Megatron serializes each ShardedObject as a BytesIO containing a list.
    dcp.save({COMMON_KEY: common}, checkpoint_id=directory)
    metadata = dcp.FileSystemReader(directory).read_metadata()
    original_shard = next(iter(metadata.storage_data.values())).relative_path
    destination = directory / common_shard
    destination.parent.mkdir(parents=True, exist_ok=True)
    (directory / original_shard).rename(destination)
    metadata.storage_data = {
        index: replace(storage, relative_path=common_shard) for index, storage in metadata.storage_data.items()
    }
    with (directory / ".metadata").open("wb") as file:
        pickle.dump(metadata, file)
    (directory / "metadata.json").write_text('{"sharded_backend":"torch_dist","sharded_backend_version":1}')
    for rank in (1, 2, 9):
        (directory / f"__{rank}_0.distcp").write_bytes(b"model shard not needed by the common-state read")
    if legacy:
        torch.save(COMMON_STATE, directory / "common.pt")
        destination.unlink()


def setup_cloud_transport(monkeypatch, module, root, cloud):
    downloads = []
    staged_paths = []
    local = root / "node-local"
    local.mkdir()
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(local))
    monkeypatch.setattr(module.io, "list_dir", lambda _: [f"{CLOUD_PATH}/{path.name}" for path in cloud.iterdir()])

    def source_path(path):
        return cloud / path.removeprefix(CLOUD_PATH + "/")

    monkeypatch.setattr(module.io, "isdir", lambda path: source_path(path).is_dir())

    def download(source, destination):
        downloads.append(source.removeprefix(CLOUD_PATH + "/"))
        shutil.copyfile(source_path(source), destination)

    monkeypatch.setattr(module.io, "download_file", download)

    def read_common_state(*, checkpoint_dir, **kwargs):
        staged_paths.append(Path(checkpoint_dir))
        legacy = Path(checkpoint_dir) / "common.pt"
        if legacy.exists():
            return torch.load(legacy, weights_only=True)
        state = {COMMON_KEY: io.BytesIO()}
        # Megatron performs this independent local read before its parallel load.
        dcp.load(state, storage_reader=dcp.FileSystemReader(checkpoint_dir), no_dist=True)
        state[COMMON_KEY].seek(0)
        return torch.load(state[COMMON_KEY], weights_only=True)[0]

    module.dist_checkpointing.load.side_effect = read_common_state
    return downloads, staged_paths


@pytest.mark.parametrize("global_rank", [0, 1])
@pytest.mark.parametrize("common_shard", ["__0_0.distcp", "__7_3.distcp", "objects/__5_2.distcp"])
def test_cloud_load_stages_common_state_on_every_node(
    strategy_module, tmp_path, monkeypatch, global_rank, common_shard
):
    cloud = tmp_path / "cloud"
    make_cloud_checkpoint(cloud, common_shard)
    downloads, staged_paths = setup_cloud_transport(monkeypatch, strategy_module, tmp_path, cloud)
    monkeypatch.setattr(strategy_module.dist, "get_rank", lambda: global_rank)
    monkeypatch.setattr(strategy_module.dist, "barrier", lambda: None)
    strategy = strategy_module.MegatronStrategy(SimpleNamespace(), node_local_rank=0)

    result = strategy._load_dist_checkpoint_from_cloud(CLOUD_PATH, {})

    assert result == COMMON_STATE
    assert downloads.count(common_shard) == 1
    if global_rank == 1:
        assert downloads.count("__1_0.distcp") == 1
    assert "__2_0.distcp" not in downloads
    assert "__9_0.distcp" not in downloads
    assert all(not path.exists() for path in staged_paths)


def test_legacy_common_pt_does_not_require_embedded_state_shard(strategy_module, tmp_path, monkeypatch):
    cloud = tmp_path / "cloud"
    make_cloud_checkpoint(cloud, "__0_0.distcp", legacy=True)
    downloads, _ = setup_cloud_transport(monkeypatch, strategy_module, tmp_path, cloud)
    monkeypatch.setattr(strategy_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(strategy_module.dist, "barrier", lambda: None)
    strategy = strategy_module.MegatronStrategy(SimpleNamespace(), node_local_rank=0)

    assert strategy._load_dist_checkpoint_from_cloud(CLOUD_PATH, {}) == COMMON_STATE

    assert downloads.count("common.pt") == 1
    assert "__0_0.distcp" not in downloads


def test_node_ranks_share_common_shard_without_duplicate_downloads(strategy_module, tmp_path, monkeypatch):
    cloud = tmp_path / "cloud"
    make_cloud_checkpoint(cloud, "__0_0.distcp")
    downloads, staged_paths = setup_cloud_transport(monkeypatch, strategy_module, tmp_path, cloud)
    rank_context = threading.local()
    barrier = threading.Barrier(2)
    monkeypatch.setattr(strategy_module.dist, "get_rank", lambda: rank_context.rank)
    monkeypatch.setattr(strategy_module.dist, "barrier", lambda: barrier.wait(timeout=10))

    def load(local_rank):
        rank_context.rank = local_rank + 1  # A node that does not host global rank 0.
        strategy = strategy_module.MegatronStrategy(SimpleNamespace(), node_local_rank=local_rank)
        return strategy._load_dist_checkpoint_from_cloud(CLOUD_PATH, {})

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(load, (0, 1)))

    assert results == [COMMON_STATE, COMMON_STATE]
    assert Counter(downloads) == Counter([".metadata", "metadata.json", "__0_0.distcp", "__1_0.distcp", "__2_0.distcp"])
    assert all(not path.exists() for path in staged_paths)
