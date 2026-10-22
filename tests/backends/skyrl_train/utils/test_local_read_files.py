from pathlib import Path
from unittest.mock import patch

from fsspec.implementations.memory import MemoryFileSystem

from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.utils.io.io import local_read_files


@patch("skyrl.backends.skyrl_train.utils.io.io.download_file")
@patch("skyrl.backends.skyrl_train.utils.io.io.exists")
@patch("skyrl.backends.skyrl_train.utils.io.io.is_cloud_path")
def test_cloud_downloads_only_requested_files(mock_is_cloud_path, mock_exists, mock_download_file):
    """
    Test that when a cloud path is provided, only the requested files are downloaded to a temporary directory.
    """
    mock_is_cloud_path.return_value = True
    mock_exists.return_value = True

    requested = [
        "model_world_size_8_rank_3.pt",
        "optim_world_size_8_rank_3.pt",
        "extra_state_world_size_8_rank_3.pt",
    ]
    with local_read_files("s3://bucket/global_step_1/policy", requested):
        pass

    downloaded = [call.args[0] for call in mock_download_file.call_args_list]
    assert len(downloaded) == len(requested)
    for name in requested:
        assert any(name in src for src in downloaded)


def test_local_path_objects(tmp_path):
    directory = tmp_path / "checkpoint"
    with io.local_work_dir(directory) as work_dir:
        assert work_dir == str(directory)
        (Path(work_dir) / "state.txt").write_text("step=42")

    with local_read_files(directory, ["state.txt"]) as read_dir:
        assert read_dir == str(directory)
        assert (Path(read_dir) / "state.txt").read_text() == "step=42"


def test_download_directory_to_path_object(tmp_path, monkeypatch):
    filesystem = MemoryFileSystem()
    cloud_path = f"gs://bucket/{tmp_path.name}/checkpoint"
    filesystem.pipe(f"{cloud_path}/state.txt", b"step=42")
    monkeypatch.setattr(io, "_get_filesystem", lambda _: filesystem)

    destination = tmp_path / "restored"
    io.download_directory(cloud_path, destination)
    assert (destination / "state.txt").read_text() == "step=42"


@patch("skyrl.backends.skyrl_train.utils.io.io.download_file")
@patch("skyrl.backends.skyrl_train.utils.io.io.exists")
@patch("skyrl.backends.skyrl_train.utils.io.io.is_cloud_path")
def test_local_path_does_not_download(mock_is_cloud_path, mock_exists, mock_download_file):
    """
    Test that when a local path is provided, no download occurs and the path is returned directly
    """
    mock_is_cloud_path.return_value = False
    mock_exists.return_value = True

    with local_read_files("/some/local/dir", ["a.pt", "b.pt"]) as read_dir:
        assert read_dir == "/some/local/dir"

    mock_download_file.assert_not_called()
