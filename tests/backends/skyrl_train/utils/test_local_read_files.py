from unittest.mock import patch

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
