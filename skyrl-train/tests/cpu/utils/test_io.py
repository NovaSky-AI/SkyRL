"""
Unit tests for cloud storage I/O utilities.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, mock_open, MagicMock, Mock, call
import json

# Import our io module
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from skyrl_train.utils.io import (
    is_cloud_path,
    makedirs,
    exists,
    write_text,
    read_text,
    save_file,
    load_file,
    _get_filesystem,
    local_work_dir,
    local_read_dir,
)
from skyrl_train.utils.trainer_utils import (
    get_latest_checkpoint_step,
    list_checkpoint_dirs,
    cleanup_old_checkpoints,
)


class TestCloudPathDetection:
    """Test cloud path detection functionality."""

    def test_is_cloud_path_s3(self):
        """Test S3 path detection."""
        assert is_cloud_path("s3://bucket/path/file.pt") == True
        assert is_cloud_path("s3://my-bucket/checkpoints/global_step_1000/model.pt") == True

    def test_is_cloud_path_gcs(self):
        """Test GCS path detection."""
        assert is_cloud_path("gs://bucket/path/file.pt") == True
        assert is_cloud_path("gcs://bucket/path/file.pt") == True

    def test_is_local_path(self):
        """Test local path detection."""
        assert is_cloud_path("/local/path/file.pt") == False
        assert is_cloud_path("./relative/path/file.pt") == False
        assert is_cloud_path("relative/path/file.pt") == False
        assert is_cloud_path("C:\\Windows\\path\\file.pt") == False


class TestLocalFileOperations:
    """Test file operations for local paths."""

    def test_makedirs_local(self):
        """Test directory creation for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = os.path.join(temp_dir, "test_checkpoints")

            # Should create directory
            makedirs(test_dir)
            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

            # Should not fail with exist_ok=True (default)
            makedirs(test_dir, exist_ok=True)
            assert os.path.exists(test_dir)

    def test_makedirs_cloud_path(self):
        """Test that makedirs does nothing for cloud paths."""
        # Should not raise an error for cloud paths
        makedirs("s3://bucket/path", exist_ok=True)
        makedirs("gs://bucket/path", exist_ok=True)
        makedirs("gcs://bucket/path", exist_ok=True)

    def test_write_read_text_local(self):
        """Test text file operations for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            test_content = "test checkpoint step: 1000"

            # Write and read text
            write_text(test_file, test_content)
            assert os.path.exists(test_file)

            read_content = read_text(test_file)
            assert read_content == test_content

    def test_exists_local(self):
        """Test file existence check for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = os.path.join(temp_dir, "existing.txt")
            non_existing_file = os.path.join(temp_dir, "non_existing.txt")

            # Create a file
            with open(existing_file, "w") as f:
                f.write("test")

            assert exists(existing_file) == True
            assert exists(non_existing_file) == False
            assert exists(temp_dir) == True


class TestCheckpointUtilities:
    """Test checkpoint-specific utilities."""

    def test_list_checkpoint_dirs_local(self):
        """Test listing checkpoint directories for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some checkpoint directories
            checkpoint_dirs = [
                "global_step_1000",
                "global_step_2000",
                "global_step_500",
                "other_dir",  # Should be ignored
            ]

            for dirname in checkpoint_dirs:
                os.makedirs(os.path.join(temp_dir, dirname))

            # List checkpoint directories
            found_dirs = list_checkpoint_dirs(temp_dir)

            # Should only include global_step_ directories, sorted
            expected = ["global_step_1000", "global_step_2000", "global_step_500"]
            assert sorted(found_dirs) == sorted(expected)

    def test_list_checkpoint_dirs_nonexistent(self):
        """Test listing checkpoint directories for non-existent path."""
        non_existent_path = "/non/existent/path"
        found_dirs = list_checkpoint_dirs(non_existent_path)
        assert found_dirs == []

    def test_get_latest_checkpoint_step_local(self):
        """Test getting latest checkpoint step for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            latest_file = os.path.join(temp_dir, "latest_ckpt_global_step.txt")

            # Test non-existent file
            assert get_latest_checkpoint_step(temp_dir) == 0

            # Test with valid step
            write_text(latest_file, "1500")
            assert get_latest_checkpoint_step(temp_dir) == 1500

            # Test with whitespace
            write_text(latest_file, "  2000  \n")
            assert get_latest_checkpoint_step(temp_dir) == 2000

    def test_cleanup_old_checkpoints_local(self):
        """Test cleanup of old checkpoints for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create checkpoint directories
            steps = [1000, 1500, 2000, 2500, 3000]
            for step in steps:
                checkpoint_dir = os.path.join(temp_dir, f"global_step_{step}")
                os.makedirs(checkpoint_dir)
                # Add a dummy file to make it more realistic
                with open(os.path.join(checkpoint_dir, "model.pt"), "w") as f:
                    f.write("dummy")

            # Keep only 3 most recent
            cleanup_old_checkpoints(temp_dir, max_checkpoints=3, current_step=3000)

            # Check remaining directories
            remaining_dirs = list_checkpoint_dirs(temp_dir)
            remaining_steps = [int(d.split("_")[2]) for d in remaining_dirs]
            remaining_steps.sort()

            # Should keep the 3 most recent: 2000, 2500, 3000
            assert remaining_steps == [2000, 2500, 3000]

    def test_cleanup_old_checkpoints_no_cleanup_needed(self):
        """Test cleanup when no cleanup is needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only 2 checkpoint directories
            steps = [1000, 2000]
            for step in steps:
                checkpoint_dir = os.path.join(temp_dir, f"global_step_{step}")
                os.makedirs(checkpoint_dir)

            # Keep 5 - should not remove any
            cleanup_old_checkpoints(temp_dir, max_checkpoints=5, current_step=2000)

            # All directories should remain
            remaining_dirs = list_checkpoint_dirs(temp_dir)
            assert len(remaining_dirs) == 2


class TestCloudFileOperationsMocked:
    """Test cloud file operations with mocked fsspec."""

    @patch("skyrl_train.utils.io.fsspec.filesystem")
    def test_get_filesystem_s3(self, mock_filesystem):
        """Test _get_filesystem for S3 paths."""
        mock_fs = Mock()
        mock_filesystem.return_value = mock_fs

        fs = _get_filesystem("s3://bucket/path")

        mock_filesystem.assert_called_once_with("s3")
        assert fs == mock_fs

    @patch("skyrl_train.utils.io.fsspec.filesystem")
    def test_get_filesystem_gcs(self, mock_filesystem):
        """Test _get_filesystem for GCS paths."""
        mock_fs = Mock()
        mock_filesystem.return_value = mock_fs

        fs = _get_filesystem("gs://bucket/path")

        mock_filesystem.assert_called_once_with("gs")
        assert fs == mock_fs

    @patch("skyrl_train.utils.io.fsspec.filesystem")
    def test_get_filesystem_local(self, mock_filesystem):
        """Test _get_filesystem for local paths."""
        mock_fs = Mock()
        mock_filesystem.return_value = mock_fs

        fs = _get_filesystem("/local/path")

        mock_filesystem.assert_called_once_with("file")
        assert fs == mock_fs

    @patch("skyrl_train.utils.io.torch.save")
    @patch("skyrl_train.utils.io.open_file")
    def test_save_file_cloud(self, mock_open_file, mock_torch_save):
        """Test save_file with cloud storage."""
        mock_file = Mock()
        mock_open_file.return_value.__enter__.return_value = mock_file

        test_obj = {"model": "weights", "step": 1000}
        cloud_path = "s3://bucket/model.pt"

        save_file(test_obj, cloud_path)

        # Verify file was opened with correct path and mode
        mock_open_file.assert_called_once_with(cloud_path, "wb")
        # Verify torch.save was called with object and file handle
        mock_torch_save.assert_called_once_with(test_obj, mock_file)

    @patch("skyrl_train.utils.io.torch.save")
    def test_save_file_local(self, mock_torch_save):
        """Test save_file with local storage."""
        test_obj = {"model": "weights", "step": 1000}
        local_path = "/tmp/model.pt"

        save_file(test_obj, local_path)

        # Should call torch.save directly with path
        mock_torch_save.assert_called_once_with(test_obj, local_path)

    @patch("skyrl_train.utils.io.torch.load")
    @patch("skyrl_train.utils.io.open_file")
    def test_load_file_cloud(self, mock_open_file, mock_torch_load):
        """Test load_file with cloud storage."""
        mock_file = Mock()
        mock_open_file.return_value.__enter__.return_value = mock_file
        mock_torch_load.return_value = {"model": "weights", "step": 1000}

        cloud_path = "s3://bucket/model.pt"

        result = load_file(cloud_path, map_location="cuda", weights_only=True)

        # Verify file was opened correctly
        mock_open_file.assert_called_once_with(cloud_path, "rb")
        # Verify torch.load was called with correct parameters
        mock_torch_load.assert_called_once_with(mock_file, map_location="cuda", weights_only=True)
        assert result == {"model": "weights", "step": 1000}

    @patch("skyrl_train.utils.io.torch.load")
    def test_load_file_local(self, mock_torch_load):
        """Test load_file with local storage."""
        mock_torch_load.return_value = {"model": "weights", "step": 1000}

        local_path = "/tmp/model.pt"

        result = load_file(local_path, map_location="cpu", weights_only=False)

        # Should call torch.load directly with path
        mock_torch_load.assert_called_once_with(local_path, map_location="cpu", weights_only=False)
        assert result == {"model": "weights", "step": 1000}

    @patch("skyrl_train.utils.io.open_file")
    def test_write_text_cloud(self, mock_open_file):
        """Test write_text with cloud storage."""
        mock_file = Mock()
        mock_open_file.return_value.__enter__.return_value = mock_file

        cloud_path = "s3://bucket/config.txt"
        content = "global_step: 1000"

        write_text(cloud_path, content)

        mock_open_file.assert_called_once_with(cloud_path, "w")
        mock_file.write.assert_called_once_with(content)

    @patch("skyrl_train.utils.io.open_file")
    def test_read_text_cloud(self, mock_open_file):
        """Test read_text with cloud storage."""
        mock_file = Mock()
        mock_file.read.return_value = "global_step: 1000"
        mock_open_file.return_value.__enter__.return_value = mock_file

        cloud_path = "s3://bucket/config.txt"

        result = read_text(cloud_path)

        mock_open_file.assert_called_once_with(cloud_path, "r")
        mock_file.read.assert_called_once()
        assert result == "global_step: 1000"

    @patch("skyrl_train.utils.io._get_filesystem")
    def test_exists_cloud(self, mock_get_filesystem):
        """Test exists with cloud storage."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_get_filesystem.return_value = mock_fs

        cloud_path = "s3://bucket/model.pt"

        result = exists(cloud_path)

        mock_get_filesystem.assert_called_once_with(cloud_path)
        mock_fs.exists.assert_called_once_with(cloud_path)
        assert result == True

    @patch("skyrl_train.utils.io._get_filesystem")
    def test_list_checkpoint_dirs_cloud(self, mock_get_filesystem):
        """Test list_checkpoint_dirs with cloud storage."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_fs.ls.return_value = [
            "s3://bucket/checkpoints/global_step_1000",
            "s3://bucket/checkpoints/global_step_2000",
            "s3://bucket/checkpoints/other_dir",
        ]
        mock_fs.isdir.side_effect = lambda path: "global_step_" in path
        mock_get_filesystem.return_value = mock_fs

        cloud_path = "s3://bucket/checkpoints"

        result = list_checkpoint_dirs(cloud_path)

        # Should return sorted checkpoint directories
        expected = ["global_step_1000", "global_step_2000"]
        assert sorted(result) == sorted(expected)

    @patch("skyrl_train.utils.io._get_filesystem")
    def test_cleanup_old_checkpoints_cloud(self, mock_get_filesystem):
        """Test cleanup_old_checkpoints with cloud storage."""
        mock_fs = Mock()
        mock_fs.exists.return_value = True
        mock_fs.ls.return_value = [
            "s3://bucket/checkpoints/global_step_1000",
            "s3://bucket/checkpoints/global_step_1500",
            "s3://bucket/checkpoints/global_step_2000",
            "s3://bucket/checkpoints/global_step_2500",
        ]
        mock_fs.isdir.return_value = True
        mock_get_filesystem.return_value = mock_fs

        cloud_path = "s3://bucket/checkpoints"

        cleanup_old_checkpoints(cloud_path, max_checkpoints=2, current_step=2500)

        # Should remove the 2 oldest checkpoints
        expected_removes = [
            "s3://bucket/checkpoints/global_step_1000",
            "s3://bucket/checkpoints/global_step_1500",
        ]

        # Check that remove was called for old checkpoints
        actual_removes = [call[0][0] for call in mock_fs.rm.call_args_list]
        assert sorted(actual_removes) == sorted(expected_removes)


class TestCheckpointScenarios:
    """Test realistic checkpoint scenarios."""

    def test_local_checkpoint_save_load_cycle(self):
        """Test a complete checkpoint save/load cycle with local storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate trainer checkpoint saving
            global_step = 1500
            global_step_folder = os.path.join(temp_dir, f"global_step_{global_step}")
            trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
            latest_checkpoint_file = os.path.join(temp_dir, "latest_ckpt_global_step.txt")

            # Save checkpoint
            makedirs(global_step_folder)

            # Simulate saving trainer state (mock torch.save for simplicity)
            trainer_state = {"global_step": global_step, "config": {"lr": 0.001}}
            with patch("skyrl_train.utils.io.torch.save") as mock_save:
                save_file(trainer_state, trainer_state_path)
                mock_save.assert_called_once_with(trainer_state, trainer_state_path)

            # Save latest checkpoint info
            write_text(latest_checkpoint_file, str(global_step))

            # Verify checkpoint was saved
            assert exists(global_step_folder)
            # Note: trainer_state_path won't exist since we mocked torch.save
            assert exists(latest_checkpoint_file)

            # Verify latest step can be retrieved
            assert get_latest_checkpoint_step(temp_dir) == global_step

    @patch("skyrl_train.utils.io.torch.save")
    @patch("skyrl_train.utils.io.fsspec.filesystem")
    def test_cloud_checkpoint_scenario_mock(self, mock_filesystem, mock_torch_save):
        """Test checkpoint scenario with mocked cloud storage."""
        # Mock the filesystem
        mock_fs = MagicMock()
        mock_file = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_file
        mock_context_manager.__exit__.return_value = None
        mock_fs.open.return_value = mock_context_manager
        mock_filesystem.return_value = mock_fs

        # Simulate S3 checkpoint paths
        base_path = "s3://my-bucket/checkpoints"
        global_step = 2000
        global_step_folder = f"{base_path}/global_step_{global_step}"
        trainer_state_path = f"{global_step_folder}/trainer_state.pt"
        latest_checkpoint_file = f"{base_path}/latest_ckpt_global_step.txt"

        # Simulate trainer checkpoint saving workflow
        makedirs(global_step_folder)  # Should do nothing for cloud paths
        save_file({"global_step": global_step, "config": {}}, trainer_state_path)
        write_text(latest_checkpoint_file, str(global_step))

        # Verify cloud filesystem was used
        mock_filesystem.assert_called_with("s3")
        # Verify files were opened for writing
        expected_calls = [call(trainer_state_path, "wb"), call(latest_checkpoint_file, "w")]
        mock_fs.open.assert_has_calls(expected_calls, any_order=True)

        # Verify torch.save was called with the file handle
        mock_torch_save.assert_called_once_with({"global_step": global_step, "config": {}}, mock_file)

        # Verify text was written to file handle
        mock_file.write.assert_called_with(str(global_step))



class TestIntegrationWithMocking:
    """Integration tests with strategic mocking."""

    @patch("skyrl_train.utils.io.torch.save")
    @patch("skyrl_train.utils.io.torch.load")
    def test_checkpoint_roundtrip_local(self, mock_torch_load, mock_torch_save):
        """Test checkpoint save/load roundtrip with local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")

            # Test data
            checkpoint_data = {
                "model_state_dict": {"layer1.weight": [1, 2, 3]},
                "optimizer_state_dict": {"step": 1000},
                "global_step": 1000,
            }

            # Save checkpoint
            save_file(checkpoint_data, checkpoint_path)
            mock_torch_save.assert_called_once_with(checkpoint_data, checkpoint_path)

            # Mock load return value
            mock_torch_load.return_value = checkpoint_data

            # Load checkpoint
            loaded_data = load_file(checkpoint_path)
            mock_torch_load.assert_called_once_with(checkpoint_path, map_location="cpu", weights_only=False)

            assert loaded_data == checkpoint_data

    @patch("skyrl_train.utils.io.torch.save")
    @patch("skyrl_train.utils.io.torch.load")
    @patch("skyrl_train.utils.io.open_file")
    def test_checkpoint_roundtrip_cloud(self, mock_open_file, mock_torch_load, mock_torch_save):
        """Test checkpoint save/load roundtrip with cloud paths."""
        # Setup mock file handles
        mock_save_file = Mock()
        mock_load_file = Mock()
        mock_open_file.return_value.__enter__.side_effect = [mock_save_file, mock_load_file]

        checkpoint_path = "s3://bucket/checkpoint.pt"

        # Test data
        checkpoint_data = {
            "model_state_dict": {"layer1.weight": [1, 2, 3]},
            "optimizer_state_dict": {"step": 1000},
            "global_step": 1000,
        }

        # Save checkpoint
        save_file(checkpoint_data, checkpoint_path)

        # Verify save operations
        assert mock_open_file.call_count == 1
        mock_open_file.assert_any_call(checkpoint_path, "wb")
        mock_torch_save.assert_called_once_with(checkpoint_data, mock_save_file)

        # Mock load return value
        mock_torch_load.return_value = checkpoint_data

        # Reset mock for load operation
        mock_open_file.reset_mock()

        # Load checkpoint
        loaded_data = load_file(checkpoint_path)

        # Verify load operations
        mock_open_file.assert_called_once_with(checkpoint_path, "rb")
        mock_torch_load.assert_called_once_with(mock_load_file, map_location="cpu", weights_only=False)

        assert loaded_data == checkpoint_data


class TestContextManagers:
    """Test the local_work_dir and local_read_dir context managers."""

    def test_local_work_dir_local_path(self):
        """Test local_work_dir with a local path."""
        with tempfile.TemporaryDirectory() as base_temp_dir:
            test_dir = os.path.join(base_temp_dir, "test_output")

            with local_work_dir(test_dir) as work_dir:
                # Should return the same path for local paths
                assert work_dir == test_dir
                # Directory should be created
                assert os.path.exists(work_dir)
                assert os.path.isdir(work_dir)

                # Write a test file
                test_file = os.path.join(work_dir, "test.txt")
                with open(test_file, "w") as f:
                    f.write("test content")

            # File should still exist after context exit
            assert os.path.exists(test_file)
            with open(test_file, "r") as f:
                assert f.read() == "test content"

    @patch("skyrl_train.utils.io.copy_tree")
    @patch("skyrl_train.utils.io.is_cloud_path")
    def test_local_work_dir_cloud_path(self, mock_is_cloud_path, mock_copy_tree):
        """Test local_work_dir with a cloud path."""
        mock_is_cloud_path.return_value = True

        cloud_path = "s3://bucket/model"

        with local_work_dir(cloud_path) as work_dir:
            # Should get a temporary directory for cloud paths
            assert work_dir.startswith("/")
            assert "tmp" in work_dir.lower()
            assert os.path.exists(work_dir)
            assert os.path.isdir(work_dir)

            # Write a test file
            test_file = os.path.join(work_dir, "model.txt")
            with open(test_file, "w") as f:
                f.write("model data")

        # Should have called copy_tree to upload to cloud
        mock_copy_tree.assert_called_once()
        # First argument should be the temp directory, second should be cloud path
        call_args = mock_copy_tree.call_args[0]
        assert call_args[1] == cloud_path
        # First argument should be the temp directory we worked with
        assert call_args[0] == work_dir  # This will be the temp dir

    def test_local_read_dir_local_path(self):
        """Test local_read_dir with a local path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            with local_read_dir(temp_dir) as read_dir:
                # Should return the same path for local paths
                assert read_dir == temp_dir

                # Should be able to read the file
                read_file = os.path.join(read_dir, "test.txt")
                assert os.path.exists(read_file)
                with open(read_file, "r") as f:
                    assert f.read() == "test content"

    @patch("skyrl_train.utils.io.copy_tree")
    @patch("skyrl_train.utils.io.is_cloud_path")
    def test_local_read_dir_cloud_path(self, mock_is_cloud_path, mock_copy_tree):
        """Test local_read_dir with a cloud path."""
        mock_is_cloud_path.return_value = True

        cloud_path = "s3://bucket/model"

        with local_read_dir(cloud_path) as read_dir:
            # Should get a temporary directory for cloud paths
            assert read_dir.startswith("/")
            assert "tmp" in read_dir.lower()
            assert os.path.exists(read_dir)
            assert os.path.isdir(read_dir)

        # Should have called copy_tree to download from cloud
        mock_copy_tree.assert_called_once_with(cloud_path, read_dir)

    def test_local_read_dir_nonexistent_local(self):
        """Test local_read_dir with a non-existent local path."""
        non_existent_path = "/non/existent/path/12345"

        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            with local_read_dir(non_existent_path) as read_dir:
                pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
