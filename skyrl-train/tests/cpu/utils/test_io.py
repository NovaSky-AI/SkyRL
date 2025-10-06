"""
Unit tests for cloud storage I/O utilities.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, Mock
import torch


from skyrl_train.utils.io.io import (
    is_cloud_path,
    makedirs,
    exists,
    open_file,
    upload_directory,
    download_directory,
    local_work_dir,
    local_read_dir,
    list_dir,
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
        assert is_cloud_path("s3://bucket/path/file.pt")
        assert is_cloud_path("s3://my-bucket/checkpoints/global_step_1000/model.pt")

    def test_is_cloud_path_gcs(self):
        """Test GCS path detection."""
        assert is_cloud_path("gs://bucket/path/file.pt")
        assert is_cloud_path("gcs://bucket/path/file.pt")

    def test_is_local_path(self):
        """Test local path detection."""
        assert not is_cloud_path("/local/path/file.pt")
        assert not is_cloud_path("./relative/path/file.pt")
        assert not is_cloud_path("relative/path/file.pt")
        assert not is_cloud_path("C:\\Windows\\path\\file.pt")


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

    def test_open_file_text_local(self):
        """Test text file operations for local paths using open_file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            test_content = "test checkpoint step: 1000"

            # Write and read text using open_file
            with open_file(test_file, "w") as f:
                f.write(test_content)
            assert os.path.exists(test_file)

            with open_file(test_file, "r") as f:
                read_content = f.read()
            assert read_content == test_content

    def test_exists_local(self):
        """Test file existence check for local paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = os.path.join(temp_dir, "existing.txt")
            non_existing_file = os.path.join(temp_dir, "non_existing.txt")

            # Create a file
            with open(existing_file, "w") as f:
                f.write("test")

            assert exists(existing_file)
            assert not exists(non_existing_file)
            assert exists(temp_dir)

    def test_list_dir_returns_strings_local(self):
        """Test that list_dir returns a list of strings for a local directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files and a subdirectory
            file_a = os.path.join(temp_dir, "a.txt")
            with open(file_a, "w") as f:
                f.write("a")
            sub_dir = os.path.join(temp_dir, "sub")
            os.makedirs(sub_dir)
            file_b = os.path.join(sub_dir, "b.txt")
            with open(file_b, "w") as f:
                f.write("b")

            entries = list_dir(temp_dir)
            print(f"\n\n\nentries: {entries}\n\n\n")

            assert isinstance(entries, list)
            assert len(entries) >= 2
            assert all(isinstance(p, str) for p in entries)


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
            with open_file(latest_file, "w") as f:
                f.write("1500")
            assert get_latest_checkpoint_step(temp_dir) == 1500

            # Test with whitespace
            with open_file(latest_file, "w") as f:
                f.write("  2000  \n")
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
            cleanup_old_checkpoints(temp_dir, max_checkpoints=3)

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
            cleanup_old_checkpoints(temp_dir, max_checkpoints=5)

            # All directories should remain
            remaining_dirs = list_checkpoint_dirs(temp_dir)
            assert len(remaining_dirs) == 2


class TestCloudFileOperationsMocked:
    """Test cloud file operations with mocked fsspec."""

    @patch("skyrl_train.utils.io.io._get_filesystem")
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

    @patch("skyrl_train.utils.io.io._get_filesystem")
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

        cleanup_old_checkpoints(cloud_path, max_checkpoints=2)

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
            with patch("torch.save") as mock_save:
                with open_file(trainer_state_path, "wb") as f:
                    torch.save(trainer_state, f)
                mock_save.assert_called_once()

            # Save latest checkpoint info
            with open_file(latest_checkpoint_file, "w") as f:
                f.write(str(global_step))

            # Verify checkpoint was saved
            assert exists(global_step_folder)
            # Note: trainer_state_path won't exist since we mocked torch.save
            assert exists(latest_checkpoint_file)

            # Verify latest step can be retrieved
            assert get_latest_checkpoint_step(temp_dir) == global_step


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

    @patch("skyrl_train.utils.io.io.upload_directory")
    @patch("skyrl_train.utils.io.io.is_cloud_path")
    def test_local_work_dir_cloud_path(self, mock_is_cloud_path, mock_upload_directory):
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

        # Should have called upload_directory to upload to cloud
        mock_upload_directory.assert_called_once()
        # First argument should be the temp directory, second should be cloud path
        call_args = mock_upload_directory.call_args[0]
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

    @patch("skyrl_train.utils.io.io.download_directory")
    @patch("skyrl_train.utils.io.io.is_cloud_path")
    def test_local_read_dir_cloud_path(self, mock_is_cloud_path, mock_download_directory):
        """Test local_read_dir with a cloud path."""
        mock_is_cloud_path.return_value = True

        cloud_path = "s3://bucket/model"

        with local_read_dir(cloud_path) as read_dir:
            # Should get a temporary directory for cloud paths
            assert read_dir.startswith("/")
            assert "tmp" in read_dir.lower()
            assert os.path.exists(read_dir)
            assert os.path.isdir(read_dir)

        # Should have called download_directory to download from cloud
        mock_download_directory.assert_called_once_with(cloud_path, read_dir)

    def test_local_read_dir_nonexistent_local(self):
        """Test local_read_dir with a non-existent local path."""
        non_existent_path = "/non/existent/path/12345"

        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            with local_read_dir(non_existent_path):
                pass


class TestUploadDownload:
    """Test upload and download directory functions."""

    def test_upload_directory_validates_cloud_path(self):
        """Test that upload_directory validates destination is a cloud path."""
        with pytest.raises(ValueError, match="Destination must be a cloud path"):
            upload_directory("/local/src", "/local/dst")

    def test_download_directory_validates_cloud_path(self):
        """Test that download_directory validates source is a cloud path."""
        with pytest.raises(ValueError, match="Source must be a cloud path"):
            download_directory("/local/src", "/local/dst")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
