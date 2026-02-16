"""
Tests for S3 checkpoint management.

uv run --isolated --extra dev pytest tests/cpu/test_s3_checkpoints.py -v
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

from integrations.fleet.s3_checkpoints import (
    cleanup_old_local_checkpoints,
    download_checkpoint_from_s3,
    wrap_trainer_with_s3_upload,
    S3CheckpointUploader,
)


# ============================================================================
# cleanup_old_local_checkpoints
# ============================================================================


def setup_checkpoint_dirs(tmpdir, steps):
    """Create fake global_step_N directories."""
    for step in steps:
        os.makedirs(os.path.join(tmpdir, f"global_step_{step}"))


def test_cleanup_old_local_checkpoints_keeps_n_most_recent():
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_checkpoint_dirs(tmpdir, [5, 10, 15, 20])

        cleanup_old_local_checkpoints(tmpdir, keep_n=2)

        remaining = sorted(os.listdir(tmpdir))
        assert remaining == ["global_step_15", "global_step_20"]


def test_cleanup_old_local_checkpoints_keeps_all_when_fewer_than_n():
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_checkpoint_dirs(tmpdir, [10, 20])

        cleanup_old_local_checkpoints(tmpdir, keep_n=5)

        remaining = sorted(os.listdir(tmpdir))
        assert remaining == ["global_step_10", "global_step_20"]


def test_cleanup_old_local_checkpoints_noop_on_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        cleanup_old_local_checkpoints(tmpdir, keep_n=2)
        assert os.listdir(tmpdir) == []


def test_cleanup_old_local_checkpoints_noop_on_missing_dir():
    # Should not raise
    cleanup_old_local_checkpoints("/nonexistent/path/abc123", keep_n=2)


def test_cleanup_old_local_checkpoints_ignores_non_checkpoint_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_checkpoint_dirs(tmpdir, [10, 20, 30])
        os.makedirs(os.path.join(tmpdir, "some_other_dir"))

        cleanup_old_local_checkpoints(tmpdir, keep_n=1)

        remaining = sorted(os.listdir(tmpdir))
        assert "some_other_dir" in remaining
        assert "global_step_30" in remaining
        assert "global_step_10" not in remaining
        assert "global_step_20" not in remaining


# ============================================================================
# download_checkpoint_from_s3
# ============================================================================


def test_download_skips_when_no_aws_credentials():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}, clear=False):
            result = download_checkpoint_from_s3(
                ckpt_path=tmpdir,
                run_name="test_run",
            )
        assert result is False


def test_download_skips_when_local_checkpoint_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create existing checkpoint
        os.makedirs(os.path.join(tmpdir, "global_step_20"))
        with open(os.path.join(tmpdir, "latest_ckpt_global_step.txt"), "w") as f:
            f.write("20")

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}, clear=False):
            result = download_checkpoint_from_s3(
                ckpt_path=tmpdir,
                run_name="test_run",
            )
        assert result is False


def test_download_finds_latest_checkpoint():
    """Verify download picks the highest global_step from S3."""
    mock_boto3 = MagicMock()
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator

    def paginate_side_effect(**kwargs):
        if "Delimiter" in kwargs:
            return [
                {
                    "CommonPrefixes": [
                        {"Prefix": "fleet-task-grpo/Qwen3-32B/test_run/global_step_10/"},
                        {"Prefix": "fleet-task-grpo/Qwen3-32B/test_run/global_step_20/"},
                        {"Prefix": "fleet-task-grpo/Qwen3-32B/test_run/global_step_5/"},
                    ]
                }
            ]
        else:
            return [
                {
                    "Contents": [
                        {
                            "Key": "fleet-task-grpo/Qwen3-32B/test_run/global_step_20/policy/model.safetensors",
                            "Size": 1000000,
                        },
                        {
                            "Key": "fleet-task-grpo/Qwen3-32B/test_run/global_step_20/trainer_state.pt",
                            "Size": 500,
                        },
                    ]
                }
            ]

    mock_paginator.paginate.side_effect = paginate_side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(sys.modules, {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()}):
            with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}, clear=False):
                result = download_checkpoint_from_s3(
                    ckpt_path=tmpdir,
                    run_name="test_run",
                )

        assert result is True

        latest_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        assert os.path.exists(latest_file)
        with open(latest_file) as f:
            assert f.read().strip() == "20"

        assert mock_s3.download_file.call_count == 2
        assert os.path.isdir(os.path.join(tmpdir, "global_step_20"))


def test_download_returns_false_when_no_checkpoints_in_s3():
    mock_boto3 = MagicMock()
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(sys.modules, {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()}):
            with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}, clear=False):
                result = download_checkpoint_from_s3(
                    ckpt_path=tmpdir,
                    run_name="test_run",
                )

        assert result is False


def test_download_uses_correct_s3_prefix():
    """Verify S3 prefix is constructed from project_name/model_name/run_name."""
    mock_boto3 = MagicMock()
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(sys.modules, {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()}):
            with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}, clear=False):
                download_checkpoint_from_s3(
                    ckpt_path=tmpdir,
                    run_name="fleet_tool_use_32b_d7167c1c",
                    project_name="fleet-task-grpo",
                    model_name="Qwen3-32B",
                )

        paginate_call = mock_paginator.paginate.call_args
        assert paginate_call.kwargs["Prefix"] == "fleet-task-grpo/Qwen3-32B/fleet_tool_use_32b_d7167c1c/"


def test_download_uses_custom_bucket_from_env():
    mock_boto3 = MagicMock()
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(sys.modules, {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()}):
            with patch.dict(
                os.environ,
                {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret", "S3_CHECKPOINT_BUCKET": "my-bucket"},
                clear=False,
            ):
                download_checkpoint_from_s3(
                    ckpt_path=tmpdir,
                    run_name="test_run",
                )

        paginate_call = mock_paginator.paginate.call_args
        assert paginate_call.kwargs["Bucket"] == "my-bucket"


# ============================================================================
# wrap_trainer_with_s3_upload
# ============================================================================


def _make_mock_trainer(ckpt_path, run_name="test_run", project_name="skyrl", model_path="Qwen/Qwen3-32B"):
    trainer = MagicMock()
    trainer.cfg.trainer.ckpt_path = ckpt_path
    trainer.cfg.trainer.run_name = run_name
    trainer.cfg.trainer.project_name = project_name
    trainer.cfg.trainer.policy.model.path = model_path
    trainer.global_step = 10
    trainer.save_checkpoints = MagicMock()
    return trainer


def test_wrap_trainer_replaces_save_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_mock_trainer(tmpdir)
        original_save = trainer.save_checkpoints

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}, clear=False):
            wrapped = wrap_trainer_with_s3_upload(trainer)

        assert wrapped.save_checkpoints is not original_save


def test_wrap_trainer_save_calls_cleanup_before_save():
    """Verify cleanup happens BEFORE save, not after."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_mock_trainer(tmpdir)
        call_order = []

        trainer.save_checkpoints = lambda: call_order.append("save")

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}, clear=False):
            with patch(
                "integrations.fleet.s3_checkpoints.cleanup_old_local_checkpoints",
                side_effect=lambda *a, **kw: call_order.append("cleanup"),
            ):
                wrapped = wrap_trainer_with_s3_upload(trainer)
                wrapped.save_checkpoints()

        assert call_order == ["cleanup", "save"]


def test_wrap_trainer_s3_upload_after_save():
    """When AWS creds are set, verify S3 upload is queued after save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_mock_trainer(tmpdir)
        trainer.global_step = 42

        def fake_save():
            os.makedirs(os.path.join(tmpdir, "global_step_42"))

        trainer.save_checkpoints = fake_save

        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"},
            clear=False,
        ):
            with patch("integrations.fleet.s3_checkpoints.cleanup_old_local_checkpoints"):
                wrapped = wrap_trainer_with_s3_upload(trainer)
                wrapped.save_checkpoints()

        assert wrapped._s3_uploader is not None


def test_wrap_trainer_no_s3_without_credentials():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_mock_trainer(tmpdir)

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}, clear=False):
            wrapped = wrap_trainer_with_s3_upload(trainer)

        assert wrapped._s3_uploader is None


def test_wrap_trainer_keep_local_controls_cleanup_count():
    """With S3 enabled, keep_n=1; without S3, keep_n=2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_mock_trainer(tmpdir)
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}, clear=False):
            with patch("integrations.fleet.s3_checkpoints.cleanup_old_local_checkpoints") as mock_cleanup:
                wrapped = wrap_trainer_with_s3_upload(trainer)
                wrapped.save_checkpoints()
                mock_cleanup.assert_called_once_with(tmpdir, keep_n=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = _make_mock_trainer(tmpdir)
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}, clear=False):
            with patch("integrations.fleet.s3_checkpoints.cleanup_old_local_checkpoints") as mock_cleanup:
                wrapped = wrap_trainer_with_s3_upload(trainer)
                wrapped.save_checkpoints()
                mock_cleanup.assert_called_once_with(tmpdir, keep_n=2)


# ============================================================================
# S3CheckpointUploader.keep_local
# ============================================================================


def test_uploader_deletes_local_after_upload():
    """After successful S3 upload, local checkpoint dir should be deleted to free disk."""
    mock_boto3 = MagicMock()
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "global_step_10")
        os.makedirs(ckpt_dir)
        with open(os.path.join(ckpt_dir, "model.pt"), "w") as f:
            f.write("fake")

        with patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": MagicMock(),
                "botocore.config": MagicMock(),
                "boto3.s3": MagicMock(),
                "boto3.s3.transfer": MagicMock(),
            },
        ):
            uploader = S3CheckpointUploader(bucket="test-bucket", prefix="test/prefix")
            result = uploader._upload_sync(ckpt_dir)

        assert result is True
        assert not os.path.exists(ckpt_dir), "Checkpoint dir should be deleted after S3 upload"
