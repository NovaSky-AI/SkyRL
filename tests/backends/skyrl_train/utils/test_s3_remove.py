"""Check S3 deletion against s3fs's handling of per-object API errors."""

from unittest.mock import AsyncMock, Mock

import pytest
from s3fs import S3FileSystem

from skyrl.backends.skyrl_train.utils.io import io


@pytest.fixture
def filesystem(monkeypatch):
    fs = S3FileSystem(anon=True, skip_instance_cache=True)
    monkeypatch.setattr(fs, "isdir", Mock(return_value=True))
    monkeypatch.setattr(fs, "_expand_path", AsyncMock(return_value=["bucket/checkpoint/model.pt"]))
    monkeypatch.setattr(fs, "_call_s3", AsyncMock(return_value={"Deleted": [{"Key": "checkpoint/model.pt"}]}))
    monkeypatch.setattr(fs, "find", Mock(return_value=[]))
    monkeypatch.setattr(io, "_get_filesystem", lambda path: fs)
    return fs


@pytest.mark.parametrize("recursive", [False, True])
def test_remove_rejects_per_object_errors_from_successful_s3_response(filesystem, recursive):
    filesystem.isdir.return_value = recursive
    filesystem._call_s3.return_value = {"Errors": [{"Key": "checkpoint/model.pt", "Code": "AccessDenied"}]}
    filesystem.find.return_value = ["bucket/checkpoint/model.pt"]

    # Real s3fs rm/_bulk_delete returns normally for this response. SkyRL must
    # detect the surviving object instead of announcing successful retention.
    with pytest.raises(OSError, match=r"1 object\(s\) remain.*s3:DeleteObject"):
        io.remove("s3://bucket/checkpoint")

    assert filesystem._call_s3.await_args.args[0] == "delete_objects"


@pytest.mark.parametrize("recursive", [False, True])
def test_remove_accepts_confirmed_s3_deletion(filesystem, recursive):
    filesystem.isdir.return_value = recursive

    io.remove("s3://bucket/checkpoint")

    assert filesystem._call_s3.await_args.args[0] == "delete_objects"


def test_remove_propagates_failed_verification(filesystem):
    filesystem.find.side_effect = PermissionError("cannot list checkpoint")

    with pytest.raises(PermissionError, match="cannot list checkpoint"):
        io.remove("s3://bucket/checkpoint")


def test_remove_propagates_failed_delete(filesystem):
    filesystem._call_s3.side_effect = PermissionError("cannot delete checkpoint")

    with pytest.raises(PermissionError, match="cannot delete checkpoint"):
        io.remove("s3://bucket/checkpoint")

    filesystem.find.assert_not_called()
