"""Verify S3 removal through native s3fs methods and fake S3 responses."""

from unittest.mock import AsyncMock, Mock

import pytest
from botocore.exceptions import ClientError
from s3fs import S3FileSystem

from skyrl.backends.skyrl_train.utils.io import io


class S3Responses:
    def __init__(self):
        self.remaining_keys = []
        self.bucket_exists = False
        self.delete_response = {"Deleted": [{"Key": "checkpoint/model.pt"}]}
        self.errors = {}
        self.calls = []

    async def call(self, method, *args, **kwargs):
        self.calls.append((method, kwargs))
        if method in self.errors:
            raise self.errors[method]
        if method == "delete_objects":
            return self.delete_response
        if method == "delete_bucket":
            return {}
        if method == "head_object":
            if kwargs["Key"] in self.remaining_keys:
                return {"ContentLength": 1}
            raise FileNotFoundError(kwargs["Key"])
        if method == "head_bucket":
            if self.bucket_exists:
                return {}
            raise FileNotFoundError(kwargs["Bucket"])
        if method == "list_objects_v2":
            keys = [key for key in self.remaining_keys if key.startswith(kwargs["Prefix"])]
            keys = keys[: kwargs.get("MaxKeys", len(keys))]
            return {"KeyCount": len(keys), "Contents": [{"Key": key, "Size": 1} for key in keys]}
        raise AssertionError(f"Unexpected S3 method: {method}")

    def get_paginator(self, method):
        assert method == "list_objects_v2"
        return self

    async def paginate(self, **kwargs):
        yield await self.call("list_objects_v2", **kwargs)


@pytest.fixture
def filesystem(monkeypatch):
    fs = S3FileSystem(anon=True, skip_instance_cache=True)
    responses = S3Responses()
    monkeypatch.setattr(fs, "isdir", Mock(return_value=True))
    monkeypatch.setattr(fs, "_expand_path", AsyncMock(return_value=["bucket/checkpoint/model.pt"]))
    monkeypatch.setattr(fs, "_is_bucket_versioned", AsyncMock(return_value=False))
    monkeypatch.setattr(fs, "_call_s3", responses.call)
    # Pagination uses the native _find/_lsdir/_iterdir chain without a network client.
    monkeypatch.setattr(fs, "set_session", AsyncMock())
    monkeypatch.setattr(fs, "get_s3", AsyncMock(return_value=responses))
    monkeypatch.setattr(io, "_get_filesystem", lambda path: fs)
    return fs, responses


@pytest.fixture(
    params=[
        ("s3://bucket/checkpoint/model.pt", False, ["bucket/checkpoint/model.pt"]),
        ("s3://bucket/checkpoint", True, ["bucket/checkpoint/model.pt"]),
        ("s3://bucket", True, ["bucket"]),
    ],
    ids=["file", "prefix", "bucket"],
)
def target(request, filesystem):
    fs, _ = filesystem
    path, recursive, expanded = request.param
    fs.isdir.return_value = recursive
    fs._expand_path.return_value = expanded
    return path


def test_remove_accepts_confirmed_deletion(filesystem, target):
    _, responses = filesystem

    io.remove(target)

    methods = [method for method, _ in responses.calls]
    assert methods[0] == ("delete_bucket" if target == "s3://bucket" else "delete_objects")
    assert methods[1] == ("head_bucket" if target == "s3://bucket" else "head_object")


def test_remove_rejects_surviving_objects(filesystem, target):
    _, responses = filesystem
    responses.remaining_keys = ["checkpoint/model.pt"]
    responses.bucket_exists = True
    responses.delete_response = {"Errors": [{"Key": "checkpoint/model.pt", "Code": "AccessDenied"}]}

    with pytest.raises(OSError, match=r"(?i)object\(s\) remain.*s3:DeleteObject"):
        io.remove(target)


def test_remove_rejects_an_empty_bucket_that_still_exists(filesystem):
    fs, responses = filesystem
    fs._expand_path.return_value = ["bucket"]
    responses.bucket_exists = True

    with pytest.raises(OSError, match="remain"):
        io.remove("s3://bucket")


@pytest.mark.parametrize("remaining", [False, True])
def test_prefix_verification_uses_one_result(filesystem, remaining):
    _, responses = filesystem
    if remaining:
        responses.remaining_keys = [f"checkpoint/shard-{i}.pt" for i in range(1001)]

    if remaining:
        with pytest.raises(OSError, match="remain"):
            io.remove("s3://bucket/checkpoint")
    else:
        io.remove("s3://bucket/checkpoint")

    listings = [kwargs for method, kwargs in responses.calls if method == "list_objects_v2"]
    assert listings == [{"Bucket": "bucket", "Prefix": "checkpoint/", "Delimiter": "/", "MaxKeys": 1}]


@pytest.mark.parametrize("method", ["head_object", "list_objects_v2", "head_bucket"])
def test_remove_propagates_verification_permission_errors(filesystem, method):
    fs, responses = filesystem
    responses.errors[method] = ClientError({"Error": {"Code": "AccessDenied", "Message": "cannot verify"}}, method)
    path = "s3://bucket/checkpoint"
    if method == "head_bucket":
        path = "s3://bucket"
        fs._expand_path.return_value = ["bucket"]

    with pytest.raises(PermissionError, match="cannot verify"):
        io.remove(path)


@pytest.mark.parametrize("remaining", [False, True])
def test_verification_ignores_cached_presence(filesystem, monkeypatch, remaining):
    fs, responses = filesystem
    if remaining:
        responses.remaining_keys = ["checkpoint/model.pt"]
    # Keep a stale positive or negative even if deletion invalidates the cache.
    monkeypatch.setattr(fs, "invalidate_cache", lambda path: None)
    fs.dircache["bucket/checkpoint"] = (
        [] if remaining else [{"name": "bucket/checkpoint/model.pt", "type": "file", "size": 1}]
    )
    fs.isdir.return_value = False

    if remaining:
        with pytest.raises(OSError, match="remain"):
            io.remove("s3://bucket/checkpoint/model.pt")
    else:
        io.remove("s3://bucket/checkpoint/model.pt")

    assert any(method == "head_object" for method, _ in responses.calls)


def test_remove_propagates_failed_delete(filesystem):
    _, responses = filesystem
    responses.errors["delete_objects"] = PermissionError("cannot delete checkpoint")

    with pytest.raises(PermissionError, match="cannot delete checkpoint"):
        io.remove("s3://bucket/checkpoint")

    assert [method for method, _ in responses.calls] == ["delete_objects"]
