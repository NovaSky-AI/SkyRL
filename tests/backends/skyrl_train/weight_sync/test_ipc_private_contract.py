"""Fail loudly when vLLM's packed IPC extension surface changes."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytestmark = pytest.mark.vllm

from vllm.distributed.weight_transfer import ipc_engine  # noqa: E402

from skyrl.backends.skyrl_train.weight_sync.ipc_lifetime import (  # noqa: E402
    FailClosedIPCLifetimeMixin,
)


def _assert_signatures(engine, producer):
    expected = {
        "_send_packed": ("self", "source"),
        "_all_gather_and_merge_handles": ("self", "handles"),
        "_do_send": ("self", "names", "dtype_names", "shapes", "ipc_handles", "tensor_sizes"),
        "_post_send_sync": (),
    }
    for name, parameters in expected.items():
        assert tuple(inspect.signature(getattr(engine, name)).parameters) == parameters, name
    assert tuple(inspect.signature(producer).parameters) == (
        "iterator",
        "gpu_uuid",
        "post_iter_func",
        "buffer_size_bytes",
    )


def _function_ast(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]


def test_installed_vllm_ipc_signatures_match_lifetime_override():
    _assert_signatures(ipc_engine.IPCTrainerWeightTransferEngine, ipc_engine.packed_ipc_producer)


def test_packed_loop_and_producer_arguments_match_installed_vllm():
    native = _function_ast(ipc_engine.IPCTrainerWeightTransferEngine._send_packed)
    protected = _function_ast(FailClosedIPCLifetimeMixin._send_packed)
    native_loop = next(node for node in native.body if isinstance(node, ast.For))
    protected_loop = next(node for node in protected.body if isinstance(node, ast.For))
    producer_assignment = next(
        node
        for node in protected.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "producer"
    )
    assert ast.dump(native_loop.iter) == ast.dump(producer_assignment.value)
    assert [ast.dump(node) for node in native_loop.body] == [ast.dump(node) for node in protected_loop.body]
    assert len(native.body) == 2, "Re-review vLLM's packed producer lifecycle before updating this contract"


def test_contract_rejects_a_changed_vllm_private_signature():
    native = ipc_engine.IPCTrainerWeightTransferEngine
    changed = SimpleNamespace(
        _send_packed=native._send_packed,
        _all_gather_and_merge_handles=native._all_gather_and_merge_handles,
        _do_send=lambda self, names: None,
        _post_send_sync=native._post_send_sync,
    )
    with pytest.raises(AssertionError, match="_do_send"):
        _assert_signatures(changed, ipc_engine.packed_ipc_producer)
