"""
Unit tests for training device selection.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.utils.device import device_module, device_type


def _mock_accelerator(monkeypatch, accelerator_type):
    """Make torch.accelerator report ``accelerator_type``, or no accelerator when it is None."""
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: accelerator_type is not None)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: SimpleNamespace(type=accelerator_type))


@pytest.mark.parametrize(
    "accelerator_type, expected",
    [
        ("cuda", "cuda"),
        ("xpu", "xpu"),
        # Accelerators that are not enabled for training fall back to cuda.
        ("mps", "cuda"),
        # Hosts without an accelerator fall back to cuda.
        (None, "cuda"),
    ],
)
def test_device_type(monkeypatch, accelerator_type, expected):
    _mock_accelerator(monkeypatch, accelerator_type)
    assert device_type() == expected


def test_device_type_without_torch_accelerator(monkeypatch):
    monkeypatch.delattr(torch, "accelerator")
    assert device_type() == "cuda"


@pytest.mark.parametrize("accelerator_type, expected", [("cuda", torch.cuda), ("xpu", torch.xpu), (None, torch.cuda)])
def test_device_module(monkeypatch, accelerator_type, expected):
    _mock_accelerator(monkeypatch, accelerator_type)
    assert device_module() is expected
