from unittest.mock import MagicMock

import pytest


def _make_mock_dispatch() -> MagicMock:
    """Worker-dispatch mock for SFT orchestration tests."""
    step_output = MagicMock()
    step_output.metrics = {"loss": 0.42, "final_loss": 0.42}
    eval_output = MagicMock()
    eval_output.metrics = {"loss": 0.31}

    dispatch_mock = MagicMock()
    dispatch_mock.forward_backward = MagicMock(return_value=step_output)
    dispatch_mock.optim_step = MagicMock(return_value=1.0)
    dispatch_mock.forward = MagicMock(return_value=eval_output)
    dispatch_mock.dp_size = MagicMock(return_value=1)
    return dispatch_mock


@pytest.fixture
def mock_dispatch() -> MagicMock:
    return _make_mock_dispatch()
