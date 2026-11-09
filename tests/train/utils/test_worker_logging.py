"""Tests for the Ray worker logging bridge."""

import logging
from types import SimpleNamespace

import pytest

from skyrl.train.utils import utils


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        ("INFO", [("WARNING", "request failed"), ("INFO", "training started")]),
        (
            "DEBUG",
            [("DEBUG", "request succeeded"), ("WARNING", "request failed"), ("INFO", "training started")],
        ),
    ],
)
def test_httpx_info_is_debug_in_ray_worker(monkeypatch, log_level, expected):
    emitted = []

    class StubLoguru:
        def remove(self):
            pass

        def level(self, name, **kwargs):
            return SimpleNamespace(name=name)

        def add(self, sink, *, level, **kwargs):
            self.min_level = getattr(logging, level)

        def opt(self, **kwargs):
            return self

        def log(self, level, message):
            if getattr(logging, level) >= self.min_level:
                emitted.append((level, message))

    monkeypatch.setattr(utils, "logger", StubLoguru())
    monkeypatch.setattr(logging.root, "handlers", logging.root.handlers[:])
    monkeypatch.setattr(logging.root, "level", logging.root.level)
    monkeypatch.setenv("LOG_LEVEL", log_level)

    utils.configure_ray_worker_logging()
    handler = logging.root.handlers[0]
    for name, level, message in [
        ("httpx._client", logging.INFO, "request succeeded"),
        ("httpx._client", logging.WARNING, "request failed"),
        ("skyrl.train", logging.INFO, "training started"),
    ]:
        handler.emit(logging.LogRecord(name, level, __file__, 1, message, (), None))

    assert emitted == expected
