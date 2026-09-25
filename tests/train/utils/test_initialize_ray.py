import threading

from skyrl.train.utils import utils


def test_ray_init_watchdog_exits_after_timeout(monkeypatch):
    exited = threading.Event()
    exit_codes = []

    def fake_exit(code):
        exit_codes.append(code)
        exited.set()

    monkeypatch.setattr(utils.faulthandler, "dump_traceback", lambda **kwargs: None)
    monkeypatch.setattr(utils.os, "_exit", fake_exit)

    completed = utils._start_ray_init_watchdog(0.01)

    assert exited.wait(timeout=1)
    assert exit_codes == [1]
    completed.set()


def test_ray_init_watchdog_can_be_disabled(monkeypatch):
    monkeypatch.setattr(utils.os, "_exit", lambda code: (_ for _ in ()).throw(AssertionError(code)))

    completed = utils._start_ray_init_watchdog(0)

    assert not completed.is_set()
