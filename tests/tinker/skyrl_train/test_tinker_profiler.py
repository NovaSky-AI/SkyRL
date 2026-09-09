from types import SimpleNamespace

from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend


class _Dispatch:
    def __init__(self):
        self.calls = []

    def start_profile(self, role):
        self.calls.append(("start", role))

    def profile_step(self, role):
        self.calls.append(("step", role))

    def dump_profiler_summary(self, role):
        self.calls.append(("summary", role))
        return [{"window_count": 1, "pairs": [("gemm", 10.0)]}]


def _backend(profiler_enabled=True):
    backend = object.__new__(SkyRLTrainBackend)
    backend._cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(torch_profiler_config=SimpleNamespace(enable=profiler_enabled))
        )
    )
    backend._dispatch = _Dispatch()
    backend._tinker_profiler_started = False
    backend._tinker_profiler_windows = 0
    return backend


def test_tinker_profiler_spans_complete_optimizer_steps():
    backend = _backend()

    backend._start_tinker_profiler()
    backend._start_tinker_profiler()
    backend._step_tinker_profiler("policy")

    assert backend._dispatch.calls == [
        ("start", "policy"),
        ("step", "policy"),
        ("summary", "policy"),
    ]
    assert backend._tinker_profiler_windows == 1


def test_tinker_profiler_ignores_critic_and_disabled_policy():
    backend = _backend()
    backend._step_tinker_profiler("critic")
    disabled_backend = _backend(profiler_enabled=False)
    disabled_backend._start_tinker_profiler()

    assert backend._dispatch.calls == []
    assert disabled_backend._dispatch.calls == []
