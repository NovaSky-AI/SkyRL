"""CPU tests for ``patch_lora_dummy_run_resident_adapter``: env gating, and which path a dummy run takes."""

import contextlib

import numpy as np
import pytest


@pytest.mark.vllm
class TestLoraDummyRunResidentAdapter:
    @pytest.fixture(autouse=True)
    def fresh(self, monkeypatch):
        pytest.importorskip("vllm")
        from vllm.v1.worker import gpu_worker
        from vllm.v1.worker import lora_model_runner_mixin as mixin_mod

        from skyrl.backends.skyrl_train.patches.vllm import (
            patch_lora_dummy_run_resident_adapter as mod,
        )

        self.orig_calls = []

        @contextlib.contextmanager
        def fake_orig(
            runner,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=0,
            *args,
            **kwargs,
        ):
            self.orig_calls.append((remove_lora, num_active_loras))
            yield

        def fake_execute_dummy_batch(worker):
            worker.seen_flag = getattr(worker.model_runner, mod._FLAG, False)
            with worker.model_runner.maybe_dummy_run_with_lora(
                worker.model_runner.lora_config, np.array([1], dtype=np.int32), None, True, 1
            ):
                pass

        monkeypatch.setattr(mixin_mod.LoRAModelRunnerMixin, "maybe_dummy_run_with_lora", fake_orig)
        monkeypatch.setattr(gpu_worker.Worker, "execute_dummy_batch", fake_execute_dummy_batch)
        monkeypatch.setattr(mod, "_APPLIED", False)
        self.mod, self.mixin_mod, self.gpu_worker = mod, mixin_mod, gpu_worker

    def _runner(self, resident):
        mixin = self.mixin_mod.LoRAModelRunnerMixin

        class FakeManager:
            def list_adapters(self):
                return set(resident)

        class FakeConfig:
            max_loras = 1

        class Runner(mixin):
            def __init__(self):
                self.lora_manager = FakeManager()
                self.lora_config = FakeConfig()
                self.set_calls = []

            def _set_active_loras(self, sample_mapping, token_mapping, requests, *args):
                self.set_calls.append((sample_mapping, token_mapping, {r.lora_int_id for r in requests}))

        return Runner()

    def _worker(self, runner):
        worker = object.__new__(self.gpu_worker.Worker)
        worker.model_runner = runner
        return worker

    def test_off_without_env(self, monkeypatch):
        monkeypatch.delenv(self.mod.ENV, raising=False)
        assert self.mod.apply_lora_dummy_run_resident_adapter_patch() is False

    def test_runtime_dummy_step_uses_resident_adapter(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        assert self.mod.apply_lora_dummy_run_resident_adapter_patch() is True
        runner = self._runner(resident={7})
        worker = self._worker(runner)
        self.gpu_worker.Worker.execute_dummy_batch(worker)
        assert worker.seen_flag is True
        assert self.orig_calls == []  # no warmup adapters built, nothing removed
        assert runner.set_calls == [((7,), (7,), {7})]
        assert getattr(runner, self.mod._FLAG) is False  # flag cleared after the step

    def test_runtime_dummy_step_without_resident_adapter_keeps_vllm_path(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        self.mod.apply_lora_dummy_run_resident_adapter_patch()
        runner = self._runner(resident=set())
        self.gpu_worker.Worker.execute_dummy_batch(self._worker(runner))
        assert self.orig_calls == [(True, 1)]
        assert runner.set_calls == []

    def test_warmup_and_capture_keep_vllm_path(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        self.mod.apply_lora_dummy_run_resident_adapter_patch()
        runner = self._runner(resident={7})  # flag not set: not a runtime dummy step
        with runner.maybe_dummy_run_with_lora(runner.lora_config, np.array([2, 3], dtype=np.int32), None, True, 1):
            pass
        assert self.orig_calls == [(True, 1)]
        assert runner.set_calls == []

    def test_no_active_loras_maps_to_no_adapter(self, monkeypatch):
        monkeypatch.setenv(self.mod.ENV, "1")
        self.mod.apply_lora_dummy_run_resident_adapter_patch()
        runner = self._runner(resident={7})
        setattr(runner, self.mod._FLAG, True)
        with runner.maybe_dummy_run_with_lora(runner.lora_config, np.array([2, 1], dtype=np.int32), None, True, 0):
            pass
        assert runner.set_calls == [((0, 0), (0, 0, 0), set())]
