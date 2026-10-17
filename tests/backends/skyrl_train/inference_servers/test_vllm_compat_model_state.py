"""CPU tests for carrying ``model_runner.model_state`` tensors across a level-2 sleep.

A level-2 sleep discards the weights pool, which also holds Model Runner V2's per-model input
state; vLLM restores only the model's buffers. Qwen4Exp's ``ngram_context_offsets`` came back as
zeros. Fake workers stand in for vLLM so no GPU is needed.
"""

import types

import torch

from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
    patch_vllm_level2_sleep_model_state,
    restore_model_state_tensors,
    save_model_state_tensors,
)


def _runner():
    state = types.SimpleNamespace(ngram_context_offsets=torch.arange(-2, 0), ngram_context_len=2)
    return types.SimpleNamespace(model_state=state)


def _fake_worker_cls():
    class Worker:
        def __init__(self):
            self.model_runner = _runner()

        def sleep(self, level=1):
            if level == 2:  # what the allocator does to the weights pool
                self.model_runner.model_state.ngram_context_offsets.zero_()

        def wake_up(self, tags=None):
            pass

    return Worker


def test_save_restore_round_trip():
    runner = _runner()
    saved = save_model_state_tensors(runner)
    assert list(saved) == ["ngram_context_offsets"]
    runner.model_state.ngram_context_offsets.zero_()
    assert restore_model_state_tensors(runner, saved) == 1
    assert runner.model_state.ngram_context_offsets.tolist() == [-2, -1]


def test_runner_without_model_state_is_a_no_op():
    runner = types.SimpleNamespace()
    assert save_model_state_tensors(runner) == {}
    assert restore_model_state_tensors(runner, {"x": torch.zeros(1)}) == 0


def test_level2_sleep_patch_restores_on_weights_wake():
    cls = _fake_worker_cls()
    assert patch_vllm_level2_sleep_model_state(cls)
    assert not patch_vllm_level2_sleep_model_state(cls)  # idempotent
    worker = cls()
    worker.sleep(level=2)
    worker.wake_up(tags=["kv_cache"])
    assert worker.model_runner.model_state.ngram_context_offsets.tolist() == [0, 0]
    worker.wake_up(tags=["weights"])
    assert worker.model_runner.model_state.ngram_context_offsets.tolist() == [-2, -1]


def test_level1_sleep_saves_nothing():
    cls = _fake_worker_cls()
    patch_vllm_level2_sleep_model_state(cls)
    worker = cls()
    worker.sleep(level=1)
    worker.model_runner.model_state.ngram_context_offsets.fill_(7)
    worker.wake_up()
    assert worker.model_runner.model_state.ngram_context_offsets.tolist() == [7, 7]
