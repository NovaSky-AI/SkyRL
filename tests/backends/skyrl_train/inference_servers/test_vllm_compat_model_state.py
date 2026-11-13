"""CPU tests for carrying lost ``model_runner.model_state`` tensors across a vLLM worker sleep.

Model Runner V2's per-model input state lives in the weights pool, but vLLM's wake path restores
only the model's parameters and buffers. Qwen4Exp's ``ngram_context_offsets`` came back as zeros
after a level-2 sleep; only the tensors listed in ``MODEL_STATE_TENSORS_LOST_ON_SLEEP`` are
carried. Fake workers stand in for vLLM so no GPU is needed.
"""

import types

import torch

from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
    patch_vllm_sleep_model_state,
    restore_model_state_tensors,
    save_model_state_tensors,
)


def _runner():
    state = types.SimpleNamespace(ngram_context_offsets=torch.arange(-2, 0), ngram_context_len=2, scratch=torch.ones(3))
    return types.SimpleNamespace(model_state=state)


def _fake_worker_cls():
    class Worker:
        def __init__(self):
            self.model_runner = _runner()

        def sleep(self, level=1):
            # What losing the weights pool does to model_state.
            self.model_runner.model_state.ngram_context_offsets.zero_()
            self.model_runner.model_state.scratch.zero_()

        def wake_up(self, tags=None):
            pass

    return Worker


def test_save_restore_round_trip_covers_only_listed_tensors():
    runner = _runner()
    saved = save_model_state_tensors(runner)
    assert list(saved) == ["ngram_context_offsets"]  # scratch is not listed
    runner.model_state.ngram_context_offsets.zero_()
    assert restore_model_state_tensors(runner, saved) == 1
    assert runner.model_state.ngram_context_offsets.tolist() == [-2, -1]


def test_models_without_listed_tensors_are_untouched():
    runner = types.SimpleNamespace(model_state=types.SimpleNamespace(scratch=torch.ones(3)))
    assert save_model_state_tensors(runner) == {}
    assert restore_model_state_tensors(runner, save_model_state_tensors(runner)) == 0


def test_runner_without_model_state_is_a_no_op():
    runner = types.SimpleNamespace()
    assert save_model_state_tensors(runner) == {}
    assert restore_model_state_tensors(runner, {"x": torch.zeros(1)}) == 0


def test_sleep_patch_restores_on_weights_wake():
    cls = _fake_worker_cls()
    assert patch_vllm_sleep_model_state(cls)
    assert not patch_vllm_sleep_model_state(cls)  # idempotent
    for level in (1, 2):
        worker = cls()
        worker.sleep(level=level)
        worker.wake_up(tags=["kv_cache"])
        assert worker.model_runner.model_state.ngram_context_offsets.tolist() == [0, 0]
        worker.wake_up(tags=["weights"])
        assert worker.model_runner.model_state.ngram_context_offsets.tolist() == [-2, -1]
        assert worker.model_runner.model_state.scratch.tolist() == [0.0, 0.0, 0.0]  # not carried


def test_full_wake_restores():
    cls = _fake_worker_cls()
    patch_vllm_sleep_model_state(cls)
    worker = cls()
    worker.sleep(level=2)
    worker.wake_up()
    assert worker.model_runner.model_state.ngram_context_offsets.tolist() == [-2, -1]
