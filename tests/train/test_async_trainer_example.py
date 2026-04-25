import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.train.utils.trainer_utils import ResumeMode

async_trainer_module = importlib.import_module("examples.train.async.async_trainer")
AsyncRayPPOTrainer = async_trainer_module.AsyncRayPPOTrainer


class _DummyPbar:
    def update(self, n: int) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_async_trainer_offloads_periodic_and_final_save_calls(monkeypatch):
    trainer = AsyncRayPPOTrainer.__new__(AsyncRayPPOTrainer)
    trainer.colocate_all = False
    trainer.resume_mode = ResumeMode.NONE
    trainer.global_step = 0
    trainer.total_training_steps = 1
    trainer.train_dataloader = [object()]
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer.ref_model = None
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            eval_interval=0,
            eval_before_train=False,
            epochs=1,
            ckpt_interval=1,
            hf_save_interval=1,
            update_ref_every_epoch=False,
        )
    )
    trainer.tracker = MagicMock()
    trainer.init_weight_sync_state = MagicMock()
    trainer.async_sync_policy_weights_to_inference_engines = AsyncMock()
    trainer._run_training = AsyncMock(return_value={"status": "ok"})
    trainer._run_generate_loop = AsyncMock(return_value=None)
    trainer.save_checkpoints = MagicMock(side_effect=AssertionError("save_checkpoints should be offloaded"))
    trainer.save_models = MagicMock(side_effect=AssertionError("save_models should be offloaded"))

    offloaded_calls = []

    async def fake_to_thread(func, *args, **kwargs):
        offloaded_calls.append((func, args, kwargs))
        return None

    monkeypatch.setattr(async_trainer_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(async_trainer_module, "tqdm", lambda *args, **kwargs: _DummyPbar())

    await trainer.train()

    assert [call[0] for call in offloaded_calls] == [
        trainer.save_checkpoints,
        trainer.save_models,
        trainer.save_checkpoints,
        trainer.save_models,
    ]
    trainer.init_weight_sync_state.assert_called_once_with()
    assert trainer.async_sync_policy_weights_to_inference_engines.await_count == 2
    trainer._run_training.assert_awaited_once()
    trainer.tracker.finish.assert_called_once_with()
