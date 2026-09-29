"""Compare native resume against uninterrupted shuffled dataloader iteration.

The real trainer loop and checkpoint serializers run on CPU. Generation and
distributed optimization are stubbed; this checks data order and update counts,
not training numerics.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.callbacks import TrainingCallback
from tests.train.test_rl_callbacks import (
    DummyDataset,
    _build_test_cfg,
    _stub_training_input,
)


class CheckpointInterruption(RuntimeError):
    pass


class InterruptAfterSave(TrainingCallback):
    def __init__(self, step):
        self.step = step

    def on_save(self, trainer, ci, control):
        if ci.global_step == self.step:
            raise CheckpointInterruption("checkpoint saved")


def _trainer(
    directory,
    monkeypatch,
    *,
    size,
    epochs,
    sampling_batches=1,
    interrupt=None,
    resume=None,
    interval=1,
    max_steps=None,
):
    cfg = _build_test_cfg()
    cfg.trainer.epochs = epochs
    cfg.trainer.ckpt_path = str(directory)
    cfg.trainer.ckpt_interval = interval
    cfg.trainer.max_training_steps = max_steps
    cfg.trainer.eval_interval = -1
    cfg.trainer.print_example_interval = 0
    cfg.trainer.algorithm.dynamic_sampling.type = (
        "filter" if sampling_batches > 1 else None
    )
    if resume is not None:
        cfg.trainer.resume_mode = "from_path"
        cfg.trainer.resume_path = str(resume)
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=DummyDataset(size=size),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
        callbacks=[InterruptAfterSave(interrupt)] if interrupt is not None else [],
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.save_weights_for_sampler = AsyncMock(return_value=None)
    trainer.dispatch.get_lcm_dp_size.return_value = 1
    monkeypatch.setattr(trainer, "init_weight_sync_state", lambda: None)
    monkeypatch.setattr(trainer, "_cleanup_old_checkpoints", lambda: None)
    monkeypatch.setattr(
        trainer, "postprocess_generator_output", lambda output, uids: (output, uids)
    )
    monkeypatch.setattr(
        trainer, "convert_to_training_input", lambda *_args: _stub_training_input()
    )
    monkeypatch.setattr(trainer, "fwd_logprobs_values_reward", lambda batch: batch)
    monkeypatch.setattr(trainer, "compute_advantages_and_returns", lambda batch: batch)
    monkeypatch.setattr(
        trainer, "train_critic_and_policy", lambda batch: {"policy_loss": 0.0}
    )
    monkeypatch.setattr(
        "skyrl.train.trainer.prepare_generator_input",
        lambda rows, *_args: ({"prompts": rows}, [str(i) for i in range(len(rows))]),
    )
    seen = []

    async def generate(batch):
        seen.append(
            (
                trainer._current_epoch,
                tuple(row[0][0]["content"] for row in batch["prompts"]),
            )
        )
        return {"rollout_metrics": None, "response_ids": [[1]], "rewards": [0.0]}

    def dynamic_sampling(output, uids):
        count = (trainer.dynamic_sampling_state or 0) + 1
        keep_sampling = count < sampling_batches
        trainer.dynamic_sampling_state = count if keep_sampling else None
        return output, uids, keep_sampling

    monkeypatch.setattr(trainer, "generate", generate)
    if sampling_batches > 1:
        monkeypatch.setattr(trainer, "handle_dynamic_sampling", dynamic_sampling)
    return trainer, seen


@pytest.mark.parametrize(
    "size,epochs,sampling_batches,interrupt",
    [(2, 2, 1, 1), (2, 2, 1, 2), (6, 2, 1, 1), (6, 2, 1, 3), (8, 3, 2, 3)],
)
def test_resume_matches_uninterrupted_batches_and_steps(
    tmp_path, monkeypatch, size, epochs, sampling_batches, interrupt
):
    kwargs = dict(size=size, epochs=epochs, sampling_batches=sampling_batches)
    baseline, expected = _trainer(tmp_path / "baseline", monkeypatch, **kwargs)
    asyncio.run(baseline.train())

    interrupted, before = _trainer(
        tmp_path / "resumed", monkeypatch, interrupt=interrupt, **kwargs
    )
    with pytest.raises(CheckpointInterruption):
        asyncio.run(interrupted.train())
    checkpoint = tmp_path / "resumed" / f"global_step_{interrupt}"
    resumed, after = _trainer(
        tmp_path / "resumed", monkeypatch, resume=checkpoint, **kwargs
    )
    asyncio.run(resumed.train())

    assert before + after == expected
    assert resumed.global_step == baseline.global_step


def test_resume_after_iterator_exhaustion_starts_the_next_epoch(tmp_path, monkeypatch):
    # Four data batches become two optimizer updates, so the final safety-net
    # checkpoint runs after exhaustion rather than at an in-loop save boundary.
    kwargs = dict(size=8, sampling_batches=2, interval=9999)
    baseline, expected = _trainer(
        tmp_path / "baseline", monkeypatch, epochs=2, **kwargs
    )
    asyncio.run(baseline.train())
    first, before = _trainer(tmp_path / "resumed", monkeypatch, epochs=1, **kwargs)
    asyncio.run(first.train())
    checkpoint = Path(first.cfg.trainer.ckpt_path) / "global_step_2"
    resumed, after = _trainer(
        tmp_path / "resumed", monkeypatch, epochs=2, resume=checkpoint, **kwargs
    )
    asyncio.run(resumed.train())
    assert before + after == expected
    assert resumed.global_step == baseline.global_step == 4


@pytest.mark.parametrize("size,limit", [(2, 1), (6, 1), (6, 3)])
def test_completed_step_limit_does_not_train_until_raised(
    tmp_path, monkeypatch, size, limit
):
    kwargs = dict(size=size, epochs=3)
    baseline, expected = _trainer(
        tmp_path / "baseline", monkeypatch, max_steps=limit + 1, **kwargs
    )
    asyncio.run(baseline.train())
    first, before = _trainer(
        tmp_path / "resumed", monkeypatch, max_steps=limit, **kwargs
    )
    asyncio.run(first.train())
    checkpoint = tmp_path / "resumed" / f"global_step_{limit}"

    completed, extra = _trainer(
        tmp_path / "resumed", monkeypatch, max_steps=limit, resume=checkpoint, **kwargs
    )
    asyncio.run(completed.train())
    assert extra == []
    assert completed.global_step == limit

    resumed, after = _trainer(
        tmp_path / "resumed",
        monkeypatch,
        max_steps=limit + 1,
        resume=checkpoint,
        **kwargs,
    )
    asyncio.run(resumed.train())
    assert before + after == expected
    assert resumed.global_step == baseline.global_step == limit + 1
