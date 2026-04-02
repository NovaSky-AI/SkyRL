"""
uv run --isolated --extra dev pytest tests/train/test_fully_async_trainer.py
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from tests.train.util import example_dummy_config


class DummyPromptDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "prompt": [{"role": "user", "content": f"question-{idx}"}],
            "env_class": None,
            "env_extras": {"idx": idx},
            "uid": f"uid-{idx}",
        }

    def collate_fn(self, batch):
        return batch


class DummyGenerator:
    async def generate(self, input_batch, disable_tqdm=False):  # noqa: ARG002
        return {
            "response_ids": [[1]],
            "rollout_metrics": {},
        }


def make_fully_async_config():
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.eval_interval = -1
    cfg.trainer.ckpt_interval = -1
    cfg.trainer.hf_save_interval = -1
    cfg.trainer.fully_async.max_staleness_steps = 0
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.generator.inference_engine.enable_http_endpoint = True
    cfg.generator.batched = False
    return cfg


def make_trainer(cfg, dataset_size: int):
    cfg.trainer.fully_async.num_parallel_generation_workers = cfg.trainer.train_batch_size
    return FullyAsyncRayPPOTrainer(
        cfg=cfg,
        tracker=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=DummyPromptDataset(dataset_size),
        eval_dataset=None,
        inference_engine_client=MagicMock(),
        generator=DummyGenerator(),
    )


def test_streaming_mode_requires_zero_staleness():
    cfg = make_fully_async_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.fully_async.max_staleness_steps = 1

    with pytest.raises(AssertionError, match="max_staleness_steps == 0"):
        make_trainer(cfg, dataset_size=4)


def test_streaming_mode_requires_single_update_epoch():
    cfg = make_fully_async_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.update_epochs_per_batch = 2

    with pytest.raises(AssertionError, match="update_epochs_per_batch == 1"):
        make_trainer(cfg, dataset_size=4)


def test_streaming_mode_disallows_advantage_batch_normalization():
    cfg = make_fully_async_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.algorithm.advantage_batch_normalize = True

    with pytest.raises(AssertionError, match="advantage_batch_normalize == false"):
        make_trainer(cfg, dataset_size=4)


def test_streaming_mode_requires_matching_critic_and_policy_minibatches():
    cfg = make_fully_async_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.critic.model.path = "critic-model"
    cfg.trainer.critic_mini_batch_size = 4

    with pytest.raises(AssertionError, match="critic_mini_batch_size == trainer.policy_mini_batch_size"):
        make_trainer(cfg, dataset_size=4)


def test_streaming_mode_uses_logical_batch_for_step_accounting():
    cfg = make_fully_async_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 2

    trainer = make_trainer(cfg, dataset_size=10)

    assert trainer.mini_steps_per_sync == 2
    assert trainer.num_steps_per_epoch == 2
    assert trainer.total_training_steps == 2
    assert trainer.async_train_dataloader._effective_dataloader_length == 8


@pytest.mark.asyncio
async def test_streaming_mode_trains_multiple_minibatches_before_sync():
    cfg = make_fully_async_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 2

    inference_engine_client = MagicMock()
    inference_engine_client.pause_generation = AsyncMock()
    inference_engine_client.resume_generation = AsyncMock()
    tracker = MagicMock()

    trainer = FullyAsyncRayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=MagicMock(),
        train_dataset=DummyPromptDataset(4),
        eval_dataset=None,
        inference_engine_client=inference_engine_client,
        generator=DummyGenerator(),
    )

    trainer.init_weight_sync_state = MagicMock()
    trainer.async_sync_policy_weights_to_inference_engines = AsyncMock()

    convert_call_sizes = []

    def fake_convert(generation_groups):
        convert_call_sizes.append(len(generation_groups))
        return "training-input"

    async def fake_run_training(training_input, mini_step_idx=None):  # noqa: ARG001
        trainer.all_metrics.update({"policy/fake_loss": float(mini_step_idx + 1)})
        return {"fake_loss": float(mini_step_idx + 1)}

    trainer.convert_generation_group_mini_batch_to_training_input = MagicMock(side_effect=fake_convert)
    trainer._run_training = AsyncMock(side_effect=fake_run_training)

    await trainer.train()

    assert convert_call_sizes == [2, 2]
    assert [call.kwargs["mini_step_idx"] for call in trainer._run_training.await_args_list] == [0, 1]
    assert trainer.async_sync_policy_weights_to_inference_engines.await_count == 2
    inference_engine_client.pause_generation.assert_awaited_once()
    inference_engine_client.resume_generation.assert_awaited_once()
