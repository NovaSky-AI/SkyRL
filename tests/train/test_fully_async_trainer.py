from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import skyrl.train.fully_async_trainer as fully_async_module
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    _AsyncStalenessManager,
)
from tests.train.util import example_dummy_config


class UidDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"uid": str(idx)}

    def collate_fn(self, batch):
        return batch


class DummyGenerator:
    async def generate(self, generator_input, **kwargs):
        return {"generator_input": generator_input}


class DummyDispatch:
    def __init__(self):
        self.forward_backward_calls = []
        self.optim_step_calls = []

    def stage_data(self, data):
        return data

    def forward_backward_from_staged(self, model, data_ref, start_idx, end_idx):
        self.forward_backward_calls.append((model, start_idx, end_idx))
        return {"loss": float(end_idx - start_idx)}

    def optim_step(self, model):
        self.optim_step_calls.append(model)
        return 1.0

    def empty_cache(self):
        return None


def make_fully_async_config(train_batch_size: int, policy_mini_batch_size: int, num_workers: int):
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = train_batch_size
    cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
    cfg.trainer.epochs = 1
    cfg.trainer.eval_interval = 0
    cfg.trainer.ckpt_interval = 0
    cfg.trainer.hf_save_interval = 0
    cfg.trainer.eval_before_train = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.fully_async.num_parallel_generation_workers = num_workers
    cfg.trainer.fully_async.max_staleness_steps = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.inference_engine.async_engine = True
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.batched = False
    return cfg


def make_trainer(cfg, dataset_size: int):
    tracker = MagicMock()
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"
    inference_engine_client = MagicMock()
    inference_engine_client.pause_generation = AsyncMock()
    inference_engine_client.resume_generation = AsyncMock()
    return FullyAsyncRayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        train_dataset=UidDataset(dataset_size),
        eval_dataset=None,
        inference_engine_client=inference_engine_client,
        generator=DummyGenerator(),
    )


def make_dummy_training_input(batch_size: int) -> TrainingInputBatch:
    data = TrainingInputBatch(
        {
            "sequences": torch.ones((batch_size, 1), dtype=torch.int64),
            "rewards": torch.ones((batch_size, 1), dtype=torch.float32),
        }
    )
    data.metadata = {"uids": [str(i) for i in range(batch_size)]}
    return data


def test_fully_async_outer_batch_accounting_uses_train_batch_size():
    cfg = make_fully_async_config(train_batch_size=4, policy_mini_batch_size=2, num_workers=4)
    trainer = make_trainer(cfg, dataset_size=10)

    assert trainer.num_steps_per_epoch == 2
    assert trainer.total_training_steps == 2
    assert trainer.async_train_dataloader._effective_dataloader_length == 8
    assert trainer.num_policy_minibatches_per_outer_step == 2
    assert trainer._staleness_manager.mini_batch_size == 4


def test_async_staleness_manager_uses_outer_train_batch_capacity():
    manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=100,
        mini_batch_size=4,
        max_staleness_steps=1,
    )

    assert manager._compute_capacity_unlocked() == 8

    manager.load_state_from_checkpoint(global_step=3)
    assert manager._stat.accepted == 8
    assert manager._stat.submitted == 8
    assert manager._compute_capacity_unlocked() == 8


@pytest.mark.asyncio
async def test_fully_async_train_aggregates_outer_batch_and_runs_inner_minibatches(monkeypatch):
    cfg = make_fully_async_config(train_batch_size=4, policy_mini_batch_size=1, num_workers=4)
    trainer = make_trainer(cfg, dataset_size=4)
    trainer.dispatch = DummyDispatch()
    trainer.init_weight_sync_state = MagicMock()
    trainer.async_sync_policy_weights_to_inference_engines = AsyncMock()
    trainer.fwd_logprobs_values_reward = MagicMock(side_effect=lambda data: data)
    trainer.compute_advantages_and_returns = MagicMock(side_effect=lambda data: data)

    monkeypatch.setattr(
        fully_async_module,
        "prepare_generator_input",
        lambda prompts, *args, **kwargs: ({"prompts": prompts}, [prompts[0]["uid"]]),
    )

    aggregated_group_counts = []

    def convert_generation_group_train_batch_to_training_input(generated_groups):
        aggregated_group_counts.append(len(generated_groups))
        return make_dummy_training_input(batch_size=len(generated_groups))

    trainer.convert_generation_group_train_batch_to_training_input = convert_generation_group_train_batch_to_training_input

    await trainer.train()

    assert aggregated_group_counts == [4]
    assert trainer.dispatch.forward_backward_calls == [
        ("policy", 0, 1),
        ("policy", 1, 2),
        ("policy", 2, 3),
        ("policy", 3, 4),
    ]
    assert trainer.dispatch.optim_step_calls == ["policy", "policy", "policy", "policy"]
    trainer.inference_engine_client.pause_generation.assert_awaited_once()
    trainer.inference_engine_client.resume_generation.assert_awaited_once()
    assert trainer.async_sync_policy_weights_to_inference_engines.await_count == 2


@pytest.mark.asyncio
async def test_fully_async_resume_uses_train_batch_size_for_consumed_uids():
    cfg = make_fully_async_config(train_batch_size=4, policy_mini_batch_size=1, num_workers=4)
    trainer = make_trainer(cfg, dataset_size=4)
    trainer.dispatch = DummyDispatch()
    trainer.init_weight_sync_state = MagicMock()
    trainer.async_sync_policy_weights_to_inference_engines = AsyncMock()
    trainer.load_checkpoints = MagicMock(
        return_value=(1, "/tmp/global_step_1", {"0", "1", "2", "3"})
    )
    trainer.async_train_dataloader.load_state_from_checkpoint = MagicMock()
    trainer._staleness_manager.load_state_from_checkpoint = MagicMock()

    await trainer.train()

    trainer.async_train_dataloader.load_state_from_checkpoint.assert_called_once_with({"0", "1", "2", "3"})
    trainer._staleness_manager.load_state_from_checkpoint.assert_called_once_with(2)
