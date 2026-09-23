from unittest.mock import MagicMock

import pytest

from skyrl.train.config.sft_config import (
    SFTConfig,
    SFTPlacementConfig,
    build_skyrl_config_for_sft,
)
from skyrl.train.sft_trainer import SFTTrainer
from tests.train.sft.util import attach_mock_sft_deps


def _build_test_sft_config() -> SFTConfig:
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    # model.path / train_datasets are unused — we never load the model and
    # monkeypatch _load_and_tokenize. eval_datasets must be non-empty so
    # load_eval_datasets actually invokes _load_and_tokenize.
    cfg.model.path = "unused"
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    cfg.train_datasets = ["unused-monkeypatched"]
    cfg.train_dataset_splits = ["train"]
    cfg.eval_datasets = ["unused-monkeypatched"]
    cfg.eval_dataset_splits = ["train"]
    # Shorthand logging name: eval metrics land under eval/evalset/...
    cfg.eval_dataset_names = ["evalset"]
    cfg.eval_interval = 1
    cfg.eval_before_train = False
    cfg.num_steps = 2
    cfg.num_epochs = None
    cfg.batch_size = 1
    cfg.micro_train_batch_size_per_gpu = 1
    cfg.max_length = 16
    cfg.remove_microbatch_padding = False
    cfg.logger = "console"
    # ckpt_path must be truthy so the save block isn't gated out. The actual
    # save is monkeypatched below so nothing is written to disk.
    cfg.ckpt_path = "/fake/sft-callback-test"
    cfg.ckpt_interval = -1
    cfg.hf_save_interval = -1
    return cfg


def _dummy_tokenized() -> list[dict]:
    """A synthetic example (10 input tokens, 4 response tokens each) for SFT."""
    example = {
        "input_ids": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "attention_mask": [1] * 10,
        "num_actions": 4,
        "loss_mask": [1, 1, 1, 1],
    }
    return [example]


def _build_minimal_trainer(dispatch_mock: MagicMock) -> SFTTrainer:
    """Build an SFTTrainer with mocked dispatch."""
    cfg = _build_test_sft_config()
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)
    attach_mock_sft_deps(trainer, dispatch_mock)
    return trainer


def test_sft_train_step_opts_out_of_per_token_outputs(mock_dispatch):
    """train_step opts out of unused per-token outputs."""
    trainer = _build_minimal_trainer(mock_dispatch)
    batch = trainer.collator(_dummy_tokenized(), batch_size=1)

    trainer.train_step(batch, step=1)

    mock_dispatch.forward_backward.assert_called_once()
    call = mock_dispatch.forward_backward.call_args
    assert call.kwargs["loss_fn"] == "cross_entropy"
    assert call.kwargs["return_per_token_outputs"] is False


def test_sft_run_eval_opts_out_of_per_token_outputs(mock_dispatch):
    """run_eval reads only ``output.metrics["loss"]``; it skips per-token outputs."""
    trainer = _build_minimal_trainer(mock_dispatch)
    trainer.eval_dataloaders = [("evalset", trainer.build_eval_dataloader(_dummy_tokenized()))]

    metrics, _ = trainer.run_eval()

    assert "evalset/loss" in metrics
    mock_dispatch.forward.assert_called()
    for call in mock_dispatch.forward.call_args_list:
        assert call.kwargs["loss_fn"] == "cross_entropy"
        assert call.kwargs["return_per_token_outputs"] is False


@pytest.mark.parametrize("async_collation", [False, True])
@pytest.mark.parametrize(
    "packing,alignment,expected_sizes",
    [(False, False, [20, 20, 4]), (True, False, [20, 20, 4]), (True, True, [22, 18, 4])],
)
def test_sft_logs_actual_example_count(mock_dispatch, monkeypatch, packing, alignment, expected_sizes, async_collation):
    cfg = _build_test_sft_config()
    cfg.strategy = "megatron" if packing else "fsdp"
    cfg.placement.num_gpus_per_node = 2
    cfg.megatron_config.tensor_model_parallel_size = 1
    cfg.megatron_config.pipeline_model_parallel_size = 1
    cfg.batch_size = 20
    cfg.max_length = 1_000
    cfg.num_steps = 3
    cfg.sampler = "sequential"
    cfg.remove_microbatch_padding = packing
    cfg.use_sequence_packing = packing
    cfg.align_packing_bins_to_dp = alignment
    cfg.packing_batch_size_allowed_variation = 0.10
    cfg.async_batch_collation = async_collation
    cfg.eval_datasets = None
    cfg.eval_dataset_splits = None
    cfg.eval_dataset_names = None
    cfg.eval_interval = 0
    cfg.ckpt_path = ""
    trainer = SFTTrainer(cfg, skyrl_cfg=build_skyrl_config_for_sft(cfg))
    mock_dispatch.dp_size.return_value = 2
    attach_mock_sft_deps(trainer, mock_dispatch)
    trainer.tracker = MagicMock()
    example = {
        "input_ids": list(range(300)),
        "attention_mask": [1] * 300,
        "num_actions": 4,
        "loss_mask": [1] * 4,
    }
    monkeypatch.setattr(trainer, "_load_and_tokenize", lambda *_args, **_kwargs: [example] * 44)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: 0)

    trainer.train()

    logs = trainer.tracker.log.call_args_list
    assert [call.kwargs["step"] for call in logs] == [1, 2, 3]
    assert [call.args[0]["train/actual_batch_size"] for call in logs] == expected_sizes
    assert [call.args[0]["train/actual_num_tokens"] for call in logs] == [size * 300 for size in expected_sizes]


@pytest.mark.parametrize("async_collation", [False, True])
def test_sft_logs_real_counts_for_short_packed_tail(mock_dispatch, monkeypatch, async_collation):
    cfg = _build_test_sft_config()
    cfg.strategy = "megatron"
    cfg.placement.num_gpus_per_node = 4
    cfg.batch_size = 4
    cfg.num_steps = 3
    cfg.sampler = "sequential"
    cfg.remove_microbatch_padding = True
    cfg.use_sequence_packing = True
    cfg.async_batch_collation = async_collation
    cfg.eval_datasets = None
    cfg.eval_dataset_splits = None
    cfg.eval_dataset_names = None
    cfg.eval_interval = 0
    cfg.ckpt_path = ""
    trainer = SFTTrainer(cfg, skyrl_cfg=build_skyrl_config_for_sft(cfg))
    mock_dispatch.dp_size.return_value = 4
    attach_mock_sft_deps(trainer, mock_dispatch)
    trainer.tracker = MagicMock()
    example = {
        "input_ids": [10, 20, 30],
        "attention_mask": [1, 1, 1],
        "num_actions": 2,
        "loss_mask": [1, 1],
    }
    monkeypatch.setattr(trainer, "_load_and_tokenize", lambda *_args, **_kwargs: [example] * 9)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: 0)

    trainer.train()

    logs = [call.args[0] for call in trainer.tracker.log.call_args_list]
    assert [log["train/actual_batch_size"] for log in logs] == [4, 4, 1]
    assert [log["train/actual_num_tokens"] for log in logs] == [12, 12, 3]
    assert [log["train/total_tokens_processed"] for log in logs] == [12, 24, 27]
    assert logs[-1]["train/tokens_per_second"] == pytest.approx(3 / logs[-1]["timing/step"])
