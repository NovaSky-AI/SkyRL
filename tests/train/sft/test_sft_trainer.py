from unittest.mock import MagicMock

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
