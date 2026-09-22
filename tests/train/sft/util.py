from unittest.mock import MagicMock

from skyrl.train.sft_trainer import SFTTrainer


def attach_mock_sft_deps(trainer: SFTTrainer, dispatch_mock: MagicMock) -> None:
    """Wire mocked setup outputs onto the trainer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer.tokenizer = tokenizer
    trainer.collator = trainer._build_collator(tokenizer)
    trainer.dispatch = dispatch_mock
