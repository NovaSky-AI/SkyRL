"""``trainer.policy.model.lora.checkpoint_export``: which checkpoints write the PEFT-format adapter.

On full ML4 the export is a 41.5 GB float32 file written by rank 0 at every save; resume reads
only the per-rank shards. ``final`` keeps it for the last checkpoint only.
"""

from types import SimpleNamespace

import pytest

from skyrl.train.config.config import SkyRLLoraConfig
from skyrl.train.trainer import RayPPOTrainer


def _trainer(rank: int, mode: str, step: int, total):
    t = RayPPOTrainer.__new__(RayPPOTrainer)
    lora = SimpleNamespace(rank=rank, checkpoint_export=mode)
    t.cfg = SimpleNamespace(trainer=SimpleNamespace(policy=SimpleNamespace(model=SimpleNamespace(lora=lora))))
    t.global_step, t.total_training_steps = step, total
    return t


class TestCheckpointExport:
    def test_default_exports_every_save(self):
        assert SkyRLLoraConfig().checkpoint_export == "every_save"

    @pytest.mark.parametrize(
        "rank,mode,step,total,expected",
        [
            (32, "every_save", 1, 10, None),
            (32, "final", 5, 10, False),
            (32, "final", 10, 10, True),
            (32, "final", 3, None, True),
            (0, "final", 5, 10, None),
        ],
        ids=["every-save", "final-mid", "final-last", "final-unknown-total", "no-lora"],
    )
    def test_export_decision(self, rank, mode, step, total, expected):
        assert _trainer(rank, mode, step, total)._export_lora_adapter_this_save() is expected


class _Dispatch:
    def __init__(self):
        self.saves, self.exports = [], []

    def save_checkpoint(self, model, ckpt_dir, tokenizer=None, export_lora_adapter=None):
        self.saves.append((ckpt_dir, export_lora_adapter))

    def export_lora_adapter(self, model, ckpt_dir):
        self.exports.append(ckpt_dir)


def _real_trainer(tmp_path, *, step: int, total: int, mode: str = "final"):
    from skyrl.train.config.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig()
    cfg.trainer.ckpt_path = str(tmp_path)
    cfg.trainer.ckpt_interval = 3
    cfg.trainer.policy.model.lora.rank = 32
    cfg.trainer.policy.model.lora.checkpoint_export = mode
    t = RayPPOTrainer.__new__(RayPPOTrainer)
    t.cfg, t.global_step, t.total_training_steps = cfg, step, total
    t.dispatch, t.tokenizer, t.all_timings = _Dispatch(), None, {}
    t._ckpt_dir_without_lora_export = None
    t.train_dataloader = SimpleNamespace(state_dict=lambda: {})
    t._cleanup_old_checkpoints = lambda: None
    t._fire = lambda *a, **k: None
    return t


class TestFinalCheckpointHasAdapter:
    """The run's last checkpoint carries the PEFT adapter under ``final``, however training ends."""

    def test_safety_net_save_exports(self, tmp_path):
        import asyncio

        # Training ended at step 7 of 10 (e.g. dynamic sampling consumed the batches); the last
        # periodic save was at step 6 (no export), so the end-of-training save runs at step 7.
        t = _real_trainer(tmp_path, step=6, total=10)
        t.save_checkpoints()
        t.global_step = 7
        asyncio.run(t._save_final_checkpoint(will_save_ckpts=False))
        assert [e for _, e in t.dispatch.saves] == [False, True]
        assert t.dispatch.exports == []

    def test_early_end_exports_into_last_checkpoint(self, tmp_path):
        import asyncio

        # Training ended at step 6 of 10 right after the periodic save of step 6 (no export), so
        # the end of training exports into that checkpoint.
        t = _real_trainer(tmp_path, step=6, total=10)
        t.save_checkpoints()
        asyncio.run(t._save_final_checkpoint(will_save_ckpts=True))
        assert [e for _, e in t.dispatch.saves] == [False]
        assert t.dispatch.exports == [str(tmp_path / "global_step_6" / "policy")]

    def test_every_save_unchanged(self, tmp_path):
        import asyncio

        t = _real_trainer(tmp_path, step=6, total=10, mode="every_save")
        t.save_checkpoints()
        asyncio.run(t._save_final_checkpoint(will_save_ckpts=True))
        assert [e for _, e in t.dispatch.saves] == [None] and t.dispatch.exports == []
