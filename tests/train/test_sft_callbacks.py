"""CPU test: callbacks fire end-to-end during an SFTTrainer training run.

Runs 2 training steps (1 epoch over a 2-example dummy dataset) with three
callbacks registered:
  * RecorderCallback — snapshots every event with the relevant CallbackInput
    fields, used for sequence + payload assertions.
  * ForceEvaluateAtStep — sets ``control.should_evaluate = True`` on the
    on_step_end of step 1, exercising the callback-driven eval path. With
    ``eval_interval=2`` step 1 would not normally eval, so any eval event
    seen at step 1 comes from this callback.
  * ForceSaveAtStep — sets ``control.should_save = True`` on the on_step_end
    of step 2, exercising the callback-driven checkpoint path.

Mocks the worker dispatch, tokenizer, and dataset loading so the test only
exercises the orchestration in ``SFTTrainer.train()`` — no GPUs / no real
training.

uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_callbacks.py -v
"""

from unittest.mock import MagicMock

import pytest

from skyrl.backends.skyrl_train.workers.worker_utils import RETURN_PER_TOKEN_OUTPUTS_KEY
from skyrl.train.config.sft_config import (
    SFTConfig,
    SFTPlacementConfig,
    build_skyrl_config_for_sft,
)
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.callbacks import (
    CallbackInput,
    TrainingCallback,
)

_FAKE_CKPT_PATH = "/fake/sft-callback-test/global_step_2"


def _make_mock_dispatch() -> MagicMock:
    """A worker-dispatch mock for the SFT orchestration tests (no GPUs).

    ``forward_backward`` / ``forward`` return objects exposing ``.metrics`` so
    ``train_step`` / ``run_eval`` complete; ``optim_step`` returns a grad_norm and
    ``dp_size`` a world size of 1.
    """
    step_output = MagicMock()
    step_output.metrics = {"loss": 0.42, "final_loss": 0.42}
    eval_output = MagicMock()
    eval_output.metrics = {"loss": 0.31}

    dispatch_mock = MagicMock()
    dispatch_mock.forward_backward = MagicMock(return_value=step_output)
    dispatch_mock.optim_step = MagicMock(return_value=1.0)
    dispatch_mock.forward = MagicMock(return_value=eval_output)
    dispatch_mock.dp_size = MagicMock(return_value=1)
    return dispatch_mock


def _attach_mock_sft_deps(trainer: SFTTrainer, dispatch_mock: MagicMock) -> None:
    """Wire the mocked tokenizer/collator/dispatch onto a trainer in lieu of ``setup()``.

    ``setup()`` would load the model + spin up real GPU workers; here we replace what
    it sets with mocks so the test only exercises ``SFTTrainer`` orchestration.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer.tokenizer = tokenizer
    trainer.collator = trainer._build_collator(tokenizer)
    trainer.dispatch = dispatch_mock


@pytest.fixture
def mock_dispatch() -> MagicMock:
    return _make_mock_dispatch()


class RecorderCallback(TrainingCallback):
    """Spy: records every event with a snapshot of the relevant CallbackInput fields."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def _snap(self, name: str, ci: CallbackInput) -> None:
        self.events.append(
            (
                name,
                {
                    "global_step": ci.global_step,
                    "epoch": ci.epoch,
                    "total_steps": ci.total_steps,
                    "steps_per_epoch": ci.steps_per_epoch,
                    "has_batch": ci.batch is not None,
                    "has_metrics": ci.metrics is not None,
                    "metrics_keys": sorted((ci.metrics or {}).keys()),
                    "has_logs": ci.logs is not None,
                    "logs_keys": sorted((ci.logs or {}).keys()),
                    "ckpt_path": ci.ckpt_path,
                },
            )
        )

    def on_train_start(self, trainer, ci, control):
        self._snap("on_train_start", ci)

    def on_train_end(self, trainer, ci, control):
        self._snap("on_train_end", ci)

    def on_epoch_start(self, trainer, ci, control):
        self._snap("on_epoch_start", ci)

    def on_epoch_end(self, trainer, ci, control):
        self._snap("on_epoch_end", ci)

    def on_step_start(self, trainer, ci, control):
        self._snap("on_step_start", ci)

    def on_step_end(self, trainer, ci, control):
        self._snap("on_step_end", ci)

    def on_eval_start(self, trainer, ci, control):
        self._snap("on_eval_start", ci)

    def on_eval_end(self, trainer, ci, control):
        self._snap("on_eval_end", ci)

    def on_save(self, trainer, ci, control):
        self._snap("on_save", ci)

    def on_log(self, trainer, ci, control):
        self._snap("on_log", ci)


class ForceSaveAtStep(TrainingCallback):
    """Sets ``control.should_save = True`` on on_step_end when the global step matches."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, trainer, ci, control):
        if ci.global_step == self.step:
            control.should_save = True


class ForceEvaluateAtStep(TrainingCallback):
    """Sets ``control.should_evaluate = True`` on on_step_end when the global step matches."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, trainer, ci, control):
        if ci.global_step == self.step:
            control.should_evaluate = True


def _build_test_sft_config() -> SFTConfig:
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    # model.path / dataset_name are unused — we never load the model and
    # monkeypatch _load_and_tokenize. eval_dataset_name must be non-empty so
    # load_eval_dataset actually invokes _load_and_tokenize.
    cfg.model.path = "unused"
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    cfg.dataset_name = "unused-monkeypatched"
    cfg.dataset_split = "train"
    cfg.eval_dataset_name = "unused-monkeypatched"
    cfg.eval_dataset_split = "train"
    # eval_interval=2 means step 1 has no interval-driven eval; only the force-evaluate
    # callback can trigger eval at step 1. Step 2 still gets an interval-driven eval.
    cfg.eval_interval = 2
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
    cfg.ckpt_interval = 0  # interval-driven saves OFF; the callback-driven one is what we test
    cfg.hf_save_interval = 0
    return cfg


def _dummy_tokenized() -> list[dict]:
    """Two synthetic examples (10 input tokens, 4 response tokens each).

    With ``batch_size=1`` and ``num_steps=2`` we get one epoch with two
    distinct steps and an epoch boundary right after step 2.
    """
    example = {
        "input_ids": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "attention_mask": [1] * 10,
        "num_actions": 4,
        "loss_mask": [1, 1, 1, 1],
    }
    return [example, example]


def test_callbacks_fire_during_sft_training(monkeypatch, mock_dispatch):
    """A 2-step SFT run fires every relevant event, in order, with the right payloads."""
    cfg = _build_test_sft_config()
    skyrl_cfg = build_skyrl_config_for_sft(cfg)

    recorder = RecorderCallback()
    force_eval = ForceEvaluateAtStep(step=1)
    force_save = ForceSaveAtStep(step=2)
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg, callbacks=[recorder, force_eval, force_save])

    # Skip setup() (which would load the model + spin up workers). Replace what
    # setup() would have set with mocks (tokenizer / collator / worker dispatch).
    _attach_mock_sft_deps(trainer, mock_dispatch)
    trainer.tracker = MagicMock()

    # Bypass HF network fetch + tokenization: both load_dataset() and
    # load_eval_dataset() funnel through _load_and_tokenize.
    monkeypatch.setattr(trainer, "_load_and_tokenize", lambda *_args, **_kw: _dummy_tokenized())

    # Stub the real checkpoint save so nothing touches disk. on_save still
    # receives the fake path the stub returns.
    monkeypatch.setattr(trainer, "save_checkpoint", lambda: _FAKE_CKPT_PATH)

    # Stamp a "load_checkpoint" marker into recorder.events so the event-order
    # assertion below proves on_train_start fires AFTER load_checkpoint.
    def _record_load_checkpoint():
        recorder.events.append(("load_checkpoint", {"global_step": trainer.global_step}))
        return 0

    monkeypatch.setattr(trainer, "load_checkpoint", _record_load_checkpoint)

    trainer.train()

    event_names = [name for name, _ in recorder.events]

    # eval_interval=2 -> step 1 has no interval-driven eval, so on_eval_start/end
    # at step 1 only appear because ForceEvaluateAtStep set should_evaluate.
    # on_save at step 2 only appears because ForceSaveAtStep set should_save.
    # Step 2's eval is interval-driven (num_steps % eval_interval == 0).
    # The leading "load_checkpoint" marker proves on_train_start fires AFTER
    # the resume read (regression guard for callback timing).
    expected = [
        "load_checkpoint",
        "on_train_start",
        "on_epoch_start",
        # --- step 1 ---
        "on_step_start",
        "on_step_end",
        "on_eval_start",  # forced
        "on_eval_end",
        "on_log",
        # --- step 2 ---
        "on_step_start",
        "on_step_end",
        "on_save",  # forced
        "on_eval_start",  # interval
        "on_eval_end",
        "on_log",
        # --- epoch boundary + cleanup ---
        "on_epoch_end",
        "on_train_end",
    ]
    assert event_names == expected, f"unexpected event sequence: {event_names}"

    # Group snapshots so we can index by event name (some events fire twice).
    snaps_by_event: dict[str, list[dict]] = {}
    for name, snap in recorder.events:
        snaps_by_event.setdefault(name, []).append(snap)

    # Both step ends carry the batch + step metrics
    for snap in snaps_by_event["on_step_end"]:
        assert snap["has_batch"], "on_step_end should see the training batch"
        assert snap["has_metrics"], "on_step_end should see step metrics"
        assert "loss" in snap["metrics_keys"], snap["metrics_keys"]

    # Both eval ends carry eval metrics
    for snap in snaps_by_event["on_eval_end"]:
        assert snap["has_metrics"], "on_eval_end should see eval metrics"
        assert "eval_loss" in snap["metrics_keys"], snap["metrics_keys"]

    # Both on_log calls carry train/loss + eval/eval_loss
    for snap in snaps_by_event["on_log"]:
        log_keys = snap["logs_keys"]
        assert "train/loss" in log_keys, log_keys
        assert "eval/eval_loss" in log_keys, log_keys

    # on_save fired exactly once at step 2, with the fake ckpt path
    assert len(snaps_by_event["on_save"]) == 1, snaps_by_event["on_save"]
    save_snap = snaps_by_event["on_save"][0]
    assert save_snap["global_step"] == 2, save_snap
    assert save_snap["ckpt_path"] == _FAKE_CKPT_PATH, save_snap

    # The step_end that triggered the force-save must come immediately before on_save
    save_idx = event_names.index("on_save")
    assert event_names[save_idx - 1] == "on_step_end"

    # Force-evaluate at step 1 means there ARE two evals (forced at step 1 + interval at step 2);
    # check the global_step of each.
    eval_steps = [snap["global_step"] for snap in snaps_by_event["on_eval_end"]]
    assert eval_steps == [1, 2], eval_steps

    # Loop metadata stays consistent across every callback event (skip the
    # synthetic "load_checkpoint" marker which doesn't carry CallbackInput).
    for name, snap in recorder.events:
        if name == "load_checkpoint":
            continue
        assert snap["total_steps"] == 2, f"{name}: total_steps={snap['total_steps']}"
        assert snap["steps_per_epoch"] == 2, f"{name}: steps_per_epoch={snap['steps_per_epoch']}"


def _build_minimal_trainer(dispatch_mock: MagicMock) -> SFTTrainer:
    """Build an SFTTrainer wired with the given mocked dispatch (no GPUs).

    Shares the tokenizer/collator/dispatch wiring with
    ``test_callbacks_fire_during_sft_training`` via ``_attach_mock_sft_deps``.
    """
    cfg = _build_test_sft_config()
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)
    _attach_mock_sft_deps(trainer, dispatch_mock)
    return trainer


def test_sft_train_step_opts_out_of_per_token_outputs(mock_dispatch):
    """train_step must tell the worker to skip the unused per-token loss_fn_outputs.

    The SFT trainer reads only ``output.metrics``, so it passes
    ``loss_fn_config={"return_per_token_outputs": False}`` to ``forward_backward``.
    """
    trainer = _build_minimal_trainer(mock_dispatch)
    batch = trainer.collator(_dummy_tokenized(), batch_size=1)

    trainer.train_step(batch, step=1)

    mock_dispatch.forward_backward.assert_called_once()
    call = mock_dispatch.forward_backward.call_args
    assert call.kwargs["loss_fn"] == "cross_entropy"
    assert call.kwargs["loss_fn_config"] == {RETURN_PER_TOKEN_OUTPUTS_KEY: False}


def test_sft_run_eval_opts_out_of_per_token_outputs(mock_dispatch):
    """run_eval reads only ``output.metrics["loss"]``; it skips per-token outputs."""
    trainer = _build_minimal_trainer(mock_dispatch)

    metrics, _ = trainer.run_eval(_dummy_tokenized())

    assert "eval_loss" in metrics
    mock_dispatch.forward.assert_called()
    for call in mock_dispatch.forward.call_args_list:
        assert call.kwargs["loss_fn"] == "cross_entropy"
        assert call.kwargs["loss_fn_config"] == {RETURN_PER_TOKEN_OUTPUTS_KEY: False}
