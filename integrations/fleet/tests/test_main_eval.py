"""Unit tests for the Fleet eval-only entrypoint.

uv run --extra dev --extra skyrl-train pytest integrations/fleet/tests/test_main_eval.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from integrations.fleet.entrypoints.main_eval import (
    FleetEvalExp,
    _strip_hydra_prefixes,
)


# ---------------------------------------------------------------------------
# _strip_hydra_prefixes
# ---------------------------------------------------------------------------


def test_strip_hydra_prefixes_handles_all_three_arg_shapes():
    args = [
        "trainer.run_name=my_run",
        "+trainer.eval_interval=1",
        "++environment.skyrl_gym.fleet_task.tasks_file=/tmp/tasks.json",
    ]
    out = _strip_hydra_prefixes(args)
    assert out == [
        "trainer.run_name=my_run",
        "trainer.eval_interval=1",
        "environment.skyrl_gym.fleet_task.tasks_file=/tmp/tasks.json",
    ]


def test_strip_hydra_prefixes_empty():
    assert _strip_hydra_prefixes([]) == []


def test_strip_hydra_prefixes_double_plus_takes_precedence_over_single():
    # "++" matches startswith("++") first, so it strips two characters, not one.
    assert _strip_hydra_prefixes(["++key=value"]) == ["key=value"]


# ---------------------------------------------------------------------------
# FleetEvalExp.get_train_dataset
# ---------------------------------------------------------------------------


def test_get_train_dataset_returns_none():
    # Bypass __init__ so we don't pull in tokenizer / placement group.
    exp = FleetEvalExp.__new__(FleetEvalExp)
    assert exp.get_train_dataset() is None


# ---------------------------------------------------------------------------
# FleetEvalExp._load_policy_only — path resolution + dispatch wiring
# ---------------------------------------------------------------------------


def _make_trainer_mock(resume_mode_value: str, ckpt_path: str, resume_path: str = "") -> MagicMock:
    """Build a minimal trainer mock for _load_policy_only tests.

    Mirrors the attribute shape `_load_policy_only` reads: trainer.resume_mode,
    trainer.cfg.trainer.{ckpt_path, ckpt_interval, resume_path}, trainer.dispatch.
    """
    from skyrl.train.utils.trainer_utils import ResumeMode

    trainer = MagicMock()
    trainer.resume_mode = ResumeMode(resume_mode_value)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            ckpt_path=ckpt_path,
            ckpt_interval=10,
            resume_path=resume_path,
        )
    )
    trainer.global_step = 0
    return trainer


def _make_exp() -> FleetEvalExp:
    """Create a FleetEvalExp bypassing __init__ (which loads a tokenizer)."""
    return FleetEvalExp.__new__(FleetEvalExp)


def test_load_policy_only_resume_none_is_noop():
    exp = _make_exp()
    trainer = _make_trainer_mock("none", ckpt_path="/tmp/does-not-matter")

    exp._load_policy_only(trainer)

    trainer.dispatch.load_checkpoint.assert_not_called()
    assert trainer.global_step == 0


def test_load_policy_only_latest_with_no_marker_file_is_noop(tmp_path):
    exp = _make_exp()
    trainer = _make_trainer_mock("latest", ckpt_path=str(tmp_path))
    # No latest_ckpt_global_step.txt written → fall through, no load.

    exp._load_policy_only(trainer)

    trainer.dispatch.load_checkpoint.assert_not_called()
    assert trainer.global_step == 0


def test_load_policy_only_latest_loads_policy_and_sets_global_step(tmp_path):
    # Build a realistic checkpoint layout that the resolver expects.
    ckpt_dir = tmp_path / "global_step_30"
    (ckpt_dir / "policy").mkdir(parents=True)
    (tmp_path / "latest_ckpt_global_step.txt").write_text("30")

    exp = _make_exp()
    trainer = _make_trainer_mock("latest", ckpt_path=str(tmp_path))

    # The consistency validator hits the filesystem in non-trivial ways; stub
    # it out so the test stays focused on this method's contract.
    with patch(
        "skyrl.train.utils.trainer_utils.validate_consistency_for_latest_checkpoint"
    ) as validator:
        exp._load_policy_only(trainer)

    validator.assert_called_once()
    trainer.dispatch.load_checkpoint.assert_called_once_with(
        "policy",
        str(ckpt_dir / "policy"),
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    assert trainer.global_step == 30


def test_load_policy_only_from_path_loads_specified_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "global_step_42"
    (ckpt_dir / "policy").mkdir(parents=True)

    exp = _make_exp()
    trainer = _make_trainer_mock("from_path", ckpt_path=str(tmp_path), resume_path=str(ckpt_dir))

    exp._load_policy_only(trainer)

    trainer.dispatch.load_checkpoint.assert_called_once_with(
        "policy",
        str(ckpt_dir / "policy"),
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    assert trainer.global_step == 42


def test_load_policy_only_from_path_missing_dir_raises(tmp_path):
    exp = _make_exp()
    trainer = _make_trainer_mock(
        "from_path",
        ckpt_path=str(tmp_path),
        resume_path=str(tmp_path / "global_step_99"),  # never created
    )

    with pytest.raises(FileNotFoundError):
        exp._load_policy_only(trainer)

    trainer.dispatch.load_checkpoint.assert_not_called()


def test_load_policy_only_from_path_invalid_dir_name_raises(tmp_path):
    # extract_step_from_path returns -1 when the dir name has no global_step prefix.
    bad_dir = tmp_path / "not_a_step_dir"
    (bad_dir / "policy").mkdir(parents=True)

    exp = _make_exp()
    trainer = _make_trainer_mock("from_path", ckpt_path=str(tmp_path), resume_path=str(bad_dir))

    with pytest.raises(ValueError, match="not a valid global_step dir"):
        exp._load_policy_only(trainer)

    trainer.dispatch.load_checkpoint.assert_not_called()
