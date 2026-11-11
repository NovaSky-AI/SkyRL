from unittest.mock import MagicMock

from tests.train.gpu_e2e_test import check_sft_trend


def test_sft_completion_uses_history_when_summary_is_stale(monkeypatch):
    rows = [{"_step": step, "train/loss": 3 - step / 100} for step in range(1, 101)]
    run = MagicMock()
    run.id = "finished-run"
    run.state = "finished"
    run.url = "https://example.test/run"
    run.summary_metrics = {"_step": 91}
    run.scan_history.return_value = iter(rows)
    api = MagicMock()
    api.runs.return_value = [run]
    monkeypatch.setattr(check_sft_trend.wandb, "Api", lambda: api)
    monkeypatch.setattr(
        "sys.argv",
        ["check_sft_trend.py", "--run_name", "sft", "--project_name", "ci", "--expected_steps", "100"],
    )

    assert check_sft_trend.main() == 0
