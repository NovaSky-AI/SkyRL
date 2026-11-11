from unittest.mock import MagicMock

from tests.train.gpu_e2e_test import check_sft_trend


def test_sft_completion_uses_history_when_summary_is_stale(monkeypatch):
    rows = [{"train/global_step": step, "train/loss": 3 - step / 100} for step in range(1, 101)]
    run = MagicMock()
    run.id = "finished-run"
    run.state = "finished"
    run.url = "https://example.test/run"
    run.summary_metrics = {"_step": 91}
    run.scan_history.side_effect = [rows[:91], rows]
    api = MagicMock()
    api.runs.return_value = [run]
    monkeypatch.setattr(check_sft_trend.wandb, "Api", lambda: api)
    monkeypatch.setattr(check_sft_trend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "sys.argv",
        ["check_sft_trend.py", "--run_name", "sft", "--project_name", "ci", "--expected_steps", "100"],
    )

    assert check_sft_trend.main() == 0
    assert run.scan_history.call_count == 2
    assert all(call.kwargs["use_cache"] is False for call in run.scan_history.call_args_list)
