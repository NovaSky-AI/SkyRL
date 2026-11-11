from unittest.mock import MagicMock

from tests.train.gpu_e2e_test import get_summary


def test_wait_for_finished_uses_final_history_row(monkeypatch):
    stale_run = MagicMock()
    stale_run.state = "finished"
    stale_run.summary_metrics = {"eval/all/avg_score": 0.47}
    stale_run.scan_history.return_value = [{"trainer/global_step": 5, "trainer/total_training_steps": 10}]

    final_run = MagicMock()
    final_run.state = "finished"
    final_run.summary_metrics = {"eval/all/avg_score": 0.47}
    final_run.scan_history.side_effect = [
        [{"trainer/global_step": 10, "trainer/total_training_steps": 10}],
        [{"trainer/global_step": 10, "eval/all/avg_score": 0.62}],
    ]

    runs = iter([stale_run, final_run])
    monkeypatch.setattr(get_summary.wandb, "Api", lambda: MagicMock(runs=lambda *args, **kwargs: [next(runs)]))
    monkeypatch.setattr(get_summary.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "get_summary.py",
            "--run_name",
            "e2e",
            "--project_name",
            "ci",
            "--wait_for_finished",
            "--asserts",
            "eval/all/avg_score >= 0.56",
        ],
    )

    assert get_summary.main() == 0
