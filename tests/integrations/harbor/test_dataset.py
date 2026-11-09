"""Harbor's task dataset: one data source per directory given, not per task."""

import pytest

from examples.train_integrations.harbor.dataset import HarborTaskDataset

pytestmark = pytest.mark.integrations


def test_each_task_reports_the_directory_it_was_found_under(tmp_path) -> None:
    for source, task in [("eval_tasks", "a"), ("eval_tasks", "b"), ("tb-dev", "c")]:
        (tmp_path / source / task).mkdir(parents=True)
        (tmp_path / source / task / "instruction.md").write_text("solve it")
    (tmp_path / "single").mkdir()
    (tmp_path / "single" / "instruction.md").write_text("solve it")

    dataset = HarborTaskDataset([str(tmp_path / name) for name in ("eval_tasks", "tb-dev", "single")])
    items = sorted((item["env_extras"]["data_source"], item["prompt"]) for item in dataset)

    assert [source for source, _ in items] == ["eval_tasks", "eval_tasks", "single", "tb-dev"]
    assert all(item["env_extras"]["task_path"] == item["prompt"] for item in dataset)
    assert dataset[0] == next(iter(dataset))
