"""Harbor's task dataset: one data source per directory given, not per task, in a stable order."""

import pathlib

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


def test_tasks_load_in_name_order_whatever_the_filesystem_lists(tmp_path, monkeypatch) -> None:
    for task in ("b", "c", "a"):
        (tmp_path / "tasks" / task).mkdir(parents=True)
        (tmp_path / "tasks" / task / "instruction.md").write_text("solve it")
    listing = pathlib.Path.iterdir
    monkeypatch.setattr(pathlib.Path, "iterdir", lambda path: iter(sorted(listing(path), reverse=True)))

    dataset = HarborTaskDataset([str(tmp_path / "tasks")])
    assert [pathlib.Path(item["prompt"]).name for item in dataset] == ["a", "b", "c"]
