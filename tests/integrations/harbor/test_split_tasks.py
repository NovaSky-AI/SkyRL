"""Splitting prepared Harbor tasks into training and held-out sets."""

import pytest

from examples.train_integrations.harbor.split_tasks import main, split

pytestmark = pytest.mark.integrations

NAMES = [f"task-{i:03d}" for i in range(100)]


def test_the_held_out_tasks_stay_the_same_whatever_the_run_trains_on() -> None:
    small_train, held_out = split(NAMES, num_eval=10, num_train=20, seed=0)
    large_train, same_held_out = split(list(reversed(NAMES)), num_eval=10, num_train=80, seed=0)

    assert held_out == same_held_out
    assert large_train[:20] == small_train
    assert not set(large_train) & set(held_out)
    assert split(NAMES, num_eval=10, num_train=20, seed=1)[1] != held_out


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(num_eval=10, num_train=91), "91 training tasks requested, but only 90"),
        (dict(num_eval=101, num_train=0), "only 100 tasks"),
        (dict(num_eval=10, num_train=0, train_names=["nope"]), "no such tasks: nope"),
    ],
    ids=["too-many-train", "too-many-eval", "unknown-name"],
)
def test_a_split_that_cannot_be_made_is_refused(kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        split(NAMES, seed=0, **kwargs)


def test_named_training_tasks_may_not_be_held_out() -> None:
    _, held_out = split(NAMES, num_eval=10, num_train=0, seed=0)
    elsewhere = next(name for name in NAMES if name not in held_out)

    assert split(NAMES, num_eval=10, num_train=0, seed=0, train_names=[elsewhere]) == ([elsewhere], held_out)
    with pytest.raises(ValueError, match=f"held-out tasks named for training: {held_out[0]}"):
        split(NAMES, num_eval=10, num_train=0, seed=0, train_names=[held_out[0]])


def test_a_run_directory_holding_anything_but_task_symlinks_is_refused_and_left_alone(tmp_path) -> None:
    tasks_dir = tmp_path / "CodeContests"
    for name in NAMES:
        (tasks_dir / name).mkdir(parents=True)
    stray = tmp_path / "run" / "tasks" / "my-notes"
    stray.mkdir(parents=True)

    with pytest.raises(ValueError, match="holds my-notes, which aren't task symlinks"):
        main(["--tasks-dir", str(tasks_dir), "--run-dir", str(tmp_path / "run"), "--eval", "5", "--train", "8"])
    assert stray.is_dir()


def test_every_launch_rebuilds_the_run_directory_from_the_seed(tmp_path) -> None:
    tasks_dir = tmp_path / "CodeContests"
    for name in NAMES:
        (tasks_dir / name).mkdir(parents=True)
    run_dir = tmp_path / "run"

    main(["--tasks-dir", str(tasks_dir), "--run-dir", str(run_dir), "--eval", "5", "--train", "20"])
    main(["--tasks-dir", str(tasks_dir), "--run-dir", str(run_dir), "--eval", "5", "--train", "8"])

    train = (run_dir / "train_tasks.txt").read_text().split()
    held_out = (run_dir / "eval_tasks.txt").read_text().split()
    assert (len(train), len(held_out)) == (8, 5)
    assert sorted(p.name for p in (run_dir / "tasks").iterdir()) == sorted(train)
    assert sorted(p.name for p in (run_dir / "eval_tasks").iterdir()) == sorted(held_out)
    assert (run_dir / "tasks" / train[0]).resolve() == (tasks_dir / train[0]).resolve()
