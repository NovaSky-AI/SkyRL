r"""A run's training and held-out task sets, from one directory of prepared Harbor tasks.

All task names are sorted and shuffled once with ``--seed``. The first ``--eval`` of that order are the
held-out tasks and the next ``--train`` are the training tasks, so the held-out set doesn't depend on how many
tasks a run trains on, and the same seed rebuilds the same split on every launch (which resuming relies on).
``--tasks`` names the training tasks instead; none of them may be held out.

Writes, under ``--run-dir``: ``tasks/`` and ``eval_tasks/`` (a symlink per task, rebuilt on every launch) and
``train_tasks.txt`` and ``eval_tasks.txt`` (one name per line).

    python3 examples/train_integrations/harbor/split_tasks.py --tasks-dir /tmp/harbor/data/CodeContests \
        --run-dir <run> --eval 32 --train 256 --seed 0

Prepared datasets that ship without a held-out split (``open-thoughts/CodeContests`` has a single one) need
this; ``harbor_skycap/run_codecontests.sh`` runs it on every launch.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def split(
    names: Sequence[str],
    *,
    num_eval: int,
    num_train: int,
    seed: int,
    train_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """The training and held-out task names. ``train_names`` replaces the training slice."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    held_out = order[:num_eval]
    if len(held_out) < num_eval:
        raise ValueError(f"{num_eval} held-out tasks requested, but there are only {len(order)} tasks")
    if train_names is not None:
        unknown = sorted(set(train_names) - set(order))
        if unknown:
            raise ValueError(f"no such tasks: {', '.join(unknown)}")
        overlap = sorted(set(train_names) & set(held_out))
        if overlap:
            raise ValueError(
                f"held-out tasks named for training: {', '.join(overlap)}. Name other tasks, or change the split "
                "with SPLIT_SEED or EVAL_TASKS."
            )
        return list(train_names), held_out
    if num_train < 1:
        raise ValueError(f"no training tasks: --train (NUM_TASKS in the recipe) must be at least 1, got {num_train}")
    train = order[num_eval : num_eval + num_train]
    if len(train) < num_train:
        raise ValueError(
            f"{num_train} training tasks requested, but only {len(order) - num_eval} are left after "
            f"{num_eval} held-out ones"
        )
    return train, held_out


def link(tasks_dir: Path, names: Sequence[str], out_dir: Path) -> None:
    """``out_dir`` holds exactly one symlink per name, into ``tasks_dir``. Anything else in it is refused, not deleted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    foreign = sorted(entry.name for entry in out_dir.iterdir() if not entry.is_symlink())
    if foreign:
        listed = ", ".join(foreign[:3]) + (f" and {len(foreign) - 3} more" if len(foreign) > 3 else "")
        raise ValueError(
            f"{out_dir} holds {listed}, which aren't task symlinks: move them or use another run directory"
        )
    for stale in out_dir.iterdir():
        stale.unlink()
    # A relative target would resolve against out_dir, not the directory the paths were given from.
    tasks_dir = tasks_dir.absolute()
    for name in names:
        (out_dir / name).symlink_to(tasks_dir / name)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval", type=int, required=True, help="held-out tasks; 0 for none")
    parser.add_argument("--train", type=int, required=True, help="training tasks, unless --tasks names them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="", help="comma-separated training task names")
    args = parser.parse_args(argv)

    names = [entry.name for entry in args.tasks_dir.iterdir() if entry.is_dir()]
    train_names = [name for name in args.tasks.split(",") if name] or None
    train, held_out = split(names, num_eval=args.eval, num_train=args.train, seed=args.seed, train_names=train_names)
    saved = [args.run_dir / "train_tasks.txt", args.run_dir / "eval_tasks.txt"]
    if all(path.exists() for path in saved) and [path.read_text().split() for path in saved] != [train, held_out]:
        raise ValueError(
            f"{args.run_dir} already holds a different split (another seed, --eval, --train or --tasks). Resuming "
            "it would put the saved data-loader position on other tasks and could evaluate on trained ones: use a "
            "new run directory, or delete train_tasks.txt and eval_tasks.txt there to replace the split."
        )
    link(args.tasks_dir, train, args.run_dir / "tasks")
    link(args.tasks_dir, held_out, args.run_dir / "eval_tasks")
    (args.run_dir / "train_tasks.txt").write_text("".join(f"{name}\n" for name in train))
    (args.run_dir / "eval_tasks.txt").write_text("".join(f"{name}\n" for name in held_out))
    print(f"{len(train)} training tasks, {len(held_out)} held-out tasks")


if __name__ == "__main__":
    main()
