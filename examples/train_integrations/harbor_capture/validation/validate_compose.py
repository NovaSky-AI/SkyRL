"""Is what `compose` hands the trainer well formed?

Fidelity says the tokens are the model's own. This says the batch built from
them holds together: that the arrays a trainer indexes in parallel are the same
length, that a rollout is grouped so one advantage can reach all of its rows,
and that nothing trains on a token no one sampled.

Reads a record directory, composes it the way the generator does, and checks
the result rather than trusting it.
"""

from __future__ import annotations

import sys
from collections import Counter

from skyrl_capture.export import formats  # noqa: E402
from skyrl_capture.export.view import view_of  # noqa: E402
from skyrl_capture.persistence.committed import DiskCommittedStore  # noqa: E402

from examples.train_integrations.harbor_capture.compose import compose  # noqa: E402

RECORD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/icap-validation"
RUN = sys.argv[2] if len(sys.argv) > 2 else None


def main() -> int:
    store = DiskCommittedStore(RECORD)
    exports, ids, rewards, stops = [], [], [], []
    for identifier in store.committed_ids():
        record = store.get_sync(identifier)
        if RUN and record.trajectory.run_id != RUN:
            continue
        exports.append(formats.token_sample_records(view_of(record)))
        ids.append(record.trajectory.id)
        rewards.append(float((record.trajectory.annotations or {}).get("reward") or 0.0))
        stops.append((record.trajectory.annotations or {}).get("stop_reason") or "complete")

    if not exports:
        print(f"no trajectories in {RECORD}" + (f" for run {RUN}" if RUN else ""))
        return 1

    out = compose(exports, trajectory_ids=ids, rewards=rewards, stop_reasons=stops)
    rows = len(out["response_ids"])
    failures = []

    # 1. Arrays a trainer indexes in parallel must agree, row by row.
    parallel = ["prompt_token_ids", "response_ids", "loss_masks", "rollout_logprobs",
                "rewards", "stop_reasons", "trajectory_ids", "is_last_step"]
    for key in parallel:
        value = out.get(key)
        if value is not None and len(value) != rows:
            failures.append(f"{key} has {len(value)} entries for {rows} rows")
    for index in range(rows):
        lengths = {
            "response_ids": len(out["response_ids"][index]),
            "loss_masks": len(out["loss_masks"][index]),
            "rollout_logprobs": len(out["rollout_logprobs"][index]),
        }
        if len(set(lengths.values())) != 1:
            failures.append(f"row {index} lengths disagree: {lengths}")

    # 2. A rollout's rows must be contiguous and marked exactly once, or the
    #    trainer cannot compute one advantage and broadcast it.
    seen_order = [str(t) for t in out["trajectory_ids"]]
    runs = [key for index, key in enumerate(seen_order) if index == 0 or key != seen_order[index - 1]]
    if len(runs) != len(set(runs)):
        failures.append("a rollout's rows are not contiguous")
    marks = Counter()
    for key, last in zip(seen_order, out["is_last_step"]):
        marks[key] += bool(last)
    bad_marks = {key: count for key, count in marks.items() if count != 1}
    if bad_marks:
        failures.append(f"rollouts not marked exactly once: {bad_marks}")

    # 3. One reward per rollout, shared by every row of it.
    by_rollout = {}
    for key, reward in zip(seen_order, out["rewards"]):
        by_rollout.setdefault(key, set()).add(reward)
    split = {key: values for key, values in by_rollout.items() if len(values) > 1}
    if split:
        failures.append(f"rows of one rollout carry different rewards: {split}")

    # 4. Every row either trains on something or is an explicit placeholder.
    empty = [index for index in range(rows) if not any(out["loss_masks"][index])]

    print(f"{len(exports)} rollouts -> {rows} rows")
    print(f"  rollouts represented        : {len(set(seen_order))}/{len(exports)}")
    print(f"  rows with trainable tokens  : {rows - len(empty)}")
    print(f"  fully masked placeholders   : {len(empty)}")
    print(f"  trainable tokens            : {sum(sum(m) for m in out['loss_masks'])}")
    print(f"  checks failed               : {len(failures)}")
    for failure in failures:
        print(f"    {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
