"""Which failed Harbor trials to mask before training, shared by the ``harbor`` and ``harbor_skycap`` generators.

Each generator classifies its trials by stop reason (``complete``, ``context_length``, ``agent_timeout`` or
``error``) and builds its own rows; ``mask_failed_trials`` decides which trials become masked placeholder rows.
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

#: Default for ``generator.timeout_mask_instance_threshold``.
TIMEOUT_MASK_INSTANCE_THRESHOLD = 2


def mask_failed_trials(
    trials: Sequence[Tuple[str, str]],
    timeout_mask_instance_threshold: int = TIMEOUT_MASK_INSTANCE_THRESHOLD,
) -> Tuple[List[bool], Dict[str, int]]:
    """Which trials to mask, from each trial's ``(instance_id, stop_reason)``, and the counts for metrics.

    An instance with an ``error`` trial is masked whole: that trial was never measured, and GRPO would count it as
    reward 0 in its group's baseline. So is an instance with at least ``timeout_mask_instance_threshold``
    ``agent_timeout`` trials, since several timeouts in one group point at the task or the infrastructure rather
    than at one rollout. Fewer timeouts mask only the timed-out trials, and the rest of the instance trains; a trial
    masked alone still counts as reward 0 in its group's baseline.
    """
    trials_of: Dict[str, List[int]] = defaultdict(list)
    for index, (instance_id, _) in enumerate(trials):
        trials_of[instance_id].append(index)

    masked = [False] * len(trials)
    counts = {
        "num_timeout_trajectories": sum(stop_reason == "agent_timeout" for _, stop_reason in trials),
        "num_error_trajectories": sum(stop_reason == "error" for _, stop_reason in trials),
        "num_masked_instances": 0,
        "num_instance_with_timeouts_unmasked": 0,
        "num_traj_masked_individually": 0,
    }
    for indices in trials_of.values():
        timeouts = [i for i in indices if trials[i][1] == "agent_timeout"]
        if any(trials[i][1] == "error" for i in indices) or len(timeouts) >= timeout_mask_instance_threshold:
            counts["num_masked_instances"] += 1
            to_mask = indices
        else:
            to_mask = timeouts
            if timeouts:
                counts["num_instance_with_timeouts_unmasked"] += 1
                counts["num_traj_masked_individually"] += len(timeouts)
        for i in to_mask:
            masked[i] = True
    return masked, counts
