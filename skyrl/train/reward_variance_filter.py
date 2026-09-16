import numpy as np


def reward_variance_filter(
    rewards: list[float],
    uids: list[str],
    loss_masks: list[list[int]] | None = None,
    strategy: str = "top_p",
    top_p: float = 0.9,
    top_k: int = 1,
    include_zero: bool = False,
    variance_ddof: int = 1,
    selection_eps: float = 0.01,
) -> tuple[list[int], dict[str, float]]:
    """Select prompt groups with the highest sample reward variance.

    ``top_p`` keeps the smallest prefix covering the requested variance mass, as
    introduced by RAGEN2. ``top_k`` keeps a fixed number of groups. Ties preserve
    first-seen order, and already loss-masked trajectories do not contribute.
    """
    if len(rewards) != len(uids):
        raise ValueError(f"rewards and uids must have equal length, got {len(rewards)} and {len(uids)}")
    if loss_masks is not None and len(loss_masks) != len(rewards):
        raise ValueError("loss_masks must have the same length as rewards")
    if strategy not in ("top_p", "top_k"):
        raise ValueError(f"strategy must be 'top_p' or 'top_k', got {strategy!r}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if variance_ddof < 0:
        raise ValueError(f"variance_ddof must be non-negative, got {variance_ddof}")
    if selection_eps < 0.0:
        raise ValueError(f"selection_eps must be non-negative, got {selection_eps}")

    is_live = [True] * len(rewards) if loss_masks is None else [sum(mask) > 0 for mask in loss_masks]
    uid2rewards: dict[str, list[float]] = {}
    for uid, reward, live in zip(uids, rewards, is_live):
        if live and np.isfinite(reward):
            uid2rewards.setdefault(uid, []).append(float(reward))

    ordered_uids = list(dict.fromkeys(uids))
    variances: dict[str, float] = {}
    for uid in ordered_uids:
        values = uid2rewards.get(uid, [])
        variances[uid] = float(np.var(values, ddof=variance_ddof)) if len(values) > variance_ddof else 0.0

    total_variance = sum(variances.values())
    selected_uids: set[str] = set()
    if total_variance <= 0.0:
        if include_zero:
            selected_uids.update(ordered_uids[:top_k] if strategy == "top_k" else ordered_uids)
    elif strategy == "top_p" and include_zero and top_p == 1.0:
        selected_uids.update(ordered_uids)
    else:
        candidates = [uid for uid in ordered_uids if include_zero or variances[uid] > 0.0]
        candidates.sort(key=lambda uid: variances[uid], reverse=True)
        if strategy == "top_k":
            selected_uids.update(candidates[:top_k])
        else:
            target = top_p * total_variance - selection_eps
            cumulative = 0.0
            # RAGEN uses this slack to skip near-zero-signal batches entirely.
            if target > 0.0:
                for uid in candidates:
                    selected_uids.add(uid)
                    cumulative += variances[uid]
                    if cumulative >= target:
                        break

    kept_indices = [index for index, uid in enumerate(uids) if uid in selected_uids]
    selected_variance = sum(variances[uid] for uid in selected_uids)
    num_groups = len(ordered_uids)
    metrics = {
        "num_groups": float(num_groups),
        "num_kept_groups": float(len(selected_uids)),
        "kept_group_ratio": float(len(selected_uids) / num_groups) if num_groups else 0.0,
        "total_variance": float(total_variance),
        "selected_variance_ratio": float(selected_variance / total_variance) if total_variance > 0.0 else 0.0,
        "num_zero_variance_groups": float(sum(value == 0.0 for value in variances.values())),
    }
    return kept_indices, metrics
