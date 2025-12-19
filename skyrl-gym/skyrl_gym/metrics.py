from typing import Any, Dict, List
from skyrl_gym.envs.registration import registry, load_env_creator


def default_aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    A minimal default metric aggregator: average numeric fields across episode-level metric dicts.
    """
    if not metrics:
        return {}
    aggregated_metrics: Dict[str, list[float]] = {}
    for m in metrics:
        for k, v in m.items():
            if isinstance(v, bool):
                v = float(v)
            elif isinstance(v, (int, float)):
                v = float(v)
            else:
                continue
            aggregated_metrics.setdefault(k, []).append(v)
    return {k: sum(vals) / len(vals) for k, vals in aggregated_metrics.items()}


def aggregate_for_environment(env_name: str, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call aggregate_metrics() for the class specified by env_name.

    Args:
        env_name: The registered environment name (e.g., "gsm8k")
        metrics: List of metric dictionaries to aggregate

    Returns:
        Aggregated metrics for the environment class
    """
    # Look up the environment spec in the registry
    env_spec = registry.get(env_name)
    if env_spec is None:
        raise ValueError(f"No registered env with id: {env_name}")

    # Get the environment class from the entry_point
    entry_point = env_spec.entry_point
    if callable(entry_point):
        env_cls = entry_point
    else:
        # Load the class from the string entry point
        env_cls = load_env_creator(entry_point)

    return env_cls.aggregate_metrics(metrics)
