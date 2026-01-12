from .data import CLASSES
# Final task reward (correctness)
async def banking77_final_reward_fn(example, pred, trace=None):
    label = pred.get("label")
    gold = example.get("label")

    if label is None:
        return 0.0

    return 1.0 if label == gold else 0.0


# Local validity / constraint reward
async def banking77_local_reward_fn(example, pred):
    assert len(pred) == 1, "Pred should have only one element"
    label = pred[0].get("label")
    gold = example.get("label")

    if label is None:
        return 0.0 if label == gold else 0.0

    return 0.5 if label in CLASSES else 0.0
