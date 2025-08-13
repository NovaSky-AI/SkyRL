import torch

from skyrl_train.dataset.replay_manager import TrainingBatchReplay
from skyrl_train.training_batch import TrainingInputBatch


def make_training_input_batch(batch_ids, seq_len=3, resp_len=2, device="cpu"):
    """Create a minimal valid TrainingInputBatch for given integer item ids.

    Each id is encoded into the sequences tensor for later verification.
    """
    batch_size = len(batch_ids)
    # Encode id in the last token position to distinguish samples
    sequences = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask[:, -resp_len - 1 :] = 1
    # Place the id token at the last position
    for i, bid in enumerate(batch_ids):
        sequences[i, -1] = int(bid)

    response_mask = torch.ones((batch_size, resp_len), dtype=torch.long, device=device)
    loss_mask = torch.ones((batch_size, resp_len), dtype=torch.long, device=device)
    action_log_probs = torch.zeros((batch_size, resp_len), dtype=torch.float32, device=device)
    base_action_log_probs = torch.zeros((batch_size, resp_len), dtype=torch.float32, device=device)
    values = torch.zeros((batch_size, resp_len), dtype=torch.float32, device=device)
    returns = torch.zeros((batch_size, resp_len), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch_size, resp_len), dtype=torch.float32, device=device)

    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "action_log_probs": action_log_probs,
            "base_action_log_probs": base_action_log_probs,
            "values": values,
            "returns": returns,
            "advantages": advantages,
        }
    )
    batch.metadata = {"response_length": resp_len}
    return batch


def extract_ids_from_batch(batch: TrainingInputBatch):
    # Ids were encoded at sequences[:, -1]
    return batch["sequences"][:, -1].cpu().tolist()


def test_append_and_capacity_eviction():
    mgr = TrainingBatchReplay(sample_batch_size=2, capacity=3, cpu_offload=True)

    # Append two batches of size 2 with distinct ids
    b0 = make_training_input_batch([10, 11])
    mgr.append(b0, behavior_version=0)
    assert len(mgr.buffer) == 2

    b1 = make_training_input_batch([20, 21])
    mgr.append(b1, behavior_version=1)
    # Capacity=3 should evict the oldest single item
    assert len(mgr.buffer) == 3

    versions = [it.info.get("behavior_version", None) for it in mgr.buffer.items]
    # Expect [0,1,1] after evicting the oldest of first append
    assert versions == [0, 1, 1]


def test_staleness_filtering():
    mgr = TrainingBatchReplay(sample_batch_size=2, capacity=10, cpu_offload=True)

    # Append 3 single-item batches with behavior versions 0,1,2 and distinct ids
    for vid, bid in zip([0, 1, 2], [30, 31, 32]):
        mgr.append(make_training_input_batch([bid]), behavior_version=vid)

    # current_policy_version = 2
    # max_staleness_steps = 0 => only behavior_version == 2 is allowed
    sampled = mgr.sample(current_policy_version=2, max_staleness_steps=0, sampling="prefer_recent", sample_batch_size=2)
    assert sampled is not None
    assert extract_ids_from_batch(sampled) == [32]

    # max_staleness_steps = 1 => versions {1,2}
    sampled = mgr.sample(current_policy_version=2, max_staleness_steps=1, sampling="fifo", sample_batch_size=2)
    assert sampled is not None
    # fifo picks earliest among candidates -> ids [31, 32]
    assert extract_ids_from_batch(sampled) == [31, 32]

    # max_staleness_steps = -1 (no candidate) is not meaningful; simulate all stale by high curr version
    sampled_none = mgr.sample(current_policy_version=100, max_staleness_steps=0, sampling="fifo", sample_batch_size=2)
    assert sampled_none is None


def test_sampling_strategies_prefer_recent_and_fifo_and_random():
    mgr = TrainingBatchReplay(sample_batch_size=3, capacity=10, cpu_offload=True)

    # Append 5 singles with behavior versions 0..4 and ids 40..44
    for vid, bid in zip(range(5), range(40, 45)):
        mgr.append(make_training_input_batch([bid]), behavior_version=vid)

    # Include all candidates
    curr_version = 10
    max_staleness = 10

    # prefer_recent -> last 3 ids: 42,43,44
    recent = mgr.sample(curr_version, max_staleness, sampling="prefer_recent", sample_batch_size=3)
    assert recent is not None
    assert extract_ids_from_batch(recent) == [42, 43, 44]

    # fifo -> first 3 ids: 40,41,42
    fifo = mgr.sample(curr_version, max_staleness, sampling="fifo", sample_batch_size=3)
    assert fifo is not None
    assert extract_ids_from_batch(fifo) == [40, 41, 42]

    # random -> deterministic with seed, compute expected via torch.randperm
    all_ids = [40, 41, 42, 43, 44]
    torch.manual_seed(0)
    perm = torch.randperm(5).tolist()[:3]
    expected = [all_ids[i] for i in perm]
    torch.manual_seed(0)
    rnd = mgr.sample(curr_version, max_staleness, sampling="random", sample_batch_size=3)
    assert rnd is not None
    assert extract_ids_from_batch(rnd) == expected


def test_sample_empty_returns_none():
    mgr = TrainingBatchReplay(sample_batch_size=2, capacity=4, cpu_offload=True)
    out = mgr.sample(current_policy_version=0, max_staleness_steps=0)
    assert out is None


