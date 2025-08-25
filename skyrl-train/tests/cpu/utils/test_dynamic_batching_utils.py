import pytest

from skyrl_train.workers.worker_utils import BatchIterator
from skyrl_train.utils.dynamic_batching import get_seqlen_balanced_partitions, calculate_num_micro_batches

from tests.gpu.utils import make_dummy_training_batch, get_test_actor_config, make_variable_length_training_batch


@pytest.mark.parametrize(
    "seq_lengths,max_tokens,expected_num_batches",
    [
        ([100] * 10, 500, 2),
        ([50, 100, 150, 200], 300, 2),
        ([500], 500, 1),
        ([50] * 8, 200, 2),
        ([100, 200, 300], 350, 2),
    ],
)
def test_dynamic_batch_iterator_core(seq_lengths, max_tokens, expected_num_batches):
    batch = make_variable_length_training_batch(seq_lengths, pad_to_length=max(seq_lengths))
    # Create a mock config for BatchIterator

    cfg = get_test_actor_config()
    cfg.trainer.use_dynamic_batching = True
    cfg.trainer.max_token_len_per_gpu_train = max_tokens
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy_mini_batch_size = len(seq_lengths)

    iterator = BatchIterator(batch, cfg=cfg, dp_size=1, dynamic_bsz=True)

    assert len(iterator) == expected_num_batches

    total_sequences = 0
    for exp in iterator:
        assert exp.attention_mask.sum().item() <= max_tokens
        assert all(hasattr(exp, f) for f in ["sequences", "attention_mask", "num_actions"])
        total_sequences += exp.sequences.shape[0]

    assert total_sequences == len(seq_lengths)


def test_dynamic_iterator_multi_epoch():
    batch = make_dummy_training_batch(batch_size=4, seq_len=100)

    cfg = get_test_actor_config()
    cfg.trainer.use_dynamic_batching = True
    cfg.trainer.max_token_len_per_gpu_train = 300
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy_mini_batch_size = 4

    iterator = BatchIterator(batch, cfg=cfg, dp_size=1, dynamic_bsz=True)

    epoch_counts = [sum(1 for _ in iterator) for _ in range(3)]
    assert len(set(epoch_counts)) == 1 and epoch_counts[0] == len(iterator)


@pytest.mark.parametrize(
    "seq_lengths,k_partitions,expected_partitions",
    [
        ([100, 200, 300], 1, [[0, 1, 2]]),
        ([100, 200, 300], 3, [[2], [1], [0]]),
        ([100, 100], 2, [[1], [0]]),
        ([50, 100, 150, 200], 2, [[0, 3], [1, 2]]),
    ],
)
def test_karmarkar_karp_partitioning(seq_lengths, k_partitions, expected_partitions):
    partitions = get_seqlen_balanced_partitions(seq_lengths, k_partitions)

    for i, partition in enumerate(partitions):
        for j, p in enumerate(partition):
            assert p == expected_partitions[i][j]
    assert len(partitions) == k_partitions


@pytest.mark.parametrize(
    "token_counts,max_tokens,min_micro_batch,expected",
    [
        ([100, 200, 300, 400], 500, None, 2),
        ([100, 100], 500, None, 1),
        ([100, 100], 500, 3, 3),
        ([50, 50, 50, 50], 150, None, 2),
        ([250, 250], 500, None, 1),
    ],
)
def test_micro_batch_calculation(token_counts, max_tokens, min_micro_batch, expected):
    num_micro = calculate_num_micro_batches(token_counts, max_tokens, min_num_micro_batch=min_micro_batch)

    assert num_micro == expected


def test_batch_iterator_with_dynamic_batching():
    """Test BatchIterator with dynamic batching enabled."""
    cfg = get_test_actor_config()
    cfg.trainer.use_dynamic_batching = True
    cfg.trainer.max_token_len_per_gpu_train = 200
    cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 2

    seq_lengths = [50, 100, 150, 200]
    batch = make_variable_length_training_batch(seq_lengths, num_actions=4)

    iterator = BatchIterator(data=batch, cfg=cfg, dp_size=1, dynamic_bsz=True, dp_group=None)

    # Check that micro-batches are created
    assert len(iterator) > 0, "Should have at least one micro-batch"

    # Iterate through all micro-batches
    total_sequences = 0
    for exp in iterator:
        assert hasattr(exp, "sequences"), "Experience should have sequences"
        assert hasattr(exp, "attention_mask"), "Experience should have attention_mask"
        total_sequences += exp.sequences.shape[0]

    assert total_sequences == len(
        seq_lengths
    ), f"Should process all {len(seq_lengths)} sequences, got {total_sequences}"
    print(f"BatchIterator with dynamic batching processed {total_sequences} sequences in {len(iterator)} micro-batches")
