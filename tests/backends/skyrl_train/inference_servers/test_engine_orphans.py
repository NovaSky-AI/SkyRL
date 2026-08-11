"""Selecting orphaned vLLM engine processes to reap.

Found on hardware, not by reading: SIGKILLing an inference engine's server actor leaves
vLLM's subprocesses running and still holding the device. On an 8xH100 that was 62.8 of
79.2 GiB, and all three restarts into that bundle failed with "Free memory on device ...
less than desired GPU memory utilization". Ray does not help -- it reaps the worker it
owns, not that worker's children, and the driver never learns their pids.

So Part 2 has to reap them, which means **killing processes**, which makes
``select_orphans`` the most dangerous function in the feature. It is pure so it can be
tested exhaustively without a GPU.

The rule is deliberately about OWNERSHIP, not memory:

  * ``ppid == 1`` -- orphaned. vLLM's subprocesses are parented to the server actor (or
    its EngineCore) while that parent lives, so a parent of 1 *means* the owner died,
    which makes the child garbage by definition. A healthy engine cannot satisfy it.
  * the name is one vLLM gives its own subprocesses.

An earlier version also required "holds memory on the GPU we want", which made the
reaper useless in the DEFAULT configuration: with
``distributed_executor_backend="ray"`` the model lives in ``ray::RayWorkerProc``
actors, so the orphanable EngineCore may hold no device memory, never appear in
nvidia-smi, and never even be considered. The tests below pin that regression.

Run:
    uv run --extra dev pytest tests/backends/skyrl_train/inference_servers/test_engine_orphans.py
"""

from skyrl.backends.skyrl_train.inference_servers.engine_orphans import select_orphans


def test_an_orphaned_engine_core_is_selected():
    """The case the whole mechanism exists for. comm is truncated to 15 chars."""
    assert select_orphans({1001: (1, "VLLM::EngineCor")}) == [1001]


def test_an_orphan_holding_no_gpu_memory_is_still_selected():
    """THE regression. Under the Ray executor the model sits in RayWorkerProc actors, so
    the orphaned EngineCore may hold nothing itself -- and killing it is precisely what
    makes Ray reclaim the actors it owned. Gating selection on device memory meant this
    process was never considered and the reaper did nothing at all."""
    holders = {}  # nvidia-smi reports nobody
    assert select_orphans({1001: (1, "VLLM::EngineCor")}, holders) == [1001]


def test_gpu_information_never_changes_the_decision():
    """It is log enrichment only; if it could gate selection the bug above returns."""
    info = {1001: (1, "VLLM::EngineCor")}
    assert select_orphans(info, None) == select_orphans(info, {1001: ("GPU-aaaa", 63774)}) == [1001]


def test_a_live_engine_is_never_selected():
    """Its subprocess is parented to a LIVE server actor, so ppid != 1. Selecting it
    would kill a serving engine to make room for a replacement of a different one."""
    assert select_orphans({2002: (2000, "VLLM::EngineCor")}) == []


def test_a_trainer_rank_is_never_selected():
    """Holds GPU memory and may be reparented, but is not a vLLM subprocess. Killing one
    would take down the training job to restart an engine."""
    assert select_orphans({3003: (1, "ray::PolicyWorke")}) == []


def test_a_producer_sidecar_is_never_selected():
    """The RDT producer server shares the trainer rank's GPU and is not ours to kill."""
    assert select_orphans({3004: (1, "ray::_RDTProduce")}) == []


def test_both_worker_flavours_match():
    """With the Ray executor and TP/PP, and with the mp executor, workers are their own
    processes holding the model."""
    assert select_orphans({5005: (1, "VLLM::Worker_TP"), 5006: (1, "VLLM::Worker_DP")}) == [5005, 5006]


def test_the_mixed_realistic_case():
    """What the 8xH100 run looked like: one dead engine's orphan among three live
    engines and four trainer ranks. Exactly one pid may be selected."""
    info = {
        1001: (1, "VLLM::EngineCor"),  # the orphan
        1002: (900, "VLLM::EngineCor"),  # live engine
        1003: (901, "VLLM::EngineCor"),  # live engine
        1004: (902, "VLLM::EngineCor"),  # live engine
        2001: (1, "ray::PolicyWorke"),  # trainer rank
        2002: (1, "ray::_RDTProduce"),  # producer sidecar
        2003: (1, "ray::RayWorkerPro"),  # someone else's Ray worker
    }
    assert select_orphans(info) == [1001]


def test_an_empty_scan_selects_nothing():
    assert select_orphans({}) == []


def test_a_comm_with_spaces_and_parens_does_not_confuse_the_matcher():
    """/proc comm can contain anything; only the prefix rule should decide."""
    assert select_orphans({9009: (1, "python (worker)"), 9010: (1, "VLLM::EngineCor")}) == [9010]


def test_zombies_are_excluded_by_the_proc_reader():
    """Killing a zombie frees nothing -- it has already exited and is only awaiting its
    parent's wait(). The first reaping run logged 16 such "reaps", all stale zombies from
    earlier runs, while the process actually holding the GPU was untouched. Excluded in
    `_read_process_info` (state 'Z') so they never reach the selector at all."""
    from unittest.mock import mock_open, patch

    from skyrl.backends.skyrl_train.inference_servers.engine_orphans import (
        _read_process_info,
    )

    zombie = "3577 (VLLM::Worker_TP) Z 1 3577 0 0 -1 0 0 0 0 0 0 0"
    live = "4001 (VLLM::EngineCor) S 1 4001 0 0 -1 0 0 0 0 0 0 0"
    with patch("builtins.open", mock_open(read_data=zombie)):
        assert _read_process_info([3577]) == {}
    with patch("builtins.open", mock_open(read_data=live)):
        assert _read_process_info([4001]) == {4001: (1, "VLLM::EngineCor")}
