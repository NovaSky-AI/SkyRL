"""Selecting orphaned vLLM engine processes to reap.

Found on hardware, not by reading: SIGKILLing an inference engine's server actor leaves
vLLM's ``EngineCore`` -- a SEPARATE child process -- running and still holding its
``gpu_memory_utilization`` share of the device. On an 8xH100 that was 62.8 of 79.2 GiB,
and all three restarts into that bundle failed with "Free memory on device ... less than
desired GPU memory utilization". Ray does not help: it reaps the worker it owns, not
that worker's children, and the driver never learns their pids.

So Part 2 has to reap them, which means **killing processes**. That makes
``select_orphans`` the most dangerous function in the feature, and it is written as a
pure function precisely so it can be tested exhaustively without a GPU. Its rule is a
conjunction of three conditions, and these tests check that dropping ANY one of them
would spare the wrong process:

  * on a GPU we are reclaiming,
  * ``ppid == 1`` (orphaned -- a live engine's core is parented to its living actor),
  * named like a vLLM subprocess.

The case that matters most is the negative one: a HEALTHY engine and a TRAINER rank must
never be selected, however much memory they hold.

Run:
    uv run --extra dev pytest tests/backends/skyrl_train/inference_servers/test_engine_orphans.py
"""

from skyrl.backends.skyrl_train.inference_servers.engine_orphans import select_orphans

# nvidia-smi reports memory per (gpu_uuid, pid). Two engine GPUs and one trainer GPU.
GPU_DEAD = "GPU-aaaa"
GPU_LIVE = "GPU-bbbb"
GPU_TRAINER = "GPU-cccc"


def test_an_orphaned_engine_core_on_a_target_gpu_is_selected():
    """The case the whole mechanism exists for."""
    apps = [(GPU_DEAD, 1001, 63774)]
    info = {1001: (1, "VLLM::EngineCor")}  # comm is truncated to 15 chars by the kernel
    assert select_orphans(apps, info, [GPU_DEAD]) == [1001]


def test_a_live_engine_is_never_selected():
    """Its EngineCore is a child of a LIVE server actor, so ppid != 1. Selecting it
    would kill a serving engine to make room for a replacement of a different one."""
    apps = [(GPU_LIVE, 2002, 61134)]
    info = {2002: (2000, "VLLM::EngineCor")}
    assert select_orphans(apps, info, [GPU_LIVE]) == []


def test_a_trainer_rank_is_never_selected():
    """Holds GPU memory and may well be orphaned-looking, but is not a vLLM subprocess.
    Killing one would take down the training job to restart an engine."""
    apps = [(GPU_TRAINER, 3003, 3696)]
    info = {3003: (1, "ray::PolicyWorke")}
    assert select_orphans(apps, info, [GPU_TRAINER]) == []


def test_an_orphan_on_someone_elses_gpu_is_left_alone():
    """Scope is the bundle being reclaimed. Another slot's orphan is that slot's
    business, and will be reaped when IT restarts."""
    apps = [(GPU_LIVE, 4004, 62000)]
    info = {4004: (1, "VLLM::EngineCor")}
    assert select_orphans(apps, info, [GPU_DEAD]) == []


def test_the_worker_subprocess_name_also_matches():
    """With the Ray executor, TP/PP workers are their own processes holding memory."""
    apps = [(GPU_DEAD, 5005, 40000)]
    info = {5005: (1, "VLLM::Worker_TP")}
    assert select_orphans(apps, info, [GPU_DEAD]) == [5005]


def test_a_pid_that_has_already_exited_is_skipped():
    """nvidia-smi's snapshot races process exit; a missing /proc entry means the goal is
    already met, not that something is wrong."""
    apps = [(GPU_DEAD, 6006, 63000)]
    assert select_orphans(apps, {}, [GPU_DEAD]) == []


def test_the_mixed_realistic_case():
    """What the 8xH100 run actually looked like: one orphan among three live engines and
    four trainer ranks. Exactly one pid may be selected."""
    apps = [
        (GPU_DEAD, 1001, 63774),  # the orphan
        (GPU_LIVE, 1002, 61134),  # live engine
        ("GPU-dddd", 1003, 65846),  # live engine
        ("GPU-eeee", 1004, 61364),  # live engine
        (GPU_TRAINER, 2001, 3696),  # trainer rank
        (GPU_TRAINER, 2002, 1636),  # trainer rank's producer sidecar
    ]
    info = {
        1001: (1, "VLLM::EngineCor"),
        1002: (900, "VLLM::EngineCor"),
        1003: (901, "VLLM::EngineCor"),
        1004: (902, "VLLM::EngineCor"),
        2001: (1, "ray::PolicyWorke"),
        2002: (1, "ray::_RDTProduce"),
    }
    assert select_orphans(apps, info, [GPU_DEAD]) == [1001]


def test_duplicate_reports_for_one_pid_collapse():
    """A process can appear once per GPU it touches; it must be killed once."""
    apps = [(GPU_DEAD, 7007, 30000), (GPU_DEAD, 7007, 30000)]
    info = {7007: (1, "VLLM::EngineCor")}
    assert select_orphans(apps, info, [GPU_DEAD]) == [7007]


def test_no_targets_selects_nothing():
    """A slot whose physical GPU ids could not be resolved must reap nothing rather than
    fall back to a broad sweep."""
    apps = [(GPU_DEAD, 8008, 63000)]
    info = {8008: (1, "VLLM::EngineCor")}
    assert select_orphans(apps, info, []) == []
    assert select_orphans(apps, info, [""]) == []


def test_a_comm_with_spaces_and_parens_does_not_confuse_the_matcher():
    """/proc comm can contain anything; only the prefix rule should decide."""
    apps = [(GPU_DEAD, 9009, 1000), (GPU_DEAD, 9010, 1000)]
    info = {9009: (1, "python (worker)"), 9010: (1, "VLLM::EngineCor")}
    assert select_orphans(apps, info, [GPU_DEAD]) == [9010]
