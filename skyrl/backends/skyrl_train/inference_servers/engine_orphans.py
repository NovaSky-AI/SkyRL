"""Reap orphaned vLLM engine processes that are still holding a GPU.

Engine fault tolerance, Part 2. Restarting a dead engine means relaunching into its
ORIGINAL placement-group bundle -- same node, same GPUs -- and that only works if the
GPUs are actually free. They frequently are not.

vLLM v1 runs its ``EngineCore`` (and, with the Ray executor, ``RayWorkerProc``) as
SEPARATE processes spawned by the server actor. When that actor dies abruptly -- a
crash, an OOM-kill, an external SIGKILL, exactly the failures fault tolerance exists
for -- those children are reparented to init and keep running, holding their
``gpu_memory_utilization`` share of the device. Observed on an 8xH100: killing one
engine's actor left 62.8 GiB of 79.2 GiB allocated, and every restart into that bundle
failed with "Free memory on device ... less than desired GPU memory utilization" until
the orphan was reaped.

Ray does not help here: it reaps the worker process it owns, not that worker's
grandchildren, and the driver never learns their pids.

**Safety.** This kills processes, so the selection rule is deliberately narrow and is a
PURE function (``select_orphans``) that is tested independently of any GPU:

  * the process must be ORPHANED -- ``ppid == 1``. A live engine's EngineCore is a
    child of its living server actor, so it can never match;
  * its name must be one vLLM gives its own subprocesses;
  * it must hold memory on a GPU we are about to reclaim.

All three must hold. A trainer rank, a healthy engine, and anything not ours fail at
least one.
"""

import logging
import os
import signal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VLLM_SUBPROCESS_NAMES = ("VLLM::EngineCore", "VLLM::Worker")
"""Process names vLLM gives the children that hold device memory. Matched as a prefix
because the kernel truncates ``comm`` to 15 characters (``VLLM::EngineCor``)."""


def select_orphans(
    compute_apps: Sequence[Tuple[str, int, int]],
    process_info: Dict[int, Tuple[int, str]],
    target_gpu_uuids: Iterable[str],
) -> List[int]:
    """Pick the pids that are safe to kill. PURE -- no side effects, no GPU, no /proc.

    Args:
        compute_apps: ``(gpu_uuid, pid, used_mib)`` as nvidia-smi reports them.
        process_info: ``pid -> (ppid, comm)``.
        target_gpu_uuids: the GPUs whose memory we intend to reclaim.

    Returns:
        Sorted pids satisfying ALL of: on a target GPU, orphaned (``ppid == 1``), and
        named like a vLLM subprocess. A pid missing from ``process_info`` has already
        exited and is skipped.
    """
    targets = {u for u in target_gpu_uuids if u}
    chosen = set()
    for gpu_uuid, pid, _used in compute_apps:
        if gpu_uuid not in targets:
            continue
        info = process_info.get(pid)
        if info is None:
            continue  # already gone
        ppid, comm = info
        if ppid != 1:
            continue  # still parented -- a LIVE engine or trainer rank, never ours to kill
        if not any(comm.startswith(n[:15]) for n in VLLM_SUBPROCESS_NAMES):
            continue
        chosen.add(pid)
    return sorted(chosen)


def _read_compute_apps() -> List[Tuple[str, int, int]]:
    """``nvidia-smi`` compute apps, or [] if it cannot be run."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ft] could not run nvidia-smi to find orphaned engines: {e}")
        return []
    apps = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            apps.append((parts[0], int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return apps


def _read_process_info(pids: Iterable[int]) -> Dict[int, Tuple[int, str]]:
    """``pid -> (ppid, comm)`` from /proc. Missing pids are simply absent."""
    info: Dict[int, Tuple[int, str]] = {}
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "r") as fh:
                stat = fh.read()
            # comm is parenthesised and may itself contain spaces/parens, so split on
            # the LAST ')': everything before is "pid (comm", after is the rest.
            head, _, tail = stat.rpartition(")")
            comm = head[head.index("(") + 1 :]
            ppid = int(tail.split()[1])
        except Exception:  # noqa: BLE001
            continue
        info[pid] = (ppid, comm)
    return info


def reap_orphaned_engines(target_gpu_uuids: Sequence[str]) -> List[int]:
    """Find and SIGKILL orphaned vLLM engine processes on the target GPUs.

    Runs ON the node holding those GPUs. Returns the pids killed.
    """
    apps = _read_compute_apps()
    if not apps:
        return []
    victims = select_orphans(apps, _read_process_info(p for _, p, _ in apps), target_gpu_uuids)
    killed = []
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            logger.warning(f"[ft] reaped orphaned vLLM engine process {pid} holding a target GPU")
        except ProcessLookupError:
            pass  # exited between selection and kill; the goal is met either way
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] could not kill orphaned process {pid}: {e}")
    return killed


def gpu_uuids_for_ids(gpu_ids: Optional[Sequence[int]]) -> List[str]:
    """Map physical GPU indices to their UUIDs, which nvidia-smi keys memory by."""
    if not gpu_ids:
        return []
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ft] could not map GPU ids to uuids: {e}")
        return []
    by_index = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                by_index[int(parts[0])] = parts[1]
            except ValueError:
                continue
    return [by_index[i] for i in gpu_ids if i in by_index]
