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
    process_info: Dict[int, Tuple[int, str]],
    gpu_holders: Optional[Dict[int, Tuple[str, int]]] = None,
) -> List[int]:
    """Pick the pids that are safe to kill. PURE -- no side effects, no GPU, no /proc.

    The rule is a conjunction of exactly two facts, and both come from ``/proc``:

    * ``ppid == 1`` -- the process is ORPHANED. vLLM's subprocesses are spawned by the
      server actor (or by its EngineCore), so while that parent lives they are parented
      to it. A parent of 1 therefore *means* the owner died, which makes the child
      garbage by definition. A healthy engine can never satisfy this.
    * the name is one vLLM gives its own subprocesses -- so a trainer rank, a Ray
      worker belonging to something else, or any other tenant is out of scope.

    Deliberately NOT filtered by "holds memory on the GPU we want". An earlier version
    was, and it made the reaper useless in the default configuration: with
    ``distributed_executor_backend="ray"`` the model lives in ``ray::RayWorkerProc``
    ACTORS, so the orphanable ``EngineCore`` may hold little or no device memory and
    never appear in nvidia-smi at all -- meaning it was never even considered. Killing
    it is what causes Ray to reclaim the worker actors it owned, which is where the
    memory actually is.

    Any orphaned vLLM subprocess is garbage regardless of which GPU it sat on, so
    dropping the GPU scope loses no safety: it cannot select a process whose owner is
    still alive.

    Args:
        process_info: ``pid -> (ppid, comm)`` for candidate processes.
        gpu_holders: optional ``pid -> (gpu_uuid, used_mib)``, used only to enrich the
            log; it never affects selection.

    Returns:
        Sorted pids that are orphaned AND vLLM-named.
    """
    del gpu_holders  # logging only; see the docstring on why it must not gate selection
    chosen = set()
    for pid, (ppid, comm) in process_info.items():
        if ppid != 1:
            continue  # still parented -- its owner is alive, so it is not ours to kill
        if not any(comm.startswith(n[:15]) for n in VLLM_SUBPROCESS_NAMES):
            continue
        chosen.add(pid)
    return sorted(chosen)


def scan_orphan_candidates() -> Dict[int, Tuple[int, str]]:
    """``pid -> (ppid, comm)`` for every process on this node, from /proc.

    Sourced from /proc rather than from nvidia-smi because the process we most need to
    find -- an orphaned EngineCore under the Ray executor -- may hold no device memory
    and so be absent from nvidia-smi entirely.
    """
    import os as _os

    out: Dict[int, Tuple[int, str]] = {}
    for entry in _os.listdir("/proc"):
        if not entry.isdigit():
            continue
        info = _read_process_info([int(entry)])
        out.update(info)
    return out


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
    """``pid -> (ppid, comm)`` from /proc, skipping ZOMBIES. Missing pids are absent.

    Zombies are excluded because killing one is a no-op -- it has already exited and is
    only waiting to be reaped by its parent -- so selecting them produces a log full of
    "reaped" lines that freed nothing. The first run with reaping enabled reported 16
    such kills, every one of them a stale zombie, while the process actually holding the
    GPU went untouched.
    """
    info: Dict[int, Tuple[int, str]] = {}
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "r") as fh:
                stat = fh.read()
            # comm is parenthesised and may itself contain spaces/parens, so split on
            # the LAST ')': everything before is "pid (comm", after is the rest.
            head, _, tail = stat.rpartition(")")
            comm = head[head.index("(") + 1 :]
            fields = tail.split()
            state, ppid = fields[0], int(fields[1])
        except Exception:  # noqa: BLE001
            continue
        if state == "Z":
            continue
        info[pid] = (ppid, comm)
    return info


def reap_orphaned_engines(target_gpu_uuids: Sequence[str]) -> List[int]:
    """Find and SIGKILL orphaned vLLM engine processes on this node.

    Runs ON the node whose GPUs we are reclaiming. ``target_gpu_uuids`` is used only to
    report how much memory was held and whether it actually came back -- selection is
    ownership-based (see ``select_orphans``), not memory-based.

    Returns the pids killed.
    """
    holders = {pid: (gpu, mib) for gpu, pid, mib in _read_compute_apps()}
    victims = select_orphans(scan_orphan_candidates(), holders)
    killed = []
    for pid in victims:
        held = holders.get(pid)
        where = f" holding {held[1]} MiB on {held[0]}" if held else " (no device memory of its own)"
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
            logger.warning(f"[ft] reaped orphaned vLLM process {pid}{where}")
        except ProcessLookupError:
            pass  # exited between selection and kill; the goal is met either way
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] could not kill orphaned process {pid}: {e}")

    # Say plainly whether the GPUs actually came back. Without this the failure mode is
    # silent: the reap "succeeds", the relaunch fails on free memory, and nothing
    # connects the two. Ray reclaims a dead owner's actors asynchronously, so a
    # still-occupied GPU here is informative rather than final.
    if target_gpu_uuids:
        wanted = {u for u in target_gpu_uuids if u}
        still = [(g, p, m) for g, p, m in _read_compute_apps() if g in wanted]
        if still:
            logger.warning(
                f"[ft] after reaping, these processes still hold the target GPU(s): "
                f"{[(p, m) for _g, p, m in still]}. If this persists the relaunch will keep "
                f"failing on free memory; distributed_executor_backend='mp' avoids the "
                f"Ray-actor ownership chain entirely."
            )
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
