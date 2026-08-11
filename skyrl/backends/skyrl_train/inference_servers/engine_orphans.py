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

    # [A/B] With SKYRL_FT_DISABLE_ORPHAN_REAP=1 the selector still runs and still
    # reports what it WOULD have killed, but nothing is signalled. That is what makes
    # the reaper's contribution measurable: two otherwise-identical runs, one with the
    # kills suppressed. It is needed because the post-reap verification below cannot
    # answer the question on its own -- Ray reclaims a dead owner's actors
    # asynchronously, so "the GPU is still held" microseconds after the kill is
    # consistent both with the reap having worked and with it having done nothing.
    if os.environ.get("SKYRL_FT_DISABLE_ORPHAN_REAP") == "1":
        logger.warning(
            f"[ft] SKYRL_FT_DISABLE_ORPHAN_REAP=1: NOT killing {len(victims)} selected "
            f"orphan(s) {victims}; holders={[(p, holders.get(p)) for p in victims]}"
        )
        return []

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


# ---------------------------------------------------------------------------
# Kill by BUNDLE -- the precise mechanism, and the reason the heuristics above
# are only a safety net now.
# ---------------------------------------------------------------------------
#
# A slot's identity in Ray's scheduler is its placement-group BUNDLE, and Ray
# publishes that on every actor scheduled into it: `required_resources` carries a
# key `bundle_group_{bundle_index}_{pg_id}`. Verified directly --
#
#   Replica pid=90020 req={'bundle_group_0_95a0d6d7...': 0.001, ...}
#   Replica pid=90435 req={'bundle_group_1_95a0d6d7...': 0.001, ...}
#
# That is exactly what was missing. vLLM's Ray executor inherits OUR placement group
# (vllm/v1/executor/ray_utils.py: `ray.util.get_current_placement_group()`) and
# schedules the model-holding `RayWorkerWrapper` into a specific bundle of it -- so the
# process that actually holds the ~62 GiB is addressable by bundle, even though it is
# unreachable by parentage (its OS parent is the raylet) and unreachable by name.
#
# This replaces inference with enumeration: instead of guessing which processes belong
# to a dead replica, ask Ray which actors are in that replica's bundles.


def bundle_resource_keys(pg_id: str, bundle_indices: Sequence[int]) -> set:
    """The resource keys Ray stamps on actors scheduled into these bundles."""
    return {f"bundle_group_{int(i)}_{pg_id}" for i in bundle_indices}


def select_bundle_actors(
    actors: Sequence[Tuple[Optional[int], str, str, Dict[str, float]]],
    keys: Iterable[str],
) -> List[Tuple[int, str]]:
    """Pick ``(pid, class_name)`` for every actor occupying one of ``keys``. PURE.

    Args:
        actors: ``(pid, state, class_name, required_resources)`` per actor, as
            ``ray.util.state.list_actors(detail=True)`` reports them.
        keys: the target bundles' resource keys (see ``bundle_resource_keys``).

    Returns:
        Sorted ``(pid, class_name)`` for ALIVE actors only. A dead actor is skipped even
        when it still carries a pid: that pid is stale, and pids recycle, so killing it
        could hit an unrelated process.

    The bundle IS the replica, and that is the whole rule. A bundle's resources were
    reserved for one engine, so anything alive in it belongs to that engine by
    construction -- a trainer rank is in a different placement group and another slot is
    at a different bundle index, so neither can match the key.

    Deliberately NOT filtered by actor class as well. A GPU run showed SkyRL's own
    ``InfoActor`` (a provisioning probe) in the listing, which looked like a reason to add
    a class allowlist -- but the probe was already DEAD, killed at provisioning, and
    nothing was ever wrongly killed. Meanwhile an allowlist has a silent failure mode that
    matters: if vLLM renames its worker class on an upgrade, the memory holder stops being
    selected, restarts start failing on free memory again, and there is no error to notice
    -- just a shorter kill list. Guarding a remote hazard by introducing a silent
    regression on upgrade is the wrong trade.

    What DOES have to be checked is liveness; see below.
    """
    wanted = set(keys)
    out = []
    for pid, state, cls, req in actors:
        if not pid:
            continue
        if str(state).upper() != "ALIVE":
            # A dead actor's pid is stale AT BEST, and pids recycle -- on a busy node it
            # may already belong to something unrelated. This is the guard that actually
            # prevents damage, and it was missing: a GPU run listed both a long-dead
            # `InfoActor` (killed at provisioning) and the just-SIGKILLed
            # `VLLMServerActor`, each still carrying a pid.
            continue
        if not any(k in req for k in wanted):
            continue
        out.append((int(pid), cls))
    return sorted(set(out))


def descendants_of(pid: int) -> List[int]:
    """Every live descendant of ``pid``, depth-first, from /proc.

    The bundle tells us which RAY ACTORS belong to a replica; it says nothing about
    processes those actors spawned themselves. Under
    ``distributed_executor_backend="mp"`` the model lives in exactly such a child, so
    killing an actor without its descendants would leave the memory held.
    """

    found: List[int] = []
    frontier = [int(pid)]
    seen = set()
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            kids = open(f"/proc/{cur}/task/{cur}/children").read().split()
        except OSError:
            continue
        for k in kids:
            k = int(k)
            found.append(k)
            frontier.append(k)
    return found


def kill_replica_processes(pids: Sequence[int]) -> List[int]:
    """SIGKILL each pid and all of its descendants. Children first, then the parent.

    Bottom-up so a supervisor-style parent cannot notice a dead child and respawn it
    before we reach it.
    """
    victims: List[int] = []
    for pid in pids:
        victims.extend(reversed(descendants_of(pid)))
        victims.append(int(pid))
    killed = []
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            pass  # already gone; the goal is met either way
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] could not kill replica process {pid}: {e}")
    if killed:
        logger.warning(f"[ft] killed {len(killed)} replica process(es) by bundle: {killed}")
    return killed


def wait_for_gpu_room(
    target_gpu_uuids: Sequence[str], need_bytes: int, timeout_s: float = 60.0
) -> Tuple[Dict[str, Tuple[int, int]], float]:
    """Poll until every target GPU has ``need_bytes`` free, or ``timeout_s`` elapses.

    Reading free memory straight after ``SIGKILL`` measures nothing: process teardown and
    CUDA reclamation are asynchronous. A GPU run made that concrete -- the kill and the
    reading were stamped the same MILLISECOND, reported "23.3 GiB free, needs 55.8 ->
    STILL OCCUPIED", and the restart then succeeded anyway because the memory came back
    while the replacement was loading its weights.

    That mattered twice over: the verdict was wrong, and it also feeds the restart-budget
    decision, so a too-early reading exempts attempts on noise.

    Returns the settled reading and how long the wait took.
    """
    import time as _time

    deadline = _time.monotonic() + max(0.0, timeout_s)
    t0 = _time.monotonic()
    free = gpu_free_bytes(target_gpu_uuids)
    while need_bytes > 0 and free and _time.monotonic() < deadline:
        if min(f for f, _t in free.values()) >= need_bytes:
            break
        _time.sleep(1.0)
        free = gpu_free_bytes(target_gpu_uuids)
    return free, _time.monotonic() - t0


def gpu_free_bytes(target_gpu_uuids: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """``uuid -> (free_bytes, total_bytes)`` for the given GPUs.

    Reads AGGREGATE per-device usage rather than the per-process compute-apps listing:
    the question here is "is this device empty enough to relaunch into", and aggregate
    usage answers it without depending on per-process visibility, which is unreliable
    in containers.
    """
    import subprocess

    wanted = {u for u in target_gpu_uuids if u}
    out: Dict[str, Tuple[int, int]] = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ft] could not read GPU memory: {e}")
        return out
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        uuid = parts[0]
        if wanted and uuid not in wanted:
            continue
        try:
            total_mib, used_mib = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        mib = 1024 * 1024
        out[uuid] = ((total_mib - used_mib) * mib, total_mib * mib)
    return out
