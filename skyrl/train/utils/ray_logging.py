"""
Helpers for Ray actor log handling.

Redirects actor stdout/stderr to a log file so infrastructure logs don't pollute the
driver's stdout, and recovers those logs' tails for diagnostics when an actor fails.
"""

import contextlib
import os
import sys
import time
from typing import TYPE_CHECKING, NoReturn, Sequence

if TYPE_CHECKING:
    from ray.actor import ActorHandle

# Caps total diagnostics size, keeping the tail, sized so the whole cap survives
# SkyRL Tracking.log_exception's truncation of the formatted exception
_MAX_DIAGNOSTICS_CHARS = 10_000


def redirect_actor_output_to_file():
    """
    Redirect stdout and stderr to log file to prevent Ray from forwarding to driver.

    Call this at the very start of any Ray actor/remote function where you want
    to suppress output from appearing on the driver's stdout. The output will
    instead be written to the log file specified by SKYRL_LOG_FILE.

    When SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1, redirection is skipped so all logs
    appear on stdout.

    Note: Do NOT call this in skyrl_entrypoint() - training progress should
    go to stdout.
    """
    from skyrl.env_vars import SKYRL_DUMP_INFRA_LOG_TO_STDOUT

    if SKYRL_DUMP_INFRA_LOG_TO_STDOUT:
        return

    log_file = os.getenv("SKYRL_LOG_FILE")
    if log_file:
        # Ensure the directory exists on this node (may differ from driver in multi-node)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", buffering=1) as log_f:
            os.dup2(log_f.fileno(), sys.stdout.fileno())
            os.dup2(log_f.fileno(), sys.stderr.fileno())


def _tail_file(path: str, max_lines: int, max_bytes: int = 256 * 1024) -> list[str]:
    """
    Return up to the last ``max_lines`` lines of ``path``, reading at most ``max_bytes``.

    The byte bound matters if the input file is unbounded in size, where reading it
    could take nontrivial time on slow filesystems.
    """
    with open(path, "rb") as f:
        file_size = f.seek(0, os.SEEK_END)
        f.seek(max(file_size - max_bytes, 0))
        return (
            f.read()
            .decode(
                # The seek can split a multibyte character;
                # don't let one bad byte kill the diagnostics
                errors="replace"
            )
            .splitlines()[-max_lines:]
        )


def get_actor_logs_tail(
    actor_ids: Sequence[str], *, max_lines_per_actor: int = 100, state_api_timeout_s: int = 10
) -> str | None:
    """
    Best-effort collection of log tails for failure diagnostics. Never raises.

    Looks in the two places actor stderr can land:
    1. The shared `SKYRL_LOG_FILE`, when set: actors calling `redirect_actor_output_to_file`
       redirect their stdout/stderr, and that of any subprocess they spawn (like vLLM's
       engine core), into this file.
    2. Each given actor's Ray worker stderr file (`worker-*.err` in the session logs dir),
       which this function fetches via the Ray state API, covering actors on remote nodes.

    Returns:
        Joined log sections, or None if nothing could be collected.
    """
    sections: list[str] = []

    # Set by SkyRL `initialize_ray` util, in both its calling process
    # and every Ray worker's runtime env
    log_file = os.getenv("SKYRL_LOG_FILE")
    if log_file:
        with contextlib.suppress(Exception):
            tail = _tail_file(log_file, max_lines=max_lines_per_actor)
            if tail:
                sections.append(
                    f"--- tail of SKYRL_LOG_FILE {log_file} "
                    f"(last {len(tail)} lines; infra actors redirect stdout/stderr here) ---\n" + "\n".join(tail)
                )

    with contextlib.suppress(Exception):  # Never raise out of diagnostics collection
        from ray._private.worker import get_dashboard_url
        from ray.util.state import get_log

        # The state API is served by Ray's dashboard HTTP server; without it, resolving the
        # server URL blocks for 20 x 2-s retries before failing, per internal_kv_get_with_retry:
        # https://github.com/ray-project/ray/blob/ray-2.51.1/python/ray/_private/utils.py#L1126-L1147
        if get_dashboard_url():
            for actor_id in actor_ids:
                with contextlib.suppress(Exception):
                    stderr_tail = "".join(
                        get_log(
                            actor_id=actor_id,
                            suffix="err",
                            tail=max_lines_per_actor,
                            timeout=state_api_timeout_s,
                            errors="replace",
                        )
                    ).rstrip()
                    if stderr_tail:
                        sections.append(
                            f"--- stderr tail of actor {actor_id} (full log: "
                            f"`ray logs actor --id {actor_id} --err`) ---\n" + stderr_tail
                        )

    if not sections:
        return None
    diagnostics = "\n\n".join(sections)
    if len(diagnostics) > _MAX_DIAGNOSTICS_CHARS:
        prefix = "...(truncated)...\n"
        diagnostics = prefix + diagnostics[len(prefix) - _MAX_DIAGNOSTICS_CHARS :]
    return diagnostics


def reraise_with_actor_diagnostics(
    e: BaseException, actors: "Sequence[ActorHandle]", context_message: str, log_flush_grace_s: float = 2
) -> NoReturn:
    """
    Re-raise a Ray actor failure as a RuntimeError carrying the actors' log tails.

    Ray surfaces actor-side failures without the actor's stderr, which is where the root
    cause actually lives when a subprocess of the actor (e.g. a vLLM engine-core child) dies;
    the driver-side exception bottoms out at vLLM's `wait_for_engine_startup` with just:
    > RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}

    Args:
        e: The original exception, chained via `__cause__`.
        actors: Actor handles to fall back to when `e` doesn't name the failed actor.
        context_message: Lead text describing what was being attempted.
        log_flush_grace_s: Seconds to wait before snapshotting the logs.
    """
    # ActorDiedError: publicly exposes the failed actor's id
    # RayTaskError (a method failed on a still-alive actor): has a private _actor_id
    failed_id: str | None = getattr(e, "actor_id", None) or getattr(e, "_actor_id", None)
    diagnostics = None
    with contextlib.suppress(Exception):  # Don't mask the original failure
        # vLLM relays the engine-core child's output to the actor's log via a pipe-reader
        # thread that can lag the failure reaching the driver; give it a moment to drain
        time.sleep(log_flush_grace_s)
        diagnostics = get_actor_logs_tail(
            actor_ids=[failed_id] if failed_id else [actor._actor_id.hex() for actor in actors]
        )
    if diagnostics is None:
        log_file = os.getenv("SKYRL_LOG_FILE")
        if log_file:
            location = f"SKYRL_LOG_FILE ({log_file})"
        else:
            logs_dir = "<ray temp dir>/session_latest/logs"
            with contextlib.suppress(Exception):  # Tolerate Ray being uninitialized
                import ray._private.worker

                logs_dir = ray._private.worker._global_node.get_logs_dir_path()
            location = f"{logs_dir}/worker-*.err on the failed actor's node"
        diagnostics = (
            f"(could not fetch actor logs; check {location}, "
            f"or run: ray logs actor --id {failed_id or '<actor_id>'} --err)"
        )
    raise RuntimeError(
        f"{context_message} The root-cause traceback usually lives in the failed actor's "
        f"logs, not in the current Ray exception.\n\n{diagnostics}"
    ) from e
