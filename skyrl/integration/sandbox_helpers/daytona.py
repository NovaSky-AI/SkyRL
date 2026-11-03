"""Daytona sandboxes a run can find and clean up again, without touching anyone else's.

A harness deletes its sandbox when a rollout ends. When the training process dies first, its sandboxes stay up,
counting against an org quota others share, until something deletes them through Daytona's API. So every
sandbox a run creates carries the run's ``labels`` and a hard ``ttl_minutes`` (``label_params``): ``cleanup``
then deletes exactly this run's sandboxes, and Daytona deletes any left over once their time is up.

    uv run --isolated --extra daytona python -m skyrl.integration.sandbox_helpers.daytona list --label owner=<you> --label run=<run>
    uv run --isolated --extra daytona python -m skyrl.integration.sandbox_helpers.daytona cleanup --label owner=<you> --label run=<run>

Harness adapters apply ``label_params`` where they create a sandbox
(``skyrl.integration.harbor.environments.LabelledDaytonaEnvironment``).
"""

import argparse
import asyncio
from typing import Any, Dict, List, Mapping, Optional

#: The labels ``cleanup`` needs at least, so it can't reach other people's or other runs' sandboxes.
SCOPE_LABELS = frozenset({"owner", "run"})
COMMANDS = ("list", "cleanup")


def check_labels(labels: Optional[Mapping[Any, Any]]) -> Dict[str, str]:
    """``labels`` as Daytona takes them (strings). Raises ``ValueError`` when there are none."""
    if not labels:
        raise ValueError("labelled Daytona sandboxes need labels, e.g. {owner: <you>, run: <run name>}")
    return {str(key): str(value) for key, value in labels.items()}


def label_params(params: Any, labels: Mapping[str, str], ttl_minutes: Optional[int] = None) -> Any:
    """Daytona's create params ``params`` with ``labels`` added over its own, and ``ttl_minutes`` if given."""
    params.labels = {**(params.labels or {}), **labels}
    if ttl_minutes is not None:
        params.ttl_minutes = ttl_minutes
    return params


def command_is_unscoped(labels: Mapping[str, str]) -> bool:
    """Whether ``labels`` lack an owner or a run label, and so could match other runs' sandboxes."""
    return not SCOPE_LABELS <= set(labels)


async def list_sandboxes(daytona: Any, labels: Mapping[str, str]) -> List[Any]:
    """The sandboxes carrying every one of ``labels``, from an ``AsyncDaytona`` client."""
    from daytona import ListSandboxesQuery

    return [sandbox async for sandbox in daytona.list(ListSandboxesQuery(labels=dict(labels)))]


async def run(command: str, labels: Mapping[str, str], daytona: Any) -> List[Any]:
    """``list`` or ``cleanup`` the sandboxes carrying every one of ``labels``. Returns them.

    Raises ``ValueError`` before reaching Daytona for any other command, or for ``labels`` without an owner and a
    run label. ``cleanup`` tries every deletion, then raises ``RuntimeError`` if any failed.
    """
    if command not in COMMANDS:
        raise ValueError(f"command must be one of {COMMANDS}, got {command!r}")
    if command_is_unscoped(labels):
        raise ValueError(f"labels need {sorted(SCOPE_LABELS)}, so cleanup can't reach other runs' sandboxes")
    sandboxes = await list_sandboxes(daytona, labels)
    if command == "list":
        for sandbox in sandboxes:
            print(sandbox.id, sandbox.state, sandbox.labels)
        print(f"{len(sandboxes)} sandboxes labelled {dict(labels)}")
        return sandboxes
    results = await asyncio.gather(*(daytona.delete(sandbox) for sandbox in sandboxes), return_exceptions=True)
    failed = [(sandbox, result) for sandbox, result in zip(sandboxes, results) if isinstance(result, BaseException)]
    for sandbox, error in failed:
        print(f"could not delete sandbox {sandbox.id}: {type(error).__name__}: {error}")
    print(f"deleted {len(sandboxes) - len(failed)} of {len(sandboxes)} sandboxes labelled {dict(labels)}")
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(sandboxes)} sandboxes could not be deleted; run cleanup again")
    return sandboxes


async def _main(command: str, labels: Dict[str, str]) -> None:
    from daytona import AsyncDaytona

    async with AsyncDaytona() as daytona:
        await run(command, labels, daytona)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or delete the Daytona sandboxes carrying every given label.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--label", action="append", required=True, metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if bad := [item for item in args.label if "=" not in item]:
        parser.error(f"labels are KEY=VALUE, got {bad}")
    args.labels = dict(item.split("=", 1) for item in args.label)
    if command_is_unscoped(args.labels):
        parser.error("give at least an owner label and a run label, so cleanup can't reach other runs' sandboxes")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(_main(args.command, args.labels))
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
