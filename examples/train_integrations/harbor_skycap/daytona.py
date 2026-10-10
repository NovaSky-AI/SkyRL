"""Daytona sandboxes this run can find and clean up again, without touching anyone else's.

Harbor deletes a trial's sandbox when the trial ends. When the training process dies first, its sandboxes
stay up, counting against an org quota others share, until something deletes them through Daytona's API.
``LabelledDaytonaEnvironment`` tags every sandbox it creates with ``labels`` and gives it a hard
``ttl_minutes``, so this module's ``cleanup`` can delete exactly this run's sandboxes, and Daytona deletes any left over
once their time is up. Use it in place of ``type: daytona``:

    harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.daytona:LabelledDaytonaEnvironment
    harbor_trial_config.environment.kwargs.labels.owner=<you> harbor_trial_config.environment.kwargs.labels.run=<run>
    harbor_trial_config.environment.kwargs.ttl_minutes=180

    python -m examples.train_integrations.harbor_skycap.daytona list --label owner=<you> --label run=<run>
    python -m examples.train_integrations.harbor_skycap.daytona cleanup --label owner=<you> --label run=<run>
"""

import argparse
import asyncio
from typing import Any, Dict, Optional

from harbor.environments.daytona import DaytonaEnvironment


class LabelledDaytonaEnvironment(DaytonaEnvironment):
    def __init__(
        self, *args: Any, labels: Optional[Dict[str, str]] = None, ttl_minutes: Optional[int] = None, **kwargs: Any
    ) -> None:
        """``labels`` go on every sandbox; ``ttl_minutes`` is how long one may live, whatever happens to us."""
        if not labels:
            raise ValueError("LabelledDaytonaEnvironment needs labels, e.g. {owner: <you>, run: <run name>}")
        super().__init__(*args, **kwargs)
        self._labels = {str(key): str(value) for key, value in labels.items()}
        self._ttl_minutes = ttl_minutes

    async def _create_sandbox(self, params: Any, daytona: Any = None) -> None:
        params.labels = {**(params.labels or {}), **self._labels}
        if self._ttl_minutes is not None:
            params.ttl_minutes = self._ttl_minutes
        await super()._create_sandbox(params=params, daytona=daytona)


async def _ours(daytona: Any, labels: Dict[str, str]) -> list:
    from daytona import ListSandboxesQuery

    return [sandbox async for sandbox in daytona.list(ListSandboxesQuery(labels=labels))]


async def _main(command: str, labels: Dict[str, str]) -> None:
    from daytona import AsyncDaytona

    async with AsyncDaytona() as daytona:
        sandboxes = await _ours(daytona, labels)
        if command == "list":
            for sandbox in sandboxes:
                print(sandbox.id, sandbox.state, sandbox.labels)
            print(f"{len(sandboxes)} sandboxes labelled {labels}")
            return
        await asyncio.gather(*(daytona.delete(sandbox) for sandbox in sandboxes))
        print(f"deleted {len(sandboxes)} sandboxes labelled {labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List or delete the Daytona sandboxes carrying every given label.")
    parser.add_argument("command", choices=["list", "cleanup"])
    parser.add_argument("--label", action="append", required=True, metavar="KEY=VALUE")
    args = parser.parse_args()
    labels = dict(item.split("=", 1) for item in args.label)
    if command_is_unscoped(labels):
        parser.error("give at least an owner label and a run label, so cleanup can't reach other runs' sandboxes")
    asyncio.run(_main(args.command, labels))


def command_is_unscoped(labels: Dict[str, str]) -> bool:
    return not {"owner", "run"} <= set(labels)


if __name__ == "__main__":
    main()
