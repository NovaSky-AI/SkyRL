"""Harbor's Daytona environment, labelling every sandbox it creates (``skyrl.integration.sandboxes.daytona``).

``LabelledDaytonaEnvironment`` tags every sandbox with ``labels`` and gives it a hard ``ttl_minutes``, so
``cleanup`` can delete exactly this run's sandboxes, and Daytona deletes any left over once their time is up.
Use it in place of ``type: daytona``:

    harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.daytona:LabelledDaytonaEnvironment
    harbor_trial_config.environment.kwargs.labels.owner=<you> harbor_trial_config.environment.kwargs.labels.run=<run>
    harbor_trial_config.environment.kwargs.ttl_minutes=180

    uv run --isolated --extra harbor python -m examples.train_integrations.harbor_skycap.daytona list --label owner=<you> --label run=<run>
    uv run --isolated --extra harbor python -m examples.train_integrations.harbor_skycap.daytona cleanup --label owner=<you> --label run=<run>
"""

from typing import Any, Dict, Optional

from harbor.environments.daytona import DaytonaEnvironment

from skyrl.integration.sandboxes.daytona import (  # noqa: F401 - the CLI, re-exported
    check_labels,
    command_is_unscoped,
    label_params,
    main,
)


class LabelledDaytonaEnvironment(DaytonaEnvironment):
    def __init__(
        self, *args: Any, labels: Optional[Dict[str, str]] = None, ttl_minutes: Optional[int] = None, **kwargs: Any
    ) -> None:
        """``labels`` go on every sandbox; ``ttl_minutes`` is how long one may live, whatever happens to us."""
        self._labels = check_labels(labels)
        super().__init__(*args, **kwargs)
        self._ttl_minutes = ttl_minutes

    async def _create_sandbox(self, params: Any, daytona: Any = None) -> None:
        await super()._create_sandbox(params=label_params(params, self._labels, self._ttl_minutes), daytona=daytona)


if __name__ == "__main__":
    main()
