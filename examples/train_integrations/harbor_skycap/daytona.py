"""Moved: ``LabelledDaytonaEnvironment`` is ``skyrl.integration.harbor.environments``, the ``list``/``cleanup``
CLI is ``python -m skyrl.integration.sandboxes.daytona``. This module keeps the old import path and command working."""

from skyrl.integration.harbor.environments import LabelledDaytonaEnvironment  # noqa: F401
from skyrl.integration.sandboxes.daytona import main

if __name__ == "__main__":
    main()
