"""Harbor as a SkyRL generator. Needs the ``harbor`` extra.

    from skyrl.integration.harbor import HarborGenerator, HarborTaskDataset

Entrypoints, run scripts and dataset preparation live in examples/train_integrations/harbor.
"""

import importlib
from typing import Any

# Imported on first use, so that importing a submodule doesn't pull in Harbor.
_EXPORTS = {
    "HarborGenerator": "skyrl.integration.harbor.harbor_generator",
    "HarborTaskDataset": "skyrl.integration.harbor.dataset",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
