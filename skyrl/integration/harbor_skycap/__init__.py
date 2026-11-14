"""Harbor through skycap, which captures each rollout's exact tokens. Needs the ``harbor`` and ``skycap`` extras.

    from skyrl.integration.harbor_skycap import HarborSkycapGenerator, SkycapRecordIndex, start_servers

Entrypoints and run scripts live in examples/train_integrations/harbor_skycap.
"""

import importlib
from typing import Any

# Imported on first use, so that `python -m skyrl.integration.harbor_skycap.record_index pull`
# doesn't import Harbor.
_EXPORTS = {
    "HarborSkycapGenerator": "skyrl.integration.harbor_skycap.harbor_generator",
    "RecordLog": "skyrl.integration.harbor_skycap.record_index",
    "SkycapRecordIndex": "skyrl.integration.harbor_skycap.record_index",
    "SkycapServers": "skyrl.integration.harbor_skycap.servers",
    "start_servers": "skyrl.integration.harbor_skycap.servers",
    "exposure_for": "skyrl.integration.harbor_skycap.servers",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
