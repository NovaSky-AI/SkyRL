"""Harbor through skycap, which captures each rollout's exact tokens. Needs the ``harbor`` and ``skycap`` extras.

    from skyrl.integration.harbor_skycap import HarborSkycapGenerator, SkycapRecordIndex, start_servers

Entrypoints and run scripts live in examples/train_integrations/harbor_skycap.
"""

import importlib
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from skyrl.integration.harbor_skycap.harbor_generator import HarborSkycapGenerator
    from skyrl.integration.harbor_skycap.record_index import (
        RecordLog,
        SkycapRecordIndex,
    )
    from skyrl.integration.harbor_skycap.servers import (
        SkycapServers,
        exposure_for,
        start_servers,
    )


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


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_EXPORTS))
