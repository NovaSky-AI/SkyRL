"""Import-time compatibility shims for third-party version skew.

Imported from ``skyrl/__init__.py`` so the shims are installed before any
``skyrl`` submodule (and therefore any of its third-party imports) is loaded.
"""

import importlib.metadata
import sys


def disable_bundled_flash_attn_cute() -> None:
    """Disable flash-attn 2's stale CuTe module unless real FA4 is installed."""
    try:
        importlib.metadata.version("flash-attn-4")
        return
    except importlib.metadata.PackageNotFoundError:
        pass

    sys.modules.setdefault("flash_attn.cute", None)
