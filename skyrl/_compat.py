"""Import-time compatibility shims for third-party version skew.

Imported from ``skyrl/__init__.py`` so the shims are installed before any
``skyrl`` submodule (and therefore any of its third-party imports) is loaded.
"""

import sys


def disable_flash_attn_cute() -> None:
    """Make ``flash_attn.cute`` (FA4) unimportable instead of AttributeError-y.

    flash-attn 2.8.3 bundles ``flash_attn/cute`` against the nvidia-cutlass-dsl
    4.0/4.1 API, but vLLM 0.26 hard-pins ``nvidia-cutlass-dsl==4.6.0`` where
    ``cutlass.cute.core.ThrMma`` no longer exists, so importing the subpackage
    raises ``AttributeError``. megatron-core probes FA4 with a bare
    ``except ImportError``, so that AttributeError escapes and takes down every
    ``megatron.core.transformer.attention`` import (i.e. all of megatron-core).

    Poisoning the module entry turns the probe into a plain ImportError, so
    megatron-core falls back to ``HAVE_FA4 = False``. Nothing in SkyRL uses FA4.
    """
    sys.modules.setdefault("flash_attn.cute", None)
