"""Keep a broken FlashAttention-4 (``flash_attn.cute``) install from taking down ``megatron.core``.

flash-attn 2.8.x ships the FA4 ``flash_attn.cute`` package alongside FA2. It is written against
a specific ``nvidia-cutlass-dsl`` release; vLLM >= 0.28.1 pins a newer cutlass DSL whose
``cutlass.cute.core`` no longer has the symbols FA4 imports (``ThrMma``), so importing
``flash_attn.cute`` raises ``AttributeError`` rather than ``ImportError``.

``megatron.core.transformer.attention`` probes FA4 with ``from flash_attn.cute import
flash_attn_varlen_func`` guarded only by ``except ImportError`` (megatron-core 0.20), so the
``AttributeError`` escapes and every ``import megatron.core`` fails. SkyRL never uses FA4 (TE
dispatches to FA2 or cuDNN fused attention), so when the probe import fails for any reason
other than a plain ``ImportError`` we register ``flash_attn.cute`` as unavailable
(``sys.modules[name] = None`` makes any later import of it raise ``ImportError``), which is
exactly the state megatron-core handles.

No-op when ``flash_attn.cute`` imports cleanly or is genuinely absent. Call it before the first
``megatron`` import in a process; ``skyrl.backends.skyrl_train.workers.megatron`` does so on
package import.

DELETE THIS PATCH once megatron-core's FA4 probe catches ``Exception`` or the flash-attn /
cutlass DSL pins agree again.
"""

import importlib
import sys

from loguru import logger

_MODULE = "flash_attn.cute"
_PATCHED_FLAG = "_skyrl_fa4_cute_import_patched"


def patch_fa4_cute_import() -> bool:
    """Return True if ``flash_attn.cute`` was marked unavailable, False if nothing was needed."""
    if getattr(sys, _PATCHED_FLAG, False):
        return sys.modules.get(_MODULE, True) is None
    setattr(sys, _PATCHED_FLAG, True)
    if _MODULE in sys.modules:
        return sys.modules[_MODULE] is None
    try:
        importlib.import_module(_MODULE)
        return False
    except ImportError:
        return False  # genuinely absent: megatron-core already handles this
    except Exception as exc:  # e.g. AttributeError from a cutlass DSL mismatch
        for name in [m for m in sys.modules if m == _MODULE or m.startswith(_MODULE + ".")]:
            del sys.modules[name]
        sys.modules[_MODULE] = None
        logger.warning(
            f"`{_MODULE}` (FlashAttention-4) failed to import with {type(exc).__name__}: {exc}. "
            "Marking it unavailable so megatron-core falls back to FA2 / cuDNN attention."
        )
        return True
