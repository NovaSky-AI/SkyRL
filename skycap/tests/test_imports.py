"""The base install must work without the ``tokens`` and ``remote`` extras."""

from __future__ import annotations

import subprocess
import sys

_CHECK = """
import sys
import skycap, skycap.cli, skycap.server, skycap.service, skycap.record, skycap.samples, skycap.tokens.backend, skycap.mirror
leaked = sorted(n for n in sys.modules if n.split('.')[0] in {'renderers', 'transformers', 'torch', 'fsspec'})
assert not leaked, leaked
"""


def test_importing_skycap_loads_nothing_from_the_optional_extras() -> None:
    # A fresh interpreter, since this process may have imported the extra already.
    result = subprocess.run([sys.executable, "-c", _CHECK], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
