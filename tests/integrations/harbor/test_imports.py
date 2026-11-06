"""Plain Harbor needs only the ``harbor`` extra: its modules don't import skycap, and its trial template ships."""

import subprocess
import sys

import pytest

pytest.importorskip("harbor")

from skyrl.integration.harbor.config import (  # noqa: E402
    HARBOR_DEFAULT_CONFIG,
    load_trial_config,
)


def test_plain_harbor_imports_no_skycap() -> None:
    modules = ", ".join(
        f"skyrl.integration.harbor.{name}" for name in ("config", "dataset", "exp", "generator", "metrics")
    )
    code = f"import sys; import {modules}; print('skycap' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip().splitlines()[-1] == "False"


def test_cli_overrides_merge_over_the_default_trial_config() -> None:
    assert HARBOR_DEFAULT_CONFIG.is_file()
    merged = load_trial_config({"agent": {"name": "mine"}})
    assert merged["agent"]["name"] == "mine" and "environment" in merged and len(merged["agent"]) > 1
