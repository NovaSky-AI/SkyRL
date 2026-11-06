"""skycap's SkyRL glue stands without a harness: it imports none, and its config and errors need none."""

import subprocess
import sys

import pytest

pytest.importorskip("skycap")

from skyrl.integration.skycap.config import (  # noqa: E402
    SkycapConfig,
    api_key_required,
    exposure_type,
)
from skyrl.integration.skycap.errors import is_context_length_message  # noqa: E402
from skyrl.train.config import SkyRLTrainConfig  # noqa: E402


def test_importing_the_package_pulls_in_no_harness() -> None:
    modules = ", ".join(
        f"skyrl.integration.skycap.{name}"
        for name in ("compose", "config", "engine", "errors", "record_index", "servers")
    )
    code = f"import sys; import {modules}; print(sorted(m for m in ('harbor', 'verifiers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "[]"


def test_any_run_config_with_a_skycap_section_is_enough() -> None:
    from dataclasses import dataclass, field

    @dataclass
    class Config(SkyRLTrainConfig):
        skycap: SkycapConfig = field(default_factory=SkycapConfig)

    cfg = Config.from_cli_overrides(["skycap.exposure.type=cloudflare"])
    assert exposure_type(cfg) == "cloudflare" and api_key_required(cfg)
    assert exposure_type(Config()) is None and not api_key_required(Config())


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"code": "context_length_exceeded"}', True),
        ("litellm.ContextWindowExceededError: too long", True),
        ("context\n    _length_exceeded", True),
        ("BadRequestError: bad temperature", False),
        ("", False),
    ],
)
def test_context_length_errors_are_read_from_messages_and_logs(text, expected) -> None:
    assert is_context_length_message(text) is expected
