"""Harbor's Codex agent, configured to sample through its trajectory's skycap route.

Harbor's ``codex`` agent runs Codex with Codex's defaults, which skycap can't serve: it tries the Responses
WebSocket transport first, retries failed calls without marking them as retries (skycap would sample again and
fork the trajectory), and sends tools that are not function tools (the ``multi_agent`` namespace, ``web_search``).
``SkycapCodex`` points Codex at a provider of its own at the trajectory's URL, with function tools only, no
retries and no WebSocket transport, and installs the Codex version these settings were validated with.
"""

import json
import shlex
from typing import Any, Optional

from harbor.agents.installed.codex import Codex

#: The Codex release these settings were validated with; bumping it is a deliberate, re-validated change.
CODEX_VERSION = "0.154.0"

#: Codex features whose tools aren't function tools.
DISABLED_FEATURES = ("multi_agent", "goals")


class SkycapCodex(Codex):
    def __init__(
        self, *args: Any, version: Optional[str] = CODEX_VERSION, context_window: Optional[int] = None, **kwargs: Any
    ) -> None:
        """``context_window`` is the engine's context length, which Codex can't look up for a local model."""
        super().__init__(*args, version=version, **kwargs)
        self._context_window = context_window

    def build_cli_flags(self) -> str:
        base_url = self._get_env("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("SkycapCodex needs OPENAI_BASE_URL in the agent's env: the trajectory's skycap URL")
        settings: dict[str, Any] = {
            "model_provider": "skycap",
            "model_providers.skycap.name": "skycap",
            "model_providers.skycap.base_url": base_url,
            "model_providers.skycap.wire_api": "responses",
            # Harbor puts the trajectory's key in OPENAI_API_KEY.
            "model_providers.skycap.env_key": "OPENAI_API_KEY",
            # Codex marks no retry, so skycap would take one for a new call and fork the trajectory.
            "model_providers.skycap.request_max_retries": 0,
            "model_providers.skycap.stream_max_retries": 0,
            # skycap streams nothing until the turn is sampled.
            "model_providers.skycap.stream_idle_timeout_ms": 3_600_000,
            "web_search": "disabled",
        }
        if self._context_window is not None:
            settings["model_context_window"] = self._context_window
        # JSON strings and integers are valid TOML values, which is what `-c` parses.
        flags = [super().build_cli_flags()]
        flags += [f"-c {shlex.quote(f'{key}={json.dumps(value)}')}" for key, value in settings.items()]
        flags += [f"--disable {feature}" for feature in DISABLED_FEATURES]
        return " ".join(flag for flag in flags if flag)
