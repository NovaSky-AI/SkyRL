"""Harbor's Codex agent, configured to sample through its trajectory's skycap route.

Harbor's ``codex`` agent runs Codex with Codex's defaults, which skycap can't serve: it tries the Responses
WebSocket transport first, retries failed calls without marking them as retries (skycap would sample again and
fork the trajectory), and sends tools that are not function tools (the ``multi_agent`` namespace, ``web_search``).
``SkycapCodex`` points Codex at a provider of its own at the trajectory's URL, with function tools only, no
retries and no WebSocket transport, and installs the Codex version these settings were validated with.

Codex has no metadata for open-weight models, and without it Codex leaves out its ``apply_patch`` tool while its
instructions still tell the model to edit files with it. ``SkycapCodex`` gives Codex a model catalog entry for the
served model: Codex's fallback metadata with ``apply_patch`` on.
"""

import asyncio
import functools
import json
import re
import shlex
import urllib.request
from typing import Any, Optional

from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from loguru import logger

#: The Codex release these settings were validated with; bumping it is a deliberate, re-validated change, which
#: includes checking `catalog_entry` against that release's `model_info_from_slug`.
CODEX_VERSION = "0.154.0"

#: Codex features whose tools aren't function tools.
DISABLED_FEATURES = ("multi_agent", "goals")

#: Where the model catalog goes in the sandbox: Harbor's run() creates, and afterwards deletes, Codex's own
#: directories; /tmp exists before that.
CATALOG_PATH = "/tmp/skycap-model-catalog.json"

#: Codex's base instructions at a git ref: the text its fallback metadata sends for a model it doesn't know.
PROMPT_URL = "https://raw.githubusercontent.com/openai/codex/{ref}/codex-rs/models-manager/prompt.md"

#: What Codex leaves out of its own instructions while its update_plan tool is off (Codex's default).
PLAN_SECTIONS = ("## Planning", "## `update_plan`", "## Plan tool", "## Plan Mode vs update_plan tool")
PLAN_TOOL_LINES = ("You have access to an `update_plan` tool", "When `update_plan` is available, follow this section")
PLAN_BULLETS = ("- Use the plan tool ", "- If you create a checklist or task list,")


def without_update_plan_instructions(instructions: str) -> str:
    """The instructions without their guidance for Codex's ``update_plan`` tool, which Codex leaves off by default.

    A port of Codex's edit of its own instructions (``codex-rs/core/src/context/update_plan_instructions.rs``).
    Codex skips that edit for instructions from a model catalog, which it treats as the user's own text.
    """
    lines, kept, i = re.findall(r"[^\n]*\n|[^\n]+", instructions), [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line in PLAN_SECTIONS:
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith(("# ", "## "))), len(lines))
            # A "## Planning" section that doesn't describe the tool stays.
            if line != "## Planning" or any(text.startswith(PLAN_TOOL_LINES) for text in lines[i:end]):
                i = end
                continue
        if (
            line == "Progress visibility:"
            and lines[i + 1 : i + 2]
            and lines[i + 1].startswith("If update_plan is available")
        ):
            i += 2
            if lines[i : i + 1] and not lines[i].strip():
                i += 1
            continue
        if line.startswith(PLAN_BULLETS):
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                i += 1
            continue
        kept.append(lines[i])
        i += 1
    return "".join(kept)


@functools.cache
def base_instructions(version: Optional[str]) -> str:
    """The instructions Codex sends for a model it doesn't know: its ``prompt.md`` at release ``version`` (at ``main``
    for an unpinned Codex), edited as Codex edits it. Fetched once per process."""
    if version is None:
        logger.warning(
            "harbor_trial_config.agent.kwargs.version is null, so Harbor installs the latest Codex; SkycapCodex gives "
            "it the instructions on Codex's main branch, which may not match that release."
        )
    url = PROMPT_URL.format(ref=f"rust-v{version}" if version else "main")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return without_update_plan_instructions(response.read().decode())
    except OSError as error:
        raise RuntimeError(f"SkycapCodex couldn't fetch Codex's instructions from {url}: {error}") from error


def catalog_entry(slug: str, instructions: str) -> dict[str, Any]:
    """Codex's fallback metadata for a model it doesn't know (``model_info_from_slug`` in
    ``codex-rs/models-manager/src/model_info.rs``), with ``apply_patch`` on."""
    return {
        "slug": slug,
        "display_name": slug,
        "description": None,
        "supported_reasoning_levels": [],
        "shell_type": "unified_exec",
        "visibility": "none",
        "supported_in_api": True,
        "priority": 99,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": instructions,
        "include_apps_usage_instructions": False,
        "supports_reasoning_summary_parameter": True,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "default_verbosity": None,
        # The fallback has none, and Codex registers apply_patch only when it is set.
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_image_detail_original": False,
        # `model_context_window` narrows this, as it does the fallback's.
        "context_window": 272000,
        "max_context_window": 272000,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_search_tool": False,
    }


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
            "model_catalog_json": CATALOG_PATH,
        }
        if self._context_window is not None:
            settings["model_context_window"] = self._context_window
        # JSON strings and integers are valid TOML values, which is what `-c` parses.
        flags = [super().build_cli_flags()]
        flags += [f"-c {shlex.quote(f'{key}={json.dumps(value)}')}" for key, value in settings.items()]
        flags += [f"--disable {feature}" for feature in DISABLED_FEATURES]
        return " ".join(flag for flag in flags if flag)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # Harbor runs `codex exec --model <the name's last component>`, which the entry's slug has to match.
        entry = catalog_entry(self.model_name.split("/")[-1], await asyncio.to_thread(base_instructions, self._version))
        catalog = self.logs_dir / "model_catalog.json"
        catalog.write_text(json.dumps({"models": [entry]}))
        await environment.upload_file(catalog, CATALOG_PATH)
        if environment.default_user is not None:
            # upload_file copies as root; Harbor hands its auth.json to the agent's user the same way.
            await self.exec_as_root(environment, command=f"chown {environment.default_user} {CATALOG_PATH}")
        await super().run(instruction, environment, context)
