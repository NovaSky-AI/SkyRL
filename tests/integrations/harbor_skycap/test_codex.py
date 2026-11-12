"""SkycapCodex: the Codex configuration it hands `codex exec`, read back the way Codex reads it."""

import io
import json
import shlex
import tomllib
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

pytest.importorskip("harbor")

from harbor.agents.installed.codex import Codex  # noqa: E402

from examples.train_integrations.harbor_skycap import (
    codex as codex_module,  # noqa: E402
)
from examples.train_integrations.harbor_skycap.codex import (  # noqa: E402
    CATALOG_PATH,
    CODEX_VERSION,
    SkycapCodex,
    base_instructions,
    without_update_plan_instructions,
)
from examples.train_integrations.harbor_skycap.harbor_generator import (  # noqa: E402
    HarborSkycapGenerator,
    agent_class,
)
from tests.integrations.harbor_skycap.test_exposure import (  # noqa: E402
    SKYCAP_CODEX,
    agent_cfg,
)
from tests.integrations.harbor_skycap.test_harbor_skycap import (
    generator_cfg,  # noqa: E402
)

pytestmark = pytest.mark.integrations

URL = "https://tunnel.example/t/tr_1/v1"


def agent(tmp_path, **kwargs) -> SkycapCodex:
    return SkycapCodex(
        logs_dir=tmp_path,
        model_name="hosted_vllm/Qwen3-1.7B",
        extra_env={"OPENAI_BASE_URL": URL, "OPENAI_API_KEY": "key"},
        **kwargs,
    )


def codex_config(flags: str) -> tuple[dict, list[str]]:
    """The `-c` settings, parsed as Codex does (a TOML value, else the raw string), and the disabled features."""
    words, settings, disabled = shlex.split(flags), {}, []
    for flag, arg in zip(words[::2], words[1::2]):
        if flag == "-c":
            key, _, raw = arg.partition("=")
            try:
                settings[key] = tomllib.loads(f"v = {raw}")["v"]
            except tomllib.TOMLDecodeError:
                settings[key] = raw
        elif flag == "--disable":
            disabled.append(arg)
    return settings, disabled


def test_codex_samples_through_its_own_provider_at_the_trajectory_url(tmp_path) -> None:
    settings, disabled = codex_config(agent(tmp_path, context_window=32768).build_cli_flags())
    assert settings["model_provider"] == "skycap"
    assert settings["model_providers.skycap.base_url"] == URL
    assert settings["model_providers.skycap.wire_api"] == "responses"
    assert settings["model_providers.skycap.env_key"] == "OPENAI_API_KEY"
    # Retried calls would fork the trajectory; tools that aren't functions are refused by skycap.
    assert settings["model_providers.skycap.request_max_retries"] == 0
    assert settings["model_providers.skycap.stream_max_retries"] == 0
    assert settings["web_search"] == "disabled" and set(disabled) == {"multi_agent", "goals"}
    assert settings["model_context_window"] == 32768
    assert settings["model_catalog_json"] == CATALOG_PATH
    # Harbor's own flags survive.
    assert settings["model_reasoning_effort"] == "high"


def test_no_context_window_leaves_codex_its_own(tmp_path) -> None:
    settings, _ = codex_config(agent(tmp_path).build_cli_flags())
    assert "model_context_window" not in settings


@pytest.mark.parametrize("kwargs, version", [({}, CODEX_VERSION), ({"version": "0.160.0"}, "0.160.0")])
def test_the_validated_codex_is_installed_unless_the_run_names_another(tmp_path, kwargs, version) -> None:
    assert agent(tmp_path, **kwargs)._version == version


def test_without_a_trajectory_url_codex_is_refused(tmp_path) -> None:
    bare = SkycapCodex(logs_dir=tmp_path, model_name="hosted_vllm/Qwen3-1.7B")
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        bare.build_cli_flags()


# -- the model catalog ---------------------------------------------------------------------
#: Copied from Codex's unit tests for the edit it makes to its instructions while update_plan is off
#: (codex-rs/core/src/context/update_plan_instructions_tests.rs at rust-v0.154.0): the instructions, then what
#: Codex sends.
UPDATE_PLAN_CASES = [
    (
        "Before.\n\n## Planning\nYou have access to an `update_plan` tool which tracks steps.\n\n### Examples\nKeep "
        "steps current.\n\n## Work\nImplement.\n\n## `update_plan`\nUpdate the checklist.\n\n# Next\n## Planning\n"
        "Discuss architecture and inspect update_plan before editing.\n",
        "Before.\n\n## Work\nImplement.\n\n# Next\n## Planning\nDiscuss architecture and inspect update_plan before "
        "editing.\n",
    ),
    (
        "# Plan Mode\nKeep planning.\n\n## Plan Mode vs update_plan tool\nThe tools are separate.\n\n## Execution\nDo "
        "not edit.\n\n## Plan tool\nWhen using the planning tool:\n- Update steps.\n",
        "# Plan Mode\nKeep planning.\n\n## Execution\nDo not edit.\n\n",
    ),
    (
        "Keep working.\n- Use the plan tool to explain the work\n    - Skip simple tasks.\n    - Keep steps current.\n"
        "- Explain discoveries.\n- If you create a checklist or task list, update its statuses.\n\nProgress "
        "visibility:\nIf update_plan is available, use it for complex work.\n\nCompletion:\nVerify the result.\n",
        "Keep working.\n- Explain discoveries.\n\nCompletion:\nVerify the result.\n",
    ),
]
# The first case also stands in for prompt.md in the run tests below.
PROMPT, INSTRUCTIONS = UPDATE_PLAN_CASES[0]


@pytest.mark.parametrize("instructions, sent", UPDATE_PLAN_CASES)
def test_the_update_plan_guidance_is_left_out_as_codex_leaves_it_out(instructions, sent) -> None:
    # Codex skips this edit for a catalog's instructions, so SkycapCodex makes it.
    assert without_update_plan_instructions(instructions) == sent


class FakeEnvironment:
    """A sandbox that keeps what the agent uploads and the commands it runs."""

    def __init__(self, default_user: str | None = None) -> None:
        self.default_user = default_user
        self.files: dict[str, str] = {}
        self.commands: list[tuple[str, str | None]] = []

    async def upload_file(self, source_path, target_path: str) -> None:
        self.files[target_path] = Path(source_path).read_text()

    async def exec(self, command: str, user=None, **_) -> SimpleNamespace:
        self.commands.append((command, user))
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.fixture
def github(monkeypatch):
    """prompt.md as GitHub serves it, without the network: the URLs fetched, and an error to raise instead."""
    served = SimpleNamespace(urls=[], error=None)

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        served.urls.append(url)
        if served.error:
            raise served.error
        return io.BytesIO(PROMPT.encode())

    monkeypatch.setattr(codex_module.urllib.request, "urlopen", urlopen)
    base_instructions.cache_clear()
    yield served
    base_instructions.cache_clear()


@pytest.fixture
def codex_starts(monkeypatch) -> list[dict[str, str]]:
    """Harbor's Codex.run, which starts `codex exec`, replaced: the sandbox's files at each start."""
    starts = []

    async def run(self, instruction, environment, context) -> None:
        starts.append(dict(environment.files))

    monkeypatch.setattr(Codex, "run", run)
    return starts


@pytest.mark.asyncio
@pytest.mark.parametrize("version, ref", [(CODEX_VERSION, f"rust-v{CODEX_VERSION}"), (None, "main")])
async def test_codex_starts_with_a_catalog_entry_that_turns_on_apply_patch(
    tmp_path, github, codex_starts, version, ref
) -> None:
    warnings = []
    sink = logger.add(warnings.append, level="WARNING")
    try:
        codex, environment = agent(tmp_path, version=version), FakeEnvironment(default_user="agent")
        for _ in range(2):
            await codex.run("task", environment, context=None)
    finally:
        logger.remove(sink)

    (entry,) = json.loads(codex_starts[0][CATALOG_PATH])["models"]
    # Harbor runs `codex exec --model Qwen3-1.7B`; the entry matches it and carries the instructions Codex sends.
    assert (entry["slug"], entry["apply_patch_tool_type"], entry["base_instructions"]) == (
        "Qwen3-1.7B",
        "freeform",
        INSTRUCTIONS,
    )
    assert any(user == "root" and f"chown agent {CATALOG_PATH}" in command for command, user in environment.commands)
    # Once per process, at the installed release; an unpinned Codex gets main's, with a warning.
    assert github.urls == [f"https://raw.githubusercontent.com/openai/codex/{ref}/codex-rs/models-manager/prompt.md"]
    assert len(codex_starts) == 2 and bool(warnings) == (version is None)


@pytest.mark.asyncio
async def test_without_prompt_md_codex_does_not_start(tmp_path, github, codex_starts) -> None:
    github.error = urllib.error.HTTPError("url", 404, "Not Found", None, None)
    with pytest.raises(RuntimeError, match=f"rust-v{CODEX_VERSION}/codex-rs/models-manager/prompt.md"):
        await agent(tmp_path).run("task", FakeEnvironment(), context=None)
    assert codex_starts == []


# -- selecting it --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, import_path",
    [("codex", None), ("codex", SKYCAP_CODEX)],
    ids=["harbor codex", "a known name wins over the import path"],
)
def test_harbors_plain_codex_is_refused_before_any_trial(name, import_path) -> None:
    with pytest.raises(ValueError, match="use SkycapCodex"):
        HarborSkycapGenerator(generator_cfg(), agent_cfg(name, import_path), ["http://x"])


def test_skycap_codex_is_selected_by_its_import_path() -> None:
    assert agent_class({"name": None, "import_path": SKYCAP_CODEX}) is SkycapCodex
    HarborSkycapGenerator(generator_cfg(), agent_cfg(None, SKYCAP_CODEX), ["http://x"])
