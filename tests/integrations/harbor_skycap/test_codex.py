"""SkycapCodex: the Codex configuration it hands `codex exec`, read back the way Codex reads it."""

import shlex
import tomllib

import pytest

pytest.importorskip("harbor")

from examples.train_integrations.harbor_skycap.codex import (  # noqa: E402
    CODEX_VERSION,
    SkycapCodex,
)
from examples.train_integrations.harbor_skycap.harbor_generator import (  # noqa: E402
    HarborSkycapGenerator,
    agent_class,
)
from tests.integrations.harbor_skycap.test_exposure import SKYCAP_CODEX, agent_cfg  # noqa: E402
from tests.integrations.harbor_skycap.test_harbor_skycap import generator_cfg  # noqa: E402

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
