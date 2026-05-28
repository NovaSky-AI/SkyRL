"""Runtime setup for Harbor mini-swe-agent containers."""

from __future__ import annotations

import os

_PATCHED = False


def patch_mini_swe_agent_environment() -> None:
    """Make Harbor's mini-swe-agent command use the shared tool install."""

    global _PATCHED
    if _PATCHED:
        return

    from harbor.agents.installed.mini_swe_agent import MiniSweAgent

    original = MiniSweAgent.create_run_agent_commands

    def create_run_agent_commands(self, instruction):
        commands = original(self, instruction)
        shared_home = os.environ.get("HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME", "/tmp/harbor-mini-swe-home")
        uv_cache_dir = os.environ.get("HARBOR_SHARED_UV_CACHE_ENV_DIR", "/harbor-shared/uv-cache")
        shared_bin = f"{shared_home}/.local/bin"
        package_ref = os.environ.get(
            "HARBOR_MINI_SWE_AGENT_PACKAGE",
            "git+https://github.com/li-boxuan/mini-swe-agent.git@8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a",
        )

        for command in commands:
            env = dict(command.env or {})
            env["PATH"] = f"{shared_bin}:{env.get('PATH') or os.environ.get('PATH', '')}"
            env.setdefault("UV_LINK_MODE", "copy")
            command.env = env
            command.command = (
                f"export HOME={shared_home}; "
                f"export UV_CACHE_DIR={uv_cache_dir}; "
                f"export PATH={shared_bin}:$PATH; "
                "if ! mini --help >/dev/null 2>&1; then "
                'mkdir -p "$HOME" "$UV_CACHE_DIR"; '
                f"uv tool install --force --python /usr/bin/python3.10 --cache-dir {uv_cache_dir} {package_ref}; "
                "hash -r; "
                "fi; "
                f"{command.command}"
            )
        return commands

    MiniSweAgent.create_run_agent_commands = create_run_agent_commands
    _PATCHED = True
