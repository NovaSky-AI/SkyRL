"""What `pip install skyrl-capture` has to be, without the extras.

Capture installs *beside* a trainer, in the same environment, and a trainer
pins its own `transformers`. So the base install must carry no ML dependency at
all: text-mode capture is a proxy, a recorder, four exporters and a viewer,
none of which tokenizes anything. `skyrl-capture[tokens]` is the only thing
that brings a renderer.

That only holds while every renderer import stays inside the function that
builds one. It is one `from renderers import ...` at module scope away from
being false, and the failure would be an ImportError on a machine that has
never needed a tokenizer -- so it is asserted rather than remembered.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Importing the runtime pulls in the data plane, the recorder, the export path
# and the TITO proxy -- everything a process running text mode composes.
PROBE = """
import sys
import skyrl_capture.runtime
import skyrl_capture.cli.main
import skyrl_capture.tito.proxy
import skyrl_capture.export.service
leaked = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {"transformers", "renderers", "torch", "jinja2"}
)
print(",".join(leaked))
"""


def test_the_base_install_never_imports_a_tokenizer():
    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, (
        f"importing the runtime pulled in {leaked}. Those live in the `tokens` extra; "
        "the import belongs inside the function that builds a renderer"
    )


def test_the_ml_dependency_is_only_in_the_tokens_extra():
    metadata = tomllib.loads(PYPROJECT.read_text())["project"]
    base = " ".join(metadata["dependencies"])
    for package in ("transformers", "renderers", "torch"):
        assert package not in base, f"{package} must not be a base dependency"
    extra = " ".join(metadata["optional-dependencies"]["tokens"])
    assert "transformers" in extra and "renderers" in extra


def test_the_package_ships_no_development_machinery():
    """What installs should describe the product, not the laboratory.

    The benchmark harness, the load generators, the mock upstream and the
    verification workflow are all real and all kept -- under `tools/` and
    `tests/support/`, run from a checkout. None of them belongs in a wheel a
    trainer installs beside itself, and each one was inside the package once.
    """
    package = PYPROJECT.parent / "src" / "skyrl_capture"
    stowaways = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if any(part in {"bench", "testing"} for part in path.parts)
        or path.name in {"verify.py", "mock_server.py", "mock_upstream.py", "loadgen.py"}
    )
    assert not stowaways, f"development machinery inside the package: {stowaways}"
