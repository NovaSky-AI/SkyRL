"""Turn a ``uv export`` of SkyRL's lock into an install list for AMD ROCm, and check the result.

SkyRL's lock resolves to CUDA wheels. A ROCm image takes the GPU-specific packages from its parent image
(torch, vLLM, Triton, flash-attn) or builds them (TransformerEngine), and takes everything else from the
lock at the pinned version. ``docker/Dockerfile.rocm`` uses the three commands below, in order.

    python docker/rocm_requirements.py snapshot > protected.json
        Record the versions of the packages the parent image provides.

    python docker/rocm_requirements.py filter export.txt docker/rocm-excluded-packages.txt > req.txt
        Print the export minus the excluded packages. Fails on any other CUDA-looking name, so a lock
        bump cannot slip a CUDA wheel in unnoticed.

    python docker/rocm_requirements.py verify protected.json req.txt
        After installing req.txt: the recorded packages are unchanged, no CUDA-tagged distribution is
        installed, and every git-pinned package is installed from exactly its pinned commit.

Install req.txt with ``uv pip install --no-config --no-deps``. Without ``--no-config``, uv applies the
``override-dependencies`` of SkyRL's pyproject.toml, which replace a git pin in the list with the newest
PyPI release of that package.
"""

import importlib.metadata as md
import json
import re
import sys
from typing import Optional

# Packages whose names look CUDA-specific but are vendor-neutral Python.
KEEP = {"nvidia-resiliency-ext", "nvidia-modelopt", "nvidia-ml-py"}
ALARM = re.compile(r"^(nvidia-|cuda-|flashinfer|nixl)|-cu1[23]$")
# Provided by the parent image or built in the image; the lock's packages must not replace them.
PROTECTED = ("torch", "torchvision", "torchaudio", "triton", "vllm", "flash-attn", "transformer_engine", "rocm")


def norm(name: str) -> str:
    """Normalizes a distribution name the way package indexes do."""
    return re.sub(r"[-_.]+", "-", name).lower()


def req_name(line: str) -> Optional[str]:
    """Returns the normalized package name of a requirement line, or None for lines such as ``-e ./pkg``."""
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return norm(m.group(1)) if m else None


def filter_requirements(export_lines, excluded_names) -> tuple:
    """Splits an export into the lines to install and the names dropped.

    Args:
        export_lines: Lines of ``uv export --no-hashes --no-annotate``.
        excluded_names: Package names that must not be installed on ROCm.

    Returns:
        ``(kept_lines, dropped_names)``.

    Raises:
        ValueError: The export has a CUDA-looking package that is neither excluded nor in ``KEEP``.
    """
    deny = {norm(name) for name in excluded_names}
    kept, dropped, unknown = [], [], []
    for raw in export_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = req_name(line)
        if name is None:
            kept.append(line)
        elif name in deny:
            dropped.append(name)
        elif ALARM.search(name) and name not in KEEP:
            unknown.append(name)
        else:
            kept.append(line)
    if unknown:
        raise ValueError(
            "CUDA-looking packages that are in neither docker/rocm-excluded-packages.txt nor the keep list: "
            f"{sorted(set(unknown))}"
        )
    return kept, dropped


def read_excluded(path: str) -> list:
    """Reads the excluded-packages file: one name per line, ``#`` comments and blank lines ignored."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def cmd_filter(export_path: str, excluded_path: str) -> None:
    excluded = read_excluded(excluded_path)
    with open(export_path) as f:
        try:
            kept, dropped = filter_requirements(f, excluded)
        except ValueError as e:
            sys.exit(str(e))
    print("\n".join(kept))
    print(f"kept {len(kept)}, dropped {len(dropped)}: {' '.join(sorted(set(dropped)))}", file=sys.stderr)
    unused = sorted({norm(name) for name in excluded} - set(dropped))
    if unused:
        print(f"excluded packages not in this export: {' '.join(unused)}", file=sys.stderr)


def versions() -> dict:
    """Returns the installed version of each protected package, or None where it is not installed."""
    out = {}
    for name in PROTECTED:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def git_pins(req_lines) -> dict:
    """Returns ``{name: commit}`` for every ``name @ git+https://...@<commit>`` line of an install list."""
    pins = {}
    for line in req_lines:
        m = re.match(r"^([A-Za-z0-9._-]+) @ git\+\S+@([0-9a-f]{7,40})", line.strip())
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def installed_commit(name: str) -> Optional[str]:
    """Returns the git commit a distribution was installed from, or None if it did not come from git."""
    try:
        text = md.distribution(name).read_text("direct_url.json")
    except md.PackageNotFoundError:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id") if text else None


def _installed_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def cmd_verify(snapshot_path: str, req_path: Optional[str] = None) -> None:
    with open(snapshot_path) as f:
        before = json.load(f)
    after = versions()
    changed = {k: (before.get(k), after[k]) for k in after if before.get(k) is not None and before.get(k) != after[k]}
    cuda = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in md.distributions()
        if "+cu" in d.version or re.search(r"-cu1[23]$", norm(d.metadata["Name"] or ""))
    )
    print("protected versions:", after)
    # A check by package name or version does not catch a PyPI release standing in for a git pin.
    wrong_git = {}
    pins = {}
    if req_path:
        with open(req_path) as f:
            pins = git_pins(f)
    for name, commit in pins.items():
        got = installed_commit(name)
        if got is None or not got.startswith(commit[:7]):
            wrong_git[name] = {
                "pinned": commit,
                "installed_commit": got,
                "installed_version": _installed_version(name),
            }
    if req_path:
        print("git pins checked:", sorted(pins))
    if changed or cuda or wrong_git:
        sys.exit(f"FAILED: changed={changed} cuda_wheels={cuda} wrong_git_pins={wrong_git}")
    print("ok: protected packages unchanged, no CUDA-tagged distribution installed, git pins match")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "filter" and len(sys.argv) == 4:
        cmd_filter(sys.argv[2], sys.argv[3])
    elif command == "verify" and len(sys.argv) in (3, 4):
        cmd_verify(*sys.argv[2:4])
    elif command == "snapshot" and len(sys.argv) == 2:
        print(json.dumps(versions()))
    else:
        sys.exit(__doc__)
