"""Exercise actual pip/uv installation and removal of the built wheel pair."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(command, env=None, succeeds=True):
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    if (result.returncode == 0) != succeeds:
        raise RuntimeError(f"Unexpected exit {result.returncode}: {command}")


def exercise(artifacts, installer, cross_platform):
    release = json.loads((artifacts / "release-manifest.json").read_text())
    wheels = [str((artifacts / item["filename"]).resolve()) for item in release["artifacts"]]
    with tempfile.TemporaryDirectory(prefix=f"skyrl-fa4-{installer}-") as directory:
        root = Path(directory)
        venv = root / "venv"
        run(["uv", "venv", "--seed", "--python", sys.executable, str(venv)])
        python = str(venv / "bin/python")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        if installer == "pip":
            install = [python, "-m", "pip", "install", "--no-deps", "--force-reinstall"]
            uninstall = [python, "-m", "pip", "uninstall", "-y"]
        else:
            install = ["uv", "pip", "install", "--python", python, "--no-deps", "--reinstall"]
            uninstall = ["uv", "pip", "uninstall", "--python", python]
        if cross_platform:
            target = root / "site"
            env["PYTHONPATH"] = str(target)
            if installer == "pip":
                install += [
                    "--target",
                    str(target),
                    "--upgrade",
                    "--platform",
                    "manylinux_2_24_x86_64",
                    "--python-version",
                    "3.12",
                    "--abi",
                    "cp312",
                    "--implementation",
                    "cp",
                ]
            else:
                install += ["--target", str(target), "--python-platform", "x86_64-manylinux_2_28"]
            # pip does not offer uninstall --target; uv owns target-directory removal.
            uninstall = ["uv", "pip", "uninstall", "--python", python, "--target", str(target)]
        verify = [python, str(HERE / "verify_install.py"), "--payload-only"]
        run(install + wheels, env)
        run(verify, env)
        run(install + [wheels[1]], env)
        run(verify, env)
        run(uninstall + ["flash-attn-4"], env)
        run(
            [
                python,
                "-c",
                "import importlib.metadata as m; from pathlib import Path; d=m.distribution('flash-attn'); assert Path(d.locate_file('flash_attn/cute/interface.py')).is_file()",
            ],
            env,
        )
        run(install + [wheels[1]], env)
        run(verify, env)
        run(uninstall + ["flash-attn"], env)
        run(verify, env, succeeds=False)
        run(install + [wheels[0]], env)
        run(verify, env)
        run(uninstall + ["flash-attn-4", "flash-attn"], env)
    return {
        "installer": installer,
        "platform": "cross-platform target" if cross_platform else "native",
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--cross-platform", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.cross_platform and (sys.platform != "linux" or sys.version_info[:2] != (3, 12)):
        parser.error("Native installation checks require Linux CPython 3.12")
    reports = [exercise(args.artifacts, installer, args.cross_platform) for installer in ("pip", "uv")]
    args.report.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
