"""Ray actor that owns one gVisor sandbox, for RaySandboxEnvironment.

Wraps ``SandboxRuntime`` instead of using ``ray.experimental.sandbox.Sandbox`` so it
can prepare the node before creating the sandbox (``/dev/net/tun``) and pass
``user`` through to exec.

Runs on the node's own Python (see ``SANDBOX_RUNTIME_ENV``), not the job's uv
environment, so sandbox nodes need only Ray. Keep this module free of imports
beyond Ray and the standard library: the class is pickled by value and rebuilt
on that interpreter.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Union

import ray
import ray.cloudpickle
from ray.experimental.sandbox import ExecResult, SandboxRuntime

# Use the node's Python, which has Ray but not the training env. Without this the
# uv runtime-env hook builds the full training env on every sandbox node first.
SANDBOX_RUNTIME_ENV = {"py_executable": "python"}

_TUN_PATH = "/dev/net/tun"


def _ensure_tun() -> None:
    """Create /dev/net/tun if missing; slirp4netns needs it for network="public".

    Some container platforms don't expose the device, and a node created at
    image build time doesn't survive container start, so create it here.
    Needs passwordless sudo.
    """
    if os.path.exists(_TUN_PATH):
        return
    # `test -e ||` tolerates another sandbox on this node creating it first.
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "sh",
            "-c",
            f"mkdir -p /dev/net && (test -e {_TUN_PATH} || mknod {_TUN_PATH} c 10 200)"
            f" && chmod 666 {_TUN_PATH}",
        ],
        capture_output=True,
        text=True,
    )
    if not os.path.exists(_TUN_PATH):
        raise RuntimeError(
            f'network="public" needs {_TUN_PATH}, and creating it failed '
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )


@ray.remote
class HarborSandbox:
    def __init__(
        self,
        image: str,
        network: str = "none",
        enforce_limits: Optional[bool] = None,
        **create_kwargs,
    ):
        if network == "public":
            _ensure_tun()
        if enforce_limits is False or (
            enforce_limits is None and not os.environ.get("RAY_SANDBOX_CGROUP_PATH")
        ):
            # Ray enforces cpu and memory through a cgroup v2 subtree that the node
            # delegates to it (RAY_SANDBOX_CGROUP_PATH), and refuses them on a node
            # without one. The actor's reservation still applies either way.
            create_kwargs.pop("cpu", None)
            create_kwargs.pop("memory", None)
        self._runtime = SandboxRuntime()
        self._id = self._runtime.create(image=image, network=network, **create_kwargs)

    def ready(self) -> str:
        return self._id

    def exec(
        self,
        command: Union[str, List[str]],
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ExecResult:
        return self._runtime.exec(
            self._id, command, timeout=timeout, cwd=cwd, env=env, shell=shell, user=user
        )

    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        self._runtime.write_file(self._id, path, content)

    def read_file(self, path: str) -> bytes:
        return self._runtime.read_file(self._id, path)

    def delete(self) -> None:
        self._runtime.delete(self._id)

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass


ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
