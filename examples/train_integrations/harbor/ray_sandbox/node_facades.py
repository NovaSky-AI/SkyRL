"""Start one Ray Sandbox gRPC facade on 127.0.0.1:<port> on every node.

Modal's SDK takes a plain-http command router only from a localhost server
URL, so a client reaches the facade at http://127.0.0.1:<port> on its own
node. SkyRL runs every Harbor trial in one Ray task that can land on any
node, so each node gets its own facade; the generator uses the one on its
node. Sandboxes themselves go wherever Ray finds CPU, not just the node whose
facade created them.

A facade bound to loopback needs no API token: sandboxes with network access
can't reach their node's loopback (slirp4netns --disable-host-loopback).

    python examples/train_integrations/harbor/ray_sandbox/node_facades.py --port 50051 --expected-nodes 4

Run it with the image's system Python, which carries the sandbox overlay.
The facades are detached actors, so they outlive this script and stop with
the cluster. The cluster must be fixed-size: nodes that join later get none.
"""

import argparse
import os
import socket
import subprocess
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
class NodeFacade:
    """Runs the facade CLI on this node's loopback as a subprocess."""

    def __init__(self, port: int, log_path: str) -> None:
        self._port = port
        self._log_path = log_path
        self._process = None

    def start(self, timeout_s: float = 120.0) -> dict:
        if self._process is None or self._process.poll() is not None:
            env = dict(os.environ)
            # Clients on this node are the only ones that can reach loopback,
            # so the facade runs without a token.
            env.pop("RAY_SANDBOX_API_TOKEN", None)
            env["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            # Popen keeps its own copy of the descriptor.
            with open(self._log_path, "ab") as log:
                self._process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "ray.experimental.sandbox.http.grpc_facade",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(self._port),
                        "--advertise-url",
                        f"http://127.0.0.1:{self._port}",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"facade exited with {self._process.returncode}; see {self._log_path}")
            try:
                socket.create_connection(("127.0.0.1", self._port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise TimeoutError(f"facade not listening after {timeout_s}s; see {self._log_path}")
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "ip": ray.util.get_node_ip_address(),
            "pid": self._process.pid,
            "log": self._log_path,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--expected-nodes", type=int, default=1, help="wait for this many alive nodes")
    parser.add_argument("--log-dir", default="/tmp/ray_sandbox_facades")
    parser.add_argument("--wait-s", type=float, default=1800)
    args = parser.parse_args()

    ray.init(address="auto", namespace="harbor_ray_sandbox")
    deadline = time.time() + args.wait_s
    while len([n for n in ray.nodes() if n["Alive"]]) < args.expected_nodes:
        if time.time() > deadline:
            raise TimeoutError(f"fewer than {args.expected_nodes} nodes joined")
        time.sleep(5)

    facades = []
    for node in ray.nodes():
        if not node["Alive"]:
            continue
        node_id = node["NodeID"]
        facade = NodeFacade.options(
            name=f"harbor-node-facade-{node_id[:12]}",
            lifetime="detached",
            get_if_exists=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        ).remote(args.port, os.path.join(args.log_dir, f"facade-{node['NodeManagerAddress']}.log"))
        facades.append(facade.start.remote())
    for info in ray.get(facades):
        print(f"facade on {info['ip']} (node {info['node_id'][:12]}): 127.0.0.1:{args.port}, log {info['log']}")


if __name__ == "__main__":
    main()
