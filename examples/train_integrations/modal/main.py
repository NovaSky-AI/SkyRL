import modal
import os
from pathlib import Path

# Read at import time both locally and inside the container. `create_modal_image` copies them into the image
# env so the container rebuilds the same function definitions (`@modal.clustered` size, GPU, timeout).
NUM_NODES = int(os.environ.get("MODAL_NUM_NODES", "1"))
GPU = os.environ.get("MODAL_GPU", "L4:1")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "3600"))
# Container memory request in MiB (Modal's default for a full GPU node can be too small for CPU optimizer offload).
MEMORY = int(os.environ["MODAL_MEMORY"]) if os.environ.get("MODAL_MEMORY") else None
# CPU cores to request (Modal's default for a GPU function can starve CPU-bound jobs such as checkpoint conversion).
CPU = float(os.environ["MODAL_CPU"]) if os.environ.get("MODAL_CPU") else None
# Comma-separated Modal secret names to attach, e.g. "huggingface,wandb".
SECRETS = [s for s in os.environ.get("MODAL_SECRETS", "").split(",") if s]
# Comma-separated extra volumes as name:mount_path, e.g. "dsv41-models:/models".
EXTRA_VOLUMES = os.environ.get("MODAL_VOLUMES", "")
# Comma-separated pyproject extras (e.g. "megatron" or "fsdp") whose packages are pre-downloaded into the image's uv
# cache, so each run's `uv run --isolated` (driver and every Ray worker on every node) links from a warm cache
# instead of downloading and building the env from scratch. The layer rebuilds only when the lockfile changes.
UV_EXTRAS = [e for e in os.environ.get("MODAL_UV_EXTRAS", "").split(",") if e]
# Semicolon-separated KEY=VALUE pairs set in the image env of every node, so the raylet and every Ray worker it
# starts inherit them (for variables SkyRL does not forward through the Ray runtime env), e.g.
# "VLLM_PP_LAYER_PARTITION=4,5".
EXTRA_ENV = dict(kv.split("=", 1) for kv in os.environ.get("MODAL_ENV", "").split(";") if kv)
UV_CACHE_DIR = "/home/ray/.cache/uv"
RAY_PORT = 6379


def _find_local_repo_root() -> Path:
    """Computes full path of local SkyRL repo robustly

    Raises:
        Exception: if cannot find local SkyRL repo

    Returns:
        Path: path object describing full path of local SkyRL repo
    """
    # If running inside Modal container, use the environment variable
    if "SKYRL_REPO_ROOT" in os.environ:
        return Path(os.environ["SKYRL_REPO_ROOT"])

    candidates = [Path(__file__).resolve(), Path.cwd()]
    for start in candidates:
        for base in [start] + list(start.parents):
            if base.exists() and (base / "skyrl-gym").exists():
                return base
    raise Exception("SkyRL root repo path not found")


def create_modal_image() -> modal.Image:
    """Creates a Modal image for Modal container. This uses the SkyRL container as
    a base image. It also mounts the local SkyRL repo to the container

    Returns:
        modal.Image: container image
    """

    local_repo_path = _find_local_repo_root()
    print(f"Root path: {local_repo_path}")

    envs = {
        "SKYRL_REPO_ROOT": "/root/SkyRL",  # where to put SkyRL in container
        "MODAL_NUM_NODES": str(NUM_NODES),
        "MODAL_GPU": GPU,
        "MODAL_TIMEOUT": str(TIMEOUT),
        "MODAL_MEMORY": str(MEMORY or ""),
        "MODAL_CPU": str(CPU or ""),
        "MODAL_SECRETS": ",".join(SECRETS),
        "MODAL_VOLUMES": EXTRA_VOLUMES,
        "MODAL_UV_EXTRAS": ",".join(UV_EXTRAS),
        "MODAL_ENV": os.environ.get("MODAL_ENV", ""),
        **EXTRA_ENV,
    }

    image = modal.Image.from_registry("novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0").env(
        {"UV_CACHE_DIR": UV_CACHE_DIR}
    )
    if UV_EXTRAS:
        # Only the files uv needs to install the locked dependencies, so source edits don't invalidate the layer.
        deps_dir = "/opt/skyrl-deps"
        image = image.add_local_file(local_repo_path / "pyproject.toml", f"{deps_dir}/pyproject.toml", copy=True)
        image = image.add_local_file(local_repo_path / "uv.lock", f"{deps_dir}/uv.lock", copy=True)
        for package in ("skyrl-gym", "skycap"):
            image = image.add_local_dir(
                local_repo_path / package, f"{deps_dir}/{package}", copy=True, ignore=["__pycache__", "*.egg-info"]
            )
        # One sync per extra (the backend extras conflict); drop each venv and keep only the cache. Modal's image
        # builder exports UV_NO_CACHE=true (uv then uses a throwaway temp cache) and a builder-only UV_INDEX_URL
        # mirror; unset both so the wheels land in UV_CACHE_DIR under the index URLs the runtime resolves.
        image = image.run_commands(
            *(
                f"cd {deps_dir} && env -u UV_NO_CACHE -u UV_INDEX_URL uv sync --frozen --no-install-project"
                f" --extra {extra} && rm -rf .venv && du -sh {UV_CACHE_DIR}"
                for extra in UV_EXTRAS
            )
        )
    if NUM_NODES > 1:
        # Userspace verbs libraries for NCCL over the cluster's RDMA network.
        image = image.run_commands(
            "sudo apt-get update -y && sudo apt-get install -y libibverbs1 ibverbs-providers && sudo apt-get clean"
        )
    return image.env(envs).add_local_dir(
        local_path=str(local_repo_path),
        remote_path="/root/SkyRL",
        ignore=[
            ".venv",
            "*.pyc",
            "__pycache__",
            ".git",
            "*.egg-info",
            ".pytest_cache",
            "node_modules",
            ".DS_Store",
        ],
    )


def create_modal_volume(volume_name: str = "skyrl-data") -> dict[str, modal.Volume]:
    """Creates volume to attach to container.

    Args:
        volume_name (str, optional): Name of volume. Creates a new
        volume if given name does not exist. Defaults to "skyrl-data".

    Returns:
        dict[str, modal.Volume]: location in container to attach the volume & volume itself
    """
    data_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    volumes = {"/root/data": data_volume}  # mounts volume at /root/data inside container
    for spec in filter(None, EXTRA_VOLUMES.split(",")):
        name, mount_path = spec.split(":")
        volumes[mount_path] = modal.Volume.from_name(name)
    return volumes


app = modal.App(os.getenv("MODAL_APP_NAME", "my_skyrl_app"))
image = create_modal_image()
volume = create_modal_volume()
secrets = [modal.Secret.from_name(name) for name in SECRETS]


def _chdir_repo_root():
    # The repo root is already set in the image environment
    repo_root = os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL")

    # Print current environment for debugging
    print(f"Container repo root: {repo_root}")
    print(f"Initial working directory: {os.getcwd()}")

    os.chdir(repo_root)
    print(f"Changed to directory: {os.getcwd()}")


def _run_streaming(command: str):
    """Runs COMMAND in a shell, streaming its output, and raises on a non-zero exit."""
    import subprocess

    print(f"Running command: {command}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    # Run the command with live output streaming
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end="")  # Print each line as it comes

    # Wait for process to complete
    returncode = process.wait()

    print("=" * 60)
    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")


@app.function(
    image=image,
    gpu=GPU,
    volumes=volume,
    secrets=secrets,
    memory=MEMORY,
    cpu=CPU,
    timeout=TIMEOUT,
)
def run_script(command: str):
    """
    Runs COMMAND inside SkyRL/
    """
    import subprocess

    _chdir_repo_root()

    print("Initializing ray cluster in command line")
    subprocess.run(
        "ray start --head",
        shell=True,
        check=True,
    )
    # Use 'auto' to automatically detect the Ray cluster instead of hardcoded IP
    os.environ["RAY_ADDRESS"] = "auto"

    _run_streaming(command)


if NUM_NODES > 1:

    @app.function(
        image=image,
        gpu=GPU,
        volumes=volume,
        secrets=secrets,
        memory=MEMORY,
        cpu=CPU,
        timeout=TIMEOUT,
    )
    @modal.clustered(size=NUM_NODES, rdma=True)
    def run_script_multinode(command: str):
        """
        Starts one Ray cluster across all NUM_NODES containers, then runs COMMAND inside SkyRL/ on rank 0.
        """
        import socket
        import subprocess
        import time

        _chdir_repo_root()

        cluster = modal.Cluster.from_context()
        rank = cluster.container_rank()
        # Ray needs IPv4; container_ips() defaults to IPv6.
        ips = cluster.container_ips(family="ipv4")
        head_ip, node_ip = ips[0], ips[rank]
        print(f"Rank {rank}/{len(ips)}: node ip {node_ip}, head ip {head_ip}")
        # Ray's uv hook builds each worker's env at startup; on a node with a cold uv cache that takes far longer
        # than the default 60s registration timeout, so workers get killed and retried until scheduling fails.
        os.environ.setdefault("RAY_worker_register_timeout_seconds", "1800")
        # NCCL's built-in IB transport over the RoCE NICs Modal configures (NCCL_IB_*). Its mlx5 Data Direct probe
        # returns EINVAL on these NICs, which fails the whole IB init; NCCL then falls through to the auto-loaded
        # aws-ofi-nccl plugin, whose TCP provider (no EFA here) picks an unrouted NIC and hangs the first cross-node
        # collective. Skip the probe, drop the plugin, and pin IB so a failure is an error, not a hang. Ray workers
        # inherit these from the raylet. On EFA hosts (no NCCL_IB_HCA) the plugin is the transport, so leave it be.
        if os.environ.get("NCCL_IB_HCA"):
            os.environ.setdefault("NCCL_IB_DATA_DIRECT", "0")
            os.environ.setdefault("NCCL_NET_PLUGIN", "none")
            os.environ.setdefault("NCCL_NET", "IB")

        if rank != 0:
            # Retry until the head's GCS is up, then stay alive until it goes away (rank 0 stops Ray when its
            # command exits). Only rank 0's result is returned to the caller.
            while (
                subprocess.run(
                    f"ray start --address={head_ip}:{RAY_PORT} --node-ip-address={node_ip}", shell=True
                ).returncode
                != 0
            ):
                time.sleep(5)
            while True:
                try:
                    socket.create_connection((head_ip, RAY_PORT), timeout=10).close()
                except OSError:
                    print("Ray head is gone, exiting")
                    return
                time.sleep(15)

        print("Initializing ray head")
        subprocess.run(
            f"ray start --head --node-ip-address={node_ip} --port={RAY_PORT} --disable-usage-stats",
            shell=True,
            check=True,
        )
        os.environ["RAY_ADDRESS"] = f"{head_ip}:{RAY_PORT}"

        print(f"Waiting for {NUM_NODES} Ray nodes")
        wait_for_nodes = f"""
import ray, time
ray.init()
while sum(n["Alive"] for n in ray.nodes()) < {NUM_NODES}:
    time.sleep(5)
print(ray.cluster_resources())
"""
        subprocess.run(["python", "-c", wait_for_nodes], check=True, timeout=900)

        try:
            _run_streaming(command)
        finally:
            subprocess.run("ray stop --force", shell=True)


@app.local_entrypoint()
def main(command: str = "nvidia-smi"):
    """Main entry-point for running a command in Modal-integrated SkyRL environmenmt.
    The given command will be run inside SkyRL/

    Args:
        command (str, optional): Command to run. Defaults to "nvidia-smi".

    Examples:
        modal run main.py --command "uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
        MODAL_GPU=A100:4 MODAL_APP_NAME=benji_skyrl_app modal run main.py --command "bash examples/train/gsm8k/run_gsm8k_modal.sh"
        MODAL_NUM_NODES=2 MODAL_GPU=H100:8 modal run main.py --command "ray status"
    """
    print(f"{'=' * 5} Submitting command to Modal: {command} {'=' * 5}")
    if NUM_NODES > 1:
        run_script_multinode.remote(command)
    else:
        run_script.remote(command)
    print(f"\n{'=' * 5} Command completed successfully {'=' * 5}")
