"""Run a SkyRL command on a multi-node Modal cluster (one Ray cluster across N GPU containers).

Like main.py, but uses ``modal.clustered`` to start ``MODAL_NODES`` containers with RDMA, joins them into
one Ray cluster (rank 0 is the head) and runs the command on rank 0 once every node has joined. Ranks > 0 stay in the
cluster until the head goes away. There is also a CPU-only ``prep`` function for jobs that need no GPU (checkpoint
download / conversion, data prep) on the same volume.

    # 2 x B200:8, the layout of examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh
    MODAL_NODES=2 MODAL_GPU=B200:8 modal run main_multinode.py --command "bash examples/train/deepseek_v4_flash/modal/run_dsv41_lora_2node_modal.sh"

    # CPU job on the volume (no Ray, no GPU)
    modal run main_multinode.py::prep --command "bash examples/train/deepseek_v4_flash/modal/prepare_dsv41_modal.sh"

Environment (read when the app is defined, i.e. on your machine):

| variable | default | meaning |
|---|---|---|
| MODAL_APP_NAME | my_skyrl_app | app name |
| MODAL_NODES | 2 | containers in the cluster |
| MODAL_GPU | B200:8 | GPUs per container |
| MODAL_MEMORY_GB | 1800 | host RAM requested per GPU container (the colocated DSv4.1 run keeps vLLM's sleep backup and the LoRA-frozen base in host RAM / disk) |
| MODAL_EPHEMERAL_DISK_GB | 3000 | local disk per container (frozen-weight offload, LoRA adapter sync) |
| MODAL_TIMEOUT_H | 24 | function timeout in hours |
| MODAL_VOLUME | skyrl-data | volume mounted at /root/data (models, data, checkpoints) |

The command sees ``SKYRL_LOCAL_DISK=/tmp/skyrl_local`` (container-local disk; same path on every node) and
``RAY_ADDRESS=auto``.
"""

import os
import subprocess
import time

import modal
import modal.experimental


def _find_local_repo_root():
    """Same as main.py: the local SkyRL checkout (or SKYRL_REPO_ROOT inside the container)."""
    from pathlib import Path

    if "SKYRL_REPO_ROOT" in os.environ:
        return Path(os.environ["SKYRL_REPO_ROOT"])
    for start in [Path(__file__).resolve(), Path.cwd()]:
        for base in [start] + list(start.parents):
            if base.exists() and (base / "skyrl-gym").exists():
                return base
    raise Exception("SkyRL root repo path not found")


def create_modal_image() -> modal.Image:
    """Same image as main.py: the SkyRL base image with the local repo at /root/SkyRL."""
    return (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0")
        .env({"SKYRL_REPO_ROOT": "/root/SkyRL"})
        .add_local_dir(
            local_path=str(_find_local_repo_root()),
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
    )


NODES = int(os.environ.get("MODAL_NODES", "2"))
GPU = os.environ.get("MODAL_GPU", "B200:8")
MEMORY_MB = int(os.environ.get("MODAL_MEMORY_GB", "1800")) * 1024
DISK_MB = int(os.environ.get("MODAL_EPHEMERAL_DISK_GB", "3000")) * 1024
TIMEOUT_S = int(float(os.environ.get("MODAL_TIMEOUT_H", "24")) * 3600)
RAY_PORT = 6379
LOCAL_DISK = "/tmp/skyrl_local"

app = modal.App(os.getenv("MODAL_APP_NAME", "my_skyrl_app"))
image = create_modal_image().env({"SKYRL_LOCAL_DISK": LOCAL_DISK})
volume = {"/root/data": modal.Volume.from_name(os.getenv("MODAL_VOLUME", "skyrl-data"), create_if_missing=True)}


def _stream(command: str, cwd: str) -> int:
    process = subprocess.Popen(
        command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end="")
    return process.wait()


def _alive_gpu_nodes() -> int:
    import ray

    return sum(1 for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0)


@app.function(
    image=image,
    gpu=GPU,
    memory=MEMORY_MB,
    ephemeral_disk=DISK_MB,
    volumes=volume,
    timeout=TIMEOUT_S,
)
@modal.clustered(size=NODES, rdma=True)
def run_cluster(command: str):
    """Rank 0: Ray head + runs COMMAND in the repo. Other ranks: Ray workers until the head is gone."""
    info = modal.experimental.get_cluster_info()
    ips = info.container_ipv4_ips or info.container_ips
    rank, my_ip, head_ip = info.rank, ips[info.rank], ips[0]
    repo_root = os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL")
    os.makedirs(LOCAL_DISK, exist_ok=True)
    print(f"[rank {rank}/{NODES}] ip {my_ip}, head {head_ip}")

    if rank == 0:
        subprocess.run(f"ray start --head --node-ip-address={my_ip} --port={RAY_PORT}", shell=True, check=True)
        os.environ["RAY_ADDRESS"] = "auto"
        import ray

        ray.init(address="auto", log_to_driver=False)
        deadline = time.time() + 900
        while _alive_gpu_nodes() < NODES:
            if time.time() > deadline:
                raise RuntimeError(f"only {_alive_gpu_nodes()} of {NODES} nodes joined the Ray cluster")
            time.sleep(5)
        ray.shutdown()
        print(f"[rank 0] all {NODES} nodes joined; running: {command}\n" + "=" * 60)
        returncode = _stream(command, repo_root)
        print("=" * 60)
        subprocess.run("ray stop --force", shell=True)
        if returncode != 0:
            raise Exception(f"Command failed with exit code {returncode}")
    else:
        # The head may still be starting: retry the join.
        for _ in range(60):
            if (
                subprocess.run(
                    f"ray start --address={head_ip}:{RAY_PORT} --node-ip-address={my_ip}", shell=True
                ).returncode
                == 0
            ):
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"could not join the Ray head at {head_ip}:{RAY_PORT}")
        # Stay until the head's GCS is gone (rank 0 finished or failed).
        misses = 0
        while misses < 3:
            time.sleep(30)
            alive = subprocess.run(f"ray status --address={head_ip}:{RAY_PORT}", shell=True, capture_output=True)
            misses = 0 if alive.returncode == 0 else misses + 1
        subprocess.run("ray stop --force", shell=True)


@app.function(
    image=image,
    cpu=64,
    memory=512 * 1024,
    ephemeral_disk=DISK_MB,
    volumes=volume,
    timeout=TIMEOUT_S,
)
def prep(command: str):
    """CPU-only: runs COMMAND in the repo (checkpoint download / conversion, data prep) and commits the volume."""
    repo_root = os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL")
    returncode = _stream(command, repo_root)
    volume["/root/data"].commit()
    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")


@app.local_entrypoint()
def main(command: str = "nvidia-smi"):
    print(f"{'=' * 5} Submitting to a {NODES} x {GPU} Modal cluster: {command} {'=' * 5}")
    run_cluster.remote(command)
    print(f"\n{'=' * 5} Command completed successfully {'=' * 5}")
