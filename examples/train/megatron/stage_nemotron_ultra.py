"""Pre-stage NVIDIA-Nemotron-3-Ultra-550B + the GSM8K dataset onto every GPU node's
local disk, for the multi-node recipe in ``run_megatron_nemotron_ultra.sh``.

Why this exists: the model is ~1.1 TB and the cluster's shared ``/home`` is small
(~255 GB), so the checkpoint must live on each node's large node-local disk
(``/mnt/local_storage``, ~28 TB). Multi-node training also needs the data present on
every node (each rank reads its data locally). This launches one Ray task per distinct
GPU node that (1) downloads the HF snapshot and (2) writes the GSM8K parquets, both under
``/mnt/local_storage``.

IMPORTANT: ``allow_patterns`` includes ``*.jinja`` so the model's official
``chat_template.jinja`` is staged. Without it the tokenizer/vLLM have NO chat template,
the instruct/reasoning model is prompted off-distribution, and reward stays 0.

Usage (from the head node, on a running Ray cluster):
    export HF_TOKEN=$(cat ~/.HF_TOKEN)   # fast authenticated download; unauth is throttled
    uv run --isolated --with ray --with huggingface_hub --with hf_transfer --with datasets \
        python examples/train/megatron/stage_nemotron_ultra.py
"""

import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes

REPO = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
HF_HOME = "/mnt/local_storage/hf_cache"
DATA_DIR = "/mnt/local_storage/data/gsm8k"

ray.init(address="auto", ignore_reinit_error=True, log_to_driver=True)

_ENV = {"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
if os.environ.get("HF_TOKEN"):
    _ENV["HF_TOKEN"] = os.environ["HF_TOKEN"]


@ray.remote(
    num_cpus=8,
    runtime_env={"pip": ["huggingface_hub", "hf_transfer", "datasets"], "env_vars": _ENV},
)
def stage(node_ip):
    import socket
    import time

    from huggingface_hub import snapshot_download

    # 1) Model snapshot (weights + config + tokenizer + chat_template.jinja).
    model_path = None
    last_err = None
    for _ in range(8):
        try:
            model_path = snapshot_download(
                REPO,
                cache_dir=f"{HF_HOME}/hub",
                max_workers=16,
                # NOTE: *.jinja is required (the chat template); *.py pulls any custom code.
                allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model", "*.jinja", "*.py"],
            )
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(15)
    if model_path is None:
        return (socket.gethostname(), node_ip, "MODEL_FAILED", last_err)

    # 2) GSM8K parquets (tiny; one copy per node so every rank reads locally).
    try:
        import re

        import datasets

        os.makedirs(DATA_DIR, exist_ok=True)
        instruction = 'Let\'s think step by step and output the final answer after "####".'

        def to_row(example, idx, split):
            q = example["question"]
            sol = re.search(r"#### (\-?[0-9\.\,]+)", example["answer"]).group(0).split("#### ")[1].replace(",", "")
            return {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": q + " " + instruction}],
                "env_class": "gsm8k",
                "reward_spec": {"method": "rule", "ground_truth": sol},
                "extra_info": {"split": split, "index": idx},
            }

        ds = datasets.load_dataset("openai/gsm8k", "main")
        ds["train"].map(lambda e, i: to_row(e, i, "train"), with_indices=True).to_parquet(f"{DATA_DIR}/train.parquet")
        ds["test"].map(lambda e, i: to_row(e, i, "test"), with_indices=True).to_parquet(
            f"{DATA_DIR}/validation.parquet"
        )
    except Exception as e:  # noqa: BLE001
        return (socket.gethostname(), node_ip, "DATA_FAILED", f"{type(e).__name__}: {str(e)[:300]}")

    return (socket.gethostname(), node_ip, "DONE", model_path)


nodes = [
    n
    for n in list_nodes(detail=True)
    if (n.get("resources_total") or {}).get("GPU", 0) > 0 and n.get("state") == "ALIVE"
]
print("staging on %d GPU nodes: %s" % (len(nodes), [n["node_ip"] for n in nodes]), flush=True)
results = ray.get(
    [
        stage.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["node_id"], soft=False)).remote(
            n["node_ip"]
        )
        for n in nodes
    ]
)
print("\n===== RESULTS =====", flush=True)
ok = sum(1 for r in results if r[2] == "DONE")
for host, ip, status, detail in results:
    print(f"[{ip} {host}] {status}: {detail}", flush=True)
print(f"\n{ok}/{len(results)} nodes staged OK", flush=True)
