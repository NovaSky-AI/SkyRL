# Cluster setup for GLM-5.3-Flash on Anyscale (8 × 8 B200)

How to bring up a Ray cluster on an Anyscale workspace so SkyRL's Megatron backend runs
GLM-5.3-Flash at full speed. Two things decide that:

- **EFA in the worker pods.** Without it, inter-node NCCL falls back to TCP sockets: ~1–2 GB/s per
  GPU instead of tens of GB/s. Nothing fails, everything is just slow. On 64 B200 over TCP, the
  MoE and context-parallel all-to-alls were most of each training step.
- **A node-local uv cache.** `uv run --isolated` builds a fresh virtualenv for the job driver and
  for every Ray worker. From a cache on network storage, that is a ~60 GB copy per process, which
  outlasts Ray's worker-registration timeout.

## 1. Compute config with EFA

On a Kubernetes-backed Anyscale cloud, EFA devices are mounted only if the pod asks for them.
[`efa_compute_config.yaml`](efa_compute_config.yaml) is a working example for the hackskyrl cloud.
It requests one device per GPU on the worker group's `ray` container:

```yaml
worker_nodes:
  - name: 8xb200
    advanced_instance_config:
      spec: {containers: [{name: ray, resources: {limits: {vpc.amazonaws.com/efa: "8"}}}]}
```

Put the request on the **worker group only**. The head node has no EFA, so its pod would never
schedule. Copy the file, change the cloud, instance types and namespace to yours, then create the
config and use it for your workspace:

```bash
anyscale compute-config create -n <config-name> -f efa_compute_config.yaml
anyscale workspace_v2 update <workspace-id> --compute-config <config-name>:1   # workspace stopped
anyscale workspace_v2 start --id <workspace-id>
```

You can also edit the worker group's advanced config in the console. If the workers stay
pending, the node pool can't provide `vpc.amazonaws.com/efa`; ask your platform team. A Kueue
priority label (commented out in the example) is also up to them.

## 2. Code

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git && cd SkyRL
git checkout glm5p3-flash-sft     # until the GLM-5.3-Flash SFT / context-parallel work is merged
```

## 3. Prepare the cluster (after every workspace start)

```bash
bash examples/train/glm5_3_flash/cluster_setup/setup_cluster.sh
```

It's idempotent, and takes ~10 minutes the first time (~5 when the shared cache is warm):

1. `uv sync --extra megatron --extra dev` into a shared cache (`SHARED_UV_CACHE`, default
   `/shared/hackskyrl/uv_cache`). This must be storage every node mounts and that survives restarts.
2. Waits for `EXPECTED_GPUS` (default 64) in `ray status`.
3. Copies the cache to `~/.cache/uv` on every node: the GPU workers in parallel, then the head
   (`seed_uv_cache.py`). Local disks are wiped on restart, so this runs every time.
4. Checks a worker pod for `/dev/infiniband` and libfabric's `efa` provider.
5. Runs [`nccl_a2a_probe.py`](nccl_a2a_probe.py) across 2 nodes and prints the bandwidth and the
   transport NCCL chose.

Good output has `NET/OFI` with the efa provider and tens of GB/s per GPU. `NET/Socket` at ~1–2
GB/s means EFA isn't in use.

## 4. Run

Every run needs a model directory, readable on every node, with `zai-org/GLM-5.3-Flash`'s
`config.json` (remove `quantization_config`) and tokenizer files. The dummy benchmark uses random
weights.

```bash
# one context length (the recipe in the script header; 1M needs CP2)
MODEL_PATH=/path/to/glm5p3_flash_cfg MAX_LENGTH=1048576 MEGATRON_TP=8 MEGATRON_CP=2 \
RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=uniform RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_MODULES='[core_attn]' \
SKYRL_OFFLOAD_CHECKPOINT_INPUTS=1 SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED=1 SKYRL_DSA_INDEXER_TP_SHARD=1 \
SKYRL_KDA_CP_EXCHANGE=allgather SKYRL_MOE_NODE_DEDUP=1 \
bash examples/train/sft/run_sft_dummy_glm5p3_flash_megatron.sh

# context-length sweep 16k..1M with the same knobs, with a summary table at the end
MODEL_PATH=/path/to/glm5p3_flash_cfg bash examples/train/sft/sweep_sft_dummy_glm5p3_flash_context.sh
```

`SKYRL_KDA_CP_EXCHANGE=allgather` and `SKYRL_MOE_NODE_DEDUP=1` cut inter-node bytes. They were
measured on TCP; with EFA, compare against runs without them.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| Ray workers time out registering, or a job sits silent with only uv's "extra-build-dependencies" warning for 15+ min | uv is building envs from network storage. Run step 3 again. Don't set `UV_CACHE_DIR` to the shared cache for runs; only `uv sync` uses it. |
| NCCL logs show `NET/OFI ... No eligible providers` / `NET/Socket` | No EFA devices in the pods (section 1). If devices exist but NCCL still uses sockets, try `FI_PROVIDER=efa` and `FI_EFA_USE_DEVICE_RDMA=1`. |
| `uv sync` fails downloading from GitHub | Retry, or keep `UV_OFFLINE=1` once the shared cache has everything. |
| Host out of memory with pinned checkpoint offload | Pinned offload holds ~1.4 MiB per token per GPU (~720 GB/node at 1M tokens, TP8 CP2). Use `SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED=0`. |
| A test kills a running job | Tests call `ray.init()` and attach to the live cluster. Run them only when the cluster is idle. |
| Step 0 of each new length is much slower | Triton, TileLang and inductor kernel caches are per node and lost on restart. They recompile once. |
