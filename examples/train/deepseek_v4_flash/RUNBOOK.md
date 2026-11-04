# DeepSeek-V4.1-Flash on branch `throughput-push-deepseek-v1`: how to run end to end

Branch: `throughput-push-deepseek-v1` on github.com/avigyabb/SkyRL. It is `NovaSky-AI/SkyRL` main plus what the
V4.1 LoRA DAPO recipe needs that main does not have yet. Recipe:
`examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh` (Megatron TP1 / EP16, one vLLM engine
TP8 x PP2, colocated, LoRA r32 without merge, R3 on).

## What this branch adds on top of main

| piece | why | upstream |
|---|---|---|
| R3 on HybridModel (sub-layer indices -> rollout blocks) | without it: `Router replay layer indices [1, 3, ...] out of range` | open PR NovaSky-AI/SkyRL#2562 (merged in) |
| `generator.cache_salt_per_trajectory` | the recipe sets it: vLLM's V4.1 decode drifts after a prefix-cache hit and R3 needs prefix caching on | open PR #2568 (merged in) |
| FP8 load in the released 32-wide blocks | V4's 128-block dequant gives wrong attention / shared-expert weights when Megatron loads the FP8 release | open PR #2610 (cherry-picked) |
| full-weight sync: 32-wide FP8 export, FP8/e8m0 kept on the wire | full FT weight sync of V4.1 (not used by the LoRA recipe) | open PR #2612 (cherry-picked) |
| V4.1 attention accepts megatron-core's `boundary_hidden` / `boundary_kv` kwargs | main's megatron-core pin (ed9c2c8c, #2558) passes them; without it the Megatron forward raises `CompressedSparseAttention2.forward() got an unexpected keyword argument 'boundary_hidden'` (DSv4.1 training is broken on main) | -- (commit d16f8cc3 here) |
| recipe engine kwarg `swa_bounded_replay: false` | vLLM 0.32 (main's pin) asserts at engine start with TP8 x PP2 (see below) | -- |
| `VLLM_PP_LAYER_PARTITION` forwarded to the engine workers | uneven PP splits (needed for layer slices) | -- |
| `dev/convert_dsv41_bf16_parallel.py` | builds the BF16 text-only checkpoint the recipe loads, per layer in parallel | parallel version of #2573's `dev/make_slice_v41.py` |

The V4.1 model support itself (Megatron bridge, CSA2 attention, Engram, vLLM patches) is on main (#2565, #2566, #2567).

### vLLM 0.32 and PP2

Two constraints on where the engine's pipeline split can go:

1. PP splits must sit on a KV-sharing group boundary (layers 2/8/14/20 of the release), else
   `PP splits inside a v4.1 kv-sharing group are not supported`.
2. With decoder SWA bounded replay on (0.32's default), the stage holding the last KV source layer (20) must also run at
   least one layer before it, else `_run_layers` asserts `residual is not None`.

The balanced 20/20 split breaks (2). The recipe turns replay off (it only trims eager prefill). The alternative is 14/26
(`VLLM_PP_LAYER_PARTITION=14,26`), which keeps replay but unbalances the stages. #2573's recipes ran on vLLM 0.31, which
has no replay.

## On Modal

`examples/train_integrations/modal/main_multinode.py` runs a command on a cluster of `MODAL_NODES` GPU containers,
built with `modal.clustered(size=..., rdma=True)`, joined into one Ray cluster. It uses the same image and
volume (`/root/data`) as `main.py`. It also has a CPU-only `prep` function. **Not tested on Modal yet**: there was no
Modal account on the machine this branch was built on.

```bash
cd examples/train_integrations/modal
# one time: release download, BF16 conversion, DAPO data (LAYERS=0,1,2,3,20,21,22,23,24 for the 9-layer slice)
modal run main_multinode.py::prep --command "HF_TOKEN=... bash examples/train/deepseek_v4_flash/modal/prepare_dsv41_modal.sh"
# training, 2 x B200:8
MODAL_NODES=2 MODAL_GPU=B200:8 modal run main_multinode.py \
  --command "WANDB_API_KEY=... bash examples/train/deepseek_v4_flash/modal/run_dsv41_lora_2node_modal.sh"
```

Things to check on the first Modal run:

- **Host RAM per container.** `MODAL_MEMORY_GB` defaults to 1800. The full model did not fit 2 TB hosts here (see
  below), so start with the slice, or more nodes.
- **Local disk.** `MODAL_EPHEMERAL_DISK_GB` defaults to 3000. It holds the LoRA-frozen base offload and the ~20 GB
  adapter sync dir. With the full model the offload is the Megatron base weights per node.
- **Environment build.** `uv run --isolated --extra megatron` builds the megatron env in every container. The first
  run takes a while (megatron-core / Bridge are built from git).
- **Ray placement.** Each container should show 8 GPUs in `ray status`; rank 0 waits up to 15 min for all nodes.

## Steps

1. **Cluster:** Ray on every node with this branch's uv env. The model and data dirs must exist on every node.
2. **Checkpoint:** download `deepseek-ai/DeepSeek-V4.1-Flash-1017` (476 GB, FP8 + MXFP4). On **every** node, build the BF16
   text-only copy onto local NVMe (1.2 TB; about 7 min with 32 workers on 192 cores):

   ```bash
   DSV41_LOCAL_DIR=<release dir> python examples/train/deepseek_v4_flash/dev/convert_dsv41_bf16_parallel.py \
     /mnt/local_storage/models/DeepSeek-V4.1-Flash-bf16-chat "" 32
   ```

   The output's `tokenizer_config.json` already has the chat-mode template (`chat_template_v41_chat.jinja`), so no symlink
   copy is needed. To build a layer slice instead, pass layer ids as the second argument (for example
   `0,1,2,3,20,21,22,23,24`).
3. **Data:** `bash examples/train/algorithms/dapo/prepare_dapo_data.sh`.
4. **Run:**

   ```bash
   MODEL_PATH=/mnt/local_storage/models/DeepSeek-V4.1-Flash-bf16-chat DATA_DIR=$HOME/data/dapo \
   LORA_SYNC_PATH=/mnt/local_storage/skyrl_lora_sync \
   SKYRL_FROZEN_OFFLOAD_DIR=/mnt/local_storage/skyrl_frozen_offload \
     bash examples/train/deepseek_v4_flash/run_dapo_dsv41_flash_lora_sync_2node.sh
   ```

   - `SKYRL_FROZEN_OFFLOAD_DIR` must be on a writable local disk. The default is `/data/...`; if that is missing, the
     LoRA-frozen base weights fall back to pinned RAM.
   - `LORA_SYNC_PATH` needs room for a ~20 GB adapter on every node.

## Host memory: the full model did not fit two 2 TB nodes here

On 2 x 8xB200 hosts with 2 TB RAM each (Anyscale), the full 40-layer model never reached step 1:

- vLLM's sleep (level 1, forced for LoRA weight sync) backs up 73 GiB per GPU to pinned host memory, about 590 GB per node.
- vLLM 0.32 keeps the Engram tables in pinned host memory (`EngramConfig(cpu_offload=True)`): 11.8 GiB per rank.
- Megatron's model load then grew host **shared** memory (`Shmem` in `/proc/meminfo`) by more than 250 GB in a minute
  and kept growing. One node was OOM-terminated; on the next try Ray's memory monitor killed the job.

What allocates that shared memory during the Megatron load is **not identified yet**. Watch `/proc/meminfo` (`Shmem`,
`MemAvailable`) on every node during the first load. Options: hosts with more RAM, or spreading the engine and Megatron
over more nodes so each node holds less.

The same recipe layout runs on a 9-layer slice (all 384 experts, every V4.1 layer role) on this cluster; see
`throughput_push_deepseek_v1/TRACKING.md` for the slice results.
