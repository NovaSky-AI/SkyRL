# Bounded-memory colocated weight sync: results

Companion to [`colocated_weight_sync.md`](colocated_weight_sync.md). Option names below are
the ones in that doc. "Default path" means every option off.

> [!NOTE]
> Rows marked * were measured on this PR's code. The rest were measured on the development
> branch, whose code for those options is the same; its offloaded-params export was an earlier
> prototype (it also prefetched the next param), and the * rows re-measure what it measured.

## How it was measured

**Harness:** `tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_colocated_weight_sync.py`.

- `test_colocated_sync_byte_parity` boots colocated vLLM engines from the checkpoint and
  records an integer checksum of every vLLM parameter and buffer on every worker
  (`skyrl_debug_weight_checksums`). It puts the engines to sleep at level 2 (weights discarded)
  and runs the production `WorkerDispatch.save_weights_for_sampler`. After each sync, every
  tensor on every worker must be byte-identical to the fresh load. A dropped, truncated,
  misrouted or stale tensor fails it.
- `test_colocated_sync_after_train_step_matches_baseline` takes one training step, syncs with
  the default path, then switches the trainer to the configured options
  (`Worker.reconfigure_for_test`) and syncs again. The vLLM bytes must be identical.
- Each sync prints one `CWS_SUMMARY` line (and a full `CWS_RECORD`). Trainer numbers come from
  `SKYRL_COLOCATED_SYNC_DEBUG=1`, which the harness sets: torch peak allocation per rank, a
  `mem_get_info` poller for peak device use, time inside the export, and round-trip count and
  time. vLLM numbers come from `skyrl_debug_memory_stats`, which includes the bytes and chunks
  each worker received. "Transfer" is `sync_weights_only_transfer`.

**Environment variables** (all optional except the model):

| Variable | Default | Meaning |
|---|---|---|
| `SKYRL_CWS_MODEL` | unset (skips the module) | HF model path |
| `SKYRL_CWS_GPUS` / `SKYRL_CWS_NODES` | `8` / `1` | Total GPUs and nodes |
| `SKYRL_CWS_TP` / `SKYRL_CWS_EP` / `SKYRL_CWS_PP` | `2` / `8` / `1` | Megatron parallelism |
| `SKYRL_CWS_INFER_TP` | `8` | vLLM TP per engine; engines = GPUs / TP |
| `SKYRL_CWS_VLLM_EP` | `0` | `1` sets `enable_expert_parallel` (needed for local shards) |
| `SKYRL_CWS_BUFFER_GB` | `1.0` | `weight_transfer_threshold_cuda_ipc_GB` (slot size) |
| `SKYRL_CWS_NUM_SYNCS` | `2` | Syncs in the byte-parity test |
| `SKYRL_CWS_GMU` | `0.6` | vLLM `gpu_memory_utilization` |
| `SKYRL_CWS_EAGER` | `1` | `0` runs with CUDA graphs |
| `SKYRL_CWS_OVERRIDES` | `{}` | JSON of dotted config overrides: the options under test |
| `SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB` | unset | Caps trainer memory during the send (forwarded to Ray workers) |

`NCCL_*` and `UV_OFFLINE` are also forwarded to the Ray workers, for multi-node runs.

**Example** (one 8-GPU node, every option in this PR):

```bash
SKYRL_CWS_MODEL=Qwen/Qwen3-30B-A3B SKYRL_CWS_VLLM_EP=1 \
SKYRL_CWS_OVERRIDES='{"trainer.policy.megatron_config.weight_sync_from_offloaded_params": true,
  "generator.inference_engine.weight_sync_num_ipc_buffers": 2,
  "generator.inference_engine.weight_sync_layer_aligned_chunks": true,
  "generator.inference_engine.weight_sync_local_shard": true}' \
uv run --isolated --extra dev --extra megatron \
  pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_colocated_weight_sync.py \
  > /tmp/cws.log 2>&1
grep CWS_SUMMARY /tmp/cws.log
```

Set `RAY_ADDRESS=local` if a Ray cluster is already up on the machine. When the command is
submitted through Ray's uv runtime-env hook, do not put `--` between `uv run` and `pytest`;
the hook mis-parses it.

## Qwen3-30B-A3B

`Qwen/Qwen3-30B-A3B`: 128 experts, 48 layers, 61 GB in BF16. 8xB200 nodes, vLLM 0.33.0.
Megatron TP2/EP8 to vLLM TP8 on one node, 1 GB slots, unless noted. Numbers are from a
steady-state sync (not the first).

**Byte parity** held in every row below: 5,408 tensors on 8 workers, 11,584 on 16.

| Configuration | Round trips | Bytes received per worker | Transfer (s) | Trainer peak alloc (GiB) |
|---|---|---|---|---|
| Default path | 58 | 61.1 GB | 7.3 | 10.9 |
| Default path, 4 GB slots | 15 | 61.1 GB | 7.0 | 13.9 |
| Offloaded-params export * (trainer capped at 6 GiB) | 58 | 61.1 GB | 6.6 | **3.0** |
| Ring, 2 slots | 58 | 61.1 GB | 7.0 | 11.9 |
| Local shards + ring, vLLM EP | **11** | **10.3 GB** | 2.6 | 11.9 |
| Offloaded params + local shards + ring, vLLM EP ‡ | 11 | 10.3 GB | **2.0** | **4.0** |
| All four options *, vLLM EP | 50 | 10.3 GB | **1.8** | 4.0 |
| Offloaded params + local shards + ring, vLLM TP4 x 2 engines (EP4 each) ‡ | 17 | 17.6 GB | 3.7 | 4.0 |
| The same, 2 nodes: Megatron TP2/EP16 to vLLM TP8 x 2 (EP), 16 workers ‡ | 11 | 10.3 GB | 2.4 | 4.0 |

**After a training step**, all four options * against the default path, on the same trained
weights: identical vLLM bytes.

| | Default path | All four options * |
|---|---|---|
| Round trips | 58 | 50 |
| Bytes received per worker | 61.1 GB | 10.3 GB |
| Transfer (s) | 5.4 | 1.8 |
| Trainer peak alloc (GiB) | 11.2 | 4.2 |

‡ These development-branch runs also had selective weight sleep on, which is not in this PR. On
its own it did not change the transfer time or the trainer peak (7.1 s and 10.9 GiB against 7.3 s
and 10.9 GiB for the default path).

**Reading the table:**

- Local shards account for the transfer-time gain. They cut what each worker receives by about
  6x, and round trips with it.
- The offloaded-params export accounts for the trainer-memory gain.
- The 2-slot ring alone gains nothing on this model. Rank 0 waited ~3.8 s on sends against a
  0.8 s export, so the receiver's clone and layerwise load set the pace.
- Bigger slots barely help the default path (7.3 s to 7.0 s at 4 GB) and cost trainer memory.
- Layer-aligned chunks add round trips (one per layer: 50 here) without slowing the sync.

**OOM reproduction *.** With the trainer capped at 6 GiB during the send
(`SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB=6`), the default path fails in
`broadcast_to_inference_engines` with `Tried to allocate 1024.00 MiB ... 6.00 GiB allowed;
8.20 GiB allocated` (the reloaded BF16 shard). The offloaded-params export completes the same
sync with byte parity at a 3.0 GiB peak.

**Serving check.** After each sync the harness compares 4 greedy 48-token generations with the
fresh engine's. In eager mode they are identical through 3 syncs. With CUDA graphs on, the fresh
engine repeated with no sync already diverges at tokens 33 and 18 on two prompts, and the
default path diverges the same way after a sync. The harness therefore asserts only that no
prompt diverges before token 8.

## Large MoE at full scale

A ~590B MoE in BF16, colocated full fine-tuning on 4 x 8xB200
nodes: Megatron TP2/EP32 next to 4 TP8 vLLM engines on the same 32 GPUs. Trainer offloaded while
vLLM generates, with optimizer state on node-local NVMe. DAPO, 50 steps. This ran on an older
vLLM (0.32) with model-specific patches.

**What was used:**

| Feature | Used |
|---|---|
| Export from offloaded params | Yes |
| Local shards | Yes. About 8x fewer expert bytes per worker. This model shards experts over TP itself instead of using vLLM EP, so it used a model hook for the local-expert ids (a follow-up PR) |
| Slot size | One 1 GB slot |
| Ring | No (one slot). A second 1 GB slot, or one 2 GB slot, ran the trainer out of memory in the EP/TP gather |
| Layer-aligned chunks | Not recorded |
| Selective sleep / wake-on-write | No (follow-up PR) |
| Clone-free receive, trainer garbage cleanup | Yes, both needed (follow-up PRs) |

**Results, BF16 rollouts:**

| | |
|---|---|
| Full-weight sync | 23.5-24 s per step, steady from step 1 |
| Before the bridge flag fix (design doc, finding 1) | ~18 min |
| Engine with weights woken | ~143-145 GiB per GPU |
| Trainer, params offloaded | ~32-35 GiB per GPU |
| Smallest free memory per GPU after a sync | ~9 GiB |
| Trainer-to-vLLM logprob gap | 0.017-0.019, flat over the run |

A flat logprob gap over 30+ steps is what a complete sync should give: a dropped or stale
tensor makes it grow as training moves the weights.

**MXFP8 rollouts.** Four 50-step runs of the same model with MXFP8 rollout experts (the
weights still sync in BF16; vLLM quantizes the experts on load) averaged 26-29 s per weight sync
(steps 2-50), of which 19-22 s was the transfer itself.
