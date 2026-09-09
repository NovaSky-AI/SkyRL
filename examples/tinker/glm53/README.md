# GLM 5.2 / 5.3: full-context GSPO profiling

First validate the client/profiler on **Qwen3-0.6B**, then use the GLM **32K, two-node** profile.
This example uses SkyRL's Tinker API directly;
no TCLI, dataset, reward model or task rollout collection is needed. These are review
candidates, not GPU-qualified recipes or performance claims.

## What the client runs

```text
create adapter → prepare full-context inputs → initial publication + short sample
  → warmup update → measured updates → checkpoint → unload adapter

each update: reference forward → one batched GSPO forward/backward → optimizer
             → publication + short sample
```

Two short seed texts are tokenized and **repeated to fill the requested context**.
For `--context 32768`, each fixture has 32,768 input tokens, 32,768 shifted targets and
32,768 nonzero loss-mask entries. One extra source token supplies the last target.
The client logs these counts and saves the actual expanded tensors.

The fixtures have distinct tokens and sequence-constant advantages +1 and -1. Old-policy
logprobs come from one batched trainer forward before each update and stay frozen through
that update. All updates use GSPO; `cross_entropy` is only the forward-only scoring label.
The profile uses sequence-mean reduction and upstream clipping defaults (0.2 each side).
`--batch-size` is the total number of full-context sequences in that one request (default 2).
SkyRL handles microbatching internally. Repeated updates exercise persistent optimizer state;
this is not a separate cross-request gradient-accumulation test.

## Server setup

Use the same checkout/environment and checkpoint path on every node of an owned Ray cluster.
Each B300 node has eight GPUs; verify physical trainer/inference separation.

| Profile | Trainer | Inference | Context |
| --- | --- | --- | --- |
| `qwen3-0.6b` | 1 GPU, TP1 | 1 separate GPU, TP1 | 32,768 |
| `glm52-32k-2n` | 1 node, TP8/CP1/EP8 | 1 node, TP8 | 32,768 |
| `glm53-32k-2n` | 1 node, TP8/CP1/EP8 | 1 node, TP8 | 32,768 |
| `glm53-256k-2n` | 1 node, TP4/CP2/EP8 | 1 node, TP8 | 262,144 |
| `glm53-256k-3n` | 2 nodes, TP8/PP2/EP8 (38/40 layers) | 1 node, TP8 | 262,144 |

All profiles preserve rank-32 attention/MLP LoRA, default FP32 publication and disabled MTP.
Inference uses BF16 weights/KV, with a 0.80 GPU-memory fraction. The default sequence ceiling
is 1024—not proven concurrent capacity. In particular, 256K KV capacity must still be measured.
The 32K trainer's 16,384-token packing target never truncates a full-context singleton.

```bash
hf download zai-org/GLM-5.3-BF16 --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
  --local-dir /shared/models/glm53-bf16

export RAY_ADDRESS=auto
uv run --isolated --extra tinker --extra megatron python examples/tinker/glm53/run_server.py glm53-32k-2n \
  --model-path /shared/models/glm53-bf16 --state-dir /shared/glm53-control \
  --database-path /local/glm53/tinker.db --profile-dir /local/glm53/traces/run-01
```

Keep the explicit `python`: the API recovers its engine's uv flags from the parent
command and does not recognize a direct `uv run ... run_server.py` invocation.

Start Ray with `ray start --head --port=6379 --num-gpus=8` on the head and
`ray start --address=HEAD_IP:6379 --num-gpus=8` on workers, if not already running.
Keep checkpoint/adapter paths shared; SQLite and compilation caches belong on node-local
scratch. The API binds localhost; use approved authenticated transport for remote clients.
Add `--print-config` for a GPU-free config inspection. Record the source SHA, image digest,
resolved package/config versions and server logs; old deployment receipts do not qualify
this newer dependency matrix (Torch 2.13 rather than historical 2.11).

For GLM 5.2, use profile `glm52-32k-2n` and the native-BF16 checkpoint
[`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2/tree/cf457fa734ab149ffef225f80893eb38c6ff5cdc)
at revision `cf457fa734ab149ffef225f80893eb38c6ff5cdc`. Download it to a separate model
directory and pass that path to both server and client. Its runtime knobs match `glm53-32k-2n`;
this is a new current-stack candidate, not a reproduction of an older FP8 GLM 5.2 run.

### Small-model control

`qwen3-0.6b.json` uses the same model/all-linear LoRA combination as
[the existing Megatron LoRA example](../../train/megatron/run_megatron_lora_qwen3-0.6b.sh),
with separate trainer/inference GPUs. It does not inherit GLM's DSA or expert settings.
Download `Qwen/Qwen3-0.6B` to `/shared/models/qwen3-0.6b` and record the checkpoint revision.
Use the server command above with profile `qwen3-0.6b`, that model path, and separate
state/database/trace directories. Run the **same client** below with the Qwen model path.
This profile needs two available GPUs, not two full B300 nodes; it is not yet GPU-qualified.
The native [full-context training example](../../train_scripts/full_context/README.md)
is a complementary backend check, but does not exercise this Tinker API/client path.

## Run the client

```bash
timeout --signal=TERM --kill-after=30s 2h \
  uv run --isolated --extra tinker examples/tinker/glm53/run_client.py \
  --model-path /shared/models/glm53-bf16 --context 32768 --batch-size 2 --steps 3 \
  --output-dir /local/glm53/control-01 > /local/glm53/control-01.log 2>&1
```

The log's parent must exist; use a new output directory. `--steps 3` means **one warmup
optimizer update plus two measured updates**. Warmup changes the adapter/optimizer state;
compare identical starting states and the same warmup protocol, not unlike runs.
The default makes four publications: one initial and one after each update.
Job settings (length, batch size, update count, learning rate) belong to the client.
Placement, precision, memory/packing budgets and worker trace setup belong to the server.
These calls target SkyRL's GSPO API and unload endpoint; this client is not yet a hosted
Tinker baseline runner. A hosted comparison must first match supported loss semantics,
fixture tokens, batch size and update count rather than silently substitute another loss.

## Profiling and artifacts

Profiling is required for this diagnostic example. Use a fresh dedicated service and a new
`--profile-dir` on each policy node. Every trainer rank records CPU/CUDA shapes and memory
as soon as the policy runtime starts, with no skipped or profiler-warmup windows. The first
client update is still labeled warmup for timing, but its operations are recorded too.
The existing hooks export a window after each successful optimizer and continue recording.
Unloading an adapter does not stop profiling on a warm service.

After an OOM, do not reuse this diagnostic runtime: the current failure guard stops and
disables that worker's profiler. It does not affect other services, but a later client on
the same warm runtime would lack that rank's trace. Per-job recovery remains an open
profiling-lifecycle issue; this example is not yet a reusable post-failure benchmark service.

On a PyTorch OOM in Megatron forward, backward or optimizer, the failing worker attempts
to export its active trace locally before re-raising, including during the first update.
No successful optimizer or follow-up controller RPC is required for this failure export.
A killed process or failed export can still lose the active window; cold model loading
and vLLM are outside these policy hooks. Retain
phase records and server logs, and verify traces were exported on every trainer node.
Profiling can retain tensors, consume substantial disk space and slow execution; these
diagnostic timings are not unprofiled throughput measurements.

- `phases.jsonl`: initial costs, labeled warmup, measured steps and subphase timing/metrics.
  Parent step times include their subphases; do not add both together.
- `datums.json`, `run.json`: expanded fixtures, hash, model/SDK and job settings.
- `step_*_batch.json`: replay inputs including reference logprobs, masks and advantages.

Successful completion checks finite outputs/gradient norms, repeated full-context training,
publication, checkpoint and adapter cleanup. It does **not** prove learning, train/sampler
parity, heterogeneous packing, full-context inference or rollout concurrency. Trainer-sourced
references are not sampler-parity evidence.

The client unloads its adapter; SkyRL keeps the service warm by default. A hard timeout can
interrupt cleanup: inspect the recorded model ID and pending work before reuse. Node/service
teardown remains the owner's responsibility; client exit does not release the deployment.
