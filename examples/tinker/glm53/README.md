# GLM 5.3: full-context GSPO profiling

Start with the **32K, two-node** profile. This example uses SkyRL's Tinker API directly;
no TCLI, dataset, reward model or task rollout collection is needed. These are review
candidates, not GPU-qualified recipes or performance claims.

## What the client runs

```text
create adapter → prepare full-context inputs → initial publication + short sample
  → warmup update → measured updates → checkpoint → unload adapter

each update: two reference forwards → two accumulated GSPO backwards → optimizer
             → publication + short sample
```

Two short seed texts are tokenized and **repeated to fill the requested context**.
For `--context 32768`, each fixture has 32,768 input tokens, 32,768 shifted targets and
32,768 nonzero loss-mask entries. One extra source token supplies the last target.
The client logs these counts and saves the actual expanded tensors.

The fixtures have distinct tokens and sequence-constant advantages +1 and -1. Old-policy
logprobs come from real trainer forwards before each update and stay frozen for both
backwards. All updates use GSPO; `cross_entropy` is only the forward-only scoring label.
The profile uses sequence-mean reduction and upstream clipping defaults (0.2 each side).
Two backward requests accumulate gradients before one optimizer update. This exercises
the second backward with gradients already resident; it is an accumulation stress case,
not a requirement that every training client issue two requests per update.

## Server setup

Use the same checkout/environment and checkpoint path on every node of an owned Ray cluster.
Each B300 node has eight GPUs; verify physical trainer/inference separation.

| Profile | Trainer | Inference | Context |
| --- | --- | --- | --- |
| `32k-2n` | 1 node, TP8/CP1/EP8 | 1 node, TP8 | 32,768 |
| `256k-2n` | 1 node, TP4/CP2/EP8 | 1 node, TP8 | 262,144 |
| `256k-3n` | 2 nodes, TP8/PP2/EP8 (38/40 layers) | 1 node, TP8 | 262,144 |

All profiles preserve rank-32 attention/MLP LoRA, default FP32 publication and disabled MTP.
Inference uses BF16 weights/KV, with a 0.80 GPU-memory fraction. The default sequence ceiling
is 1024—not proven concurrent capacity. In particular, 256K KV capacity must still be measured.
The 32K trainer's 16,384-token packing target never truncates a full-context singleton.

```bash
hf download zai-org/GLM-5.3-BF16 --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
  --local-dir /shared/models/glm53-bf16

export RAY_ADDRESS=auto
uv run --isolated --extra tinker --extra megatron examples/tinker/glm53/run_server.py 32k-2n \
  --model-path /shared/models/glm53-bf16 --state-dir /shared/glm53-control \
  --database-path /local/glm53/tinker.db
```

Start Ray with `ray start --head --port=6379 --num-gpus=8` on the head and
`ray start --address=HEAD_IP:6379 --num-gpus=8` on workers, if not already running.
Keep checkpoint/adapter paths shared; SQLite and compilation caches belong on node-local
scratch. The API binds localhost; use approved authenticated transport for remote clients.
Add `--print-config` for a GPU-free config inspection. Record the source SHA, image digest,
resolved package/config versions and server logs; old deployment receipts do not qualify
this newer dependency matrix (Torch 2.13 rather than historical 2.11).

## Run the client

```bash
timeout --signal=TERM --kill-after=30s 2h \
  uv run --isolated --extra tinker examples/tinker/glm53/run_client.py \
  --model-path /shared/models/glm53-bf16 --context 32768 --batch-size 1 --steps 3 \
  --output-dir /local/glm53/control-01 > /local/glm53/control-01.log 2>&1
```

The log's parent must exist; use a new output directory. `--steps 3` means **one warmup
optimizer update plus two measured updates**. Warmup changes the adapter/optimizer state;
compare identical starting states and the same warmup protocol, not unlike runs.
The default makes four publications: one initial and one after each update.
Job settings (length, batch size, update count, learning rate) belong to the client.
Placement, precision, memory/packing budgets and worker trace setup belong to the server.

## Profiling and artifacts

Measure an **unprofiled control** first. For a separate profiled run, start a fresh dedicated
service with `--profile-dir /local/glm53/traces/run-01`, then run the identical client.
Existing Tinker hooks are runtime-scoped: the server profiler warms up through the first
optimizer and records until the second. This includes warmup publication and the next
reference/backward passes. It is not a client-session profiling API: unloading an adapter
from a warm service does not reset the schedule for another job.

An OOM before the first optimizer completes can leave no recorded GPU trace: that interval
is profiler warmup. Phase records and server logs are still needed for failure diagnosis.
This preset is for steady-state profiling, not warmup-OOM capture; even an active trace
can be lost if its worker is killed before export.

Traces cover CPU/CUDA shapes and memory on **policy rank 0**, not vLLM workers, all-rank peaks
or cold model loading. Verify traces were actually exported. Profiling can retain tensors
and slow execution; use unprofiled repeats to credit speedups.

- `phases.jsonl`: initial costs, labeled warmup, measured steps and subphase timing/metrics.
  Parent step times include their subphases; do not add both together.
- `datums.json`, `run.json`: expanded fixtures, hash, model/SDK and job settings.
- `step_*_batch_*.json`: replay inputs including reference logprobs, masks and advantages.

Successful completion checks finite outputs/gradient norms, repeated full-context training,
publication, checkpoint and adapter cleanup. It does **not** prove learning, train/sampler
parity, heterogeneous packing, full-context inference or rollout concurrency. Trainer-sourced
references are not sampler-parity evidence.

The client unloads its adapter; SkyRL keeps the service warm by default. A hard timeout can
interrupt cleanup: inspect the recorded model ID and pending work before reuse. Node/service
teardown remains the owner's responsibility; client exit does not release the deployment.
