# Harbor KV cache offloading

Generation-only CodeContests evaluation on two independent Qwen3-8B engines
(TP=1, PP=1). The examples use the same Harbor generator and thinking template as
`../run_harbor_gen.sh`. No training workers are started.

`main_generate.py` selects a seed-42 subset from sorted task paths, runs Harbor,
and saves raw metrics from each engine. Unlike the ten-task debug entrypoint,
the sample count is configurable. Each mode gets a fresh cache.

## Run

Prepare CodeContests with the existing dataset script:

```bash
uv run --isolated --extra harbor \
  examples/train_integrations/harbor/prepare_harbor_dataset.py \
  --dataset open-thoughts/CodeContests > /tmp/harbor-prepare.log 2>&1
```

Run from the repository root. Set `DATA_DIR` to the directory containing the
extracted task folders and `ENV_FILE` to an absolute credentials file path.
The launcher passes credentials and a generated backend env file through
`uv --env-file`; Ray's uv worker hook reuses these arguments. Ray 2.58 is supplied
with `--with ray==2.58.0` for the evaluation cluster.

```bash
export DATA_DIR="$HOME/data/harbor/CodeContests"
export ENV_FILE="$HOME/hackskyrl/.env.srh"
export RAY_ADDRESS=<cluster-address>
unset DAYTONA_API_KEY  # Let the credentials file supply the key.

export RUN_DIR=/tmp/skyrl-kv-offload/baseline
uv run --isolated --no-project bash \
  examples/train_integrations/harbor/kv_offload/run.sh baseline \
  > /tmp/kv-baseline-launch.log 2>&1

export RUN_DIR=/tmp/skyrl-kv-offload/cpu
uv run --isolated --no-project bash \
  examples/train_integrations/harbor/kv_offload/run.sh cpu \
  > /tmp/kv-cpu-launch.log 2>&1
```

Use a fresh `RUN_DIR` for each run; the launcher rejects existing evaluation
logs. Run matched comparisons sequentially. For a
four-task smoke test, set `NUM_SAMPLES=4 MAX_CONCURRENCY=4`. `START_PORT` sets
SkyRL's engine base port; each subsequent engine uses a 100-port stride. Give
overlapping jobs separate port ranges.

The defaults are 500 tasks, admission at 5 trials/s, maximum concurrency 200,
32,768-token context, and 12,288-token output cap. `GPU_BLOCKS` controls the cache
in units of 16 FP8 KV tokens per engine. The default `GPU_BLOCKS=50000` gives
800,000 tokens; `GPU_BLOCKS=45000` gives 720,000 tokens. Keep it identical across a matched
baseline/offloading comparison. Change concurrency and cache capacity together
only when starting a new comparison. Additional SkyRL CLI overrides can be
appended to either launcher.

On this cluster, the system library setup used for the runs was:

```bash
export LD_LIBRARY_PATH="/home/ray/hackskyrl/logs/0929/runtime-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
```

This supplies the C++ runtime required by the cluster's Python SQLite build.
Other machines should use their own runtime library path.

## CPU cache

| Backend | Connector | CPU cache |
| --- | --- | --- |
| Baseline | None | None |
| CPU | `OffloadingConnector` | 128,000,000,000 bytes per engine; 64-token offload chunks |

The CPU connector's byte budget is shared across workers **within an engine**;
two independent engines each allocate that budget. See the
[vLLM CPU offloading guide](https://docs.vllm.ai/en/stable/features/kv_offloading_usage/).

## Metrics and validation

`driver.log` contains the generation log, `infra/` contains infrastructure logs
when redirection is enabled, and `trials/` contains Harbor results. `tasks.json`
records the exact task IDs; `config.json` records engine settings and package
versions. `result.json` records the generation/evaluation window and rollout metrics.
Elapsed time excludes model loading, engine startup, and the final metric flush.
`metrics.jsonl` preserves five-second per-engine scrapes, including a ten-second
flush interval after generation; `engine-*-latest.prom` contains terminal
scrapes. HTTP engine metrics are enabled by disabling Ray-only metric routing.

```bash
uv run --isolated --extra fsdp --with ray==2.58.0 \
  examples/train_integrations/harbor/kv_offload/summarize.py "$RUN_DIR" \
  > "$RUN_DIR/analysis.log" 2>&1
```

The summary takes counter differences from the initial and terminal scrapes,
keeps each engine separate, and sums their preemptions. Missing counters are
reported as unavailable. CPU traffic uses the canonical
`kv_offload_store_bytes_total` / `kv_offload_load_bytes_total`; do not add the
deprecated direction-labeled counters to these totals. Positive writes prove
the store path; positive reads prove cache reuse. Stock vLLM does not expose an
exact CPU eviction counter. Standard prefix-hit metrics exclude lookups for
already-preempted request resumptions, while transfer bytes include them.

Context-limit trajectories can have zero reward while remaining accepted.
Masked tasks make an evaluation invalid. Sampling, session placement, and
sandbox timing vary between runs, so task IDs and cache settings alone do not
make throughput differences deterministic.

## Recorded results

Runs used one eight-B200 node, two TP1/PP1 engines, vLLM 0.30.0, Ray 2.58.0,
and 500 tasks from `/mnt/local_storage/data/harbor/CodeContests-500`.
Raw artifacts remain in `/tmp/skyrl-kv-offload/`; compact results are saved in
[`results.json`](results.json).

| GPU KV tokens per engine | Mode | Preemptions (engine 0 + 1) | Evaluation time | Masked tasks |
| --- | --- | --- | --- | --- |
| 800,000 | Baseline | 1 + 4 = 5 | 751.40 s | 0 |
| 800,000 | CPU | 0 + 1 = 1 | 681.47 s | 0 |
| 720,000 | Baseline | 0 + 44 = 44 | 748.89 s | 1 (invalid) |
| 720,000 | CPU | 0 + 5 = 5 | 685.86 s | 0 |

The valid 800,000-token pair measured a 69.93-second reduction (9.3%) in
comparison time. CPU mode stored 435.86 GB and loaded 56.61 GB across both
engines. These are cumulative transfer bytes, rather than peak CPU occupancy.
Both runs had zero masked tasks and zero timeouts.

The smaller-cache baseline had a Daytona sandbox failure that masked one task,
so its timing is not a valid comparison with CPU mode. The CPU run stored
430.04 GB and loaded 51.61 GB, with zero masked tasks and zero timeouts.
Approximately 30 baseline preemptions has not yet been established in a valid
run; CPU mode stayed below ten in both completed evaluations.

These are single runs. Sampling, engine placement, sandbox timing, output
length, and overlapping cluster activity can affect elapsed time. A controlled
repeat is needed before attributing the full measured speedup to offloading.
