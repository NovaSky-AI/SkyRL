# SkyRL Fireworks training backend: progress and validation

Last updated: 2026-07-21 07:00 UTC

This is the working record of the Fireworks Training API integration in SkyRL:
what was implemented, why the boundaries look the way they do, what was tested
live, and what remains incomplete. The longer design rationale is in
[`fireworks_training_backend.md`](fireworks_training_backend.md).

## Current status

The backend now runs policy-only GRPO through Fireworks serverless or dedicated
training APIs and uses Fireworks-hosted rollout sampling. Dedicated access is
enabled on this account; serverless training reached the service but returned
HTTP 403, so the live work moved to dedicated B200 shapes.

The following paths are implemented:

- synchronous dedicated GSM8K with LoRA or full-parameter training;
- fully asynchronous dedicated GSM8K with bounded SkyRL staleness;
- exact response-token and rollout-logprob transport into Fireworks' built-in
  importance-sampling loss;
- one stable dedicated sampler across Fireworks hot-loads;
- data-parallel trainer replicas and independent rollout replica counts;
- exact-ID resource cleanup and deletion audit.

The implementation is GRPO, not PPO. There is no critic, value model, GAE, or
PPO update path.

## Architecture and design decisions

### Fireworks is a hosted backend, not a `WorkerDispatch` target

`WorkerDispatch` assumes Ray actor groups, locally placed models, CPU/GPU
offload, and direct weight broadcasts. A Fireworks trainer is a remote service
with a different lifecycle. We therefore added a provider-specific runtime and
policy dispatch adapter while preserving SkyRL's outer training loop:

```text
SkyRL dataset / generator / reward / GRPO / async staleness / W&B
                              |
                    RayPPOTrainer-compatible loop
                       /                 \
        FireworksPolicyDispatch    FireworksInferenceClient
                       \                 /
                         FireworksRuntime
             training client + snapshots + hosted sampler
```

SkyRL still owns prompt grouping, rewards, group-relative advantages, loss
masks, staleness admission, and metrics. Fireworks owns forward/backward,
optimizer state, sampler checkpoint creation, and deployment hot-loading.

### The training data conversion is explicit

SkyRL tensors are converted to shifted `tinker.Datum` objects without decoding
and retokenizing. For `tokens = prompt + response`:

```text
model_input   = tokens[:-1]
targets       = zeros over prompt prediction positions + response token IDs
logprobs      = zeros over prompt prediction positions + behavior logprobs
advantages    = zeros over prompt prediction positions
                + GRPO advantage * response loss mask
```

The converter strips padding from `attention_mask`, respects right-aligned
response masks, preserves step-wise masked spans, enforces the Fireworks
context limit, and rejects non-finite or misaligned rollout logprobs.

The policy objective is configured as SkyRL `rollout_is` and dispatched to
Fireworks' `importance_sampling` loss. This is ordinary token-level rollout
importance sampling. SkyRL's optional truncated-IS and geometric/product
sequence masking are not yet translated and are rejected at config validation.

### Full fine-tuning and LoRA

`trainer.policy.model.lora.rank=0` means full-parameter training on dedicated
Fireworks. A positive rank selects LoRA. Serverless is currently restricted to
LoRA in this integration. All Qwen3-4B learning-curve runs described below used
full-parameter training unless explicitly labeled LoRA.

### Dedicated ASYNC weight transition

Dedicated mode retains one stable SDK sampling client. Publishing a policy
version saves an initial base snapshot and then delta snapshots; Fireworks
loads them into the linked rollout deployment. Logs show Fireworks-managed
cloud checkpoint transfer followed by deployment hot-load, not NCCL/RDMA or a
trainer-to-rollout GPU collective.

With Fireworks' documented `ASYNC` transition, an active request pauses during
the swap and resumes on the same stream with its live KV state. New requests
queue during the transition. SkyRL does not need to abort or recreate active
requests, and it does not need multiple clients to keep old requests alive.

The runtime's sampler lease is only a lifetime guard:

- it counts active calls so final teardown cannot close the SDK sampler under
  a live stream;
- in serverless mode it also lets an old snapshot-qualified client drain after
  an RCU pointer swap; and
- in dedicated mode it does **not** pin a request to one policy or perform the
  pause/resume—the Fireworks deployment performs the in-flight transition.

The SDK helper currently relies on Fireworks' documented ASYNC default; SkyRL
does not yet explicitly set or verify the deployment transition field.

### Fully asynchronous scheduling

SkyRL's existing fully async trainer remains the scheduler. It tracks each
trajectory group's submission step and applies `max_staleness_steps` as an
admission bound while rollout generation, remote optimization, and weight
publication overlap. `pause_generation` is a local admission gate. Active
dedicated requests remain alive and are handled by the Fireworks ASYNC
transition.

The first fully async GSM8K configuration used:

```text
train_batch_size = 64 prompt groups
n_samples_per_prompt = 5
max_staleness_steps = 2
num_parallel_generation_workers = (2 + 1) * 64 = 192
policy_loss_type = rollout_is
```

This run uses importance sampling, but not SkyRL's additional TIS/geometric
sequence mask.

### Trainer and rollout scaling

The account's validated full-training profile is
`accounts/fireworks/trainingShapes/qwen3-4b-minimum` (profile version
`m06lkf12`). One shaped replica contains one node and one B200 180 GB
accelerator. Thus `1x` means one chip for this shape, not an eight-GPU node.

The shape topology cannot be changed by setting `accelerator_count=2` or
`node_count=2`; the SDK configuration rejects those overrides. It can be
replicated with `trainer_replica_count=2`, which produces two data-parallel
HSDP replicas, each holding the complete one-B200 shape. This is data parallel
training, not tensor/pipeline sharding of one model across two chips.

Rollout scaling is independent through `replica_count`. The read-only probe is
[`check_fireworks_qwen3_4b_trainer_scaling.py`](../../examples/train/gsm8k/check_fireworks_qwen3_4b_trainer_scaling.py),
and the paid two-replica demonstration wrapper is
[`run_gsm8k_fireworks_2x_b200_trainer_replicas.sh`](../../examples/train/gsm8k/run_gsm8k_fireworks_2x_b200_trainer_replicas.sh).

## Implementation inventory

The main additions are:

- `skyrl/backends/fireworks/runtime.py`: service, trainer, snapshots, stable
  dedicated sampler, serverless RCU leases, and teardown;
- `skyrl/backends/fireworks/inference.py`: exact token-in/token-out sampling,
  and pause/resume admission;
- `skyrl/backends/fireworks/grpo.py`: SkyRL batch to Fireworks datum conversion;
- `skyrl/backends/fireworks/training_backend.py`: remote policy dispatch;
- `skyrl/train/entrypoints/main_fireworks.py`: direct hosted GSM8K entrypoint;
- Fireworks config, validation, dependency extra, and tests; and
- dedicated GSM8K sync/async scripts plus exact-ID cleanup tooling.

Secrets remain environment variables. `FIREWORKS_API_KEY` and `WANDB_API_KEY`
are propagated without being serialized into SkyRL config.

## Validation completed

### Unit and contract tests

The Fireworks tests cover config capability gates, credential redaction, datum
shape/alignment, runtime publication and cleanup, stable dedicated clients,
serverless RCU retirement, native async sampling, trainer replica
pass-through, and entrypoint selection.

### Live runs

| Validation | Configuration | Result |
| --- | --- | --- |
| Serverless entitlement probe | Qwen small-model LoRA attempt | Reached Fireworks and returned HTTP 403 before a paid training session; dedicated access was used afterward. |
| Dedicated lifecycle smoke | Qwen3-4B LoRA, 1 trainer + 1 rollout, batch 2 × 4, one step | Forward/backward, optimizer, initial and updated hot-loads passed; exact resources deleted and independently audited. W&B [`8gtfo4gy`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/8gtfo4gy). |
| Reward diagnosis | Same small batch with 256 max generated tokens | Reward was uniformly zero because all responses ended inside `<think>` at the cap, before GSM8K's `####` answer delimiter. Raising the cap fixed it. |
| Ten-step small-batch run | Qwen3-4B, batch 2 × 4, 1,024 tokens | Mean reward 0.35 over 80 completions; nonzero on 9/10 steps. This validated learning transport but was intentionally too noisy for a trend. W&B [`au5dmfx9`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/au5dmfx9). |
| Ten-step full-parameter curve | Qwen3-4B full, batch 64 × 5, 1 trainer + 2 rollout | Raw reward rose from 0.390625 to 0.890625; ten-step mean 0.781562. W&B [`euznxf4s`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/euznxf4s). |
| Local SkyRL comparison | Qwen3-4B FSDP, batch 64 × 5, microbatch 1, no offload | Generated one batch with raw reward 0.384375, then OOMed during Adam optimizer state update on 80 GB GPUs. W&B [`b13jq803`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/b13jq803). |
| Long synchronous Fireworks comparison | Qwen3-4B full, batch 64 × 5, 1 trainer + 1 rollout | Stopped on request after publishing version 84 and logging 85 reward batches; raw reward moved from 0.375 to 0.928125, mean 0.929301. Exact cleanup reported trainer `JOB_STATE_DELETED` and deployment absent. W&B [`vdb3pf53`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/vdb3pf53). |
| Long fully async Fireworks run | Qwen3-4B full, batch 64 × 5, staleness 2, 192 workers, 1 trainer + 2 rollout | Completed 42 optimizer updates before the requested stop; 43 generated reward batches moved from 0.41875 to 0.9625, mean 0.862573. Exact trainer/deployment cleanup passed. W&B [`ymgzevsh`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/ymgzevsh). |
| Fully async Harbor/CodeContests experiment | Qwen3-4B full, batch 16 × 8, staleness 4, 32 group workers, 2 trainer + 6 rollout | Stopped by request after validating remote multi-turn execution. Step-wise prefix merging reduced 328→328, 347→347, and 570→570 sequences, so the experimental chat proxy and launch path were removed. W&B [`uozx19sz`](https://wandb.ai/sky-posttraining-uc-berkeley/harbor/runs/uozx19sz). |

The large-batch synchronous and asynchronous curves both rose into roughly the
0.9–0.98 range. This is consistent with functional GRPO updates and much less
noisy than the original two-prompt experiment. It is not a controlled sync vs
async comparison because their rollout replica counts and sample order differ.

### Performance observations

- Full Qwen3-4B remote forward/backward plus optimizer was typically about
  15–21 seconds for the batch used in the async run.
- Snapshot creation used a base snapshot first and delta snapshots afterward,
  commonly about 7–11 seconds.
- Deployment load/update dominated at roughly 45 seconds, making total weight
  publication about 50–60 seconds.
- Adding a second rollout replica did not materially shorten hot-load latency.
- Fully async steady-state steps were roughly 65–75 seconds. The workload was
  weight-sync-bound rather than rollout-bound, motivating 1 trainer + 1
  rollout as the economical default for future GSM8K comparisons.

## Abandoned Harbor/CodeContests experiment

The experiment confirmed that Harbor could execute multi-turn Daytona trials
against the hosted sampler, but its turns were not prefix-mergeable. In a
representative three-turn trajectory, the first prompt plus completion was
14,326 tokens while the next prompt was only 1,714 tokens. The next prompt
diverged one token before the generation boundary and omitted the separated
Qwen `reasoning_content`, so `prompt[i] + completion[i]` was never an exact
prefix of `prompt[i+1]`. Step-wise training remained token-correct, but every
turn stayed a separate training sequence.

The CodeContests run was stopped, its exact trainer/deployment were deleted and
audited, and the experimental local OpenAI proxy, affinity-client machinery,
Fireworks Harbor entrypoint branch, tests, and launch wrapper were removed.

## Current target: Harbor Apex with TITO

The next requested target is the private Harbor Apex fully-async workflow,
ported from
`SkyRL-private/examples/train_integrations/harbor_apex/run_tito_qwen36_35b_apex_dev_fully_async_8nodes_h200_split.sh`.
Unlike the abandoned chat-completions bridge, this uses TITO (token-in,
token-out): the agent retains one exact token sequence plus loss mask and
behavior logprobs across model and tool turns. It therefore does not depend on
adjacent chat prompts satisfying SkyRL's prefix-merging condition.

Requested live configuration:

```text
base model             accounts/fireworks/models/qwen3p5-9b
maximum context        128,000 tokens
training               4 accelerators total, two trainer replicas
rollout                12 accelerators
training dataset       apex-agents-dev-1928
evaluation dataset     apex-agents-eval-99-062926 (99 tasks)
maximum concurrency    500
W&B project            mercor-rl (same project as the source Apex runs)
```

Implementation status:

- the previous CodeContests process is stopped and its exact Fireworks trainer
  and deployment have been deleted and audited;
- the experimental local OpenAI proxy has been removed;
- all 53 tracked files in the private Apex integration were copied into
  `examples/train_integrations/harbor_apex`, while `pyproject.toml` now uses the
  editable absolute `/home/ray/default/harbor-private` checkout;
- the copied directory has two substantive edits relative to SkyRL-private:
  `entrypoints/common.py` conditionally handles the private-only curriculum
  field and forwards the Modal environment plus Fireworks/Daytona credentials;
  `tito_harbor_generator.py` selects the native Fireworks `/completions`
  endpoint, configures TITO/session affinity, forwards the worker environment,
  and pins hosted-inference trials to the Ray entrypoint node. A new
  Fireworks-specific Qwen3.5 launcher was also added. The differences in the
  two `__init__.py` files and `wandb_config.txt` are newline-only, and
  `__pycache__` entries are generated artifacts;
- the Qwen3.5-9B full-training shape is
  `accounts/fireworks/trainingShapes/qwen3p5-9b-256k/versions/p7qlru0q`.
  It is a one-node, two-B200 shape, so two trainer replicas are four training
  B200s. Its linked rollout shape is
  `accounts/fireworks/deploymentShapes/rft-qwen3p5-9b-v2/versions/pcxyecdg`,
  which owns one B200 per replica, so 12 replicas are 12 rollout B200s;
- `FireworksRuntime` exposes the native SDK-managed deployment model and
  `https://api.fireworks.ai/inference/v1` base URL to agent integrations;
- the Apex generator now sends Harbor's TITO requests directly to that managed
  deployment. It does not recreate the removed chat proxy and does not pass
  model requests through SkyRL's token sampler wrapper;
- `harbor-private` gained an explicit `session_affinity_backend=fireworks`
  mode. It sends the documented `x-multi-turn-session-id` and
  `x-session-affinity` headers on every turn, while omitting SkyRL-local
  `session_id` and `cache_salt` request-body fields that Fireworks does not
  accept;
- hosted Harbor trial tasks pin to the Ray entrypoint node, because there are
  no local inference nodes. The current cluster has one 23-CPU/240-GiB node;
  the task's 0.2-CPU Ray reservation limits actual local concurrency to about
  115 even though the requested limiter ceiling remains 500;
- dev-1928 is present with 1,928 task directories and the requested dated
  evaluation set is present as `apex-agents-eval-99-062926` with 99 tasks;
- the long-running wrapper is
  `examples/train_integrations/harbor_apex/run_tito_qwen3p5_9b_apex_dev_fully_async_fireworks.sh`.
  It uses full-parameter GRPO, batch 16 x 16, staleness 3, 64 prompt-group
  workers, 128k context, W&B project `mercor-rl`, eval every 10 steps, and a
  Fireworks DCP checkpoint every 5 steps; and
- credentials will be loaded from `SkyRL-private/.env.apex` and the existing
  Fireworks shell environment without copying values into configs or this log.

Local validation after the port:

- `uv lock` resolved successfully with the editable Harbor checkout;
- the dedicated runtime endpoint and inference-client tests pass (13 tests);
- Harbor's Fireworks-versus-SkyRL affinity behavior passes its three new unit
  tests;
- Ruff passes on the changed SkyRL and Harbor Python files; and
- the paid wrapper's no-provision preflight resolves exactly 2 x 2 training
  B200s, 12 x 1 rollout B200s, the two dated datasets, 128k context, and a
  500-trajectory concurrency ceiling.

### Live Apex launch (2026-07-21)

Three no-cost boot attempts failed before provider provisioning and were
cleanly audited with both exact resources absent:

1. the cluster runs Ray 2.51.1 while the project was pinned to 2.56.0;
2. Ray's staged working directory could not resolve an editable relative
   `../harbor-private` dependency; and
3. copied private code expected the optional private
   `data.curriculum_config`, which public SkyRL does not expose.

The project now matches cluster Ray 2.51.1, the editable Harbor source is the
absolute local `/home/ray/default/harbor-private` checkout (required because
the run is pinned to this one node), and curriculum ordering is conditionally
enabled only when both public APIs support it. A remote Ray import preflight
confirmed that workers load Harbor from that exact checkout.

The first paid launch (`apex4`) validated the provider and inference setup:

- the two-replica trainer reached `JOB_STATE_RUNNING` and the 12-replica
  deployment reached `READY` on the expected linked one-B200 shape;
- W&B run `bkxumks3` started in `sky-posttraining-uc-berkeley/mercor-rl`;
- the full base sampler snapshot was created, hot-loaded in 25 seconds, and
  published as SkyRL policy version 0; and
- Harbor constructed the native
  `text-completion-openai/accounts/.../deployments/...` TITO target at
  `https://api.fireworks.ai/inference/v1`.

The evaluation then exposed a sandbox configuration bug before making an LLM
request. `.env.apex` contains the permitted `MODAL_ENVIRONMENT`, but the copied
Apex Ray-worker whitelist did not forward that variable. Workers therefore
fell back to Modal environment `main`, where the supplied identity has no write
access. The invalid evaluation was stopped before training; its exact trainer
is `JOB_STATE_DELETED` and its deployment is absent. `MODAL_ENVIRONMENT` is now
forwarded through both the entrypoint and per-trial worker environments and is
required by the launcher. This retains the source Apex Modal/ECR path; Daytona
does not yet have the private-ECR credential handling used by these task images.

The second paid launch used:

```text
run name       harbor-apex-fireworks-qwen3p5-9b-dev1928-2x2train-12rollout-20260721074002-apex5
trainer ID     skyrl-smoke-harbor-apex-20260721074002-apex5-trainer
deployment ID  skyrl-smoke-harbor-apex-20260721074002-apex5-rollout
driver log     /home/ray/data/harbor/fireworks_runs/harbor-apex-fireworks-qwen3p5-9b-dev1928-2x2train-12rollout-20260721074002-apex5/driver.log
```

Its remote-worker preflight listed `MODAL_ENVIRONMENT` alongside the two Modal
credentials, Fireworks, Daytona, and tool-service credentials without logging
their values. The 12-replica rollout deployment reached `READY`; the
two-replica trainer
reached `JOB_STATE_RUNNING` at 07:51:50 UTC. W&B run `99ok27of` started at
`sky-posttraining-uc-berkeley/mercor-rl`. Fireworks created and hot-loaded the
full version-0 base snapshot in 63 seconds, after which the requested 99-task
evaluation began. The Modal environment fix worked: private images started,
agents loaded all 73 MCP tools, and the previous `PermissionDeniedError` count
remained zero.

That evaluation exposed the next pre-inference compatibility issue:
`TITOAgentState` unconditionally loaded `generation_config.json`, which
`Qwen/Qwen3.5-9B` does not publish. The run was stopped before any optimizer
step, and its exact trainer/deployment were audited as deleted/absent. Harbor's
TITO state now falls back to generation defaults derived from `config.json` and
also accepts the tokenizer EOS. For Qwen3.5 this correctly recognizes both
`<|endoftext|>` (embedded model default) and `<|im_end|>` (chat-template turn
delimiter). A regression test plus a real cached Qwen3.5 tokenizer preflight
pass. The forced stop initially left 18 Modal sandboxes; all 18 exact
containers belonging to the Apex app `harbor-apex-fw-qwen35` (app ID
`ap-iGAkrrtoJu1KWdiC5vcnAO`) were terminated and that exact app was re-audited
at zero before the next evaluation. No workspace-wide container cleanup was
used. The launcher now gives every future run a unique Modal app name derived
from its resource suffix. Manual cleanup must resolve that exact app name and
app ID and may stop only containers belonging to it; unrelated Modal apps in
the shared workspace are out of scope.

The third paid launch is active with the two validated fixes:

```text
run name       harbor-apex-fireworks-qwen3p5-9b-dev1928-2x2train-12rollout-20260721080020-apex6
trainer ID     skyrl-smoke-harbor-apex-20260721080020-apex6-trainer
deployment ID  skyrl-smoke-harbor-apex-20260721080020-apex6-rollout
driver log     /home/ray/data/harbor/fireworks_runs/harbor-apex-fireworks-qwen3p5-9b-dev1928-2x2train-12rollout-20260721080020-apex6/driver.log
```

The launcher is detached in process group/session `3941570`. At the 08:06 UTC
monitoring check, the 12-replica rollout deployment was `READY`, the
two-replica trainer was still `JOB_STATE_CREATING` with provider status `OK`,
and all local driver processes were healthy. At the first ten-minute boundary
(08:10 UTC), these states remained unchanged and provider status was still
`OK`; no trial files existed yet because SkyRL was waiting for trainer
allocation. The new log contained no `generation_config`, trajectory-attempt,
Modal-permission, or traceback errors. Monitoring is scheduled every 10 minutes
until the first successful optimizer step, then hourly.

## Known gaps

- Fireworks DCP save/resume has been validated on GSM8K, but the new Apex run
  has not yet reached its first step-5 checkpoint/resume boundary.
- The geometric/product sequence mask and truncated-IS options are not
  translated to Fireworks.
- Per-token Fireworks snapshot identities are not yet returned by the SDK
  adapter, so a request crossing a hot-load cannot yet be labeled token by
  token in SkyRL metrics.
- Dedicated ASYNC is currently the provider default rather than an explicit
  field set and audited by this SDK path.
