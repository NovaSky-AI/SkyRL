# Fireworks Training API backend for SkyRL

Status: synchronous serverless/dedicated MVP and fully-async scheduler wiring implemented; dedicated ten-step GRPO live-validated  
Initial scope: text-only, policy-only GRPO; LoRA or dedicated full-parameter training  
Target order: serverless GSM8K sync -> dedicated GSM8K sync -> dedicated GSM8K fully async

Serverless remains account-gated: on July 20, 2026, its one-step GSM8K smoke
reached the API but received HTTP 403 before a training session or paid model
operation was created. Dedicated access is enabled on this account. On July
21, 2026, the bounded Qwen3-4B dedicated run provisioned a trainer and linked
rollout deployment, completed one GRPO forward/backward plus optimizer step,
hotloaded both the initial and updated sampler snapshots, and deleted both
resources. See [Live dedicated validation](#live-dedicated-validation).

## Decision summary

Fireworks should be added as a remote training and inference runtime, not as a
conditional branch inside `WorkerDispatch`. `WorkerDispatch` assumes Ray actor
groups, local GPU placement, CPU/GPU offload, and direct weight broadcasts.
Those concepts do not exist for a hosted trainer.

The Fireworks entrypoint runs the orchestration loop directly in its driver;
it does not attach to a Ray cluster. This keeps hosted training usable when the
local cluster's Ray version differs and avoids allocating cluster resources for
a backend whose model computation is entirely remote.

The native SkyRL loop should continue to own datasets, grouped rollout
assembly, rewards, GRPO advantages, staleness accounting, logging, and
checkpoint metadata. A Fireworks backend should own:

- the Fireworks service, training client, and sampler lifecycle;
- conversion from `TrainingInputBatch` to `tinker.Datum`;
- remote `forward_backward(..., "importance_sampling")` and `optim_step`;
- sampler snapshot creation and installation; and
- provider-specific checkpoint, retry, and cleanup behavior.

The first implementation is deliberately **not PPO**. It has no critic, value
model, GAE, PPO loss literal, or PPO update loop. It uses SkyRL's existing GRPO
advantage estimator and the `rollout_is` policy objective, translated to
Fireworks' built-in `importance_sampling` loss.

## Why the boundary is not `WorkerDispatch`

`RayPPOTrainer` currently calls one object for several unrelated concerns:

1. policy/ref/critic model execution;
2. Ray data staging and DP sharding;
3. optimizer execution;
4. local GPU offload and profiling;
5. checkpoint/export operations; and
6. local trainer-to-vLLM weight transfer.

Making `WorkerDispatch` itself switch on `fsdp | megatron | fireworks` would
leave most methods as provider-specific no-ops and make remote behavior depend
on local Ray types. Instead, introduce a narrow trainer-facing protocol and
keep the existing class as its local implementation:

```python
class TrainingBackend(Protocol):
    def get_batch_alignment(self) -> int: ...
    def stage_policy_batches(self, batch, boundaries): ...
    def forward_backward_policy(self, staged_batch) -> WorkerOutput: ...
    def optim_step_policy(self) -> dict[str, float]: ...
    async def publish_sampler_weights(self) -> SamplerVersion: ...
    def save_checkpoint(self, path, trainer_state) -> CheckpointRef: ...
    def load_checkpoint(self, ref) -> None: ...
    async def close(self) -> None: ...
```

`RayWorkerTrainingBackend` can initially be a thin adapter around the current
`WorkerDispatch`. `FireworksTrainingBackend` implements only remote policy
training. Ref/critic support should be added later as explicit capabilities,
not emulated through dummy methods.

The inference side remains a separate `InferenceEngineInterface`
implementation. Both objects share a `FireworksRuntime`, which owns the SDK
service and current sampler:

```text
SkyRL dataset/generator/reward/GRPO/staleness
                 |
          RayPPOTrainer loop
           /             \
TrainingBackend       InferenceEngineInterface
       |                       |
FireworksTrainingBackend  FireworksInferenceClient
           \             /
             FireworksRuntime
       service + trainer + sampler(s)
```

This separation also permits a later mixed mode: local SkyRL training with a
Fireworks hot-load inference deployment. The MVP uses a policy-only dispatch
adapter with the existing trainer-facing method names; extracting the formal
protocol and adapting the local path can happen independently afterward.

## Phase 1 algorithm contract: GRPO only

The first vertical slice must validate these settings and fail early on any
incompatible configuration:

```text
trainer.algorithm.advantage_estimator = grpo
trainer.algorithm.policy_loss_type = rollout_is
trainer.algorithm.use_kl_loss = false
trainer.algorithm.use_kl_in_reward = false
trainer.critic.model.path = null
trainer.update_epochs_per_batch = 1
trainer.placement.colocate_all = false
trainer.fireworks.infrastructure = serverless | dedicated
trainer.policy.model.lora.rank >= 0
trainer.resume_mode = null
```

Dedicated mode additionally requires stable `trainer_job_id`, `deployment_id`,
and `training_shape_id` values, one or more rollout replicas, and explicit
cleanup-on-exit. A zero LoRA rank selects full-parameter training. Serverless
requires neither resource ID and remains LoRA-only, so its rank must be
positive.

The reason for `update_epochs_per_batch=1` is that the rollout logprobs are the
behavior-policy anchor. Multiple optimizer passes over the same samples are a
separate off-policy feature and should not be enabled accidentally.

SkyRL continues to perform the GRPO math:

1. generate `n_samples_per_prompt` responses;
2. score them in the existing environment;
3. group by SkyRL UID;
4. calculate group-relative advantages with the existing
   `advantage_estimator="grpo"`; and
5. apply SkyRL's configured loss-reduction scaling before dispatch.

Fireworks receives the already-computed, already-scaled token advantages and
computes the differentiable importance ratio from current trainer logprobs and
rollout logprobs. No critic or reference request is made.

KL and SkyRL's geometric off-policy sequence mask are intentionally out of the
first slice. Both require either an additional remote forward or a Fireworks
custom loss, and should be added only after numeric parity for plain GRPO is
established.

## `TrainingInputBatch` to `tinker.Datum`

SkyRL stores a left-padded full sequence and right-aligned response tensors.
Fireworks expects shifted next-token arrays with the same length as the model
input. For each unpadded sample:

```text
tokens          = prompt + response
model_input     = tokens[:-1]
target_tokens   = [0] * (prompt_len - 1) + response
logprobs        = [0] * (prompt_len - 1) + rollout_response_logprobs
advantages      = [0] * (prompt_len - 1)
                  + normalized_response_advantages * response_loss_mask
```

All four arrays have length `len(tokens) - 1`. Multiplying advantages by
`loss_mask` preserves SkyRL's masked assistant spans, including multi-turn
user/tool tokens inside a flattened response. Non-trainable positions have
zero advantage and zero behavior logprob.

The converter must:

- remove batch padding using `attention_mask`, never by searching for the pad
  token ID;
- obtain `response_len` from `response_mask` and slice the right-aligned tails
  of `loss_mask`, `advantages`, and `rollout_logprobs`;
- reject missing or non-finite rollout logprobs on trainable tokens;
- reject a sample with fewer than two total tokens;
- enforce Fireworks `max_seq_len` before submitting it; and
- preserve sample order and mini-batch boundaries.

Unit tests should cover unequal prompt/response lengths, a pad token that also
appears in real text, masked multi-turn spans, a zero-variance group, and a
round-trip shape comparison against SkyRL's existing Tinker conversion path.

## Serverless lifecycle

Serverless is a good first integration because Fireworks supplies a shared
trainer and sampler pool. It is LoRA-only, currently private preview, and model
availability is account-specific. The documentation's current example uses
`accounts/fireworks/models/qwen3p6-27b`; it does not publish a stable global
"smallest supported model" list.

Therefore the SkyRL config should require both values rather than silently
guessing them:

```text
trainer.fireworks.base_model
trainer.policy.model.path
```

For a cheap GSM8K smoke test, prefer
`accounts/fireworks/models/qwen3-1p7b` **only if it is enabled for serverless
training on the account**. Otherwise use the model Fireworks has enabled and
reported for the account. Do not fall back to a larger model without showing
the resolved model and run shape to the operator.

The runtime sequence is:

1. create `FiretitanServiceClient` at
   `https://api.fireworks.ai/training/v1/serverless`;
2. call `create_lora_training_client(base_model, rank)`;
3. before the first rollout, save an initial sampler snapshot;
4. create a sampling client bound to that exact snapshot;
5. after every optimizer step, save a uniquely named snapshot and atomically
   install a new sampling client; and
6. close retired samplers after their in-flight reference count reaches zero,
   then close the service during trainer teardown.

Serverless `sleep()` and `wake_up()` are no-ops. They are memory-management
operations for colocated local GPUs and must not be overloaded to mean hosted
resource creation or deletion.

## Live dedicated validation

The account's read-only shape catalog did not expose a compatible dedicated
LoRA training shape for Qwen3-1.7B. The smallest matching option found was:

- base model `accounts/fireworks/models/qwen3-4b` and tokenizer
  `Qwen/Qwen3-4B`;
- training shape `qwen3-4b-minimum-lora`, pinned provider version `z433rsow`;
- linked rollout shape `rft-qwen3-4b`, pinned version `j1px5wbo`;
- 32,768-token context and LoRA rank 8; and
- one `NVIDIA_B200_180GB` accelerator for the trainer plus one for rollout.

For these particular shapes, the provider reports one node and one accelerator
per resource. Thus `1x` means one B200 chip here, not a multi-GPU node bundle.
The trainer and rollout are separate paid resources, so both being active is
two B200s total. At the public July 2026 price of USD 10/B200-hour, the combined
rate is approximately USD 20/hour. The smoke script has a 20-minute hard cap,
prints the resolved topology before provisioning, and requires
`FIREWORKS_RUN_CONFIRMED=1`.

The failed and successful attempts had trainer create-to-delete windows of
about 188 and 310 seconds. Charging both B200 resources for both entire windows
would be a conservative total estimate of about USD 2.77. Fireworks returned
an empty `acceleratorSeconds` map on the deleted trainer records, so this is not
an invoice measurement; the account billing console remains authoritative.

The first provisioning attempt proved account access but exposed a naming
constraint before sampling: managed checkpoint creation appends its own
eight-character suffix, and the resulting checkpoint exceeded the 63-character
DNS-label limit. The runtime now emits lowercase DNS labels of at most 54
characters, leaving room for that provider suffix. That failed attempt deleted
the deployment and trainer immediately.

The retry `20260721000951-02` then completed the full bounded path:

1. provisioned the Qwen3-4B LoRA trainer and linked deployment;
2. saved and hotloaded initial snapshot version 0 in 10 seconds;
3. generated eight trajectories (two prompt groups times four completions) in
   1.34 seconds, with 2,048 response tokens and 2,920 total submitted tokens;
4. ran one GRPO policy forward/backward and optimizer step in 3.25 seconds;
5. saved and hotloaded post-update snapshot version 1 in 10 seconds; and
6. deleted both exact resource IDs on close, then independently observed the
   trainer as `JOB_STATE_DELETED` and the deployment as absent.

All eight sampled rewards were zero, so the final policy loss was zero. This is
a transport, lifecycle, and optimizer-path acceptance test, not evidence of
learning quality. The W&B record is
[`8gtfo4gy`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/8gtfo4gy).

The script is `examples/train/gsm8k/run_gsm8k_fireworks_dedicated.sh`. It uses
unique `skyrl-smoke-*` IDs, SDK cleanup in `finally`, an outer wall-clock
supervisor, and a secondary exact-ID-only cleanup/audit utility. It never
enumerates or modifies pre-existing account resources.

### Ten-step reward validation

The initial one-step run used a 256-token generation cap. Every completion
ended at exactly that cap while still inside Qwen's `<think>` section, before
emitting the GSM8K `#### <number>` answer delimiter. The all-zero reward was
therefore a generation-length failure rather than a Fireworks training-path
failure. The dedicated smoke script now defaults to 1,024 generated tokens and
has explicit `MAX_TRAINING_STEPS` and `TRAINING_EPOCHS` controls.

Run `20260721005311-04` completed ten policy-only GRPO optimizer steps with two
prompt groups and four completions per group at each step:

| Step | Raw reward | Pass@4 | Mean response tokens | Policy loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.125 | 0.5 | 950.250 | 0.029414 |
| 2 | 0.500 | 0.5 | 709.000 | 0.000000 |
| 3 | 0.250 | 0.5 | 702.750 | 0.050977 |
| 4 | 0.750 | 1.0 | 429.375 | 0.047664 |
| 5 | 0.250 | 0.5 | 898.000 | -0.021341 |
| 6 | 0.125 | 0.5 | 971.750 | 0.014749 |
| 7 | 0.625 | 1.0 | 683.875 | -0.014356 |
| 8 | 0.000 | 0.0 | 1,024.000 | 0.000000 |
| 9 | 0.250 | 0.5 | 874.875 | -0.008668 |
| 10 | 0.625 | 1.0 | 926.375 | 0.051288 |

The mean raw reward was 0.35 (28 correct completions out of 80), and 12 of 20
prompt groups had at least one correct completion. This establishes that
rewards, GRPO advantages, remote optimizer updates, and repeated snapshot
hotloads are wired correctly. It does not establish a learning trend: the
first-five mean was 0.375 and the last-five mean was 0.325, and a batch of only
two changing prompts gives very high sampling variance.

Generation length still dominates the noise. Successful completions averaged
619 response tokens while zero-reward completions averaged 923; the correlation
between step mean reward and step mean response length was -0.76. At step 8,
all eight completions reached the 1,024-token cap and all scored zero. Two
logged examples also solved the arithmetic but emitted `#### $5` or
`#### $10`; the existing GSM8K parser deliberately accepts only a number
immediately after `####`, so those were scored zero. Thus 1,024 fixes the
original universal truncation but is not yet a clean quality-measurement
configuration for thinking-enabled Qwen3.

The zero losses at steps 2 and 8 are expected GRPO behavior, not dropped
updates. At step 2 one prompt group was uniformly correct and the other
uniformly incorrect; at step 8 both groups were uniformly incorrect. With
`zero_variance_filter=true`, neither batch has within-group reward variance and
the resulting advantages are zero.

For a learning-curve experiment, disable Qwen's thinking mode for this simple
format-sensitive task (or raise the cap again), evaluate a fixed held-out set,
and use more prompt groups per step. The live run is recorded in W&B as
[`au5dmfx9`](https://wandb.ai/sky-posttraining-uc-berkeley/gsm8k-fireworks/runs/au5dmfx9).
It reached `max_training_steps=10`, reported `Training done!`, deleted the exact
deployment, and was independently audited with the trainer in
`JOB_STATE_DELETED` and the deployment absent.

## Fully async semantics

Fully async is possible on both Fireworks modes, but the weight-boundary
semantics differ.

| Backend | In-flight call at weight publication | New call after publication | Meaning of `pause_generation` |
| --- | --- | --- | --- |
| Local vLLM | Frozen, then resumed after local weight update | Uses new weights | Real scheduler pause |
| Fireworks serverless | Provider-managed; verify the hosted deployment transition behavior for the account | Acquires the newly installed sampler client | Local admission gate plus provider-managed transition |
| Fireworks dedicated, ASYNC hot-load | Pauses at the swap and resumes on the same HTTP stream with the new weights and existing active KV | Queues during the swap, then uses new weights | Provider-managed by hot-load |
| Fireworks dedicated, SYNC hot-load | Finishes on old weights before the swap | May receive HTTP 425 and retry | Drain semantics |

Fireworks documents `ASYNC` as the default and recommended transition for RL
rollout deployments. It pauses an active stream at the weight swap and resumes
the same HTTP connection with its existing KV state; new requests queue while
the swap is in progress. Streaming chunks expose `model@snapshot_identity`,
which can change within one request. `SYNC` instead drains existing requests on
the old weights and can return HTTP 425 to new requests. The transition is
selectable when creating a deployment, but the installed SDK-managed
`from_firetitan_config` helper does not expose that field. The current SkyRL
path therefore relies on Fireworks' documented `ASYNC` default and must not
claim to have explicitly changed the provider setting. The earlier live GSM8K
test validated sequential hotloads; the fully-async run is the first live test
with generation and hot-loads overlapping.

For serverless, use a read-copy-update sampler pointer:

```text
generate(): acquire current sampler/version -> sample -> release
publish():  save snapshot -> build new sampler -> atomic pointer swap
retire():   close old sampler when its active count becomes zero
```

This overlaps rollout and training safely, but it does not splice new weights
into an active generation. SkyRL's existing fully-async staleness gate remains
the admission bound: a rollout group records the SkyRL training step at
submission, while the generator's cache salt reads the sampler version. New
groups stop being admitted when the configured head budget is full.
As in the Fireworks cookbook, this is a submission/accounting guarantee, not a
proof that every slow completion finishes within the bound.

For dedicated ASYNC hot-load, Fireworks already performs the behavior SkyRL
wants from `pause_generation`: active streams pause for the actual weight swap
and resume without client-side abort/retry. SkyRL should not issue a separate
pause request. Aborting an agent turn is the wrong default because retrying can
duplicate tool effects and loses the active HTTP/KV state. SkyRL's local
dedicated runtime retains one stable sampling client across every hot-load; it
does not need one client or lease generation per checkpoint. The lease merely
prevents final teardown from closing that shared client with a live stream. It
neither pins a stream to one weight version nor performs the provider's
pause/resume operation. Serverless still uses an RCU lease because its sampling
route is snapshot-qualified and the SDK returns a new client per snapshot.

Dedicated streaming responses can report `model@snapshot_identity` per chunk.
The adapter should record the observed snapshot identity for every generated
token when available. A turn that crosses a hot-load boundary is then visible
rather than being mislabeled as a single policy version.

## Inference adapter

The current `FireworksInferenceClient` implements SkyRL's token-in/token-out
contract over the sampling client returned by either hosted runtime. In
dedicated mode, the SDK prepares the linked deployment by hotloading the saved
snapshot before returning that client. The adapter:

- uses the SDK's native async streaming coroutine rather than occupying one
  Python executor thread for every active rollout request;
- maps SkyRL sampling parameters to `tinker.SamplingParams`;
- returns exact response token IDs and one rollout logprob per response token;
- maps finish reasons into SkyRL's strings;
- exposes an integer SkyRL `weight_version`; and
- treats `finish_session` as a no-op for the token-in/token-out path.

The experimental process-local OpenAI adapter was removed after the
CodeContests validation. The current client intentionally exposes only
token-in/token-out generation. Streaming snapshot identity is still not
exposed to SkyRL.

The adapter must not decode text and re-tokenize it to reconstruct training
tokens. Fireworks and SkyRL must use the tokenizer corresponding exactly to the
configured base model.

## Harbor/TITO status

Harbor support is deferred. If revisited, it should consume Fireworks' native
OpenAI-compatible `/v1/chat/completions` endpoint directly rather than recreate
a local chat proxy. An endpoint descriptor could carry non-secret routing
metadata:

```python
@dataclass(frozen=True)
class InferenceEndpoint:
    base_url: str
    model: str
    headers: dict[str, str]
    auth_env_var: str | None = None
```

For dedicated Fireworks this would describe the inference gateway,
deployment/model headers, and authentication environment-variable name without
containing the credential. Any future multi-turn implementation must preserve
Fireworks-returned token IDs per turn and explicitly handle separated reasoning
history; assuming that adjacent chat turns are prefix-mergeable is unsafe.

## Configuration shape

Add a provider-specific config rather than reusing local placement fields:

```python
@dataclass
class FireworksConfig(BaseConfig):
    infrastructure: Literal["serverless", "dedicated"] = "serverless"
    base_url: str = "https://api.fireworks.ai"
    base_model: str | None = None
    max_seq_len: int | None = None
    request_timeout_s: int = 3600
    sampling_timeout_s: int = 600
    trainer_timeout_s: int = 900
    deployment_timeout_s: int = 900
    hotload_timeout_s: int = 600
    adam_eps: float = 1e-8
    snapshot_prefix: str = "skyrl"
    training_shape_id: str | None = None
    trainer_job_id: str | None = None
    deployment_id: str | None = None
    replica_count: int = 1
    cleanup_on_exit: bool = True
    cleanup_deployment_on_close: Literal["delete", "scale_to_zero"] = "delete"
```

The tokenizer identifier remains `trainer.policy.model.path`, and the LoRA
rank remains `trainer.policy.model.lora.rank`. Reusing those existing SkyRL
fields avoids two sources of truth.

The API key is read only from `FIREWORKS_API_KEY`; it must not be serialized
into Hydra/OmegaConf config, Ray task payloads, W&B config, checkpoint
metadata, or Harbor trial files.

Fireworks does not consume local placement topology. `colocate_all=true` and
`colocate_policy_ref=true` are hard errors in the initial backend.

The transition mode and three-way prompt-cache reset policy deliberately are
not configuration fields yet. The former belongs to the deployment template;
the installed SDK exposes only a boolean cache reset in its managed sampler
path, while the public hot-load API supports `all | new_session | none`. The
Harbor streaming adapter should expose the public API semantics once it owns
hot-load calls directly.

## Checkpoint model

SkyRL needs to store two different kinds of state:

- provider training state: Fireworks DCP checkpoint/reference; and
- local orchestration state: dataloader cursor, `global_step`, fully-async UID
  state, sampler version, and W&B metadata.

A local checkpoint manifest should contain the provider checkpoint reference,
base model, tokenizer, infrastructure mode, and sampler snapshot identity. It
must never contain credentials.

Serverless cross-run resume is not currently supported, so Phase 1 requires
`resume_mode=null`. Same-run snapshots are still used for sampling. Dedicated
support should use Fireworks DCP for weights+optimizer and SkyRL's local
manifest for orchestration state; restore is complete only after both sides
have loaded.

## Implementation plan

### Milestone 0: contracts and unit tests (MVP complete)

- Add a policy-only trainer-facing adapter using the current dispatch method
  names; extract the formal `TrainingBackend` protocol later without changing
  the local path.
- Add the pure `TrainingInputBatch -> tinker.Datum` GRPO converter.
- Add config capability validation and secret-redaction tests.
- Add a fake Fireworks service/training/sampler test double. No network calls.

### Milestone 1: synchronous serverless GSM8K (implementation complete; paid smoke entitlement-blocked)

- Implement `FireworksRuntime`, `FireworksTrainingBackend`, and
  `FireworksInferenceClient`.
- Wire policy-only GRPO with `rollout_is`, KL disabled, LoRA, and one update
  per batch.
- Add `examples/train/gsm8k/run_gsm8k_fireworks_serverless.sh` with a one-step
  smoke-test mode.
- Compare datum contents and GRPO advantages against native SkyRL on a fixed
  synthetic rollout before making a paid call.

### Milestone 2: fully async serverless GSM8K (scheduler/runtime wiring complete; paid smoke pending)

- Add the reference-counted sampler pointer and version metadata.
- Reuse SkyRL's `FullyAsyncRayPPOTrainer` admission/staleness manager.
- Select that scheduler in the direct Fireworks entrypoint when
  `trainer.fully_async.enabled=true`; despite its legacy class name, the
  configured algorithm remains policy-only GRPO.
- Test delayed old-sampler completions, pointer swaps, cleanup, and restart
  failure behavior with fakes.
- Run `max_staleness_steps=0`, then 1, then a small bounded value.

### Milestone 3: dedicated Fireworks (synchronous MVP and cleanup live-validated)

- SDK-managed trainer/deployment provisioning, LoRA snapshot hot-load, stable
  resource IDs, timeouts, delete/scale-to-zero selection, and exact-ID fallback
  cleanup are implemented.
- The one-step Qwen3-4B GSM8K acceptance run and both snapshot publications
  passed; see the live-validation record above.
- DCP resume, an explicit provider-resource manifest for intentional reattach,
  streamed snapshot identities, and a long request crossing a hot-load remain.
- ASYNC versus SYNC is a provider deployment-template setting; resolve it with
  Fireworks rather than defaulting it in SkyRL configuration.

### Milestone 4: Harbor/TITO

- Deferred after the CodeContests experiment showed that the current Qwen
  reasoning/chat history was not prefix-mergeable.
- The experimental local proxy and Harbor-specific Fireworks code were removed.
- If revisited, start from Fireworks' native HTTP endpoint and validate exact
  per-turn token IDs before adding a new SkyRL integration.

## Acceptance gates

Before the first paid serverless run:

- all unit tests use fakes and pass locally;
- the resolved account-enabled base model, tokenizer, LoRA rank, max sequence
  length, number of prompt groups, completions per prompt, token caps, and
  estimated paid operations are printed for explicit confirmation;
- the run is capped at one optimizer step with minimal sampling tokens;
- W&B uses a new run name and no secret fields; and
- every opened sampler/service has a tested `finally` cleanup path.

Before Harbor:

- a dedicated one-step GSM8K run has completed and resumed from DCP;
- a synthetic long streaming request has crossed an ASYNC hot-load and exposed
  both snapshot identities without restarting;
- a multi-turn session has retained affinity across tool turns;
- abort is disabled by default; and
- scale-to-zero/delete behavior has been explicitly selected and verified.

## References

- [Fireworks Training API introduction](https://docs.fireworks.ai/fine-tuning/training-api/introduction)
- [Serverless training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- [Dedicated training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)
- [Training and sampling](https://docs.fireworks.ai/fine-tuning/training-api/training-and-sampling)
- [Loss functions](https://docs.fireworks.ai/fine-tuning/training-api/loss-functions)
- [Checkpoints and resume](https://docs.fireworks.ai/fine-tuning/training-api/saving-and-loading)
- [Fireworks async RL cookbook](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl)
- [Inference and policy versions for RL rollouts](https://docs.fireworks.ai/guides/rollout-inference)
- [Checkpoint-swap behavior](https://docs.fireworks.ai/fine-tuning/rl-rollout-debugging)
- [Dedicated GPU pricing](https://fireworks.ai/pricing)
- `other_frameworks/fw_cookbook/training/examples/serverless_rl/countdown_rl.py`
- `other_frameworks/fw_cookbook/training/recipes/async_rl_loop.py`
- `other_frameworks/rllm/rllm/trainer/fireworks/fireworks_backend.py`
- `other_frameworks/rllm/rllm/trainer/fireworks/fireworks_trainer.py`
- `other_frameworks/rllm/rllm/trainer/tinker/transform.py`
