# SkyRL Integration v2

## Update — 2026-09-22 00:26:04 UTC

This update supersedes conflicting statements later in this document about requiring non-step-wise mode or changing SkyRL's generator contract.

The integration will reuse SkyRL's existing step-wise machinery as a rollout-grouping and advantage-broadcast mechanism, without treating capture rows as actual sequential turns:

- Set `generator.step_wise_trajectories=true` so one input rollout may emit multiple output rows without changing SkyRL's current validation or trainer implementation.
- Set `generator.merge_stepwise_output=false`. Prefix merging exists to combine sequential per-turn Harbor rows when `prompt[i] + response[i]` is a prefix of `prompt[i+1]`. Capture rows are already complete multi-turn root-to-leaf samples, so merging them again is unnecessary and could incorrectly combine compatible-looking paths.
- Emit every trainable capture path as one complete multi-turn sample with its existing loss mask.
- Emit all paths for a physical rollout contiguously with the same `TrajectoryID`.
- Emit `is_last_step=false` for every path except the final emitted path, which gets `is_last_step=true`. Here the marker means "representative reward row and end of this rollout group," not "chronologically final LLM turn."
- Give every path the same physical rollout reward. SkyRL selects the row marked `is_last_step=true`, computes the rollout-level advantage once, and broadcasts it to every path in the group. Because every path has the same reward, path ordering does not affect the advantage.
- No SkyRL trainer or `GeneratorOutput` contract change is required for this integration. The reuse is intentional even though `step_wise` is an imperfect name for grouped complete-path samples.

Two caveats remain:

- Step-wise evaluation and some metrics retain only the `is_last_step=true` row. Token-exact parity across every capture path must therefore inspect the raw generator output before final-step filtering.
- Zero-variance filtering currently observes flattened rows rather than unique physical rollouts. This does not affect eval-only validation. Before using this integration for training with a single repetition that can produce multiple paths, either disable zero-variance filtering or make it trajectory-aware.

## Status

Proposed implementation plan.

## Summary

Upgrade the Harbor/SkyRL integration to the current `skyrl_capture` service and SDK while keeping Harbor unmodified. Harbor continues to make ordinary OpenAI-compatible requests in text space. A per-trajectory capture route renders those requests, sends exact token IDs to SkyRL's token-in/token-out endpoint, records the resulting graph, and exports complete multi-turn token samples for SkyRL.

The exported samples are not step-wise samples. Each row is a complete root-to-leaf, multi-turn sequence whose loss mask selects every model-generated token that should be trained. SkyRL must accept multiple such rows from one physical rollout without enabling `step_wise_trajectories` or requiring `is_last_step`.

The shipping criterion is token-exact eval parity with the current harness-side TITO collector, with no Harbor source changes.

## Goals

- Keep Harbor source code unchanged.
- Run Harbor without `collect_rollout_details`; capture owns token accounting.
- Use the current `skyrl_capture` configuration, service, SDK, persistence, and TITO protocol APIs.
- Send exact token IDs through SkyRL's `/skyrl/v1/generate` endpoint.
- Forward the policy-version `cache_salt` to SkyRL.
- Preserve routed-expert indices when requested and returned.
- Export complete multi-turn samples with token-level loss masks.
- Support one or more exported paths from a physical rollout without step-wise training.
- Compute one rollout-level advantage and share it across all exported paths from that rollout.
- Validate the integration in eval/sample-collection mode against the existing harness-side TITO implementation.

## Non-goals

- No changes to Harbor source code.
- No per-turn reconstruction in SkyRL.
- No `is_last_step` output or step-wise merge path.
- No optimizer or training smoke test in this iteration.
- No configurable treatment of model-added reasoning prefixes in this iteration. The integration will match the current harness-side TITO behavior.
- No PostgreSQL-backed capture service or dynamic target registry.
- No generic multi-upstream routing inside one capture process.

## Architecture

```text
Harbor (unchanged)
  │ OpenAI/LiteLLM chat request
  │ api_base = per-trajectory capture URL
  │ extra_body.cache_salt = policy version
  ▼
skyrl_capture token proxy
  │ render messages and tools
  │ maintain the trajectory graph
  │ exact prompt token IDs
  ▼
SkyRLTitoProtocol
  │ POST /skyrl/v1/generate
  │ X-Session-ID: <capture trajectory ID>
  │ cache_salt at the SkyRL endpoint's expected location
  ▼
SkyRL inference router / vLLM workers
  │ completion token IDs, selected-token logprobs,
  │ optional routed-expert indices
  ▼
capture journal and token-samples projection
  │ one complete multi-turn sample per root-to-leaf path
  ▼
non-step-wise SkyRL GeneratorOutput
```

Capture remains framework-neutral. SkyRL-specific request wiring and `GeneratorOutput` composition stay in the SkyRL integration.

## Sample semantics

A `token-samples` export row contains one complete root-to-leaf path through the captured message graph:

- `input_ids` contains the full rendered multi-turn sequence.
- `loss_mask` marks all model-generated tokens that should train.
- `rollout_logprobs` aligns one-to-one with `input_ids`.
- `node_ids` and `node_spans` preserve provenance.
- shared sampled nodes are trainable in only one exported path, so branching does not train the same generation twice.
- optional routed-expert data must align with the same token positions.

The SkyRL adapter splits a row at its first trainable token:

```text
prompt_token_ids = input_ids[:first_trainable]
response_ids     = input_ids[first_trainable:]
loss_masks       = loss_mask[first_trainable:]
rollout_logprobs = rollout_logprobs[first_trainable:]
```

The response suffix may contain alternating generated and replayed tokens. The loss mask, rather than a turn boundary, determines what trains.

Rows with no trainable tokens do not become independent training samples. If an entire physical rollout has no trainable row or is invalid, emit one fully masked placeholder so batch and rollout lineage remain explicit.

## Rollout grouping and advantages

One Harbor execution is one physical rollout and has one reward. Compaction or history rewriting may cause capture to export multiple complete path rows for that rollout. Those rows are sample shards of the same rewarded execution, not independent reward observations.

SkyRL must therefore:

1. Preserve the source `TrajectoryID` on every exported row.
2. Compute the rollout's reward and advantage once.
3. Replicate that advantage across every exported row belonging to the rollout.
4. Let each row's loss mask decide which tokens receive the shared advantage.
5. Count the physical rollout once during reward-group normalization, regardless of its number of exported paths.

This prevents graph shape, compaction frequency, or branch count from changing GRPO reward weighting.

This grouping is not step-wise training. Each row is already a complete multi-turn sample; grouping exists only to preserve rollout-level reward semantics across sample shards.

## Compatibility behavior for `step_wise`

The integration may retain its existing `step_wise` argument temporarily to minimize call-site churn, but it is ignored. The output contract is always non-step-wise:

- `is_last_step=None`.
- `generator.step_wise_trajectories` must be `False`.
- branching never causes a trajectory to be masked merely because multiple rows were exported.
- no step-wise merge or per-turn advantage path is used.

The ignored parameter should be documented as deprecated and removed after downstream configurations have migrated.

## SkyRL-specific TITO protocol

Rewrite `examples/train_integrations/harbor/icap/upstream.py` using the current protocol extension API:

```python
from skyrl_capture.tito.upstream import VLLMTitoProtocol, register


class SkyRLTitoProtocol(VLLMTitoProtocol):
    name = "skyrl"
    generate_path = "/skyrl/v1/generate"
```

The protocol is responsible for:

- using SkyRL's endpoint path;
- preserving `X-Session-ID` affinity;
- putting `cache_salt` at the location expected by SkyRL;
- filtering the remaining sampling parameters through vLLM's accepted set;
- decoding and validating SkyRL's routed-expert response when enabled;
- returning normalized completion IDs, selected-token logprobs, stop reason, and optional routed experts.

If native vLLM gains the exact routed-expert contract SkyRL needs, this plugin can be removed and the capture upstream type can become `vllm`.

## `cache_salt` forwarding

`cache_salt` is a feature of the capture token proxy's request path to the SkyRL/vLLM upstream. It is not generic trajectory metadata and is not forwarded to arbitrary capture upstreams.

The request path is:

```text
Harbor configuration
  llm_kwargs.extra_body.cache_salt
    → OpenAI-compatible request received by capture
    → token-proxy sampling/request metadata
    → SkyRLTitoProtocol consumes cache_salt
    → top-level field accepted by /skyrl/v1/generate
```

The capture chat parser must retain `cache_salt` for token-mode requests. `SkyRLTitoProtocol` removes it before vLLM sampling-parameter filtering and places it in the SkyRL request body. The ordinary `VLLMTitoProtocol` must not send it inside `SamplingParams`.

The value is derived from `inference_engine_client.weight_version` when `generator.use_cache_salt` is enabled.

## Service lifecycle

Replace the old target-registry and embedded-PostgreSQL bootstrap with the current capture configuration:

```python
config = Config(
    record_dir=record_dir,
    upstream=TitoUpstream(
        type="skyrl",
        url=engine_url,
        tokenizer=tokenizer_name,
        model=model_name,
        max_model_len=max_model_len,
    ),
    upstream_modules=(
        "examples.train_integrations.harbor.icap.upstream",
    ),
)
service = CaptureService(config=config, port=port)
```

Remove:

- `ensure_target` and dynamic target names;
- database/control-key setup;
- embedded PostgreSQL assumptions;
- the obsolete `num_workers` service argument;
- per-trajectory target and upstream configuration.

One capture process owns one configured upstream. Multiple policy endpoints require separate capture processes.

Both in-process and externally managed service modes remain supported. The external mode only needs a base URL; it does not create or update targets.

## Generator lifecycle

Update `harbor_generator.py` to the current SDK:

- import `skyrl_capture.sdk`;
- use a shared `CaptureClient` owned by the generator;
- create one fresh capture trajectory per attempt;
- pass first-class `project`, `run_id`, `task_id`, and `step` metadata;
- point the copied Harbor trial configuration at `trajectory.base_url`;
- use a placeholder API key accepted by the capture route;
- add `llm_kwargs.extra_body.cache_salt` when enabled;
- call `finish(format="token-samples", annotations=...)` and consume `result["records"]`;
- reject incomplete or delivery-uncertain records;
- call `inference_engine_client.finish_session(trajectory.id)` in `finally`;
- close trajectory/client resources on success, error, timeout, and cancellation;
- retain retry isolation by using a fresh trajectory ID for every attempt;
- use the same rollout rate limiter as the sibling Harbor integration.

A failed finish is a failed attempt. The generator must never train from an uncommitted or partially exported capture.

Harbor itself remains unchanged. Only its per-trial configuration copy receives the capture `api_base`, API-key placeholder, and optional `extra_body` field.

## Composer changes

Update `compose.py` so that every trainable token-sample row becomes one full non-step-wise output row.

- Keep the current first-trainable-token split.
- Preserve the entire response suffix and its loss mask.
- Preserve aligned logprobs and routed-expert indices.
- Emit all trainable exported paths.
- Copy the source physical `TrajectoryID` onto every path row.
- Do not emit `is_last_step`.
- Do not assign independent reward observations to path rows; attach rollout lineage so SkyRL can compute and share one advantage.
- Keep `step_wise` temporarily if compatibility requires it, but ignore its value.
- Mask an entire rollout on Harbor timeout/error or capture-integrity failure.

The composer must validate, before returning:

- `input_ids`, `loss_mask`, and `rollout_logprobs` have equal lengths;
- every routed-expert slice has the expected token alignment;
- every emitted response has at least one trainable token unless it is the explicit masked placeholder;
- all rows from one physical rollout retain the same rollout lineage and reward.

## Required SkyRL generator-contract change

SkyRL currently equates non-step-wise mode with exactly one output row per input prompt. That prevents a capture trajectory from returning multiple complete multi-turn paths.

Generalize the contract so non-step-wise output may expand when rollout lineage is provided:

- permit `len(response_ids) >= len(input_prompts)` when output `trajectory_ids` are present and valid;
- derive output UIDs from `generator_output["trajectory_ids"]` when cardinality expands;
- distinguish physical rollout count from sample-row count;
- normalize rewards and compute advantages over physical rollouts;
- broadcast each physical rollout's advantage to its sample rows;
- keep `is_last_step` absent and do not enter step-wise trainer code;
- keep ordinary one-output-per-prompt generators unchanged.

If a new explicit field makes the contract clearer, introduce `rollout_ids` or `sample_group_ids` rather than overloading turn-oriented fields. The field must identify all rows produced from the same physical execution and must be independent of `node_ids` and capture path IDs.

## Failure semantics

- Harbor timeout: mask the physical rollout.
- Harbor error after retries: mask the physical rollout.
- Capture create/commit/finish failure: retry with a fresh capture trajectory, then mask if retries are exhausted.
- Incomplete integrity or delivery uncertainty: do not train any row from that physical rollout.
- Context-length stop: retain the sample according to the same masking policy as the existing harness-side TITO collector.
- One failed repetition in a GRPO prompt group: use the existing SkyRL group-failure policy so capture-backed and harness-side collection behave identically.
- Cancellation: release the SkyRL router session and SDK resources, then propagate cancellation.

## Validation strategy

Validation runs Harbor and sample collection in eval mode. No optimizer step is required.

### Unit and contract tests

Test the SkyRL protocol with fixed request/response fixtures:

- endpoint path is `/skyrl/v1/generate`;
- `X-Session-ID` equals the capture trajectory ID;
- `cache_salt` reaches the expected top-level request field;
- it is absent from vLLM `SamplingParams`;
- token IDs and selected-token logprobs are decoded exactly;
- routed-expert payloads decode and align correctly;
- malformed IDs, logprobs, and routed-expert shapes fail closed.

Test composition with current `token-samples` fixtures:

- linear multi-turn trajectory;
- tool-call trajectory;
- compaction/history rewrite producing multiple paths;
- shared model node appearing on multiple paths with train-once masking;
- fully masked path;
- context-length stop;
- incomplete integrity;
- routed experts enabled and disabled;
- multiple path rows sharing one physical rollout and one advantage lineage.

Assert that composition is non-step-wise even when the retained compatibility argument is passed as `True`.

### Token-exact eval parity

Run the same Harbor eval set through:

```text
A. current harness-side TITO collection
B. unmodified Harbor → skyrl_capture → SkyRL TITO endpoint
```

Use deterministic decoding or replay the same completed transcript. For every physical rollout and exported path, compare:

- full rendered token sequence;
- prompt/response split;
- token-level loss mask;
- selected-token logprobs where both paths expose them;
- stop reason;
- reward;
- routed-expert indices when enabled;
- sample count and lineage after compaction or history rewriting.

Any intentional incompatibility must be documented and approved. The default requirement is exact equality for token IDs and loss-mask positions.

### Eval-mode operational validation

During the parity run, also verify:

- capture and router sessions are released;
- HTTP clients and file descriptors do not grow with completed trials;
- concurrent rollouts respect the configured rate limit;
- committed records survive capture restart;
- a recovered trajectory can finish and export identically;
- interrupted or incomplete trajectories never enter generator output;
- finish-time records match offline re-export;
- shutdown drains committed records.

## Acceptance criteria

The integration is ready when all of the following hold:

- Harbor has no source changes.
- Harbor uses ordinary OpenAI/LiteLLM calls and does not collect rollout token details.
- The integration uses current `skyrl_capture` APIs and filesystem journal persistence.
- `generator.step_wise_trajectories=False`.
- No `is_last_step` or per-turn reconstruction is used.
- A token-sample row is consumed as one complete multi-turn sample with its loss mask intact.
- Multiple paths from one physical rollout share one rollout-level advantage and count once in reward normalization.
- Eval output is token-exact with current harness-side TITO for token IDs, prompt/response boundaries, and loss-mask positions.
- `cache_salt` reaches `/skyrl/v1/generate` and is not passed as a vLLM sampling parameter.
- Routed-expert indices are either correctly aligned or explicitly absent.
- Incomplete capture data never trains.
- Capture clients, files, and router sessions do not leak during the eval harness run.

## Implementation order

1. Rewrite `upstream.py` against `skyrl_capture.tito.upstream` and add protocol contract tests.
2. Add token-proxy `cache_salt` retention and SkyRL-specific forwarding.
3. Rewrite capture service bootstrap and external-service handling.
4. Update generator SDK usage, lifecycle cleanup, rate limiting, and finish-time export.
5. Simplify the composer to emit complete non-step-wise samples and ignore the compatibility `step_wise` argument.
6. Generalize SkyRL's non-step-wise generator contract for multiple sample rows per physical rollout and shared advantages.
7. Replace stale tests with current-schema unit and socket-level integration tests.
8. Run token-exact eval parity and operational validation.
9. Update the integration README and development launcher to describe the final architecture and commands.

## Deferred work

- Per-trajectory control over whether model-added reasoning prefixes are prompt tokens or predicted tokens.
- Migration from `/skyrl/v1/generate` to native vLLM when routed-expert support is equivalent.
- Durable database-backed record storage for deployments that need shared, multi-host persistence beyond the current filesystem journal.
