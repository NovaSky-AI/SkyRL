# R2EGym 32B ThunderAgent Recipe — Test Plan

> Scope: validate the Harbor + ThunderAgent recipe under
> `examples/train/thunder_agent/` after the post-merge fixes
> (port env name, llm_call_kwargs.timeout, EXTERNAL_PROXY_URL,
> configurable retry budget, dataset empty-guard).
>
> Each tier is a hard gate for the next one. Do not skip up the ladder.

---

## Tier 1 — Static & import checks

Goal: catch typos, OmegaConf field errors, and Python import breakage
before any GPU is allocated. Runs in <60s on a CPU-only login node.

```bash
cd $REPO_ROOT  # skyrl-ta-pr-core

# 1.1 shell syntax
bash -n examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh

# 1.2 entrypoint + transitive imports
python -c "from examples.train.thunder_agent.main_harbor_thunder_agent import HarborThunderAgentFullyAsyncExp; print('OK')"
python -c "from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator; print('OK')"
python -c "from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset; print('OK')"
python -c "from examples.train.thunder_agent.skyrl_integration.remote_inference_client import ThunderAgentRemoteInferenceClient; print('OK')"

# 1.3 pause() signature is a strict superset of upstream
python - <<'PY'
import inspect
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient, PauseMode
from examples.train.thunder_agent.skyrl_integration.remote_inference_client import ThunderAgentRemoteInferenceClient

base = set(inspect.signature(RemoteInferenceClient.pause).parameters)
sub  = set(inspect.signature(ThunderAgentRemoteInferenceClient.pause).parameters)
assert base.issubset(sub), f"missing: {base - sub}"
default = inspect.signature(ThunderAgentRemoteInferenceClient.pause).parameters["mode"].default
assert default is PauseMode.KEEP, default
print("OK")
PY

# 1.4 CLI override roundtrip — including keys NOT present in default.yaml
python - <<'PY'
from examples.train.thunder_agent.training_config import ThunderAgentHarborConfig
cfg = ThunderAgentHarborConfig.from_cli_overrides([
    "max_train_tasks=8", "max_eval_tasks=2",
    "generator.rate_limit.enabled=true",
    "generator.rate_limit.trajectories_per_second=1",
    "generator.rate_limit.max_concurrency=2",
    "harbor_trial_config.agent.name=mini-swe-agent",
    "harbor_trial_config.trials_dir=/tmp/x",
    "harbor_trial_config.agent.kwargs.llm_call_kwargs.timeout=1200",
])
assert cfg.max_train_tasks == 8
assert cfg.generator.rate_limit.enabled is True
assert cfg.harbor_trial_config["trials_dir"] == "/tmp/x"
assert cfg.harbor_trial_config["agent"]["kwargs"]["llm_call_kwargs"]["timeout"] == 1200
print("OK")
PY
```

Pass criteria: every command exits 0.

---

## Tier 2 — Generator unit tests (CPU only)

Goal: verify the four hot paths in the generator without spinning up
Harbor, vLLM, or Ray. Place under
`examples/train/thunder_agent/tests/test_harbor_generator_units.py`.

Cases to cover:

1. **`_attach_trial_routing_ids`** — assert
   `cfg["agent"]["kwargs"]["session_id"]
    == cfg["agent"]["kwargs"]["llm_call_kwargs"]["extra_body"]["program_id"]`
   and that calling it twice on a fresh deepcopy of the template never
   leaks IDs across attempts.

2. **`_apply_sampling_params_to_trial_config`** — feed
   `{"temperature": 0.3, "top_p": 0.9, "top_k": 20, "min_p": 0.05,
    "max_tokens": 2048, "logprobs": 1}` and assert:
   - `agent.kwargs.temperature == 0.3`
   - `llm_call_kwargs.top_p == 0.9`, `llm_call_kwargs.max_tokens == 2048`
   - `extra_body.top_k == 20`, `extra_body.min_p == 0.05`
   - `agent.kwargs.collect_rollout_details is True`
   - `top_k=-1`, `min_p=0`, `repetition_penalty=1.0` are NOT added to extra_body.

3. **`_best_effort_release_program`** — use `httpx.MockTransport` to
   simulate (200 ok / 404 / 500 / TimeoutException) and assert:
   - 200 returns immediately, no retry
   - 404 returns immediately, no retry
   - 500 logs a warning, no retry
   - `TimeoutException` retries up to `THUNDERAGENT_RELEASE_MAX_ATTEMPTS`
     with exponential backoff, then gives up without raising.

4. **`_get_response_ids_and_loss_mask_from_harbor_rollout`** — using a
   real Qwen tokenizer + the `qwen3_acc_thinking.jinja2` template:
   - feed `[{"role":"user", ...}, {"role":"assistant", ...},
     {"role":"user", ...}, {"role":"assistant", ...}]`
   - assert `len(response_ids) == len(loss_mask) == len(rollout_logprobs)`
   - assert user-segment slots in `loss_mask` are all 0
   - assert assistant-segment slots are 1
   - assert `rollout_logprobs` aligned with the assistant slots only
     (other positions are 0.0).

5. **Empty-dataset guard (Tier 1 corollary)** — instantiating
   `ThunderAgentHarborDataset(["/tmp"])` on a directory with no
   `instruction.md` subdirs must raise `ValueError`.

Run: `pytest examples/train/thunder_agent/tests/`. Pass: all green.

---

## Tier 3 — vLLM `/pause` & `/resume` probe (1 GPU, target image)

Goal: prove the rollout image actually exposes the native
weight-sync endpoints SkyRL relies on. Without this, every fully-async
weight sync fails on the first iteration.

```bash
# Start ONE rollout server on the target image with VLLM_SERVER_DEV_MODE=1
SERVER_URL="http://$(hostname -i):8001"

curl -fsS "$SERVER_URL/health" >/dev/null
curl -fsS -X POST "$SERVER_URL/pause?mode=keep&clear_cache=false"
curl -fsS -X POST "$SERVER_URL/resume"
curl -fsS "$SERVER_URL/get_world_size"
```

Pass criteria: all four calls return HTTP 200. If any returns 404, the
recipe will not work on this image and we must rebuild before Tier 4.

---

## Tier 4 — Smoke stage end-to-end (1 node × 1 GPU × 1 vLLM)

Goal: prove a full pipe from CLI parse → Trial run → weight sync round-
trip on minimal hardware. Uses `STAGE=smoke` defaults
(`MAX_TRAIN_TASKS=8`, `N_SAMPLES=2`, `EPOCHS=1`).

```bash
ROLLOUT_SERVER_URLS='["http://localhost:8001"]' \
EXTERNAL_PROXY_URL=http://localhost:8080 \
TRAIN_NUM_NODES=1 TRAIN_GPUS_PER_NODE=1 \
ROLLOUT_ENGINES=1 ROLLOUT_TP_SIZE=1 \
DATA_ROOT=/path/to/r2egym/smoke \
bash examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh smoke
```

Pass criteria (greppable):

- ≥1 line `[ThunderAgent] /weight_sync/begin` followed by
  `/weight_sync/end` (i.e. `pause_generation()`/`resume_generation()`
  closes the loop).
- ≥1 `POST /programs/release` line in the TA proxy log (best-effort
  release fired in `finally`).
- `rollout_metrics["generate/avg_num_turns"] > 0` in the trainer log.
- `rollout_metrics["generate/num_error_trajectories"] < N_SAMPLES *
   MAX_TRAIN_TASKS` (not every trial errored).
- The training process exits 0 after `EPOCHS=1`.

If smoke fails with `TypeError` on `pause`, regression on Phase 1 fix.
If smoke fails with `OmegaConf ConfigKeyError`, regression on Phase 3
config classes. If smoke fails with an httpx connection error to the TA
proxy, EXTERNAL_PROXY_URL is mis-routed.

---

## Tier 5 — Pilot stage (1 node × 8 GPU)

Goal: smoke-out scaling and rate-limiter behaviour. Uses
`STAGE=pilot`: `MAX_TRAIN_TASKS=64`, `EPOCHS=1`,
`TRAIN_BATCH_SIZE=$TRAINING_WORLD_SIZE` (= 8).

```bash
TRAIN_NUM_NODES=1 TRAIN_GPUS_PER_NODE=8 \
ROLLOUT_ENGINES=2 ROLLOUT_TP_SIZE=2 \
ROLLOUT_SERVER_URLS='["http://r0:8001","http://r1:8001"]' \
EXTERNAL_PROXY_URL=http://r0:8080 \
DATA_ROOT=/path/to/r2egym \
bash examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh pilot
```

Pass criteria:

- ≥4 training steps complete.
- `eval_interval=20` should NOT trigger inside one epoch (i.e. no eval
  loop runs — that is a configured property, not a regression).
- TA proxy `/programs/release` count ≈ #completed_trials_so_far
  (allow ±10%; some 404s are expected from already-released programs).
- Step-0 mean reward and stop-reason distribution within the historical
  baseline range for `r2egym-train256-medium-hard-v1` first-epoch.

---

## Tier 6 — Full benchmark replay (4 nodes × 8 GPU)

Goal: reproduce the `thunderagent_medium_hard_256_10epoch_no_preflight`
benchmark variant.

```bash
TRAIN_NUM_NODES=4 TRAIN_GPUS_PER_NODE=8 \
ROLLOUT_ENGINES=4 ROLLOUT_TP_SIZE=2 \
ROLLOUT_SERVER_URLS='["http://r0:8001",...,"http://r3:8001"]' \
EXTERNAL_PROXY_URL=http://r0:8080 \
DATA_ROOT=/path/to/r2egym \
bash examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh full
```

Pass criteria:

- 10 epochs × `ceil(MAX_TRAIN_TASKS * N_SAMPLES / TRAIN_BATCH_SIZE)`
  steps complete without hard failure.
- Per-step `policy_loss` and `reward_mean` track the historical
  baseline within ±5% (envelope, not exact).
- `eval_interval=4` triggers ~10 evals over the run; `eval_reward_mean`
  monotonic-ish over epochs (allow short regressions).
- TA proxy steady-state metrics: `programs_open` bounded
  (no monotonic growth across epochs — that would indicate a release
  leak).

If Tier 6 deviates from baseline, bisect first by toggling
`EXTERNAL_PROXY_URL` (rules out router selection), then by setting
`HARBOR_MAX_NUM_RETRIES_PER_TRIAL=1` (rules out retry-mask interactions),
then by setting `generator.sampling_params.logprobs=0` (rules out the
rollout-logprob extraction path).

---

## Pre-flight environment variables (for reference)

| Variable | Default | Effect |
|---|---|---|
| `THUNDER_AGENT_ROUTER_PORT` | `$SKYRL_INFERENCE_ROUTER_PORT` (8080) | Embedded TA router port (only used when `EXTERNAL_PROXY_URL` is unset). |
| `EXTERNAL_PROXY_URL` | unset | URL of an externally-launched TA proxy. Skips the embedded router. |
| `MINI_SWE_MODEL_TIMEOUT_SEC` | 1200 | LiteLLM single-call timeout passed via `harbor_trial_config.agent.kwargs.llm_call_kwargs.timeout`. |
| `HARBOR_MAX_NUM_RETRIES_PER_TRIAL` | 2 | Per-trial retry budget in the generator. |
| `THUNDERAGENT_RELEASE_TIMEOUT_SEC` | 30 | Release HTTP timeout. |
| `THUNDERAGENT_RELEASE_MAX_ATTEMPTS` | 4 | Release retry count. |
| `THUNDERAGENT_RELEASE_RETRY_BACKOFF_SEC` | 0.5 | Release retry exponential base. |
| `THUNDERAGENT_RELEASE_MAX_INFLIGHT` | 64 | Release-call concurrency cap. |
| `HARBOR_HARD_FAILURE_EXCEPTION_TYPES` | `RewardFileNotFoundError,VerifierTimeoutError` | Exception types that trip the per-task circuit breaker. |
| `HARBOR_TASK_CIRCUIT_BREAKER_ENABLED` | true | Master switch for the circuit breaker. |
| `HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD` | 2 | Consecutive hard failures before a task is dropped. |
| `SKYRL_DISABLE_THUNDERAGENT` | unset (=enabled) | Set to `1` only for non-TA debugging. |
