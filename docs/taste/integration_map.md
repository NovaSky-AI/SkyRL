# Fleet GRPO Reward Integration Map

Repo: `https://github.com/fleet-ai/skyrl-fleet` (cloned to `/tmp/skyrl-fleet-2` in sandbox; `git clone` into `/sessions/.../outputs` failed because the existing mount blocked write to `.git/`, so we cloned to `/tmp/skyrl-fleet-2`).

The `skyrl-train` package has been merged into `skyrl/` (per `skyrl-train/README.md`). Modern code paths live under `skyrl/train/...`.

---

## Reward emit point

The Fleet env returns reward in **two places**, both in `skyrl-gym/skyrl_gym/envs/fleet_task/env.py`:

### Per-step reward — `step_async()` returns
File: `skyrl-gym/skyrl_gym/envs/fleet_task/env.py`

The reward is initialized to `0.0` at line **552**, populated from OpenEnv at lines **590–592** and **615–617**, and finally emitted on the `BaseTextEnvStepOutput` returns at lines **674, 708, 762**.

```
552        reward = 0.0
...
588            try:
589                mcp_start = time.time()
590                obs, reward, done, info = (
591                    await self.openenv_task_env.step_async(openenv_action)
592                )
...
613            try:
614                mcp_start = time.time()
615                obs, reward, done, info = (
616                    await self.openenv_task_env.step_async(openenv_action)
617                )
...
672            return BaseTextEnvStepOutput(
673                observations=[],
674                reward=reward,
675                done=True,
676                metadata={...},
677            )
...
706                return BaseTextEnvStepOutput(
707                    observations=[new_obs],
708                    reward=reward,
709                    done=episode_done,
710                    metadata=metadata,
711                )
...
760        return BaseTextEnvStepOutput(
761            observations=[new_obs],
762            reward=reward,
763            done=episode_done,
764            metadata=metadata,
765        )
```

### Final reward fallback — `close()` / `close_async()`
For trajectories that get terminated by SkyRL (context overflow, timeout) **without** the agent emitting `<done>`, OpenEnv's verifier is run inside `close()` / `close_async()` and the result is stashed on `self.last_reward` (lines **789–790** and **805–806**). It then surfaces via `get_metrics()` as `final_reward` (line **824**):

```
784    def close(self):
785        """Close the Fleet environment and cleanup resources."""
786        if self.openenv_task_env:
787            try:
788                self.openenv_task_env.close()
789                if self.openenv_task_env.final_reward is not None:
790                    self.last_reward = self.openenv_task_env.final_reward
...
796    async def close_async(self):
...
802        if self.openenv_task_env:
803            try:
804                await self.openenv_task_env.close_async()
805                if self.openenv_task_env.final_reward is not None:
806                    self.last_reward = self.openenv_task_env.final_reward
```

The terminal reward used by the GRPO trainer comes from the last step where `done=True`, i.e. one of the three return sites above. **The clean place to inject `taste_score` is inside `step_async()` immediately before each of those three returns, when `episode_done is True`.**

---

## Verifier source

The binary `0.0 / 1.0` reward is **not computed inside this repo**. It comes back from OpenEnv's `FleetTaskEnv.step_async()` (and `close_async()`) at:

- `skyrl-gym/skyrl_gym/envs/fleet_task/env.py:590-592` — happy path during the step where the agent submits its tool call
- `skyrl-gym/skyrl_gym/envs/fleet_task/env.py:615-617` — when the agent emits `<done>` with no tool call
- `skyrl-gym/skyrl_gym/envs/fleet_task/env.py:788-790, 804-806` — orphaned-trajectory fallback via `openenv_task_env.final_reward`

OpenEnv runs a programmatic Python verifier server-side; the Fleet wrapper only consumes its return value. There is also a **partial-reward** mode (not binary) toggled by `env_config.partial_reward` (constructor lines **176–181**); the VL launch script enables it (`scripts/fleet-vl-run.sh:42` — `environment.skyrl_gym.fleet_task.partial_reward=true`). Per `reward_metrics.py:79-82`, only `reward >= 1.0` counts as a "pass" in pass@n, so partial values land in `(0,1)`.

For task-generation runs there is also `integrations/fleet/task_gen_reward.py` which applies a derived "mixed result" reward — orthogonal to the browser-use loop but worth noting because it's a precedent for shaping rewards inside this repo.

---

## LLM-as-judge example

Path: `examples/train/llm_as_a_judge/`. Four files (5 with `__init__.py`):

| File | Purpose |
|---|---|
| `llm_judge_env.py` | The env: `GSM8kLLMJudgeEnv(BaseTextEnv)` with a synchronous `step()` that calls the OpenAI client to score an answer. |
| `main_llm_judge.py` | Ray entrypoint that registers the env id `"llm_as_a_judge"` and calls `BasePPOExp(cfg).run()`. |
| `gsm8k_dataset_judge.py` | Dataset prep: emits parquet with `env_class="llm_as_a_judge"` and `reward_spec.ground_truth`. |
| `run_llm_judge.sh` | GRPO launch (Qwen2.5-1.5B-Instruct, 4× GPU). Sets `environment.skyrl_gym.llm_as_a_judge.model="gpt-4o-mini"`. |

What the env actually does — quoting the **only** reward-relevant section of `llm_judge_env.py`:

```python
def _get_reward(self, action: str) -> float:
    message = PROMPT + f"\n\nGOLD SOLUTION:\n{self.ground_truth}\n\nPREDICTED SOLUTION:\n{action}\n\nAnswer:"
    try:
        response = self.llm_judge_client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": message}]
        )
        reply = response.choices[0].message.content.strip()
        match = re.search(r"### Final Score:\s*([01](?:\.0)?)", reply)
        if match:
            return float(match.group(1))
        if reply.strip() in {"1", "0"}:
            return float(reply.strip())
        return 0.0
    except Exception as e:
        print(f"LLM Judge error: {type(e).__name__}: {e}")
        return 0.0

def step(self, action: str) -> BaseTextEnvStepOutput:
    done = True
    reward = self._get_reward(action)
    return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
```

Properties:

- **Synchronous and blocking.** `step()` is sync and uses the `openai.OpenAI` blocking client. Each rollout's `step()` call blocks the worker thread for the full judge latency.
- **Single-turn.** The env always returns `done=True` on the first step, so the judge runs exactly once per trajectory at the very end.
- **No batching.** One judge call per rollout, no aggregation across the GRPO group of `n_samples_per_prompt` trajectories.
- **No async / no thread pool / no retry / no timeout.** Errors swallow to `0.0` (silent failure mode).
- **Caller is the env itself.** Reward is computed inline in `step()` — the trainer never knows there is an LLM judge in the loop.

The reason this is acceptable in the GSM8k example: it is **single-turn**, run on cheap CPU-side I/O, with a tiny batch (the script uses `train_batch_size=32`, `n_samples_per_prompt=5`), and rollouts in SkyRL's generator already run concurrently via the async generator + Ray, so blocking calls in different envs proceed in parallel. The pattern does **not** scale cleanly to long multi-turn browser_use rollouts where you don't want to hold the env alive for an extra 1–3 s × group_size at the very end.

---

## Async strategy

**Recommendation: post-hoc, parallel, and out-of-step.** Specifically:

1. **Do not call the judge inside `step_async()` per turn.** Browser-use trajectories have 50–80 turns (`MAX_TURNS=80` in the YAML); judging every step is wasteful and the judge can't reasonably score before the trajectory is done anyway.
2. **At episode end** (the `episode_done` branch in `step_async()` and inside `close_async()`), kick off the judge call **as an awaitable**. Two options, in order of cleanliness:
   - **Option A (preferred):** make `score_trajectory` an `async def` that uses `httpx.AsyncClient` or `openai.AsyncOpenAI`, with `asyncio.wait_for(..., timeout=judge_timeout_s)`. SkyRL's generator already runs `step_async` inside an asyncio task per rollout, so judge calls across the entire GRPO group naturally overlap. With `n_samples_per_prompt=4` and ~50 prompts, you get 200 judge calls running concurrently and the wall-clock cost collapses to ~max(judge_latency).
   - **Option B (escape hatch):** wrap the sync judge in `asyncio.to_thread(...)` (Python 3.9+) so the existing sync OpenAI/Anthropic client doesn't block the event loop. Slightly worse than A under load but a one-line change.
3. **Use `asyncio.gather` or `asyncio.wait_for` with a hard timeout** of e.g. 10 s. On timeout/exception, log a warning and fall back to `verifier_reward` only (i.e. effectively `alpha = 1.0` for that trajectory). This keeps a slow Anthropic API outage from stalling a training step.
4. **Do not gate trajectory cleanup on the judge.** Resolve the judge future, attach the score to the final `BaseTextEnvStepOutput`, and let `close_async()` proceed independently if the judge is still pending. (In practice, since `step_async` returns the terminal `done=True` step, you must either `await` the judge before the final return or do post-hoc reward attribution at the trainer level.)
5. **Optional optimization — batch by prompt-group at the trainer layer.** A more invasive variant: store the trajectory transcripts in `metadata`, then have the trainer call the judge once per GRPO group (with all `n` trajectories in one prompt) before computing advantages. This gives the judge cross-trajectory context for relative ranking and is what most production RLAIF setups do. Requires patching the trainer's reward post-processing path (where `flatten_rewards` in `integrations/fleet/reward_metrics.py` is called), not the env. Out of scope for the minimal patch but worth flagging.

**The existing `llm_as_a_judge` example uses none of these**: it is sync, inline, single-call, single-turn, no timeout, no retry. **Do not copy it as-is for browser_use** — copy the *interface shape* (judge runs inside the env at episode end and emits a scalar in `[0,1]`) and rewrite the call to be async + timed-out.

---

## GRPO config knobs

Defaults in `skyrl/train/config/ppo_base_config.yaml` (lines 96–124), with VL overrides from `scripts/fleet-vl-run.sh`:

| Knob | Default | VL launch override | Interaction with shaped reward |
|---|---|---|---|
| `trainer.algorithm.advantage_estimator` | `"grpo"` | `grpo` | Computes per-prompt-group advantages from raw rewards. A continuous `taste_score` increases within-group variance and produces non-zero advantages even when all trajectories pass/fail the binary verifier — exactly the desired effect. |
| `trainer.algorithm.grpo_norm_by_std` | `true` | (default) | GRPO divides advantage by group-level reward std. With binary rewards, std is 0 when the whole group passes/fails; mixing in `taste_score` raises std, which **also damps the advantage magnitude**. Watch for: groups where verifier is unanimous now produce small but non-zero advantages — the gradient signal will be tiny. May want `grpo_norm_by_std=false` once shaped reward is on. |
| `trainer.algorithm.zero_variance_filter` | `false` | `true` (line 73) | Currently masks out prompts where all rewards are identical (no signal). With shaped reward this filter would fire **far less often** since `taste_score` is approximately continuous → almost every prompt now contributes a gradient. This is good for sample efficiency but may also amplify judge noise into the policy. Consider keeping it on but with a tolerance threshold. |
| `trainer.algorithm.use_kl_loss` | `true` | `true` | KL is on the policy loss, so it is independent of reward scale. Good. |
| `trainer.algorithm.kl_loss_coef` | `0.001` | (default) | Independent of reward, no change needed. |
| `trainer.algorithm.use_kl_in_reward` | `false` | (default, mutually exclusive with `use_kl_loss`) | If you ever flip to `use_kl_in_reward=true`, the KL term gets *added to the reward* and competes directly with `taste_score`. Keep this `false`. |
| `trainer.algorithm.eps_clip_low / eps_clip_high` | `0.2 / 0.2` | (default) | PPO ratio clip. Independent of reward magnitude (operates on log-prob ratio), so safe. |
| `trainer.algorithm.advantage_batch_normalize` | `false` | (default) | If turned on, would re-normalize advantages across the whole batch. Consider enabling if the taste_score's scale + verifier mix produces unstable cross-prompt advantage magnitudes. |
| `trainer.algorithm.loss_reduction` | `"token_mean"` | `"sequence_mean"` (line 47) | Doesn't touch reward, but `sequence_mean` is what's used for VL — keep aware that gradient is per-trajectory averaged. |

**Concrete suggestions:**
- Start with `alpha=0.5` (balanced).
- Keep `grpo_norm_by_std=true` initially; if you observe gradient norm collapse, set it to `false`.
- Bound `taste_score` to `[0,1]` (same range as verifier) so the mixed reward stays in `[0,1]` and existing pass@n / signal-ratio metrics in `integrations/fleet/reward_metrics.py` still parse correctly.
- Consider reporting `verifier_reward` and `taste_reward` separately as wandb metrics so you can disentangle their contributions — fits naturally into the existing metric schema.

---

## Existing evals

**Eval entrypoint:** `integrations/fleet/entrypoints/main_eval.py` — `FleetEvalExp(BasePPOExp).run()`. Resumes FSDP weights from S3, calls `await trainer.eval()` once (line 125), logs the dict via `trainer.tracker.log(...)`, and (optionally) uploads dump to S3.

**Metric computation:** `integrations/fleet/reward_metrics.py` exposes:
- `flatten_rewards(rewards)` — collapses token-level rewards to scalars.
- `compute_pass_at_n(rewards, uids)` — fraction of unique prompts with **at least one rollout `>= 1.0`**.
- The module's docstring documents the wandb naming convention: `reward/{group}/pass_at_n`, `reward/{group}/variance_per_prompt`, `reward/{group}/signal_ratio`, `reward/{group}/mean_positive_reward`.

**What gets measured today:**
- Final reward distribution (pass@n with threshold ≥ 1.0).
- Within-prompt reward variance (the GRPO learning-signal proxy).
- Signal ratio (% prompts with non-zero variance).
- Mean positive reward.
- Per-env metrics emitted from `FleetTaskEnv.get_metrics()` at lines 812–835: `task_key`, `env_key`, `turns`, `tool_calls`, `tool_errors`, `is_hinted`, `final_reward`, `verifier_stdout`, `verifier_error`, `tool_error_messages`, `chat_history`.

**How to add a new metric (e.g. `taste_reward_mean`):**
1. In `step_async()`'s terminal returns, also stash `self.last_taste_reward` and `self.last_verifier_reward` on the env.
2. Append both to the metadata dict and to `get_metrics()` output (line 814 onward) so they flow into the trainer's metric aggregator alongside `final_reward`.
3. The trainer's `_get_response_level_rewards`/eval-dump path picks up env metadata — no further patching needed if the keys are scalar-typed. For aggregated metrics (group-level), add a function to `reward_metrics.py` modeled on `compute_pass_at_n` and call it from wherever `pass_at_n` is logged in the trainer (search `compute_pass_at_n` to find the call sites — they live in `skyrl/train/trainer.py` and `integrations/fleet/entrypoints/main_fleet_tinker.py`).
4. Test path: `integrations/fleet/tests/test_main_eval.py`.

---

## Files referenced (absolute, in cloned repo)

- `/tmp/skyrl-fleet-2/skyrl-gym/skyrl_gym/envs/fleet_task/env.py` — env, reward emit point (lines 525–765, 784–810).
- `/tmp/skyrl-fleet-2/examples/train/llm_as_a_judge/llm_judge_env.py` — sync judge example.
- `/tmp/skyrl-fleet-2/examples/train/llm_as_a_judge/main_llm_judge.py`
- `/tmp/skyrl-fleet-2/examples/train/llm_as_a_judge/gsm8k_dataset_judge.py`
- `/tmp/skyrl-fleet-2/examples/train/llm_as_a_judge/run_llm_judge.sh`
- `/tmp/skyrl-fleet-2/tasks/openenv-fleet-grpo-vl.yaml` — VL launch task.
- `/tmp/skyrl-fleet-2/scripts/fleet-vl-run.sh` — actual GRPO CLI args.
- `/tmp/skyrl-fleet-2/skyrl/train/config/ppo_base_config.yaml` — GRPO/PPO defaults (lines 96–124).
- `/tmp/skyrl-fleet-2/integrations/fleet/reward_metrics.py` — metric helpers.
- `/tmp/skyrl-fleet-2/integrations/fleet/entrypoints/main_eval.py` — eval entrypoint.
- `/tmp/skyrl-fleet-2/integrations/fleet/task_gen_reward.py` — precedent for shaping rewards inside this repo (task-gen, not browser-use).
