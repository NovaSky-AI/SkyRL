# On-Policy Distillation

On-policy distillation (OPD) trains a student on its own rollouts with a frozen teacher grading
every token: the per-token reverse KL to the teacher, `log π_student − log π_teacher`, is the
advantage. It combines the on-policy character of RL with a dense, per-token signal, and was
popularized by [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
(earlier: [Agarwal et al.](https://arxiv.org/abs/2306.13649), [Gu et al.](https://arxiv.org/abs/2306.08543),
the [Qwen3 report](https://arxiv.org/abs/2505.09388)).

The entrypoint, `skyrl.train.entrypoints.main_opd` (library code in `skyrl/train/opd/`; this
directory holds the run scripts), serves the teacher from an **inference engine**, not a training worker: the
teacher never syncs weights, can be much larger than the student, and scores each prompt's group of
rollouts as soon as it finishes, while the other groups are still generating, so teacher latency
overlaps with generation. The generator is untouched: any `GeneratorInterface` works.
It replaces the 2025-11 example that reused the reference model as the teacher
([writeup](https://novasky-ai.notion.site/on-policy-distillation)).

## Quickstart

```bash
export FIREWORKS_API_KEY=<your key>

# Fully serverless smoke test: gpt-oss-20b student, gpt-oss-120b teacher on Fireworks serverless.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash examples/train/on_policy_distillation/run_opd_gsm8k_gptoss_fireworks.sh

# The math recipe from the writeup: Qwen3-4B-Base (or 1.7B-Base) student, DAPO-17k prompts, AIME24 eval,
# teacher = a Qwen3-family model on a dedicated Fireworks deployment.
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
TEACHER_MODEL=accounts/<account>/deployments/<id> \
  bash examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_4b.sh
```

The only OPD-specific flags are the teacher and, optionally, the mixing knobs:

```bash
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.policy.model.path=Qwen/Qwen3-4B-Base \
  trainer.teacher.model=accounts/<account>/deployments/<id> \
  data.train_data="['$HOME/data/dapo/dapo-math-17k-cleaned.parquet']" \
  environment.env_class=aime \
  ...   # the usual placement, batch and sampling flags
```

The student and the teacher must share a tokenizer: the teacher scores the student's token ids
verbatim.

## Teacher backends

| `trainer.teacher.backend` | What it is | Notes |
|---|---|---|
| `fireworks` (default) | A Fireworks model id (`accounts/fireworks/models/<id>`) or dedicated deployment (`accounts/<account>/deployments/<id>`), scored through the completions API with an integer prompt and `echo_last` | Serverless is fine for checking the plumbing; for training use a dedicated deployment. Serverless replicas disagree on logprobs by more than the OPD signal itself, some serverless models do not support echo, and the Qwen3 (2025) family is not serverless. Custom checkpoints can be uploaded with `firectl model create` and deployed. |
| `vllm` | vLLM servers you started, given by `trainer.teacher.server_urls`, scored through the OpenAI-compatible `/v1/completions` with vLLM's `prompt_logprobs` parameter | Works with a stock `vllm serve` and with SkyRL's `serve` entrypoint (`examples/train/remote_inference_server/run_vllm_server.sh`). Any model vLLM serves; `max_model_len` must cover prompt + response + 1. Requests round-robin across the URLs. The servers are never weight-synced or slept. |

Before any model is loaded, the entrypoint sends the same short sequence to the teacher
`trainer.algorithm.opd.self_test_samples` times concurrently and refuses to train if the answers
differ by more than `self_test_max_abs_diff` nats (default 0.05). This catches replica-dependent
serving, models that reject echo, and a mismatched tokenizer, at minute 0 instead of minute 40.

### A vLLM teacher

Start the teacher on GPUs the training job does not use, either with vLLM directly or with SkyRL's
standalone server (`examples/train/remote_inference_server/run_vllm_server.sh`, which logs its
`server_urls`), then point the run at it:

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000   # on the teacher node(s)

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.teacher.backend=vllm \
  trainer.teacher.model=Qwen/Qwen3-32B \
  trainer.teacher.server_urls="['http://teacher-host:8000']" \
  ...
```

The self-test at startup catches the two usual server-side problems: a `max_model_len` shorter than
prompt + response + 1, and a vLLM version that rejects `prompt_logprobs` while prefix caching is on
(start such a server with `--no-enable-prefix-caching`). Scoring requests bypass the prefix cache
on the versions that allow them, so each request prefills the full sequence. No authentication is
sent to vLLM servers.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `trainer.teacher.backend` | `fireworks` | `fireworks` or `vllm` |
| `trainer.teacher.model` | — | Fireworks model / deployment id, or the served model name of the vLLM servers |
| `trainer.teacher.base_url` | Fireworks data plane | Server root without `/v1` |
| `trainer.teacher.api_key_var` | `FIREWORKS_API_KEY` | Environment variable holding the key (Fireworks only) |
| `trainer.teacher.server_urls` | — | `vllm` only: base URLs, e.g. `"['http://host:8000']"` |
| `trainer.teacher.max_concurrency` | 32 | Teacher requests in flight |
| `trainer.teacher.request_timeout_s`, `max_retries` | 120, 3 | Request timeout and retries with backoff (a retry moves to the next URL) |
| `trainer.algorithm.opd.kl_coef` | 1.0 | `advantages -= kl_coef · (log π_student − log π_teacher)` |
| `trainer.algorithm.opd.use_task_reward` | `false` | `false`: pure distillation (env reward is only logged). `true`: the reward's advantages plus the teacher term |
| `trainer.algorithm.opd.self_test_samples`, `self_test_max_abs_diff` | 8, 0.05 | The startup determinism check |

The entrypoint changes two algorithm defaults: `trainer.algorithm.use_kl_loss=false` (no reference
model is instantiated; turn it on to add a reference-KL loss alongside the teacher) and
`policy_loss_type=importance_sampling` (the Thinking Machines recipe; `regular`, PPO-clip, also
works). Everything else keeps the core default: `advantage_estimator=grpo` (with zero rewards it
emits zeros, so pure OPD needs no special estimator), `zero_variance_filter=false`, no dynamic
sampling, `advantage_batch_normalize=false`, temperature and top-p 1.0. `validate_opd_cfg` rejects
settings that would silently break the signal (zero-variance filtering, dynamic sampling,
batch-normalized advantages, losses that skip the old-logprob forward pass).

## How it works

- `skyrl.train.opd.trainer.OPDTrainer.generate` splits the batch into one `GeneratorInput` per prompt
  (its `n_samples_per_prompt` rows) and runs one `generator.generate` call per group concurrently, the
  way the fully-async trainer's `_run_generate_for_a_group_loop` does. As soon as a group's rollouts
  are back it calls the teacher for each row's `prompt + response` and stores the per-token logprobs
  as `GeneratorOutput["teacher_logprobs"]`; `concatenate_generator_outputs` then reassembles the
  batch in input order. Only the groups that finish last pay for the teacher. Eval batches take the
  base path and are never scored.
- The same trainer zeroes the per-token rewards after their metrics are logged (pure mode), pads
  `teacher_logprobs` into the training batch right-aligned like `rollout_logprobs`, and after the
  advantage estimator subtracts `kl_coef · (action_log_probs − teacher_logprobs) · loss_mask`.
  Applying the term after the estimator is what makes it compose with any estimator: GRPO would
  otherwise sum the per-token signal into one scalar per sequence.
- `skyrl.train.opd.teacher_client.TeacherLogprobClient` is the abstract base class for teachers; it owns the concurrency limit,
  the length check and the self-test, and a backend implements one method, `_compute_logprobs`.

## Metrics

`opd/reverse_kl` is the headline curve (it approaches 0 as the student matches the teacher; the
writeup's runs plateau at 0.01–0.09 nats), with `opd/reverse_kl_abs_max`, `opd/adv_rl_abs_mean`
and `opd/adv_opd_abs_mean` to compare the scale of the reward term and the teacher term when
tuning `kl_coef` in mixed mode. `opd/teacher_time_exposed` is the seconds the step waited on the
teacher after the last rollout of the batch came back, the only scoring cost the overlap cannot
hide; `opd/teacher_time_per_group_mean` is the mean scoring time of one prompt's group. `reward/avg_pass_at_n` still reports the verifier's pass rate on training
rollouts in pure mode.

## Caveats

- Mixed mode adds a z-scored GRPO advantage (order 1 on every token) to a raw log-probability
  difference (0.01–1 nats per token); `kl_coef` trades them off and the two `opd/adv_*` metrics
  show the ratio.
- A failed teacher request fails the batch loudly after the client's retries.
- Step-wise trajectories and top-k / full-distribution distillation losses are not supported;
  the sampled-token estimate needs one logprob per token, which every backend provides.
