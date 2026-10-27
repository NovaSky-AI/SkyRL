# On-Policy Distillation

On-policy distillation (OPD) trains a student on its own rollouts with a frozen teacher grading
every token: the negative per-token reverse KL to the teacher, `log π_teacher − log π_student`,
is the advantage. It combines the on-policy character of RL with a dense, per-token signal, and was
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
# The math recipe from the writeup: Qwen3-4B-Base (or 1.7B-Base) student, DAPO-17k prompts, AIME24 eval.
# The job launches the teacher (Qwen3-32B, TP=4) on 4 GPUs of an 8-GPU node, next to the student's 4.
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
bash examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_4b.sh
```

The only OPD-specific flags are the teacher and, optionally, the mixing knobs. By default the job launches
the teacher's vLLM deployment on its own GPUs, so a teacher is a model path plus its engine count and
parallelism:

```bash
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.policy.model.path=Qwen/Qwen3-4B-Base \
  trainer.teacher.model=Qwen/Qwen3-32B \
  trainer.teacher.inference_engine.num_engines=1 \
  trainer.teacher.inference_engine.tensor_parallel_size=4 \
  data.train_data="['$HOME/data/dapo/dapo-math-17k-cleaned.parquet']" \
  environment.env_class=aime \
  ...   # the usual placement, batch and sampling flags; the teacher's GPUs come on top of them
```

The student and the teacher must share a tokenizer: the teacher scores the student's token ids
verbatim.

## Teacher backends

Where the teacher lives is one choice with two answers, and each backend reads only its own fields
(setting the other's fields is a config error):

| `trainer.teacher.backend` | What it is | Notes |
|---|---|---|
| `skyrl` (default) | A vLLM deployment this job launches from `trainer.teacher.inference_engine` (the same block as `generator.inference_engine`) on its own GPUs, driven through a `RemoteInferenceClient` exactly as the student's engines are | `trainer.teacher.model` is an HF id or local path. Own placement group (never the colocate group), a router, a port window past the student's; never weight-synced or slept. Defaults that differ from the student's block: prefix caching off (vLLM skips the prefix-cache lookup for any request that asks for prompt logprobs, and every scoring request does), `gpu_memory_utilization` 0.9, Ray Prometheus stats off. Set `engine_init_kwargs.max_model_len` to cover prompt + response + 1, as the example script does; unset, vLLM uses the model's native context, as for the student. Weight-sync, sleep, LoRA, PD, speculative-decoding, routed-expert and external-URL fields are rejected. |
| `vllm` | One vLLM endpoint you run per teacher model, given by `trainer.teacher.server_url`, scored through the OpenAI-compatible `/v1/completions` with vLLM's `prompt_logprobs` parameter | A single `vllm serve`, a data-parallel one, or a router in front of several servers: SkyRL's `serve` entrypoint (`examples/train/remote_inference_server/run_vllm_server.sh`) starts servers behind a router and logs its `proxy_url`, which is the URL to use. Spreading requests over replicas is the endpoint's job; the client does not round-robin. Any model vLLM serves; `max_model_len` must cover prompt + response + 1. The servers are never weight-synced or slept. |

### A teacher launched by the job

The default. Budget the GPUs: the teacher's `num_engines · tensor_parallel_size · pipeline_parallel_size ·
data_parallel_size` come on top of the student's engines (and the training workers when not colocated).
On one 8-GPU node, a colocated 4-GPU student plus a TP=4 teacher:

```bash
uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.teacher.model=Qwen/Qwen3-32B \
  trainer.teacher.inference_engine.num_engines=1 \
  trainer.teacher.inference_engine.tensor_parallel_size=4 \
  trainer.placement.colocate_all=true trainer.placement.policy_num_gpus_per_node=4 \
  generator.inference_engine.num_engines=4 \
  ...
```

Prefer independent replicas (`num_engines=N`) over one data-parallel group. A budget the cluster cannot
satisfy surfaces as the teacher placement group's timeout (`SKYRL_RAY_PG_TIMEOUT_IN_S`); there is no
upfront check yet. The teacher is launched before the student's engines and the training workers, and
goes down with the Ray job when the run ends, like the student's engines.

### A vLLM teacher you run

Start the teacher on GPUs the training job does not use, either with vLLM directly or with SkyRL's
standalone server (`examples/train/remote_inference_server/run_vllm_server.sh`, which starts servers
behind a router and logs its `proxy_url`), then point the run at that one endpoint with `backend=vllm`:

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000   # on the teacher node(s)

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  trainer.teacher.backend=vllm \
  trainer.teacher.model=Qwen/Qwen3-32B \
  trainer.teacher.server_url=http://teacher-host:8000 \
  ...
```

Two usual server-side problems to rule out yourself: a `max_model_len` shorter than the longest
input + longest response + 1, and a vLLM version that rejects `prompt_logprobs` while prefix caching
is on (start such a server with `--no-enable-prefix-caching`). Scoring requests bypass the prefix cache
on the versions that allow them, so each request prefills the full sequence. No authentication is
sent to vLLM servers.

## Configuration

Every OPD field, and the two algorithm defaults the entrypoint changes (`trainer.algorithm.use_kl_loss=false`,
`policy_loss_type=importance_sampling`), is documented in `skyrl/train/opd/config.py`. `validate_opd_cfg` in
the same file rejects the settings that would break the teacher signal.

To match the 2025-11 writeup it reproduces, the math script sets
`trainer.algorithm.loss_reduction=seq_mean_token_sum_norm` with `trainer.algorithm.max_seq_len` set to
the longest prompt plus the longest response (2048 + 8192). That sums the batch's token losses and
divides by a constant, where the default `token_mean` divides by the batch's own token count.

## How it works

- `skyrl.train.opd.trainer.OPDTrainer.generate_on_input_batch` splits the batch into one `GeneratorInput` per prompt
  (its `n_samples_per_prompt` rows) and runs one `generator.generate` call per group concurrently, the
  way the fully-async trainer's `_run_generate_for_a_group_loop` does. As soon as a group's rollouts
  are back it calls the teacher for each row's `prompt + response` and stores the per-token logprobs
  as `GeneratorOutput["teacher_logprobs"]`; `concatenate_generator_outputs` then reassembles the
  batch in input order. Only the groups that finish last pay for the teacher. Eval batches take the
  base path and are never scored.
- The same trainer zeroes the per-token rewards after their metrics are logged (pure mode), pads
  `teacher_logprobs` into the training batch right-aligned like `rollout_logprobs`, and after the
  advantage estimator subtracts `kl_coef · (action_log_probs − teacher_logprobs) · loss_mask`.
  Applying the term after the estimator is what makes it compose with any estimator in mixed mode,
  and with any reward-only estimator in pure mode: GRPO would otherwise sum the per-token signal
  into one scalar per sequence.
- `skyrl.train.opd.teacher_client.TeacherLogprobClient` is the abstract base class for teachers; it owns the concurrency limit,
  the length, missing-logprob and finiteness checks, and a backend implements one method, `_compute_logprobs`.

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
