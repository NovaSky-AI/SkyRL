# Self-Distillation Policy Optimization (SDPO)

This example implements **SDPO** from
[Hübotter et al., *Reinforcement Learning via Self-Distillation* (2026)](https://arxiv.org/abs/2601.20802)
(original implementation: [lasgroup/SDPO](https://github.com/lasgroup/SDPO)), and
**SRPO** from
[Li et al., *Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing* (2026)](https://arxiv.org/abs/2604.02288).

For each prompt, the student samples a group of responses. Successful responses
(and, optionally, environment feedback) become extra context for a self-teacher,
which scores the student's own responses. The student is trained with a
top-k distillation loss (forward KL, reverse KL, or JSD) toward the teacher's
next-token distribution. SRPO routes each response instead: correct responses
take a GRPO update, failed responses with a teacher hint take an
entropy-weighted SDPO update.

Everything lives in this folder; no SkyRL core changes are required.

## Layout

| File | Contents |
|---|---|
| `main_sdpo.py` | `SDPOGenerator`, `SDPOTrainer`, `SDPOExp`, and the CLI/Ray entrypoint. Recipes subclass `SDPOExp`. |
| `sdpo_config.py` | `trainer.distillation.*` config dataclasses and validation of supported SkyRL settings. |
| `fsdp_worker.py` | FSDP policy worker: teacher scoring, the distillation loss, and SRPO routing. |
| `megatron/` | Megatron policy worker (`megatron_worker.py`) with vocab-parallel loss (`loss.py`), EMA teacher (`teacher.py`), and response-only logit projection (`response_logits.py`). |
| `teachers.py` | FSDP teacher providers: `self`, `ema`, `frozen`. |
| `utils.py` | Teacher prompt construction, paired student/teacher inputs, routing masks, and losses. |
| `reproduction/` | The SDPO paper's Section 3 science and tool-use benchmarks: data conversion, reward environments, and the `run_sdpo.sh` / `run_srpo.sh` launchers. |

## Quickstart: SDPO paper benchmarks

The Section 3 / Table 3 benchmarks are SciKnowEval `biology`, `chemistry`,
`material`, `physics`, and ToolAlpaca `tooluse`. Convert the original splits
(pinned to SDPO commit `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, no resplitting):

```bash
git clone https://github.com/lasgroup/SDPO $HOME/SDPO
uv run --isolated --extra fsdp -m examples.train.sdpo.reproduction.prepare_data \
  --sdpo-repo $HOME/SDPO --output-dir $HOME/data/sdpo
```

Then launch a task. Both scripts use 4 GPUs (2 FSDP training, 2 vLLM) and
Qwen3-8B in non-thinking mode; set `MODEL=allenai/Olmo-3-7B-Instruct` for OLMo-3.
This placement is sized for 141 GB GPUs (H200). On 80 GB GPUs, full-parameter
Qwen3-8B with an EMA teacher does not fit on 2 training GPUs; raise
`NUM_POLICY_GPUS`.

```bash
export WANDB_API_KEY=<your_key_here>

# SDPO: EMA self-teacher conditioned on a successful sibling, JSD (alpha=0.5) over top-100 + tail
bash examples/train/sdpo/reproduction/run_sdpo.sh chemistry

# SRPO: same recipe with sample routing and teacher-entropy weighting
bash examples/train/sdpo/reproduction/run_srpo.sh chemistry
```

Extra arguments are forwarded as config overrides, e.g.
`bash examples/train/sdpo/reproduction/run_sdpo.sh physics trainer.max_training_steps=2`.

The recipe follows the paper: 32 prompts × 8 responses per step, learning rate
1e-5, EMA rate 0.05, rollout importance-ratio cap 2.0, 8192-token responses,
and avg@16 evaluation every 5 steps (temperature 0.6, top-p 0.95). The paper
reports the best avg@16 within 1 and 5 hours of training time; the scripts run
up to 30 epochs, so compare on the logged `eval/all/avg_score` curve at
matching wall time.

## Megatron backend

`trainer.strategy=megatron` selects the [Megatron SDPO worker](megatron/megatron_worker.py).
It uses the same teacher prompt construction and student-selected top-k/tail
KL/JSD objective as FSDP, normalizing across vocabulary shards without gathering
full vocabulary tensors. The EMA tracks effective weights (base plus merged LoRA
update) with an FP32 CPU accumulator, optional teacher CPU storage, and per-rank
checkpoint state.

Supported: dense models with TP/SP, PP=1, 2 or 3, CP=EP=1, sample-based
microbatching, the distribution loss, global-token-mean reduction, teacher-entropy
weighting, rollout importance-ratio clipping, and advantage clipping with
`alpha=1`. SRPO, FP8, packing, context parallelism, and more than three pipeline
stages are rejected at validation.

The teacher is loaded once to score a batch, then offloaded before student
forward/backward. Response logits stay vocabulary-sharded in a CPU cache, and
each entry is released after its student backward.
`trainer.distillation.megatron_teacher_cache_gib` (default 128) caps the cache
per final-stage TP rank; batches that exceed it are scored and trained in bounded
windows (at least one microbatch per window). Megatron projects only
response-predicting hidden states onto the vocabulary through the model's
`output_processor` hook; the transformer still processes the full context.

```text
trainer.strategy=megatron
trainer.policy.megatron_config.tensor_model_parallel_size=2
trainer.policy.megatron_config.pipeline_model_parallel_size=1
trainer.policy.megatron_config.context_parallel_size=1
trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=false
trainer.distillation.limit_response_logits=false
trainer.distillation.teacher.storage=cpu
trainer.distillation.teacher_first=true
```

## Objective and conditioning knobs

All fields below are under `trainer.distillation` unless qualified otherwise.
For response position $t$, write $p_t(v)=\pi_\theta(v\mid x,y_{<t})$ for the
student and $q_t(v)=\pi_{\bar\theta}(v\mid x,h,y_{<t})$ for the detached teacher.
The hint $h$ can contain an expert answer, judge feedback, and the student's
previous answer. Both models score the same sampled continuation. Probabilities
use the configured training temperature. The teacher does not receive gradients.

### Distribution objective: `loss`, `alpha`, `topk`, `add_tail`

`loss=distribution` uses these per-token losses:

- `alpha=0`: forward KL, $D_{KL}(q_t\|p_t)$.
- `alpha=1`: reverse KL, $D_{KL}(p_t\|q_t)$.
- $0<\alpha<1$: generalized JSD,
  $(1-\alpha)D_{KL}(p_t\|m_t)+\alpha D_{KL}(q_t\|m_t)$,
  where $m_t=(1-\alpha)p_t+\alpha q_t$.

The endpoint values are explicitly special-cased to KL; they are not the limits
of the interior JSD expression. `alpha=0.5` is ordinary equally weighted JSD.

`topk=k` selects the student's top $k$ vocabulary entries at each position and
scores both distributions on those same entries. `add_tail=true` adds one
category with the remaining probability mass, $p_{tail}=1-\sum_{v\in K}p(v)$,
and likewise for the teacher; the tail is summed in log space so it survives a
peaked distribution. Thus top-100 plus tail has 101 categories; it is
not top-1 supervision. With no tail, each selected distribution is renormalized.
`topk=null` retains the full vocabulary.

### Advantage clipping: `advantage_is_clip`

Here "advantage" denotes a teacher/student log-probability difference, not a
normalized task reward. Let $\delta_t(v)=\log p_t(v)-\log q_t(v)$ and let
$\operatorname{sg}$ mean stop-gradient. The clip threshold for update $u$ is
$C_u=c\,\overline{|\delta|}_{<u}$, where $c$ is `advantage_is_clip` and the mean
uses cumulative statistics from prior updates. The first update only collects
statistics and is unclipped. These running statistics are checkpointed.

For **distribution reverse KL** (FSDP and Megatron), the mean is student-probability-weighted
over eligible positions and vocabulary categories. The clipped gradient is the
gradient of

$$\sum_v p_t(v)\,\operatorname{clip}(\operatorname{sg}[\delta_t(v)],-C_u,C_u).$$

The implementation keeps the original KL as the reported forward loss while
substituting this surrogate gradient. Clipping works on the compressed categories
when top-k/tail is enabled. Advantage clipping requires `alpha=1`; JSD and forward
KL are rejected.

For **FSDP `loss=sampled_reverse_kl`**, the implementation uses the sampled-token
surrogate

$$\operatorname{clip}(\operatorname{sg}[\delta_t(y_t)],-C_u,C_u)\log p_t(y_t),$$

with an unweighted historical mean over eligible sampled tokens. The running
statistics are restored on resume only when `loss`, `alpha`, `topk`, `add_tail`
and the temperature match the checkpoint. `null` disables clipping.
Smaller $c$ suppresses extreme distillation gradients
more aggressively; it does not impose a hard KL trust region.

### Teacher entropy weighting: `use_entropy_weighting`, `entropy_beta`

Compute entropy from the teacher's full vocabulary distribution:

$$H_t=-\sum_v q_t(v)\log q_t(v),\qquad a_t=\exp(-\beta H_t).$$

For valid SDPO-token mask $M_t$, normalize across the entire optimizer batch,
including all microbatches and data-parallel ranks:

$$\mu=\frac{\sum_t M_t a_t}{\sum_t M_t},\qquad w_t=a_t/\mu.$$

With `reduction=global_token_mean`, the weighted objective is
$L=\sum_t M_t w_t\ell_t/\sum_t M_t$. The denominator remains the valid-token
count, not the number of microbatches. Weights are detached. Lower-entropy
(confident) teacher positions receive more weight. Increasing $\beta$ sharpens
that preference; $\beta=0$ gives unit weights. This is not an entropy bonus for
the student and does not itself encourage exploration. GRPO-routed tokens are
unaffected. Both backends use batch-wide normalization.

### Rollout importance clipping: `rollout_is_clip`

For the actual sampled token and behavior-policy probability $b_t(y_t)$,

$$r_t=\operatorname{sg}\!\left[\min\left(\exp(\log p_t(y_t)-\log b_t(y_t)),c_{IS}\right)\right].$$

This multiplies the SDPO per-token loss, so the combined objective becomes
$\sum_t M_t w_t r_t\ell_t/\sum_t M_t$. It requires stored rollout log-probabilities.
The cap is one-sided, with no lower clipping bound, and is distinct from PPO's
$[1-\epsilon,1+\epsilon]$ clipping. `null` means no importance correction.

### Teacher updates and prompt contents

`teacher.kind=ema` updates effective teacher weights after an optimizer step:
$\bar\theta_{u+1}=(1-\eta)\bar\theta_u+\eta\theta_{u+1}$, where
$\eta=$ `teacher.update_rate` in $[0,1]$. Zero freezes the initial teacher; one
copies the latest student; 0.01 moves 1% toward it per update. For LoRA, EMA
tracks effective base-plus-adapter weights. `teacher.kind=frozen` is shorthand
for `kind=ema` with `update_rate=0`: the teacher never moves, keeps no FP32
accumulator, and its checkpoints hold no weights. `teacher.kind=self` instead
uses the current student weights with detached scoring and extra teacher context.

- `demonstration_source`: choose expert traces, successful siblings, fallback
  combinations, or `feedback_only` (no demonstration).
- `include_student_response`: include the previous sampled answer in the hint.
- `include_environment_feedback`: include the judge/environment's feedback.
- `remove_thinking`: strip thinking from copied answer demonstrations/hints.
- `train_on_reasoning_tokens`: independently choose whether student thinking
  positions receive loss. False masks those positions but preserves their tokens
  as context for subsequent answer predictions.
- `skip_correct_singletons`: skip already-successful G=1 responses when enabled.
- `use_sample_routing`: FSDP SRPO routes rows between SDPO and GRPO; disabled in
  the Megatron implementation. Entropy weighting is independent of routing.

### Memory and execution controls

`teacher.storage`, `teacher_first`, `megatron_teacher_cache_gib`, and
`loss_chunk_size` control storage, execution order, cache windows, and loss
computation chunking. They do not intentionally change the objective.
The Megatron cache cap is per final-pipeline-stage TP rank; a batch may span
multiple windows, and at least one microbatch is processed even if it exceeds
the cap. With entropy weighting, multiple windows require a scalar-weight
prepass to retain batch-wide normalization. Response-only projection saves
vocabulary-logit memory while the transformer still reads the full context.
