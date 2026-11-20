# Self-Distillation Policy Optimization (SDPO)

This example implements [SDPO](https://arxiv.org/abs/2601.20802) (Hübotter et al., 2026; original code [lasgroup/SDPO](https://github.com/lasgroup/SDPO)) and [SRPO](https://arxiv.org/abs/2604.02288) (Li et al., 2026).

For each prompt, the student samples a group of responses. The teacher sees one successful response from that group (a sibling), and can also see environment feedback. It scores the sampled responses. The student is trained to match those scores with a top-k distillation loss: forward KL, reverse KL, or JSD.

SRPO uses a different update for each outcome where correct responses go through GRPO. Failed responses that have a teacher hint go through an entropy-weighted SDPO update.

## Layout

- `main_sdpo.py`: SkyRL endpoint and defines`SDPOGenerator`, `SDPOTrainer`, and `SDPOExp`
- `sdpo_config.py`: defines and validates `trainer.distillation` config and instantiates them
- `fsdp_worker.py`: handles FSDP worker scoring, the distillation loss, and SRPO routing
- `teachers.py`: implements the `self`, `ema`, and `frozen` teachers as separate models
- `megatron/`: the FSDP worker, teacher, and loss do not apply here. Each rank holds one vocabulary shard, and pipeline stages pass hidden states rather than logits. Prompt building and routing masks still come from `utils.py`.
  - `megatron_worker.py`: runs the student and teacher through Megatron's pipeline schedule. Only the last stage keeps response logits, and a CPU teacher is offloaded before the student step. `fsdp_worker.py` cannot do this: it calls one Hugging Face module. Sample routing stays there.
  - `loss.py`: the same top-k, tail, and JSD objective as `utils.sdpo_loss`, on each rank's vocabulary shard. Log-softmax, top-k, and the tail are reduced across the tensor-parallel group, so the full vocabulary is never gathered onto one GPU.
  - `teacher.py`: EMA or frozen copy of the Megatron weights. LoRA is merged before the average, because averaging the adapters is not the same as averaging the merged weight. `teachers.py` only handles FSDP shards and PEFT.
  - `response_logits.py`: keeps response positions before the vocabulary projection. Megatron has no Hugging Face `logits_to_keep`. Sequence-parallel hidden states are gathered, sliced, and sharded again so the output layer's gradients stay valid.
- `utils.py`: builds teacher prompts, paired inputs, routing masks, and the loss
- `reproduction/run_sdpo.sh` and `run_srpo.sh`: scripts for reproducing across SciKnowEval `biology`, `chemistry`, `material`, and `physics`, and ToolAlpaca `tooluse`

## Quickstart

```bash
git clone https://github.com/lasgroup/SDPO $HOME/SDPO
uv run --isolated --extra fsdp -m examples.train.sdpo.reproduction.prepare_data \
  --sdpo-repo $HOME/SDPO --output-dir $HOME/data/sdpo

export WANDB_API_KEY=<your_key_here>
bash examples/train/sdpo/reproduction/run_sdpo.sh chemistry   # EMA teacher, JSD, top-100 + tail
bash examples/train/sdpo/reproduction/run_srpo.sh chemistry   # same recipe, with sample routing
```

Both scripts use 4 GPUs: 2 for FSDP and 2 for vLLM. The default model is `Qwen/Qwen3-8B` with thinking disabled. This setup is sized for 141 GB GPUs. On 80 GB GPUs, raise `NUM_POLICY_GPUS` so the model and the EMA teacher fit.

Each step uses 32 prompts and 8 responses per prompt. The learning rate is `1e-5`, the EMA update rate is `0.05`, and the rollout importance-sampling cap is `2.0`. Responses can be up to 8192 tokens. Every 5 steps, the run evaluates avg@16 at temperature 0.6 and top-p 0.95.

Training runs for up to 30 epochs. The paper reports the best avg@16 within 1 hour and within 5 hours, so compare `eval/all/avg_score` at the same wall-clock time. Arguments after the task name are config overrides.

```bash
# Olmo-3-7B. On 80 GB GPUs, set NUM_POLICY_GPUS=4 so the model and the EMA teacher fit.
MODEL=allenai/Olmo-3-7B-Instruct NUM_POLICY_GPUS=4 \
  bash examples/train/sdpo/reproduction/run_sdpo.sh physics

# alpha=0 is forward KL, alpha=1 is reverse KL, and 0 < alpha < 1 is generalized JSD (0.5 is symmetric).
# topk is the student's top-k, and the teacher uses the same set. add_tail keeps the remaining probability as one bin.
# topk=null keeps the full vocabulary. advantage_is_clip requires alpha=1.
bash examples/train/sdpo/reproduction/run_sdpo.sh chemistry \
  trainer.distillation.alpha=1 \
  trainer.distillation.topk=null \
  trainer.distillation.add_tail=false \
  trainer.distillation.teacher.kind=frozen \
  trainer.distillation.rollout_is_clip=2.0 \
  trainer.distillation.advantage_is_clip=3.0

# teacher.kind is ema (set teacher.update_rate in [0, 1]), frozen, or self.
# demonstration_source is sibling, expert_only, expert_then_sibling, sibling_then_expert, or feedback_only.
bash examples/train/sdpo/reproduction/run_sdpo.sh tooluse \
  trainer.distillation.teacher.kind=self \
  trainer.distillation.demonstration_source=feedback_only \
  trainer.distillation.include_environment_feedback=true \
  trainer.distillation.include_student_response=true \
  trainer.distillation.remove_thinking=true \
  trainer.distillation.train_on_reasoning_tokens=false

# What run_srpo.sh adds to the SDPO recipe. entropy_beta=0 uses uniform weights.
bash examples/train/sdpo/reproduction/run_sdpo.sh chemistry \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.distillation.use_sample_routing=true \
  trainer.distillation.use_entropy_weighting=true \
  trainer.distillation.entropy_beta=1.0

# Stop after 2 steps.
bash examples/train/sdpo/reproduction/run_sdpo.sh physics trainer.max_training_steps=2
```

The scripts above pass `--extra fsdp`. To run the same module with Megatron, pass `--extra megatron` and set `trainer.strategy=megatron`.

Megatron supports dense models with tensor and sequence parallelism. Pipeline parallelism can be 1–3. Context parallelism and expert parallelism must stay at 1. Use `loss=distribution` and `reduction=global_token_mean`. Sample routing is only available on the FSDP worker.

With the settings below, the teacher is stored on CPU, scored once, and offloaded before the student step:

```text
trainer.strategy=megatron
trainer.policy.megatron_config.tensor_model_parallel_size=2
trainer.policy.megatron_config.pipeline_model_parallel_size=1
trainer.policy.megatron_config.context_parallel_size=1
trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=false
trainer.distillation.limit_response_logits=false
trainer.distillation.reduction=global_token_mean
trainer.distillation.teacher.storage=cpu
trainer.distillation.teacher_first=true
trainer.distillation.megatron_teacher_cache_gib=128
```

