# DAPO

## Reproduction Runs
| Recipe  | Model  |  AIME24 Pass@32 | AIME24 Avg Score | Hardware  | Training Steps | WandB |
|---|---|---|---|---|---|---|
|  DAPO (w/o Dynamic Sampling) | Qwen/Qwen-2.5-7B-Math  |  0.633 |  -0.304 | 8xH100 | 320 | [Link](https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q) |
|  DAPO (w/o Dynamic Sampling) | Qwen/Qwen3-1.7B |  0.4 |  -0.702 | 8xH100 | 225 | [Link](https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q) |
|  DAPO (w/o Dynamic Sampling) | Qwen/Qwen3-4B  |  0.6 |  -0.51 | 8xH100 | 90 | [Link](https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q) |

## Data
`prepare_dapo_data.sh` downloads the training set and the AIME eval sets into `$DATA_DIR` (default `~/data/dapo`) and writes the deduplicated `*-cleaned.parquet` files used by the run scripts. DAPO-Math-17k and AIME-2024 are processed by `data_preprocess_dapo_aime.py`; AIME-2025 is processed by `data_preprocess_aime2025.py` into its own file:

```bash
DATA_DIR=$HOME/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
```

| File | Source | `data_source` | Problems |
|---|---|---|---|
| `dapo-math-17k-cleaned.parquet` | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | `math_dapo` | train |
| `aime-2024-cleaned.parquet` | [BytedTsinghua-SIA/AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024) | `math_dapo` | 30 |
| `aime-2025-cleaned.parquet` | [zhuzilin/aime-2025](https://huggingface.co/datasets/zhuzilin/aime-2025) (pinned revision) | `aime-2025` | 30 (AIME 2025 I + II) |

AIME-2025 is optional for eval: the DAPO training scripts evaluate only on `aime-2024-cleaned.parquet` unless you add `aime-2025-cleaned.parquet` to `data.val_data` (see below). To prepare it on its own, download the pinned `aime-2025.jsonl` (the `TEST_FILE_2025` URL in `prepare_dapo_data.sh`) and run:

```bash
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.data_preprocess_aime2025 --data-dir $DATA_DIR --input-file $DATA_DIR/aime-2025.jsonl
```

`data_preprocess_aime2025.py` converts AIME-2025 to the same schema, prompt template (`... Answer: $Answer ...`), and `rule-lighteval/MATH_v2` reward as AIME-2024, so it is graded by the same `aime` environment. All labels are integers in [0, 999] (the upstream `336^\circ` label is normalized to `336`). Rows are in problem order and `extra_info.index` is the problem position: 0-14 are AIME 2025 I problems 1-15, and 15-29 are AIME 2025 II problems 1-15.

## Evaluation on AIME-2024 and AIME-2025
AIME-2024 keeps `data_source=math_dapo`, so existing metric names are unchanged; AIME-2025 uses `aime-2025`. To include AIME-2025 in eval, add its file to `data.val_data`. Passing both files as eval data reports each one separately, under `eval/math_dapo/...` and `eval/aime-2025/...`, plus the combined `eval/all/...`:

```bash
data.val_data="['$DATA_DIR/aime-2024-cleaned.parquet', '$DATA_DIR/aime-2025-cleaned.parquet']"
```

This works for the eval during training in the run scripts above and for eval-only runs. `run_generation_aime.sh` evaluates a model without training through the eval-only entrypoint `skyrl.train.entrypoints.main_generate`, using DeepSeek-R1-Distill-Qwen-1.5B with MathArena's sampling settings (temperature 0.6, top_p 0.95, 32000 max generated tokens, 16 samples per problem):

```bash
bash examples/train/algorithms/dapo/run_generation_aime.sh
# AIME-2024 and AIME-2025 together:
bash examples/train/algorithms/dapo/run_generation_aime.sh \
  data.val_data="['$HOME/data/dapo/aime-2024-cleaned.parquet', '$HOME/data/dapo/aime-2025-cleaned.parquet']"
```

The `aime` reward is +1 for a correct answer and -1 otherwise. With `generator.eval_n_samples_per_prompt=k`, each data source reports:

| Metric | Meaning for the `aime` env |
|---|---|
| `eval/<data_source>/mean_positive_reward` | avg@k accuracy: the fraction of correct samples |
| `eval/<data_source>/avg_score` | mean reward, `2 * accuracy - 1` (so accuracy = `(avg_score + 1) / 2`) |
| `eval/<data_source>/pass_at_k` | pass@k: the fraction of problems with at least one correct sample |

The `aime` env grades the last `Answer: X` line of the response. A response that puts its final answer only in `\boxed{}`, with no `Answer:` line, is scored as incorrect, which lowers scores for models that box their answers. For example, DeepSeek-R1-Distill-Qwen-1.5B on AIME-2025 with `run_generation_aime.sh` scores `mean_positive_reward=0.125` (avg@16 of 12.5%) against MathArena's 20.0%; the gap comes mostly from correct answers that were only boxed.

## DAPO Settings
The DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) algorithm consists of the following components on top of a GRPO baseline:

1. Clip-Higher: Promotes the diversity of the system and avoids entropy collapse;

2. Dynamic Sampling: Improves training efficiency and stability;

3. Token-Level Policy Gradient Loss: Critical in long-CoT RL scenarios;

4. Overlong Reward Shaping: Reduces reward noise and stabilizes training.

In this README, we walk through how to enable each of these components in SkyRL. 

### Clip-Higher
To use clip-higher, you can simply configure `trainer.algorithm.eps_clip_high` separately from `trainer.algorithm.eps_clip_low`.

```bash
trainer:
  algorithm:
    eps_clip_low: 0.2
    eps_clip_high: 0.28
```

### Dynamic Sampling
In DAPO style dynamic sampling, we sample rollouts until we have a full batch with non-zero advantages (meaning that we have a non-zero std deviation of rewards for the n rollouts for a given prompt).

To configure DAPO style dynamic sampling, you can set `trainer.algorithm.dynamic_sampling.type` to `filter` and configure `trainer.algorithm.dynamic_sampling.max_sample_batches` to the maximum number of batches to sample. If `max_sample_batches > 0` and is exceeded, SkyRL-Train will raise an error. If `max_sample_batches <= 0`, SkyRL-Train will sample until a full batch with non-zero advantages is accumulated.

```bash
trainer:
  algorithm:
    dynamic_sampling:
      type: filter
      max_sample_batches: 30
```

### Token-Level Policy Gradient Loss
DAPO uses token-level policy gradient loss, which can be enabled by setting `trainer.algorithm.loss_reduction` to `token_mean`. This is the default setting in SkyRL-Train.

```bash
trainer:
  algorithm:
    loss_reduction: "token_mean"
```

### Overlong Reward Shaping
The DAPO paper proposes two methods for overlong reward shaping:

- **Overlong Filtering**: Sets loss mask to be all zeros for responses that exceed the max response length.

- **Soft Overlong Punishment**: Penalizes responses that exceed the max response length within a punishment interval. Within this interval, the longer the response, the greater the punishment it receives. This penalty is added to the original reward.

#### Overlong Filtering
To enable overlong filtering, which sets the loss mask to be all zeros for responses that do not finish with a stop token (i.e. responses that are too long), you can set `generator.apply_overlong_filtering` to `true`.

```bash
generator:
  apply_overlong_filtering: true
```

#### Soft Overlong Punishment
To enable soft overlong punishment, you can create a custom trainer class and override the `RayPPOTrainer` `postprocess_generator_output` method to additionally apply soft overlong punishment to rewards. We provide an example of this in `main_dapo.py`, and show an overview of the implementation below:


```python
class DAPOTrainer(RayPPOTrainer):
  @torch.no_grad()
  def postprocess_generator_output(
      self, generator_output: GeneratorOutput, uids: List[str]
  ) -> Tuple[GeneratorOutput, List[str]]:
      # apply soft overlong punishment
      overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer_len
      overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer_penalty_factor
      ...
      # use base class impl for metrics and per-token reward conversion
      return super().postprocess_generator_output(generator_output, uids)

class DAPOExp(BasePPOExp):
  def get_trainer(self, *args, **kwargs):
      return DAPOTrainer(*args, **kwargs)

@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = DAPOExp(cfg)
    exp.run()
```

`main_dapo.py` handles both reward formats. For sequence-level rewards (`List[float]`, e.g. from batched generation), the penalty is subtracted from each reward. For per-token rewards (`List[List[float]]`, e.g. from non-batched generation with `generator.batched=false`, which is required to pass `generator.chat_template_kwargs`), the penalty is subtracted from the last token's reward. Responses longer than `generator.sampling_params.max_generate_length` get a reward of 0. `main_dapo.py` does not support `generator.step_wise_trajectories=true`.

To add the overlong buffer length and penalty factor parameters to the config, you can subclass ``AlgorithmConfig`` with additional fields and use ``make_config`` to create a custom config class. See ``main_dapo.py`` for the full example:

```bash
trainer.algorithm.overlong_buffer_len=512 \
trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
```
