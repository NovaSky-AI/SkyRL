# DAPO

## Reproduction Runs
| Recipe  | Model  |  AIME24 Pass@32 | AIME24 Avg Score | Hardware  | Training Steps | WandB |
|---|---|---|---|---|---|---|
|  DAPO (w/o Dynamic Sampling) | Qwen/Qwen-2.5-7B-Math  |  0.633 |  -0.304 | 8xH100 | 320 | [Link](https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q) |
|  DAPO (w/o Dynamic Sampling) | Qwen/Qwen3-1.7B |  0.4 |  -0.702 | 8xH100 | 225 | [Link](https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q) |
|  DAPO (w/o Dynamic Sampling) | Qwen/Qwen3-4B  |  0.6 |  -0.51 | 8xH100 | 90 | [Link](https://api.wandb.ai/links/sky-posttraining-uc-berkeley/ijmo1v6q) |

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
  def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
      # apply soft overlong punishment
      overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer.len
      overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer.penalty_factor
      ...
      # use base class impl for metrics and per-token reward conversion
      return super().postprocess_generator_output(generator_output, uids)

class DAPOExp(BasePPOExp):
  def get_trainer(self, *args, **kwargs):
      return DAPOTrainer(*args, **kwargs)

@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = DAPOExp(cfg)
    exp.run()
```

To add the overlong buffer length and penalty factor parameters to the config, you can add the following lines to the run_dapo_gsm8k.sh script:

```bash
+trainer.algorithm.overlong_buffer.len=512 \
+trainer.algorithm.overlong_buffer.penalty_factor=1.0 \
```
