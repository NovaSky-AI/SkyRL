# ScaleRL

This recipe packages the remaining Issue #495 ingredients into one standard SkyRL training run:

- `trainer.algorithm.policy_loss_type=cispo`
- `trainer.algorithm.zero_variance_filter=true`
- `trainer.algorithm.loss_reduction=prompt_mean`
- `trainer.algorithm.adaptive_prompt_filtering.enabled=true`
- `trainer.policy.model.upcast_logits_to_fp32=true`

The example script is:

- `examples/train/algorithms/scalerl/run_scalerl_gsm8k.sh`

## V1 Constraints

- `prompt_mean` is only supported for non-step-wise training.
- Adaptive prompt filtering is only supported in the standard trainer, not fully async training.
- Adaptive prompt filtering uses prompt pass rate, where a sampled response counts as positive when its scalar reward is greater than `0`.
- Filtering happens only at epoch boundaries and keeps a minimum active prompt floor via `min_active_prompts` / `min_active_ratio`.

## Launch

```bash
export WANDB_API_KEY=your_wandb_api_key
bash examples/train/algorithms/scalerl/run_scalerl_gsm8k.sh
```
