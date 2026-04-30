# VisGym Multi-Image Multi-Turn VLM RL

Multi-turn RL in VisGym — each environment step returns a new image observation, and the model must accumulate visual context across turns to solve the task.

Two recipes are provided:

- **`run_visgym_from_sft.sh`** — starts from an SFT-trained Qwen3-VL checkpoint that emits structured `<observation>/<justification>/<action>` output with tuple actions. Uses a mixed task + format reward.
- **`run_visgym_from_instruct.sh`** — starts from a vanilla `Qwen/Qwen3-VL-8B-Instruct`, uses keyword actions (`<action>left</action>` etc.), task-only reward, KL regularization on.

See [the docs](https://docs.skyrl.ai/docs/examples/visgym) and the [Vision-Language RL tutorial](https://docs.skyrl.ai/docs/tutorials/vision_language_rl) for walkthroughs.

## Quick start

```bash
# Instruct recipe (no SFT checkpoint required)
bash examples/train/visgym/run_visgym_from_instruct.sh

# SFT recipe (set MODEL_PATH to your checkpoint)
MODEL_PATH=/path/to/your/sft_ckpt bash examples/train/visgym/run_visgym_from_sft.sh
```

Both scripts auto-generate the stub dataset on first run. Tested on 1× 8×H100.
