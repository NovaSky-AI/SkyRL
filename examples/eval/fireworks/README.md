# Eval-only generation with Fireworks

Run SkyRL evaluation rollouts against a hosted [Fireworks](https://fireworks.ai) endpoint instead
of a self-hosted vLLM server — no GPUs, no engine startup, no weight loading. The stock
`SkyRLGymGenerator` and eval loop are unchanged; only the inference client is swapped.

## Why

Standing up vLLM is the expensive part of iterating on everything *around* the model: environment
logic, reward functions, chat templates, dataset formatting, eval metrics. For that kind of quick
prototyping you don't need your own server at all — you need an endpoint that behaves like one.
This example points the eval-only entrypoint at Fireworks so you can validate a full
generate → env step → reward → metrics loop from a laptop, then switch to the real vLLM-backed
setup for training with no changes to your environment or data.

Fireworks was chosen specifically (rather than any OpenAI-compatible API) because SkyRL's generator is
token-in/token-out: it sends prompts as token ids and expects the generated token ids back.
Fireworks' `/completions` accepts a pre-tokenized integer-array `prompt` and, with
`return_token_ids=true`, returns the generated `token_ids` — so the tokens SkyRL records are
exactly what the served model consumed and produced, with no re-tokenization drift.

This is generation/eval-only: there is no weight sync to a hosted endpoint, so training still
requires the vllm-backed entrypoints.

## Quickstart

```bash
# 1. Prepare the GSM8K dataset
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

# 2. Set your Fireworks API key
export FIREWORKS_AI_API_KEY=<your_key_here>

# 3. Run eval (defaults: gpt-oss-20b on Fireworks serverless)
bash examples/eval/fireworks/run_eval_fireworks.sh
```

## Configuration conventions

No custom config fields — the example reuses existing knobs:

| Knob | Meaning here |
|---|---|
| `trainer.policy.model.path` | The served model's HF id (e.g. `openai/gpt-oss-20b`). Consumed **only** for the tokenizer in eval-only mode — it must be the served model's tokenizer, since prompts are sent as raw token ids (a mismatch silently degrades generations). |
| `generator.inference_engine.served_model_name` | The Fireworks model id (e.g. `accounts/fireworks/models/gpt-oss-20b`), routed as the request `model`. |
| `FIREWORKS_AI_API_KEY` (env) | API key, sent as `Authorization: Bearer`. |
| `FIREWORKS_BASE_URL` (env, optional) | Server root override (no `/v1` suffix) for self-hosted OpenAI-compatible endpoints. |

## Files

- `fireworks_client.py` — `FireworksInferenceClient`, an `InferenceEngineInterface` implementation
  over the `fireworks-ai` SDK. Converts the vLLM-shaped sampling params the stock eval path emits
  to the subset Fireworks accepts, and no-ops the control plane (weight sync raises).
- `main_eval_fireworks.py` — `FireworksEvalOnlyEntrypoint`, an `EvalOnlyEntrypoint` subclass that
  overrides `get_inference_client()` to build the client above.
- `run_eval_fireworks.sh` — GSM8K launcher (installs the `fireworks` uv extra).
