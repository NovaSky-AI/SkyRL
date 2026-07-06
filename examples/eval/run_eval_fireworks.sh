set -x

# Evaluation-only generation for GSM8K against the external Fireworks endpoint
# (generator.inference_engine.backend=fireworks). No local inference engines and
# no vLLM: prompts are sent as token ids and Fireworks returns the generated
# token ids (return_token_ids), so the stock generator works unchanged.
#
# trainer.policy.model.path must be the served model's tokenizer (token ids are
# sent raw), and served_model_name is the Fireworks model id.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export FIREWORKS_AI_API_KEY=<your_key_here>
# bash examples/eval/run_eval_fireworks.sh

: "${FIREWORKS_AI_API_KEY:?export FIREWORKS_AI_API_KEY first}"

DATA_DIR="$HOME/data/gsm8k"
TOKENIZER="openai/gpt-oss-20b"
FW_MODEL="accounts/fireworks/models/gpt-oss-20b"
LOGGER="console"

uv run --isolated --extra fireworks \
  -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="$TOKENIZER" \
  trainer.logger="$LOGGER" \
  trainer.placement.colocate_all=false \
  generator.inference_engine.backend=fireworks \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.served_model_name="$FW_MODEL" \
  generator.inference_engine.api_key="$FIREWORKS_AI_API_KEY" \
  generator.sampling_params.logprobs=null \
  generator.eval_sampling_params.max_generate_length=2048 \
  generator.eval_sampling_params.temperature=0.7 \
  environment.env_class=gsm8k \
  "$@"
