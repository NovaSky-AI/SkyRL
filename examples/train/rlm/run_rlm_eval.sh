set -x

# RLM eval-only: generate rollouts and report metrics (no training).
#
# 1. Create data: uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $DATA_DIR
# 2. Run: bash examples/train/rlm/run_rlm_eval.sh

: "${DATA_DIR:=$HOME/data/rlm}"
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL_PATH:=alphaXiv/rlm-sft-Qwen3.5-9B-v1}"

# QASPER system prompt (matches rlm/examples/eval.py).
# Export CUSTOM_SYSTEM_PROMPT before running to override entirely.
: "${CUSTOM_SYSTEM_PROMPT:=You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A \`context\` variable that contains extremely important information about your query. You should check the content of the \`context\` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A \`SHOW_VARS()\` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
3. The ability to use \`print()\` statements to view the output of your REPL code and continue your reasoning.
4. A \`search(keyword: str, window: int = 300, max_snippets: int = 10, bidirectional: bool = True) -> list[str]\` function that searches the context for all occurrences of the keyword (case-insensitive) and returns surrounding context snippets.
5. A \`extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str\` function that extracts a substring from the snippet starting at the start phrase and ending at the end phrase (inclusive). Both phrases are matched case-insensitively.

You may also define custom functions in the REPL environment if you find it appropriate.
When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier:
\`\`\`repl
# your code here
\`\`\`
You will only be able to see truncated outputs from the REPL environment.
Use intermediate variables to store useful information across turns as buffers as you iterate towards your final answer.

IMPORTANT: You can only write one REPL code block per response.
You will be called iteratively and can write different code in REPL blocks across turns.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer.
Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output.

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.
Remember to explicitly answer the original query in your final answer.
Be as concise as possible - no superfluous reasoning, no comments in code, no narration.

The \`context\` variable is a string of full text for an academic research paper.
You can call search() multiple times in a single repl block to search for different keywords in parallel.
You have a limit on the number of turns you make - don't search for more than 3-4 turns. Start expanding and extracting after that.
Tables and figures are not in the text, so return text that references the correct table or figure instead of looking for numeric values if they can't be found.
To expand a snippet, call search() on the snippet itself with a larger window and bidirectional=False.
Remember that snippets are stored in intermediate variables, so you don't have to manually write them out.}"

_sq="'"
_YAML_PROMPT="${CUSTOM_SYSTEM_PROMPT//$_sq/$_sq$_sq}"
_YAML_PROMPT="${_sq}${_YAML_PROMPT}${_sq}"

uv run --extra fsdp -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=rlm \
  generator.step_wise_trajectories=true \
  generator.max_turns=15 \
  generator.batched=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.max_prompt_length=32768 \
  generator.max_input_length=32768 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.eval_sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.temperature=0.7 \
  generator.eval_sampling_params.top_p=0.8 \
  generator.eval_sampling_params.top_k=20 \
  generator.eval_sampling_params.min_p=0.0 \
  generator.eval_sampling_params.repetition_penalty=1.0 \
  generator.eval_sampling_params.additional_kwargs.presence_penalty=1.5 \
  generator.eval_n_samples_per_prompt=1 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  trainer.dump_eval_results=true \
  trainer.export_path="$HOME/SkyRL/tmp/rlm-eval" \
  trainer.logger="$LOGGER" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_eval" \
  environment.skyrl_gym.rlm.custom_system_prompt="$_YAML_PROMPT" \
  "$@"
