# ECHO Terminal-Agent Training

This example trains terminal agents with ECHO, an environment cross-entropy hybrid objective. ECHO combines standard policy-gradient RL with an auxiliary cross-entropy loss on terminal-output tokens observed in the same rollout.

SkyRL provides the core RL training stack, distributed worker execution, and vLLM-backed inference. This example adds the terminal-agent dataset loader, prompt formatting, tool-call parsing, Harbor-backed environment execution, rollout construction, token masks, and the optional environment-prediction loss.

## Structure

```text
examples/train_integrations/echo_terminal/
  entrypoint.py                         # Training entrypoint
  generator.py                          # Terminal rollout loop and SkyRL trajectory construction
  dataset.py                            # Parquet dataset loader and prompt tokenization
  harbor_environment.py                 # Harbor container execution wrapper
  interaction.py                        # Rollout transcript and token-mask bookkeeping
  parsers.py                            # Tool-call parsing
  prompts.py                            # Terminal-agent system prompts
  tools.py                              # Tool schemas
  chat_template.py                      # Chat-template loading helpers
  chat_templates/qwen3_xml_tool_calling.jinja
  world_modeling/
    config.py                           # ECHO config extensions
    fsdp_worker.py                      # FSDP auxiliary-loss hook implementation
    loss.py                             # Environment-token CE loss
    trainer.py                          # Training-batch conversion for ECHO masks
  configs/
    qwen3_8b_rl.yaml                    # Vanilla GRPO baseline
    qwen3_8b_rl_wm05.yaml               # GRPO + ECHO loss, lambda=0.05
```

## Quick Start

Install SkyRL with the FSDP and Harbor dependencies:

```bash
cd SkyRL
pip install -e ".[fsdp,harbor]"
```

Edit the train and validation parquet paths in the config you want to run:

```yaml
data:
  train_data:
    - name: terminal_agent_train
      path: /path/to/train.parquet
  val_data:
    - name: terminal_agent_train
      path: /path/to/val.parquet
```

Set an output directory and launch the vanilla GRPO baseline:

```bash
export OUTPUT_DIR=/path/to/outputs/qwen3_8b_rl
export CONFIG_PATH=examples/train_integrations/echo_terminal/configs/qwen3_8b_rl.yaml
bash examples/train_integrations/echo_terminal/run_echo_terminal.sh
```

Launch ECHO with the auxiliary environment-prediction loss:

```bash
export OUTPUT_DIR=/path/to/outputs/qwen3_8b_rl_wm05
export CONFIG_PATH=examples/train_integrations/echo_terminal/configs/qwen3_8b_rl_wm05.yaml
bash examples/train_integrations/echo_terminal/run_echo_terminal.sh
```

Checkpoints are written to `${OUTPUT_DIR}/ckpts`, and SkyRL logs are written to `${OUTPUT_DIR}/skyrl_logs`.

## Design

The rollout loop is handled directly in this example rather than through Harbor's full rollout API. Harbor is used as the terminal task backend: it starts the task containers, runs shell commands, returns terminal observations, and executes verifiers. SkyRL/vLLM owns model generation so the training code has direct, batched access to generated token ids, logprobs, attention masks, sampling controls, and ECHO-specific token masks.

During training, the standard GRPO loss is computed on model-generated action tokens. When `trainer.algorithm.world_model_coeff > 0`, ECHO also computes cross entropy on selected terminal-output tokens from the same trajectory:

```text
L = L_GRPO(action tokens) + world_model_coeff * CE(terminal-output tokens)
```

Setting `world_model_coeff: 0.0` recovers the vanilla GRPO baseline. The included ECHO config uses `world_model_coeff: 0.05` and `generator.world_loss_target: env_only`, which trains on terminal environment-output tokens while leaving the RL action-token mask unchanged.
