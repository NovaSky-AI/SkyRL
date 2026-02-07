# SkyRL-Train Examples
Welcome to the SkyRL-Train examples! In this folder you can find the following examples.

## Algorithms

- `algorithms/`: Examples for how to configure and run RL with various algorithms and policy-loss variants (e.g., DAPO, SAPO, GRPO, CISPO, GSPO, or your own custom advantage estimators and custom policy losses).
- `ppo/`: Vanilla PPO training (with a critic, ref, and policy model)
- `on_policy_distillation/`: [On-policy distillation recipe](https://novasky-ai.notion.site/on-policy-distillation) that uses a teacher model to provide dense token-level rewards during training, reproducing results from the [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/).
- `tis_correction/`: Applying [Flash-RL TIS](https://fengyao.notion.site/off-policy-rl) correction to improve off-policy stability.
- `turn_level_rewards/`: GSM8K multi-turn environment illustrating turn-level rewards and custom advantage estimators.

## Async RL

- `async/`: One-step off-policy GRPO with an asynchronous generator–trainer loop.
- `fully_async/`: Fully asynchronous (PipelineRL/AReal-style) GRPO training with in-flight weight updates. [See docs for full design + details](https://docs.skyrl.ai/docs/tutorials/one_step_off_async).

## Tasks

- `gsm8k/`: Basic GSM8K math word-problem dataset utilities and baseline training/generation scripts.
- `llm_as_a_judge/`: GSM8K training with an external LLM as a judge to produce rewards instead of strict exact-match grading.
- `multiply/`: Toy arithmetic environment for multiplying numbers, useful for quick sanity checks and debugging.
- `livecodebench/`: LiveCodeBench code-generation task setup and training scripts.
- `text_to_sql/`: [Text-to-SQL (SkyRL-SQL)](https://docs.skyrl.ai/docs/examples/multi_turn_text2sql) environment and training scripts for mapping natural language questions to SQL queries.
- `step_wise/`: Step-wise training for chat-template agnostic multi-turn RL training.
- `search/`: Multi-turn search agent training with the SearchR1 dataset, backed by a FAISS-based retriever server.

## Integrations

- `flash_rl/`: Integration with [FlashRL’s](https://fengyao.notion.site/flash-rl) patched vLLM inference engine for high-throughput RL training.
- `terminal_bench/`: Custom [Terminal-Bench/Harbor](https://harborframework.com/) Generator for training agents to solve command-line tasks.
- `mini_swe_agent/`: Integration with [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) to train coding agents on SWE-Bench via SkyRL.
- `../integrations/verifiers/`: Integration with PrimeIntellect's [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers) + [Environments Hub](https://app.primeintellect.ai/dashboard/environments?_gl=1*1vogwn8*_gcl_au*NjA1ODI2MTMxLjE3NjczOTkwMTM)
- `../integrations/openenv/`: Integration with HuggingFace/Meta [OpenEnv](https://github.com/meta-pytorch/OpenEnv)

## Large Scale Model Training
- `megatron/`: Examples for running SkyRL with the Megatron Backend for 5D parallelism.
- `moe/`: Work-in-progress MoE training example used for development and testing large-scale multi-node Mixture-of-Experts support.
- `gptoss/`: Training example for the GPT-OSS-20B model using patched attention to support attention sinks.

## Features and More
- `lora/`: LoRA RL fine-tuning recipes.
- `remote_inference_engine/`: Scripts for running remote vLLM/sglang inference servers and connecting them to SkyRL.
- `training_backends/`: Runner scripts demonstrating how to use different training backends on SkyRL.
