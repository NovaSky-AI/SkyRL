# LLM-as-a-Judge with Local vLLM Reward Model

This example demonstrates using a **locally-hosted vLLM reward model** (LLM-as-a-Judge) for GRPO training on GSM8K, without requiring any changes to SkyRL core.

Instead of calling an external API (e.g., OpenAI) for reward scoring, it uses a `FrozenRewardInferenceClient` — a subclass of `InferenceEngineClient` that creates vLLM engines without weight-sync capabilities (since the reward model never changes). The reward engine is exposed as a named Ray actor, so environments discover it automatically with no HTTP servers, no port conflicts, and no stale subprocesses.

## Quick Start

```bash
cd skyrl-train

# 1. Prepare dataset
uv run examples/llm_as_a_judge_local/gsm8k_dataset_local.py \
    --output_dir ~/data/gsm8k_llm_judge_local

# 2. Prepare rule-based dataset (for comparison runs)
uv run examples/gsm8k/gsm8k_dataset.py \
    --output_dir ~/data/gsm8k

# 3. Run training (pick one)
bash examples/llm_as_a_judge_local/run_llm_judge_local.sh        # sync + LLM judge
bash examples/llm_as_a_judge_local/run_rule_based.sh              # sync + rule-based
bash examples/llm_as_a_judge_local/run_llm_judge_local_async.sh   # async + LLM judge
bash examples/llm_as_a_judge_local/run_rule_based_async.sh        # async + rule-based
```

## Four Configurations

This example includes four launch scripts that share **identical hyperparameters** (model, learning rate, batch size, group size) so the only variables are *reward mechanism* and *training mode*:

| Script | Reward | Training | GPUs | Entrypoint |
|--------|--------|----------|------|------------|
| `run_rule_based.sh` | Rule-based (string match) | Sync | 1 | `skyrl_train.entrypoints.main_base` |
| `run_llm_judge_local.sh` | LLM judge (Qwen2.5-1.5B-Instruct) | Sync | 2 | `main_llm_judge_local.py` |
| `run_rule_based_async.sh` | Rule-based (string match) | Async | 2 | `examples.async.main_async` |
| `run_llm_judge_local_async.sh` | LLM judge (Qwen2.5-1.5B-Instruct) | Async | 3 | `main_llm_judge_local_async.py` |

### GPU Layouts

```
Sync Rule-Based (1 GPU):           Sync LLM Judge (2 GPUs):
┌────────────────┐                ┌────────────────┐ ┌────────────────┐
│ Policy (0.5B)  │                │ Policy (0.5B)  │ │ Reward (1.5B)  │
│ + vLLM Infer.  │                │ + vLLM Infer.  │ │ Frozen vLLM    │
│ (sleep/wake)   │                │ (sleep/wake)   │ │ (always active) │
│ Reward: string │                │ Reward: →GPU 2 │ │ ← scores here  │
└────────────────┘                └────────────────┘ └────────────────┘

Async Rule-Based (2 GPUs):         Async LLM Judge (3 GPUs):
┌────────┐ ┌────────┐            ┌────────┐ ┌────────┐ ┌────────┐
│ Policy │ │ vLLM   │            │ Policy │ │ vLLM   │ │ Reward │
│ (train)│ │ (gen)  │            │ (train)│ │ (gen)  │ │ (1.5B) │
│ ◄─sync─► │       │            │ ◄─sync─► │       │ │ Frozen │
└────────┘ └────────┘            └────────┘ └────────┘ └────────┘
Gen runs 1 step ahead             Gen runs 1 step ahead
```

## Experimental Results

All experiments use **Qwen2.5-0.5B-Instruct** as the policy model, GRPO with group size 4, batch size 16, and learning rate 1e-6 on NVIDIA L4 GPUs.

### Throughput

| Metric | Sync Rule-Based | Sync LLM Judge | Async Rule-Based | Async LLM Judge |
|--------|:-:|:-:|:-:|:-:|
| **Generate** | 10.7s | 18.0s | 11.3s | 20.3s |
| **Train** | 4.7s | 5.3s | 4.8s | 5.1s |
| **Weight sync** | 1.2s | 1.2s | 1.9s | 1.9s |
| **Step time** | **21.9s** | **30.6s** | **13.8s** | **22.1s** |
| **GPUs** | 1 | 2 | 2 | 3 |

- **Async speedup:** 37% for rule-based (21.9→13.8s), 28% for LLM judge (30.6→22.1s).
- **LLM judge overhead:** ~40% slower per step than rule-based (reward model inference dominates).

### Reward Trajectories

```
         ── Sync ──────────────    ── Async ─────────────
Step    Rule-Based  LLM Judge    Rule-Based  LLM Judge
  1      0.016       0.766        0.000       0.766
  5      0.031       0.859        0.016       0.859
 10      0.063       0.859        0.094       0.922
 20      0.188       0.875        0.141       0.750
 30      0.328       0.953        0.422       0.922
 40      0.453       0.953        0.406       0.859
 50      0.594       1.000         —          0.875
```

**Key findings:**
- **Rule-based starts near zero** because the model must learn both correctness *and* the `#### <number>` format. This high variance produces strong GRPO gradients.
- **LLM judge starts at ~0.77** because the reward model recognizes correct answers regardless of formatting — the model already knows the math.
- **Sync and async learning curves are nearly identical**, validating that single-step weight staleness is benign for GRPO with small learning rates.

## Architecture

```
main_llm_judge_local.py
  ├── start_reward_service()           # Spawns RewardInferenceService actor
  │     └── RewardInferenceService     # Ray actor (reward_inference.py)
  │           └── FrozenRewardInferenceClient
  │                 └── create_frozen_vllm_engines()  # No weight sync
  ├── register("llm_as_a_judge_local") # Points to llm_judge_local_env.py
  └── BasePPOExp.run()                 # Standard SkyRL training loop
        └── GSM8kLLMJudgeLocalEnv.step()
              └── ray.get_actor("reward_inference_service").score()
```

### Key Design Decisions

1. **Self-contained** — No modifications to SkyRL core. `create_frozen_vllm_engines()` duplicates the relevant engine-creation logic without the weight-sync `worker_extension_cls`.
2. **Ray actor for reward** — Environments discover the reward service by name (`ray.get_actor("reward_inference_service")`). No HTTP, no port management.
3. **Scalable** — Increase `REWARD_NUM_ENGINES` to add more frozen vLLM engines; load balancing is automatic via `InferenceEngineClient`.

## Files

| File | Lines | Description |
|------|:-----:|-------------|
| `reward_inference.py` | 447 | `FrozenRewardInferenceClient` + `RewardInferenceService` |
| `main_llm_judge_local.py` | 215 | Sync entrypoint (starts reward service → registers env → trains) |
| `main_llm_judge_local_async.py` | 120 | Async entrypoint (uses `AsyncRayPPOTrainer`) |
| `llm_judge_local_env.py` | 156 | Environment: prompts reward model, parses score |
| `gsm8k_dataset_local.py` | 95 | Dataset preparation |
| `run_llm_judge_local.sh` | 101 | Launch: sync + LLM judge (2 GPUs) |
| `run_llm_judge_local_async.sh` | 92 | Launch: async + LLM judge (3 GPUs) |
| `run_rule_based.sh` | 78 | Launch: sync + rule-based (1 GPU) |
| `run_rule_based_async.sh` | 76 | Launch: async + rule-based (2 GPUs) |
