"""
GPU integration tests for TinkerTrainingAdapter.

Tests the full training path: TinkerTrainingAdapter → WorkerDispatch → Workers → Loss computation

# Run tests:
uv run --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_tinker_training_adapter_integration.py -m "vllm" -v
"""

import pytest
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.training.tinker_adapter import TinkerTrainingAdapter
from skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl_train.workers.ppo_ray_actor_group import PPORayActorGroup


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        cfg.trainer.run_name = "tinker_training_test"
        # Use token_mean for loss_reduction (standard SkyRL setting)
        cfg.trainer.algorithm.loss_reduction = "token_mean"
        cfg.trainer.algorithm.use_kl_loss = False  # Disable KL for supervised tests
        cfg.trainer.algorithm.use_entropy_loss = False  # Disable entropy for clarity
        return cfg


def init_worker_dispatch(cfg: DictConfig, tp_size: int = 1) -> WorkerDispatch:
    """Initialize a minimal WorkerDispatch for testing."""
    from skyrl_train.workers.fsdp.worker import PolicyFSDPWorker

    # Create policy actor group with single worker
    policy_actor_group = PPORayActorGroup.remote(
        worker_class=PolicyFSDPWorker,
        num_nodes=1,
        num_gpus_per_node=tp_size,
        cfg=cfg,
        model_name="policy",
    )

    # Initialize WorkerDispatch
    worker_dispatch = WorkerDispatch(
        cfg=cfg,
        policy_actor_group=policy_actor_group,
    )

    return worker_dispatch


@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, marks=pytest.mark.vllm),
    ],
    ids=["tp1"],
)
def test_tinker_adapter_cross_entropy_forward_backward(ray_init_fixture, tp_size: int):
    """Test TinkerTrainingAdapter with cross_entropy loss through real workers."""
    cfg = get_test_config()
    cfg.trainer.algorithm.policy_loss_type = "cross_entropy"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Initialize worker dispatch
    worker_dispatch = init_worker_dispatch(cfg, tp_size)

    # Create adapter
    adapter = TinkerTrainingAdapter(worker_dispatch)

    # Create Tinker-style training data
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    # Extend with a simple completion
    response_tokens = tokenizer.encode(" The answer is 4.", add_special_tokens=False)
    full_tokens = prompt_tokens + response_tokens

    # Create datum with cross-entropy inputs
    data = [
        {
            "model_input": {"tokens": full_tokens},
            "loss_fn_inputs": {
                "target_tokens": full_tokens[1:] + [tokenizer.eos_token_id],  # Shifted targets
                "weights": [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens),  # Only train on response
            },
        },
    ]

    # Call forward_backward
    import asyncio

    async def run_forward_backward():
        return await adapter.forward_backward(
            data=data,
            loss_fn="cross_entropy",
        )

    result = asyncio.run(run_forward_backward())

    # Verify result structure
    assert "metrics" in result.to_dict(), "Result should have metrics"
    assert "loss_fn_outputs" in result.to_dict(), "Result should have loss_fn_outputs"

    metrics = result.metrics
    assert "final_loss" in metrics, "Should have final_loss metric"
    assert "policy_loss" in metrics, "Should have policy_loss metric"

    # Verify loss is reasonable (should be positive for cross-entropy)
    assert metrics["final_loss"] > 0, f"Cross-entropy loss should be positive, got {metrics['final_loss']}"
    assert metrics["policy_loss"] > 0, f"Policy loss should be positive, got {metrics['policy_loss']}"

    # Verify clip_ratio is 0 for cross-entropy (no clipping)
    assert metrics.get("ppo_clip_ratio", 0.0) == 0.0, "Cross-entropy should have zero clip ratio"

    print(f"\n=== Cross-Entropy Test Results ===")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Policy loss: {metrics['policy_loss']:.4f}")


@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, marks=pytest.mark.vllm),
    ],
    ids=["tp1"],
)
def test_tinker_adapter_importance_sampling_forward_backward(ray_init_fixture, tp_size: int):
    """Test TinkerTrainingAdapter with importance_sampling loss through real workers."""
    cfg = get_test_config()
    cfg.trainer.algorithm.policy_loss_type = "importance_sampling"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Initialize worker dispatch
    worker_dispatch = init_worker_dispatch(cfg, tp_size)

    # Create adapter
    adapter = TinkerTrainingAdapter(worker_dispatch)

    # Create Tinker-style RL training data
    prompt_text = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    response_tokens = tokenizer.encode(" The answer is 4.", add_special_tokens=False)
    full_tokens = prompt_tokens + response_tokens

    # Create datum with RL inputs (logprobs and advantages)
    data = [
        {
            "model_input": {"tokens": full_tokens},
            "loss_fn_inputs": {
                "target_tokens": full_tokens[1:] + [tokenizer.eos_token_id],
                "logprobs": [-0.1] * len(full_tokens),  # Sampling policy logprobs
                "advantages": [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens),  # Positive reward for response
            },
        },
    ]

    # Call forward_backward
    import asyncio

    async def run_forward_backward():
        return await adapter.forward_backward(
            data=data,
            loss_fn="importance_sampling",
        )

    result = asyncio.run(run_forward_backward())

    # Verify result structure
    metrics = result.metrics
    assert "final_loss" in metrics, "Should have final_loss metric"
    assert "policy_loss" in metrics, "Should have policy_loss metric"

    # Verify clip_ratio is 0 for importance sampling (no clipping)
    assert metrics.get("ppo_clip_ratio", 0.0) == 0.0, "Importance sampling should have zero clip ratio"

    print(f"\n=== Importance Sampling Test Results ===")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Policy loss: {metrics['policy_loss']:.4f}")


@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, marks=pytest.mark.vllm),
    ],
    ids=["tp1"],
)
def test_tinker_adapter_ppo_forward_backward(ray_init_fixture, tp_size: int):
    """Test TinkerTrainingAdapter with PPO loss through real workers."""
    cfg = get_test_config()
    cfg.trainer.algorithm.policy_loss_type = "regular"  # PPO

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Initialize worker dispatch
    worker_dispatch = init_worker_dispatch(cfg, tp_size)

    # Create adapter
    adapter = TinkerTrainingAdapter(worker_dispatch)

    # Create Tinker-style RL training data
    prompt_text = "Count to 5:"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    response_tokens = tokenizer.encode(" 1 2 3 4 5", add_special_tokens=False)
    full_tokens = prompt_tokens + response_tokens

    # Create datum with RL inputs
    data = [
        {
            "model_input": {"tokens": full_tokens},
            "loss_fn_inputs": {
                "target_tokens": full_tokens[1:] + [tokenizer.eos_token_id],
                "logprobs": [-0.2] * len(full_tokens),  # Sampling policy logprobs
                "advantages": [0.0] * len(prompt_tokens) + [1.5] * len(response_tokens),  # High reward for response
            },
        },
    ]

    # Call forward_backward with PPO config
    import asyncio

    async def run_forward_backward():
        return await adapter.forward_backward(
            data=data,
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.2, "clip_high_threshold": 0.2},
        )

    result = asyncio.run(run_forward_backward())

    # Verify result structure
    metrics = result.metrics
    assert "final_loss" in metrics, "Should have final_loss metric"
    assert "policy_loss" in metrics, "Should have policy_loss metric"
    assert "ppo_clip_ratio" in metrics, "PPO should have clip_ratio metric"

    # PPO clip_ratio should be between 0 and 1
    assert 0 <= metrics["ppo_clip_ratio"] <= 1, f"Clip ratio should be in [0,1], got {metrics['ppo_clip_ratio']}"

    print(f"\n=== PPO Test Results ===")
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Policy loss: {metrics['policy_loss']:.4f}")
    print(f"Clip ratio: {metrics['ppo_clip_ratio']:.4f}")


@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, marks=pytest.mark.vllm),
    ],
    ids=["tp1"],
)
def test_tinker_adapter_forward_backward_then_optim_step(ray_init_fixture, tp_size: int):
    """Test full training cycle: forward_backward → optim_step."""
    cfg = get_test_config()
    cfg.trainer.algorithm.policy_loss_type = "cross_entropy"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Initialize worker dispatch
    worker_dispatch = init_worker_dispatch(cfg, tp_size)

    # Create adapter
    adapter = TinkerTrainingAdapter(worker_dispatch)

    # Create training data
    prompt_text = "Hello"
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    response_tokens = tokenizer.encode(" world!", add_special_tokens=False)
    full_tokens = prompt_tokens + response_tokens

    data = [
        {
            "model_input": {"tokens": full_tokens},
            "loss_fn_inputs": {
                "target_tokens": full_tokens[1:] + [tokenizer.eos_token_id],
                "weights": [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens),
            },
        },
    ]

    import asyncio

    async def run_training_cycle():
        # Forward-backward pass
        fb_result = await adapter.forward_backward(data=data, loss_fn="cross_entropy")

        # Optimizer step
        grad_norm = await adapter.optim_step(learning_rate=1e-4)

        return fb_result, grad_norm

    fb_result, grad_norm = asyncio.run(run_training_cycle())

    # Verify forward_backward result
    assert "final_loss" in fb_result.metrics
    assert fb_result.metrics["final_loss"] > 0

    # Verify optimizer step
    if grad_norm is not None:
        assert grad_norm >= 0, f"Grad norm should be non-negative, got {grad_norm}"
        print(f"\n=== Training Cycle Test Results ===")
        print(f"Loss: {fb_result.metrics['final_loss']:.4f}")
        print(f"Grad norm: {grad_norm:.4f}")
    else:
        print(f"\n=== Training Cycle Test Results ===")
        print(f"Loss: {fb_result.metrics['final_loss']:.4f}")
        print("Grad norm: None (not returned)")
