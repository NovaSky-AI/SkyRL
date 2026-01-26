"""Tinker-compatible training adapter for skyrl-train.

This module provides an adapter that enables Tinker-style training operations
through skyrl-train's WorkerDispatch.

The adapter works with plain Python types (dict, list) rather than Tinker's
pydantic models, allowing skyrl-train to remain decoupled from Tinker dependencies.
skyrl-tx can use this adapter with a thin wrapper for Tinker type conversion.

Architecture:
    Tinker API -> TinkerTrainingAdapter -> WorkerDispatch -> Workers

Supported loss functions:
    - cross_entropy: Supervised learning cross-entropy loss
    - importance_sampling: REINFORCE with importance sampling correction
    - ppo: Proximal Policy Optimization with clipping

Usage:
    from skyrl_train.training.tinker_adapter import TinkerTrainingAdapter

    adapter = TinkerTrainingAdapter(worker_dispatch)
    result = await adapter.forward_backward(
        data=[
            {
                "model_input": {"tokens": [1, 2, 3]},
                "loss_fn_inputs": {
                    "target_tokens": [2, 3, 4],
                    "weights": [0, 1, 1],
                }
            }
        ],
        loss_fn="cross_entropy",
    )
    grad_norm = await adapter.optim_step(learning_rate=1e-4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import torch

if TYPE_CHECKING:
    from skyrl_train.workers.worker_dispatch import WorkerDispatch


# Type aliases
LossFnName = Literal["cross_entropy", "importance_sampling", "ppo"]
DatumDict = Dict[str, Any]


@dataclass
class ForwardBackwardOutput:
    """Result from a forward_backward() call.

    This is a simple container class using plain Python types,
    avoiding dependencies on Tinker's pydantic models.
    """

    loss_fn_outputs: List[Dict[str, Any]]
    """Per-datum output tensors (e.g., logprobs for each token)."""

    metrics: Dict[str, float]
    """Aggregated training metrics (e.g., loss, clip_ratio)."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "loss_fn_outputs": self.loss_fn_outputs,
            "metrics": self.metrics,
        }


class TinkerTrainingAdapter:
    """Adapter for Tinker-compatible training through skyrl-train.

    This adapter provides the conversion logic between Tinker-style API calls
    and skyrl-train's WorkerDispatch, using plain Python types.

    For full Tinker type support, use skyrl-tx's wrapper which handles
    Tinker pydantic model conversion.

    Supported loss functions:
        - cross_entropy: For supervised learning
          Required inputs: target_tokens, weights
        - importance_sampling: For RL with importance sampling
          Required inputs: target_tokens, logprobs (sampling), advantages
        - ppo: For PPO with clipping
          Required inputs: target_tokens, logprobs (sampling), advantages
          Optional config: clip_low_threshold, clip_high_threshold
    """

    # Map Tinker loss function names to SkyRL policy loss types
    LOSS_FN_MAP = {
        "cross_entropy": "cross_entropy",
        "importance_sampling": "importance_sampling",
        "ppo": "regular",  # SkyRL's regular PPO
    }

    def __init__(
        self,
        worker_dispatch: "WorkerDispatch",
        model_name: str = "policy",
    ):
        """Initialize the adapter.

        Args:
            worker_dispatch: skyrl-train's WorkerDispatch for training operations.
            model_name: Name of the model in WorkerDispatch (default: "policy").
        """
        self.worker_dispatch = worker_dispatch
        self.model_name = model_name

    async def forward_backward(
        self,
        data: List[DatumDict],
        loss_fn: LossFnName,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> ForwardBackwardOutput:
        """Run forward pass and compute gradients.

        Args:
            data: List of Datum dicts, each containing:
                - model_input: Dict with "tokens" key (flat list of token IDs)
                - loss_fn_inputs: Dict with loss-function-specific inputs
            loss_fn: Loss function name ("cross_entropy", "importance_sampling", "ppo")
            loss_fn_config: Optional config dict for loss function (e.g., clip thresholds)

        Returns:
            ForwardBackwardOutput with per-datum outputs and aggregated metrics.

        Raises:
            ValueError: If loss_fn is not supported or required inputs are missing.
        """
        if loss_fn not in self.LOSS_FN_MAP:
            raise ValueError(f"Unsupported loss function: {loss_fn}. " f"Supported: {list(self.LOSS_FN_MAP.keys())}")

        # Convert Tinker data format to SkyRL TrainingInputBatch
        training_batch = self._convert_data_to_batch(data, loss_fn)

        # Update metadata with loss_fn info (don't overwrite existing metadata)
        training_batch.metadata.update(
            {
                "loss_fn": self.LOSS_FN_MAP[loss_fn],
                "loss_fn_config": loss_fn_config or {},
            }
        )

        # Call WorkerDispatch forward_backward
        # Note: WorkerDispatch.forward_backward is synchronous, but we make this
        # method async for consistency with Tinker's API
        metrics = self.worker_dispatch.forward_backward(self.model_name, training_batch)

        # Extract per-datum logprobs from the forward pass
        # For now, we return the batch metrics; per-datum outputs would need
        # worker changes to return them
        loss_fn_outputs = self._extract_loss_fn_outputs(data, metrics)

        return ForwardBackwardOutput(
            loss_fn_outputs=loss_fn_outputs,
            metrics=metrics,
        )

    async def optim_step(
        self,
        learning_rate: Optional[float] = None,
    ) -> Optional[float]:
        """Apply accumulated gradients with optimizer step.

        Args:
            learning_rate: Optional learning rate override.
                Note: SkyRL uses scheduler-based LR, so this is currently ignored.
                To change LR, configure the scheduler in the trainer config.

        Returns:
            Gradient norm if available, else None.
        """
        # Note: SkyRL's optim_step doesn't take learning_rate as an arg;
        # LR is controlled by the scheduler. Tinker's API accepts it for
        # compatibility, but we ignore it here.
        grad_norm = self.worker_dispatch.optim_step(self.model_name)
        return grad_norm

    def _convert_data_to_batch(
        self,
        data: List[DatumDict],
        loss_fn: LossFnName,
    ):
        """Convert Tinker datum list to SkyRL TrainingInputBatch.

        Args:
            data: List of Datum dicts
            loss_fn: Loss function name (determines which inputs to extract)

        Returns:
            TrainingInputBatch compatible with WorkerDispatch
        """
        from skyrl_train.training_batch import TrainingInputBatch

        batch_size = len(data)
        if batch_size == 0:
            raise ValueError("Data list cannot be empty")

        # Find max sequence length for padding
        max_seq_len = max(len(d["model_input"].get("tokens", [])) for d in data)

        # Initialize batch tensors
        sequences = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        response_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        # For RL losses
        action_log_probs = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        advantages = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

        # Required but unused for supervised learning (will be zeros if not needed)
        base_action_log_probs = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        values = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        returns = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

        num_actions = 0

        for i, datum in enumerate(data):
            tokens = datum["model_input"].get("tokens", [])
            seq_len = len(tokens)

            # Left-pad sequences (SkyRL convention)
            pad_len = max_seq_len - seq_len
            sequences[i, pad_len:] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, pad_len:] = 1

            loss_fn_inputs = datum.get("loss_fn_inputs", {})

            if loss_fn == "cross_entropy":
                # SL: weights indicate which tokens to train on
                weights = loss_fn_inputs.get("weights", [1] * seq_len)
                loss_mask[i, pad_len:] = torch.tensor(weights, dtype=torch.float)
                response_mask[i, pad_len:] = 1  # All tokens are part of response

                # Track num_actions as the number of weighted tokens
                num_actions = max(num_actions, sum(1 for w in weights if w > 0))

            else:
                # RL: need logprobs and advantages
                logprobs = loss_fn_inputs.get("logprobs", [0.0] * seq_len)
                advs = loss_fn_inputs.get("advantages", [0.0] * seq_len)

                action_log_probs[i, pad_len:] = torch.tensor(logprobs, dtype=torch.float)
                advantages[i, pad_len:] = torch.tensor(advs, dtype=torch.float)
                response_mask[i, pad_len:] = 1  # All tokens are part of response

                # For RL, loss_mask = 1 where we have advantages
                loss_mask[i, pad_len:] = torch.tensor(
                    [1.0 if a != 0 else 0.0 for a in advs],
                    dtype=torch.float,
                )

                num_actions = max(num_actions, seq_len)

        # Create TrainingInputBatch with all required keys
        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "response_mask": response_mask,
                "action_log_probs": action_log_probs,
                "base_action_log_probs": base_action_log_probs,
                "advantages": advantages,
                "values": values,
                "returns": returns,
            }
        )

        # Add metadata (including response_length required by batch_to_experience)
        # response_length is the padded response length (max_seq_len)
        batch.metadata = {
            "num_actions": num_actions,
            "response_length": max_seq_len,
        }

        return batch

    def _extract_loss_fn_outputs(
        self,
        data: List[DatumDict],
        metrics: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Extract per-datum outputs from the forward pass.

        For now, we don't have per-datum logprobs from the worker,
        so we return placeholder outputs. This would need worker
        changes to fully support.

        Args:
            data: Original datum list
            metrics: Aggregated metrics from forward_backward

        Returns:
            List of per-datum output dicts
        """
        # TODO: Extend worker to return per-datum logprobs
        # For now, return empty outputs as placeholder
        return [{"logprobs": []} for _ in data]

    @staticmethod
    def extract_tokens_from_model_input(model_input: Dict[str, Any]) -> List[int]:
        """Extract flat token list from Tinker ModelInput dict.

        Helper for converting Tinker's ModelInput format to a flat token list.

        Args:
            model_input: Dict with either:
                - "tokens": flat list of token IDs, or
                - "chunks": list of dicts with "tokens" key

        Returns:
            Flat list of token IDs.
        """
        if "tokens" in model_input:
            return model_input["tokens"]

        # Handle chunked format
        tokens: List[int] = []
        for chunk in model_input.get("chunks", []):
            tokens.extend(chunk.get("tokens", []))
        return tokens
