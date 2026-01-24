"""Tinker-compatible inference adapter for skyrl-train.

This module provides an adapter that enables Tinker-style token-in/token-out
sampling through skyrl-train's InferenceEngineClient.

The adapter works with plain Python types (dict, list) rather than Tinker's
pydantic models, allowing skyrl-train to remain decoupled from Tinker dependencies.
skyrl-tx can use this adapter with a thin wrapper for Tinker type conversion.

Architecture:
    Tinker API -> TinkerInferenceAdapter -> InferenceEngineClient.sample()

Usage:
    from skyrl_train.inference_engines.tinker_adapter import TinkerInferenceAdapter

    adapter = TinkerInferenceAdapter(inference_client)
    result = await adapter.sample(
        prompt_tokens=[1, 2, 3],
        num_samples=3,
        sampling_params={"temperature": 0.7, "max_tokens": 100},
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


# Type aliases for clarity
PromptTokens = list[int]
SamplingParamsDict = dict[str, Any]
StopReason = Literal["length", "stop"]


class TinkerSampleResult:
    """Result from a Tinker-style sample() call.

    This is a simple container class using plain Python types,
    avoiding dependencies on Tinker's pydantic models.
    """

    def __init__(
        self,
        sequences: list[dict[str, Any]],
        prompt_logprobs: list[float] | None = None,
    ):
        """Initialize sample result.

        Args:
            sequences: List of generated sequences, each containing:
                - tokens: list[int] - Generated token IDs
                - logprobs: list[float] - Log probabilities for each token
                - stop_reason: "length" | "stop" - Why generation stopped
            prompt_logprobs: Optional log probabilities for prompt tokens.
        """
        self.sequences = sequences
        self.prompt_logprobs = prompt_logprobs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sequences": self.sequences,
            "prompt_logprobs": self.prompt_logprobs,
        }


class TinkerInferenceAdapter:
    """Adapter for Tinker-compatible inference through skyrl-train.

    This adapter provides the conversion logic between Tinker-style API calls
    and skyrl-train's InferenceEngineClient, using plain Python types.

    For full Tinker type support, use skyrl-tx's SkyRLInferenceClient which
    wraps this adapter with Tinker pydantic model conversion.
    """

    def __init__(self, inference_client: "InferenceEngineClient"):
        """Initialize the adapter.

        Args:
            inference_client: skyrl-train's InferenceEngineClient with engines initialized.
        """
        self.inference_client = inference_client

    async def sample(
        self,
        prompt_tokens: PromptTokens,
        num_samples: int,
        sampling_params: SamplingParamsDict,
        session_id: str | int | None = None,
    ) -> TinkerSampleResult:
        """Generate multiple independent samples from a single prompt.

        This is the main entry point for Tinker-style sampling.

        Args:
            prompt_tokens: Token IDs for the prompt.
            num_samples: Number of independent samples to generate.
            sampling_params: Sampling parameters dict with keys:
                - temperature: float
                - max_tokens: int
                - top_k: int (optional, default -1)
                - top_p: float (optional, default 1.0)
                - seed: int (optional)
                - stop_token_ids: list[int] (optional)
                - stop: list[str] (optional, stop strings)
            session_id: Optional session ID for consistent engine routing.

        Returns:
            TinkerSampleResult with generated sequences.
        """
        # Call skyrl-train's sample() method
        output = await self.inference_client.sample(
            prompt_token_ids=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params,
            session_id=session_id,
        )

        # Convert InferenceEngineOutput to TinkerSampleResult
        return self._convert_output(output)

    def _convert_output(self, output: dict[str, Any]) -> TinkerSampleResult:
        """Convert skyrl-train's InferenceEngineOutput to TinkerSampleResult.

        Args:
            output: InferenceEngineOutput from skyrl-train's sample().

        Returns:
            TinkerSampleResult with generated sequences.
        """
        sequences = []
        num_samples = len(output["response_ids"])

        for i in range(num_samples):
            # Map stop_reason to Tinker's expected values
            stop_reason = output["stop_reasons"][i]
            if stop_reason in ("stop", "eos"):
                tinker_stop_reason: StopReason = "stop"
            else:
                tinker_stop_reason = "length"

            # Extract logprobs if available
            logprobs: list[float] = []
            if output.get("response_logprobs") and output["response_logprobs"][i]:
                logprobs = output["response_logprobs"][i]

            sequences.append({
                "tokens": output["response_ids"][i],
                "logprobs": logprobs,
                "stop_reason": tinker_stop_reason,
            })

        # Note: prompt_logprobs not supported yet in skyrl-train's sample()
        return TinkerSampleResult(sequences=sequences, prompt_logprobs=None)

    @staticmethod
    def extract_prompt_tokens(model_input: dict[str, Any]) -> PromptTokens:
        """Extract flat token list from Tinker ModelInput dict.

        This is a helper for converting Tinker's ModelInput format to a flat token list.

        Args:
            model_input: Dict with "chunks" key, each chunk having "tokens" list.

        Returns:
            Flat list of token IDs.
        """
        tokens: list[int] = []
        for chunk in model_input.get("chunks", []):
            tokens.extend(chunk.get("tokens", []))
        return tokens

    @staticmethod
    def convert_sampling_params(params: dict[str, Any]) -> SamplingParamsDict:
        """Convert Tinker SamplingParams dict to skyrl-train format.

        This normalizes parameter names between Tinker and skyrl-train formats.

        Args:
            params: Dict with Tinker-style parameter names.

        Returns:
            Dict compatible with skyrl-train's sampling.
        """
        result: SamplingParamsDict = {
            "temperature": params.get("temperature", 1.0),
            "max_tokens": params.get("max_tokens", 100),
            "top_k": params.get("top_k", -1),
            "top_p": params.get("top_p", 1.0),
        }

        if params.get("seed") is not None:
            result["seed"] = params["seed"]

        # Handle stop tokens/strings (Tinker uses different names)
        if params.get("stop_tokens"):
            result["stop_token_ids"] = params["stop_tokens"]
        if params.get("stop_strings"):
            result["stop"] = params["stop_strings"]

        return result
