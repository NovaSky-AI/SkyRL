"""External inference engine integration for vLLM/SGLang."""

import json
from typing import Any
from pathlib import Path

import requests
from tx.tinker import types
from tx.utils.log import logger


class ExternalInferenceClient:
    """Client for external inference engines (vLLM/SGLang) via OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        """Initialize the external inference client.
        
        Args:
            base_url: Base URL of the inference server (e.g., http://localhost:8000)
            api_key: API key for authentication (default: "EMPTY" for local servers)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to external inference engine at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to external inference engine: {e}")

    def generate_batch(
        self,
        prompts: list[list[int]],
        sampling_params: list[types.SamplingParams],
        model: str | None = None,
        lora_path: str | None = None,
    ) -> list[types.GeneratedSequence]:
        """Generate completions for a batch of prompts.
        
        Args:
            prompts: List of token sequences (prompt tokens)
            sampling_params: Sampling parameters for each prompt
            model: Model name (optional, uses server default if None)
            lora_path: Path to LoRA adapter on the server (optional)
            
        Returns:
            List of generated sequences with tokens and logprobs
        """
        # Send all prompts in parallel using batch API
        # vLLM/SGLang will handle batching internally for better throughput
        
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = []
            for prompt_tokens, params in zip(prompts, sampling_params):
                future = executor.submit(
                    self._generate_single,
                    prompt_tokens,
                    params,
                    model,
                    lora_path,
                )
                futures.append(future)
            
            # Collect results in order
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error generating with external engine: {e}")
                    # Return empty sequence on error
                    results.append(
                        types.GeneratedSequence(
                            stop_reason="stop",
                            tokens=[],
                            logprobs=[],
                        )
                    )
        
        return results

    def _generate_single(
        self,
        prompt_tokens: list[int],
        params: types.SamplingParams,
        model: str | None,
        lora_path: str | None,
    ) -> types.GeneratedSequence:
        """Generate completion for a single prompt."""
        # Build request payload
        payload: dict[str, Any] = {
            "prompt": prompt_tokens,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "seed": params.seed,
            "logprobs": 1,  # Request logprobs for generated tokens
        }
        
        if model:
            payload["model"] = model
        
        if lora_path:
            payload["lora_path"] = lora_path
        
        # Add stop tokens/strings if specified
        if params.stop_tokens:
            payload["stop_token_ids"] = params.stop_tokens
        if params.stop_strings:
            payload["stop"] = params.stop_strings
        
        # Make request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        response = requests.post(
            self.completions_url,
            headers=headers,
            json=payload,
            timeout=300,  # 5 minute timeout for long generations
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        choice = result["choices"][0]
        
        # Extract tokens and logprobs
        tokens = choice.get("tokens", [])
        logprobs_data = choice.get("logprobs", {})
        
        # Extract token logprobs (vLLM format)
        if logprobs_data and "token_logprobs" in logprobs_data:
            logprobs = logprobs_data["token_logprobs"]
        else:
            logprobs = [0.0] * len(tokens)
        
        # Determine stop reason
        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = "length" if finish_reason == "length" else "stop"
        
        return types.GeneratedSequence(
            stop_reason=stop_reason,
            tokens=tokens,
            logprobs=logprobs,
        )

    def upload_lora_adapter(
        self,
        adapter_path: Path,
        model_id: str,
    ) -> str:
        """Upload LoRA adapter to the external inference server.
        
        Args:
            adapter_path: Local path to the LoRA adapter directory
            model_id: Model ID for the adapter
            
        Returns:
            Path to the adapter on the server
        """
        # For vLLM, we need to use the LoRA management API
        # This is a simplified version - actual implementation depends on server setup
        
        # vLLM expects adapters to be accessible via filesystem
        # In production, you'd sync adapters to a shared storage location
        logger.info(f"LoRA adapter at {adapter_path} should be accessible to inference server")
        
        # Return the path that the server can access
        return str(adapter_path)
