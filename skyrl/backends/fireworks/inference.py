"""SkyRL inference adapter for Fireworks hosted snapshot samplers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional, Tuple

from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)


class FireworksInferenceClient(InferenceEngineInterface):
    """Token-in/token-out client over the runtime's current hosted sampler."""

    def __init__(
        self,
        *,
        runtime: FireworksRuntime,
        default_sampling_params: dict[str, Any],
    ):
        self.runtime = runtime
        self._default_sampling_params = dict(default_sampling_params)
        self._admission = threading.Event()
        self._admission.set()

    async def _wait_for_admission(self) -> None:
        if not self._admission.is_set():
            await asyncio.to_thread(self._admission.wait)

    @property
    def model_name(self) -> str:
        return self.runtime.inference_endpoint.model

    @property
    def weight_version(self) -> int:
        return self.runtime.weight_version

    @staticmethod
    def _tinker_sampling_params(values: dict[str, Any]):
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Fireworks sampling requires the tinker package") from exc

        kwargs = {
            "max_tokens": values.get("max_tokens"),
            "temperature": values.get("temperature", 1.0),
            "top_k": values.get("top_k", -1),
            "top_p": values.get("top_p", 1.0),
            "stop": values.get("stop"),
        }
        if values.get("seed") is not None:
            kwargs["seed"] = values["seed"]
        return tinker.SamplingParams(**kwargs)

    @staticmethod
    def _stop_reason(value: Any) -> str:
        raw = getattr(value, "value", value)
        return "stop" if raw is None else str(raw).lower()

    async def _sample_one(
        self,
        tokens: list[int],
        *,
        params: Any,
        request_logprobs: bool,
    ) -> tuple[str, list[int], str, list[float] | None]:
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Fireworks sampling requires the tinker package") from exc

        prompt = tinker.ModelInput.from_ints(tokens)
        result, identity = await self.runtime.sample_async(
            prompt=prompt,
            sampling_params=params,
        )
        sequences = list(getattr(result, "sequences", []) or [])
        if len(sequences) != 1:
            raise RuntimeError(
                f"Fireworks sampler returned {len(sequences)} sequences for a one-sample request "
                f"at snapshot {identity.snapshot_path}"
            )
        sequence = sequences[0]
        response_ids = [int(token) for token in list(sequence.tokens)]
        raw_logprobs = getattr(sequence, "logprobs", None)
        response_logprobs = (
            None
            if raw_logprobs is None
            else [float(value) for value in list(raw_logprobs)]
        )
        if request_logprobs and (
            response_logprobs is None or len(response_logprobs) != len(response_ids)
        ):
            raise RuntimeError(
                "Fireworks did not return one rollout logprob per generated token"
            )
        response = self.runtime.tokenizer.decode(response_ids, skip_special_tokens=True)
        return (
            response,
            response_ids,
            self._stop_reason(sequence.stop_reason),
            response_logprobs,
        )

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: Optional[str] = None,
    ) -> InferenceEngineOutput:
        del model
        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None or input_batch.get("prompts") is not None:
            raise ValueError(
                "Fireworks hosted generation currently requires prompt_token_ids only"
            )

        await self._wait_for_admission()
        values: dict[str, Any] = dict(self._default_sampling_params)
        sampling_overrides = input_batch.get("sampling_params")
        if sampling_overrides is not None:
            for key, value in sampling_overrides.items():
                values[key] = value
        request_logprobs = values.get("logprobs") is not None
        params = self._tinker_sampling_params(values)

        rows = await asyncio.gather(
            *(
                self._sample_one(
                    list(tokens), params=params, request_logprobs=request_logprobs
                )
                for tokens in prompt_token_ids
            )
        )
        return InferenceEngineOutput(
            responses=[row[0] for row in rows],
            response_ids=[row[1] for row in rows],
            stop_reasons=[row[2] for row in rows],
            response_logprobs=[row[3] for row in rows] if request_logprobs else None,
            prompt_logprobs=None,
            rollout_expert_indices=None,
        )

    def get_endpoint_url(self) -> str:
        return self.runtime.inference_endpoint.api_base

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        del request_payload
        raise NotImplementedError(
            "Fireworks hosted sampling currently uses token-in/token-out generation"
        )

    async def render_chat_completion(
        self, request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        del request_payload
        raise NotImplementedError(
            "Fireworks hosted rendering is performed by the SkyRL tokenizer"
        )

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        del request_payload
        raise NotImplementedError(
            "Use token-in/token-out generate() with Fireworks hosted sampling"
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        return None

    async def sleep(self, *args: Any, **kwargs: Any):
        return None

    async def init_weight_update_communicator(self, init_info):
        raise NotImplementedError("Fireworks owns hosted weight synchronization")

    async def update_named_weights(self, request):
        raise NotImplementedError("Fireworks owns hosted weight synchronization")

    async def teardown(self):
        await self.runtime.close()

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        del reset_running_requests
        return None

    async def pause_generation(self) -> None:
        self._admission.clear()

    async def resume_generation(self) -> None:
        self._admission.set()

    async def finish_session(self, session_id: str) -> None:
        del session_id
        return None

    async def get_world_size(self) -> Tuple[int, int]:
        return 1, 1
