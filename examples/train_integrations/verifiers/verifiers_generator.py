from copy import deepcopy
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI
from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs, RolloutInput

from skyrl.train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl.train.config import GeneratorConfig
from skyrl.train.generators.utils import get_rollout_metrics


_VERIFIERS_EXTRA_BODY_KEYS = (
    "include_stop_str_in_output",
    "min_p",
    "min_tokens",
    "repetition_penalty",
    "skip_special_tokens",
    "top_k",
)


class VerifiersGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: GeneratorConfig object containing the generator configuration
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name

        ie_cfg = generator_cfg.inference_engine
        assert ie_cfg.enable_http_endpoint, "HTTP endpoint must be enabled for VerifiersGenerator"
        self.base_url = f"http://{ie_cfg.http_endpoint_host}:{ie_cfg.http_endpoint_port}/v1"
        self.client = self._setup_client(connection_limit=None)  # None means unlimited connections

    def _setup_client(self, connection_limit: Optional[int]) -> AsyncOpenAI:
        timeout = httpx.Timeout(timeout=600, connect=5.0)
        limits = httpx.Limits(
            max_connections=connection_limit,  # OAI default: 1000
            max_keepalive_connections=connection_limit,  # OAI default: 100
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key="dummy",  # Make OAI client happy.
            max_retries=10,  # OAI default: 2
            http_client=http_client,
        )

    def _build_verifiers_sampling_args(self, input_batch: GeneratorInput) -> dict[str, Any]:
        """Build sampling args for Verifiers' OpenAI client path.

        Verifiers' current chat parsing expects prompt token IDs in the response, which vLLM
        exposes when `return_token_ids` is requested. We pass this through `extra_body`
        because the request is made via the OpenAI client.
        """
        sampling_params = deepcopy(input_batch.get("sampling_params", {}))
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = 1

        extra_body = dict(sampling_params.pop("extra_body", {}) or {})
        extra_body["return_token_ids"] = True
        # Preserve existing behavior for token strings in logprob content until we confirm
        # Verifiers no longer needs it.
        extra_body.setdefault("return_tokens_as_token_ids", True)

        for key in _VERIFIERS_EXTRA_BODY_KEYS:
            if key in sampling_params:
                extra_body[key] = sampling_params.pop(key)

        sampling_params["extra_body"] = extra_body
        return sampling_params

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        assert "env_extras" in input_batch, "Verifiers dataset fields are passed through env_extras"

        # Defaults are based on Verifiers' defaults.
        verifiers_dicts = [sample["verifiers"] for sample in input_batch["env_extras"]]
        rollout_inputs = []
        for i, item in enumerate(verifiers_dicts):
            rollout_inputs.append(
                RolloutInput(
                    prompt=input_batch["prompts"][i],
                    answer=item.get("answer", ""),
                    example_id=item["example_id"],
                    info=item.get("info", {}),
                    task=item.get("task", "default"),
                )
            )

        # Assumes all training samples correspond to the same Verifiers environment.
        # For now, if multiple environments are needed, use Verifiers' EnvGroup abstraction.
        environment_id = verifiers_dicts[0]["environment"]
        vf_env = load_environment(environment_id)

        sampling_params = self._build_verifiers_sampling_args(input_batch)

        # Generate the trajectories.
        try:
            generate_outputs: GenerateOutputs = await vf_env.generate(
                inputs=rollout_inputs,
                client=self.client,
                model=self.model_name,
                sampling_args=sampling_params,
            )
        except Exception as exc:
            if "prompt_token_ids" in str(exc) or (
                isinstance(exc, TypeError) and "NoneType" in str(exc) and "len()" in str(exc)
            ):
                raise RuntimeError(
                    "Verifiers generation failed while parsing token metadata from "
                    "/chat/completions. This integration requires the HTTP endpoint to return "
                    "prompt_token_ids, completion token_ids, and logprobs. Verify that "
                    "`return_token_ids` is being requested through `extra_body` and that the "
                    "served OpenAI-compatible vLLM endpoint supports it."
                ) from exc
            raise

        processed_outputs: ProcessedOutputs = vf_env.process_env_results_vllm(
            prompts=generate_outputs.prompt,
            completions=generate_outputs.completion,
            states=generate_outputs.state,
            rewards=generate_outputs.reward,
            processing_class=self.tokenizer,
            max_seq_len=self.generator_cfg.max_input_length + self.generator_cfg.sampling_params.max_generate_length,
            mask_env_responses=True,
        )

        # Convert output to SkyRL format.
        return GeneratorOutput(
            prompt_token_ids=processed_outputs.prompt_ids,
            response_ids=processed_outputs.completion_ids,
            rewards=processed_outputs.rewards,
            loss_masks=processed_outputs.completion_mask,
            rollout_logprobs=processed_outputs.completion_logprobs,
            rollout_metrics=get_rollout_metrics(processed_outputs.completion_ids, processed_outputs.rewards),
        )
