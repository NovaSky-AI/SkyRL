from typing import Optional
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from omegaconf import DictConfig
from openai import AsyncOpenAI
import httpx
from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs, GenerateInputs
from skyrl_train.generators.utils import get_rollout_metrics


class VerifiersGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name

        assert generator_cfg.enable_http_endpoint, "HTTP endpoint must be enabled for VerifiersGenerator"
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}/v1"
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

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        assert "env_extras" in input_batch, "Verifiers dataset fields are passed through env_extras"

        # Defaults are based on Verifiers' defaults.
        verifiers_dicts = [sample["verifiers"] for sample in input_batch["env_extras"]]
        generate_inputs = GenerateInputs(
            prompt=input_batch["prompts"],
            answer=[item.get("answer", "") for item in verifiers_dicts],
            info=[item.get("info", {}) for item in verifiers_dicts],
            task=[item.get("task", "default") for item in verifiers_dicts],
        )

        # Assumes all training samples correspond to the same Verifiers environment.
        # For now, if multiple environments are needed, use Verifiers' EnvGroup abstraction.
        environment_id = verifiers_dicts[0]["environment"]
        vf_env = load_environment(environment_id)

        # Verifiers requires logprobs from vLLM for post-processing.
        sampling_params = input_batch.get("sampling_params", {}).copy()
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = 1
        sampling_params["extra_body"] = {
            "return_tokens_as_token_ids": True,
        }

        # Clean the sampling params for Verifiers' a_generate.
        extra_body_keys = [
            "min_tokens",
            "skip_special_tokens",
            "include_stop_str_in_output",
            "top_k",
            "min_p",
            "repetition_penalty",
        ]
        for key in extra_body_keys:
            if key in sampling_params:
                sampling_params["extra_body"][key] = sampling_params[key]
                del sampling_params[key]

        # Generate the trajectories.
        generate_outputs: GenerateOutputs = await vf_env.a_generate(
            inputs=generate_inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=sampling_params,
        )

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
