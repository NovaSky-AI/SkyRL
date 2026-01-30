from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Any, Optional, Hashable, TYPE_CHECKING

if TYPE_CHECKING:
    from skyrl_train.weight_sync import WeightUpdateRequest
    from skyrl_train.weight_sync.transfer_strategy import WeightSyncInitInfo

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class InferenceEngineInput(TypedDict):
    # Either prompts or prompt_token_ids must be provided, but not both.
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    session_ids: Optional[List[Hashable]]


class InferenceEngineOutput(TypedDict):
    # We always return both tokens and text outputs. The tokens are the outputs
    # of inference engine, and the text is the decoded text output. Therefore,
    # it is guaranteed that tokenizer.decode(response_token_ids, skip_special_tokens=True) == responses,
    # but the reverse is not guaranteed, since there are multiple ways to
    # represent the same text with tokens. Therefore, for multi-turn generation,
    # please use token-in-token-out to ensure correctness.
    # `skip_special_tokens=True` is needed because string responses do not include EOS tokens like `<|im_end|>`
    responses: List[str]
    response_ids: List[List[int]]
    stop_reasons: List[str]
    response_logprobs: Optional[List[List[float]]]


class InferenceEngineInterface(ABC):

    @abstractmethod
    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        raise NotImplementedError

    async def sample(
        self,
        prompt_token_ids: List[int],
        num_samples: int,
        sampling_params: Dict[str, Any],
    ) -> InferenceEngineOutput:
        """Generate multiple independent samples from a single prompt.

        This method provides Tinker-compatible token-in/token-out sampling semantics.

        Args:
            prompt_token_ids: Token IDs for a single prompt.
            num_samples: Number of independent samples to generate.
            sampling_params: Sampling parameters.

        Returns:
            InferenceEngineOutput containing num_samples results:
                - response_ids: List of num_samples token ID lists
                - responses: List of num_samples decoded strings
                - stop_reasons: List of num_samples stop reasons
                - response_logprobs: Optional list of num_samples logprob lists
        """
        all_response_ids = []
        all_responses = []
        all_stop_reasons = []
        all_response_logprobs = []

        for _ in range(num_samples):
            input_batch: InferenceEngineInput = {
                "prompts": None,
                "prompt_token_ids": [prompt_token_ids],  # Wrap in list for batch of 1
                "sampling_params": sampling_params,
                "session_ids": None,
            }
            output = await self.generate(input_batch)

            # Extract single result from batch of 1
            all_response_ids.append(output["response_ids"][0])
            all_responses.append(output["responses"][0])
            all_stop_reasons.append(output["stop_reasons"][0])
            if output.get("response_logprobs") is not None:
                all_response_logprobs.append(output["response_logprobs"][0])

        return {
            "response_ids": all_response_ids,
            "responses": all_responses,
            "stop_reasons": all_stop_reasons,
            "response_logprobs": all_response_logprobs if all_response_logprobs else None,
        }

    @abstractmethod
    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a ChatCompletionRequest.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError

    @abstractmethod
    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a CompletionRequest.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        """Initialize weight update communicator from init info.

        Args:
            init_info: WeightSyncInitInfo from the sender containing all info needed
                to create the appropriate receiver.
        """
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: "WeightUpdateRequest"):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError

    @abstractmethod
    def tp_size(self) -> int:
        """Return the tensor parallel size of this inference engine."""
        raise NotImplementedError

    @abstractmethod
    def pp_size(self) -> int:
        """Return the pipeline parallel size of this inference engine."""
        raise NotImplementedError

    @abstractmethod
    def dp_size(self) -> int:
        """Return the data parallel size of this inference engine."""
        raise NotImplementedError

    @abstractmethod
    async def abort_generation(self) -> None:
        """
        Abort all running and waiting requests, which make the ongoing requests return the
        already-generated tokens with a stop_reason of "abort". If the request was waiting,
        it returns a response with zero completion tokens.
        """
        raise NotImplementedError
