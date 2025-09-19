from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Any, Optional, Hashable

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


class NamedWeightsUpdateRequest(TypedDict):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    extras: Optional[List[Dict[str, Any]]]


class InferenceEngineInterface(ABC):

    @abstractmethod
    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        raise NotImplementedError()

    @abstractmethod
    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a ChatCompletionRequest.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError()

    @abstractmethod
    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a CompletionRequest.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError()

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError()

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError()

    @abstractmethod
    def tp_size(self) -> int:
        """Return the tensor parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    def dp_size(self) -> int:
        """Return the data parallel size of this inference engine."""
        raise NotImplementedError()
