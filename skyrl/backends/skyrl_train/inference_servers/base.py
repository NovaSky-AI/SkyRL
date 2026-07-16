from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Tuple, TypedDict

from skyrl.utils.routed_experts import RoutedExpertIndices

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.weight_sync import WeightUpdateRequest
    from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
        WeightSyncInitInfo,
    )

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class MMPlaceholderRangeInfo(TypedDict):
    offset: int
    length: int


class MultiModalFeatures(TypedDict):
    mm_hashes: dict[str, list[str]]
    mm_placeholders: dict[str, list[MMPlaceholderRangeInfo]]
    kwargs_data: Optional[dict[str, list[str | None]]]


class InferenceEngineInput(TypedDict):
    # Either prompts or prompt_token_ids must be provided, but not both.
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    session_ids: Optional[List[Hashable]]
    mm_features: Optional[List[MultiModalFeatures]]
    # Optional prefix-cache salt forwarded to vLLM as the request ``cache_salt`` so cache blocks are
    # only shared between requests carrying the same salt. See ``GeneratorConfig.use_cache_salt``.
    cache_salt: Optional[str]
    routed_experts_prompt_starts: Optional[List[int]]


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
    prompt_logprobs: Optional[List[List[float]]]  # per-prompt-token logprobs under the current model
    rollout_expert_indices: Optional[List[RoutedExpertIndices]]


class InferenceEngineInterface(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The base model identifier the inference server was started with.

        Generators pass this as the ``model`` field on requests (e.g. when
        rendering a chat completion) when they don't otherwise specify one.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: Optional[str] = None,
    ) -> InferenceEngineOutput:
        raise NotImplementedError

    @abstractmethod
    def get_endpoint_url(self) -> str:
        """Return the base URL of the data-plane (OpenAI-compatible) endpoint.

        Generators point external clients at the inference server/router through
        this URL (e.g. LiteLLM via ``OPENAI_BASE_URL``) without depending on the
        concrete client type or how the URL is derived.
        """
        raise NotImplementedError

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
    async def render_chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the chat template and tokenize without generating.

        Accepts the same ``{"json": <request-body>}`` payload as
        ``chat_completion`` and returns the rendered prompt / token IDs. Used by
        generators that need token-in/token-out rendering (e.g. multi-modal).
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
    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        raise NotImplementedError

    @abstractmethod
    async def pause_generation(self) -> None:
        """Pause generation, freezing in-flight requests so they can be resumed later."""
        raise NotImplementedError

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume generation after a pause, continuing any frozen in-flight requests."""
        raise NotImplementedError

    @abstractmethod
    async def finish_session(self, session_id: str) -> None:
        """Notify the inference server that a session (trajectory) is complete.

        Best-effort: lets session-aware routing release the replica capacity the
        session held. Generators call this in trajectory-cleanup paths.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_world_size(self) -> Tuple[int, int]:
        """Return ``(total_world_size, world_size_per_server)`` across all inference workers."""
        raise NotImplementedError
