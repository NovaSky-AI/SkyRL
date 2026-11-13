"""Names of the fields that rollouts and training batches carry.

Every producer and reader of an ``InferenceEngineInput``/``Output``, a ``GeneratorInput``/``Output``
or a ``TrainingInputBatch`` names its fields through these, never as string literals. The members are ``str``, so they index the plain dicts directly.
"""

from enum import StrEnum


class InferenceEngineInputKey(StrEnum):
    """``InferenceEngineInput``'s keys."""

    PROMPTS = "prompts"
    PROMPT_TOKEN_IDS = "prompt_token_ids"
    SAMPLING_PARAMS = "sampling_params"
    SESSION_IDS = "session_ids"
    MM_FEATURES = "mm_features"
    CACHE_SALT = "cache_salt"
    ROUTED_EXPERTS_PROMPT_STARTS = "routed_experts_prompt_starts"
    RETURN_SAMPLE_SUPPORT = "return_sample_support"


class InferenceEngineOutputKey(StrEnum):
    """``InferenceEngineOutput``'s keys."""

    RESPONSES = "responses"
    RESPONSE_IDS = "response_ids"
    STOP_REASONS = "stop_reasons"
    RESPONSE_LOGPROBS = "response_logprobs"
    PROMPT_LOGPROBS = "prompt_logprobs"
    ROLLOUT_EXPERT_INDICES = "rollout_expert_indices"
    ROLLOUT_SAMPLE_SUPPORT = "rollout_sample_support"


class GeneratorInputKey(StrEnum):
    """``GeneratorInput``'s keys."""

    PROMPTS = "prompts"
    ENV_CLASSES = "env_classes"
    ENV_EXTRAS = "env_extras"
    SAMPLING_PARAMS = "sampling_params"
    TRAJECTORY_IDS = "trajectory_ids"
    BATCH_METADATA = "batch_metadata"


class GeneratorOutputKey(StrEnum):
    """``GeneratorOutput``'s keys."""

    PROMPT_TOKEN_IDS = "prompt_token_ids"
    RESPONSE_IDS = "response_ids"
    REWARDS = "rewards"
    LOSS_MASKS = "loss_masks"
    STOP_REASONS = "stop_reasons"
    ROLLOUT_METRICS = "rollout_metrics"
    ROLLOUT_LOGPROBS = "rollout_logprobs"
    TRAJECTORY_IDS = "trajectory_ids"
    TRAJECTORY_GENERATION_TIMES = "trajectory_generation_times"
    TRAJECTORY_TIME_SPLITS = "trajectory_time_splits"
    ROLLOUT_EXPERT_INDICES = "rollout_expert_indices"
    ROLLOUT_SAMPLE_SUPPORT = "rollout_sample_support"
    IS_LAST_STEP = "is_last_step"
    ENV_METRICS = "env_metrics"
    PIXEL_VALUES = "pixel_values"
    IMAGE_GRID_THW = "image_grid_thw"


# The keys holding a row's token payload. A stored trajectory keeps these in the Ray object store;
# every other key is metadata the driver may read.
PAYLOAD_KEYS = frozenset(
    {
        GeneratorOutputKey.PROMPT_TOKEN_IDS,
        GeneratorOutputKey.RESPONSE_IDS,
        GeneratorOutputKey.LOSS_MASKS,
        GeneratorOutputKey.ROLLOUT_LOGPROBS,
        GeneratorOutputKey.ROLLOUT_EXPERT_INDICES,
        GeneratorOutputKey.ROLLOUT_SAMPLE_SUPPORT,
        GeneratorOutputKey.PIXEL_VALUES,
        GeneratorOutputKey.IMAGE_GRID_THW,
    }
)


class TrainingField(StrEnum):
    """``TrainingInput``'s keys: the fields of a ``TrainingInputBatch``."""

    # Real rows, including real trajectories with no loss tokens; absent means every row is real.
    REAL_SAMPLE_MASK = "real_sample_mask"
    SEQUENCES = "sequences"
    ATTENTION_MASK = "attention_mask"
    LOSS_MASK = "loss_mask"
    RESPONSE_MASK = "response_mask"
    ACTION_LOG_PROBS = "action_log_probs"
    BASE_ACTION_LOG_PROBS = "base_action_log_probs"
    VALUES = "values"
    RETURNS = "returns"
    ADVANTAGES = "advantages"
    KL = "kl"
    REWARDS = "rewards"
    ROLLOUT_LOGPROBS = "rollout_logprobs"
    ROLLOUT_EXPERT_INDICES = "rollout_expert_indices"
    ROUTER_PADDING_MASK = "router_padding_mask"
    ROLLOUT_SAMPLE_SUPPORT = "rollout_sample_support"
    PIXEL_VALUES = "pixel_values"
    IMAGE_GRID_THW = "image_grid_thw"
    # Per row, the lengths of the packed segments it holds.
    SUB_SEQ_LENGTHS = "sub_seq_lengths"


class ForwardOutputKey(StrEnum):
    """Keys a worker's forward pass writes: its ``TrainingOutputBatch``, the model output and step metrics."""

    # ``TrainingOutputBatch``: the per-token logprobs or values.
    OUTPUT = "output"
    # The model output: per-token entropy, added by the model wrapper.
    ENTROPY = "entropy"
    # A microbatch's metrics: its per-sample loss outputs, gathered into ``WorkerOutput.loss_fn_outputs``.
    LOSS_FN_OUTPUTS = "loss_fn_outputs"


class BatchMetadataKey(StrEnum):
    """Keys of a ``TrainingInputBatch``'s ``metadata``."""

    UIDS = "uids"
    IS_LAST_STEP = "is_last_step"
    RESPONSE_LENGTH = "response_length"
    AVG_RESPONSE_LENGTH = "avg_response_length"
    PAD_SIZE = "pad_size"
    IS_PADDING_BATCH = "is_padding_batch"
    METRICS = "metrics"
    GLOBAL_STEP = "global_step"
    POLICY_MINI_BATCH_BOUNDARIES = "policy_mini_batch_boundaries"
    CRITIC_MINI_BATCH_BOUNDARIES = "critic_mini_batch_boundaries"
    POLICY_PROMPT_BOUNDARIES = "policy_prompt_boundaries"
