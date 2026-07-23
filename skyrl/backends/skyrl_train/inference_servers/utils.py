import copy
import json
import logging
from argparse import Namespace
from typing import Any, Dict, List, Optional

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    SKYRL_LORA_ADAPTER_NAME,
)
from skyrl.backends.skyrl_train.weight_sync import get_transfer_strategy
from skyrl.backends.skyrl_train.weight_sync.serialized_fp8 import (
    SERIALIZED_BLOCKWISE_FP8,
    get_qwen35_fp8_ignored_layers,
    get_serialized_fp8_quantization_config,
    is_qwen35_config,
    should_use_serialized_fp8,
)
from skyrl.train.config import (
    InferenceEngineConfig,
    SkyRLTrainConfig,
    get_config_as_dict,
)

logger = logging.getLogger(__name__)


def _serialized_fp8_ignored_layers(model_path: Optional[str]) -> list[str]:
    if not model_path:
        raise ValueError("A model path is required when serialized FP8 weight sync is enabled")
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not inspect the model config required to derive serialized FP8 ignored layers: "
            f"model_path={model_path!r}"
        ) from exc
    if not is_qwen35_config(hf_config):
        raise ValueError(
            "Serialized FP8 weight sync currently supports only Qwen3.5 checkpoint layouts; "
            f"model_path={model_path!r}"
        )
    return get_qwen35_fp8_ignored_layers(hf_config)


def _set_or_validate(mapping: Dict[str, Any], key: str, expected: Any, *, context: str) -> None:
    if key in mapping and mapping[key] != expected:
        raise ValueError(
            f"{context}.{key} must be {expected!r} when serialized FP8 weight sync is enabled, " f"got {mapping[key]!r}"
        )
    mapping[key] = copy.deepcopy(expected)


def _apply_serialized_fp8_weight_sync_defaults(
    ie_cfg: InferenceEngineConfig,
    engine_kwargs: Dict[str, Any],
    model_path: Optional[str] = None,
) -> None:
    """Configure vLLM for checkpoint-format blockwise FP8 weight reloads."""

    mode = ie_cfg.fp8_weight_sync_mode
    if mode is None:
        return
    if not should_use_serialized_fp8(mode):
        raise ValueError(
            f"Unsupported fp8_weight_sync_mode={mode!r}. " f"Supported value: {SERIALIZED_BLOCKWISE_FP8!r}."
        )

    _set_or_validate(engine_kwargs, "quantization", "fp8", context="engine_init_kwargs")
    # Build FP8 modules without a bootstrap checkpoint; the first full-weight
    # sync replaces the dummy values.
    _set_or_validate(engine_kwargs, "load_format", "dummy", context="engine_init_kwargs")

    hf_overrides_value = engine_kwargs.get("hf_overrides")
    hf_overrides = {} if hf_overrides_value is None else copy.deepcopy(hf_overrides_value)
    if not isinstance(hf_overrides, dict):
        raise ValueError("engine_init_kwargs.hf_overrides must be a dict when serialized FP8 weight sync is enabled")

    qcfg_value = hf_overrides.get("quantization_config")
    qcfg = {} if qcfg_value is None else copy.deepcopy(qcfg_value)
    if not isinstance(qcfg, dict):
        raise ValueError(
            "engine_init_kwargs.hf_overrides.quantization_config must be a dict "
            "when serialized FP8 weight sync is enabled"
        )

    ignored_layers = _serialized_fp8_ignored_layers(model_path)
    if ignored_layers:
        logger.info(
            "Serialized FP8 weight sync will leave %d vLLM modules unquantized "
            "to match the Qwen3.5 serialized FP8 policy.",
            len(ignored_layers),
        )

    for key, value in get_serialized_fp8_quantization_config(
        ignored_layers=ignored_layers,
    ).items():
        _set_or_validate(
            qcfg,
            key,
            value,
            context="engine_init_kwargs.hf_overrides.quantization_config",
        )
    hf_overrides["quantization_config"] = qcfg
    engine_kwargs["hf_overrides"] = hf_overrides


def _uses_lora_weight_sync(cfg: SkyRLTrainConfig) -> bool:
    """Return True when the trainer syncs LoRA adapters (not merged weights).

    FSDP always syncs LoRA adapters when ``lora.rank > 0``.
    Megatron merges LoRA into the base weights by default
    (``merge_lora=True``), so the inference engine should not enable LoRA.
    """
    if cfg.trainer.policy.model.lora.rank <= 0:
        return False
    if cfg.trainer.strategy == "megatron":
        return not cfg.trainer.policy.megatron_config.lora_config.merge_lora
    return True


def resolve_policy_model_name(cfg: SkyRLTrainConfig) -> str:
    """Return the model identifier the inference engine knows the policy by.

    Mirrors the weight-sync code path: when the worker registers a LoRA
    adapter on the inference engine (FSDP + LoRA, or Megatron + LoRA with
    ``merge_lora=False``), the policy is that adapter and callers must pass
    ``SKYRL_LORA_ADAPTER_NAME`` as ``model`` on data-plane calls. Otherwise
    — including Megatron + LoRA with ``merge_lora=True``, where merged
    weights are pushed as a full weight update — the policy is the base
    model itself, or its configured served alias.

    This is the single source of truth for "which name does the inference
    server know the policy by?" and should be used wherever a caller needs
    to issue a ``generate``/``sample``/``chat_completion``/``completion`` /
    ``render_chat_completion`` request against the current policy.
    """
    if _uses_lora_weight_sync(cfg):
        return SKYRL_LORA_ADAPTER_NAME
    return cfg.generator.inference_engine.served_model_name or cfg.trainer.policy.model.path


# TODO: Add a test for validation
def build_vllm_cli_args(cfg: SkyRLTrainConfig) -> Namespace:
    """Build CLI args for vLLM server from config."""
    from vllm import AsyncEngineArgs
    from vllm.config import WeightTransferConfig
    from vllm.entrypoints.openai.cli_args import FrontendArgs
    from vllm.platforms import current_platform
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # This function may run a GPU-less Ray head
    # node, where ``current_platform`` resolves to ``UnspecifiedPlatform`` with
    # ``device_type == ""``. vLLM's ``add_cli_args`` walks ``VllmConfig`` defaults
    # and instantiates ``DeviceConfig()`` (device="auto"), which in turn runs
    # ``DeviceConfig.__post_init__`` and raises ``Failed to infer device type``.
    # Explicitly pin the platform's device type to ``cuda`` so the autodetection
    # path in ``DeviceConfig.__post_init__`` succeeds during arg parsing.
    # NOTE: mutating current_platform.device_type relies on vLLM 0.20.2's singleton
    # initialization where instance attrs shadow class attrs. This should be re-verified on vLLM bumps.
    if not current_platform.device_type:
        current_platform.device_type = "cuda"

    # Create common CLI args namespace
    parser = FlexibleArgumentParser()
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)
    # parse args without any command line arguments
    args: Namespace = parser.parse_args(args=[])

    ie_cfg = cfg.generator.inference_engine
    overrides = dict(
        model=cfg.trainer.policy.model.path,
        tensor_parallel_size=ie_cfg.tensor_parallel_size,
        pipeline_parallel_size=ie_cfg.pipeline_parallel_size,
        dtype=ie_cfg.model_dtype,
        data_parallel_size=ie_cfg.data_parallel_size,
        seed=cfg.trainer.seed,
        gpu_memory_utilization=ie_cfg.gpu_memory_utilization,
        enable_prefix_caching=ie_cfg.enable_prefix_caching,
        enforce_eager=ie_cfg.enforce_eager,
        max_num_batched_tokens=ie_cfg.max_num_batched_tokens,
        enable_expert_parallel=ie_cfg.expert_parallel_size > 1,
        max_num_seqs=ie_cfg.max_num_seqs,
        # Sleep mode is required for colocated (offload/backload each step) and also when
        # non-colocated weight sync opts into offloading the KV cache around the sync.
        enable_sleep_mode=cfg.trainer.placement.colocate_all or ie_cfg.offload_kv_for_weight_sync,
        enable_return_routed_experts=ie_cfg.enable_return_routed_experts,
        weight_transfer_config=WeightTransferConfig(
            backend=get_transfer_strategy(ie_cfg.weight_sync_backend, cfg.trainer.placement.colocate_all),
        ),
        worker_extension_cls=VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS,
        # NOTE (sumanthrh): We set generation config to be vLLM so that the generation behaviour of the server is same as using the vLLM Engine APIs directly
        generation_config="vllm",
        # NOTE: vllm expects a list entry for served_model_name
        served_model_name=(
            [cfg.generator.inference_engine.served_model_name]
            if cfg.generator.inference_engine.served_model_name
            else None
        ),
        language_model_only=ie_cfg.language_model_only,
        mm_processor_cache_gb=0,
        kv_cache_metrics=True,
        # models with custom modeling code (MiMo, Qwen3.5, DeepSeek-V3, ...) require it to load.
        # Overridable via generator.inference_engine.engine_init_kwargs.trust_remote_code below.
        trust_remote_code=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    # Enable LoRA on the inference engine only when the trainer will sync
    # LoRA adapters (not merged weights).  Megatron merges by default
    # (merge_lora=True), so the inference engine must NOT have LoRA wrapping.
    if _uses_lora_weight_sync(cfg):
        lora_cfg = cfg.trainer.policy.model.lora
        args.enable_lora = True
        args.max_lora_rank = lora_cfg.rank
        args.max_loras = lora_cfg.max_loras
        if lora_cfg.max_cpu_loras is not None:
            args.max_cpu_loras = lora_cfg.max_cpu_loras
        args.fully_sharded_loras = ie_cfg.fully_sharded_loras

        if not cfg.trainer.placement.colocate_all:
            lora_path = cfg.trainer.policy.model.lora.lora_sync_path
            logger.warning(
                "LoRA weight sync is enabled but training and inference are not "
                "colocated (placement.colocate_all=false). The trainer saves LoRA "
                "adapters to disk for the inference engine to load, so both must "
                "share a filesystem. Set trainer.policy.model.lora.lora_sync_path "
                "to a shared mount (current value: %s).",
                lora_path,
            )
    else:
        args.enable_lora = False

    # Speculative decoding (e.g. MTP): passed straight through to vLLM's speculative_config.
    if ie_cfg.speculative_config is not None:
        spec_cfg = get_config_as_dict(ie_cfg.speculative_config)
        args.speculative_config = spec_cfg
        logger.info(f"vLLM speculative decoding enabled: speculative_config={spec_cfg}")

    engine_kwargs = get_config_as_dict(ie_cfg.engine_init_kwargs)
    _apply_serialized_fp8_weight_sync_defaults(
        ie_cfg,
        engine_kwargs,
        cfg.trainer.policy.model.path,
    )
    for key, value in engine_kwargs.items():
        setattr(args, key, value)

    return args


def get_pd_cli_args(cli_args: Namespace, role: str = "prefill") -> Namespace:
    """Build PD-specific CLI args by injecting ``kv_role=kv_both``.

    Reads ``kv_transfer_config`` from the args namespace (set via
    ``engine_init_kwargs`` pass-through) and injects ``kv_role=kv_both``.
    ``VLLMServerActor._setup_nixl_side_channel`` later enriches the dict
    with ``engine_id``.

    Args:
        cli_args: Base CLI args from :func:`build_vllm_cli_args`.
        role: Currently unused (kv_role is always ``kv_both``).
            Kept for future flexibility.

    Returns:
        A deep copy of *cli_args* with ``kv_transfer_config`` as a dict
        containing ``kv_role=kv_both``.
    """
    args = copy.deepcopy(cli_args)

    kv_config = getattr(args, "kv_transfer_config", None)
    if kv_config is None:
        raise ValueError(
            "engine_init_kwargs.kv_transfer_config must be set when enable_pd=True "
            "(e.g. engine_init_kwargs.kv_transfer_config.kv_connector=NixlConnector)"
        )

    # kv_transfer_config arrives as a dict from Hydra's nested key resolution
    if isinstance(kv_config, str):
        kv_config = json.loads(kv_config)

    if "kv_connector" not in kv_config:
        raise ValueError("kv_transfer_config.kv_connector must be set when enable_pd=True")

    if kv_config["kv_connector"].lower() != "NixlConnector".lower():
        raise ValueError(f"Only NixlConnector is supported, got {kv_config['kv_connector']}")

    kv_config["kv_role"] = "kv_both"
    args.kv_transfer_config = kv_config

    return args


def build_router_args(
    ie_cfg: InferenceEngineConfig,
    server_urls: Optional[List[str]] = None,
    prefill_urls: Optional[List[str]] = None,
    decode_urls: Optional[List[str]] = None,
):
    """Build ``RouterArgs`` for vllm-router from SkyRL config.

    Constructs the dataclass used by ``vllm_router.Router``.  PD mode is
    activated when *prefill_urls* and *decode_urls* are provided; otherwise
    uniform mode uses *server_urls*.

    User overrides from ``cfg.generator.inference_engine.router_init_kwargs``
    are applied last so they can override any computed default.

    Args:
        ie_cfg: Inference engine config.
        server_urls: Backend URLs for uniform (non-PD) routing.
        prefill_urls: Prefill backend URLs (PD mode).
        decode_urls: Decode backend URLs (PD mode).

    Returns:
        A populated ``RouterArgs`` instance.
    """
    from vllm_router.router_args import RouterArgs

    from skyrl.backends.skyrl_train.inference_servers.common import get_open_port

    is_pd = prefill_urls is not None and decode_urls is not None

    port = get_open_port()

    kwargs: Dict[str, Any] = dict(
        host="0.0.0.0",
        port=port,
        policy="consistent_hash",
    )

    if is_pd:
        # prefill_urls in RouterArgs expects List[Tuple[str, Optional[int]]]
        kwargs["prefill_urls"] = [(url, None) for url in prefill_urls]
        kwargs["decode_urls"] = decode_urls
        kwargs["vllm_pd_disaggregation"] = True
        kwargs["prefill_policy"] = "consistent_hash"
        kwargs["decode_policy"] = "consistent_hash"
    else:
        if server_urls is None:
            raise ValueError("Either server_urls or prefill_urls/decode_urls must be provided")
        kwargs["worker_urls"] = server_urls

    # Apply user overrides from config
    router_overrides = get_config_as_dict(ie_cfg.router_init_kwargs)
    kwargs.update(router_overrides)

    return RouterArgs(**kwargs)
