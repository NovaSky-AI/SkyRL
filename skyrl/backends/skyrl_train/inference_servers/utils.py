import copy
import json
import logging
from argparse import Namespace
from typing import Any, Dict, List, Optional

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS,
)
from skyrl.backends.skyrl_train.weight_sync import get_transfer_strategy
from skyrl.train.config import (
    InferenceEngineConfig,
    SkyRLTrainConfig,
    get_config_as_dict,
)

logger = logging.getLogger(__name__)


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


# TODO: Add a test for validation
def build_vllm_cli_args(cfg: SkyRLTrainConfig) -> Namespace:
    """Build CLI args for vLLM server from config."""
    from vllm import AsyncEngineArgs
    from vllm.config import WeightTransferConfig
    from vllm.entrypoints.openai.cli_args import FrontendArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

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
        enable_sleep_mode=cfg.trainer.placement.colocate_all,
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
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    # Enable LoRA on the inference engine only when the trainer will sync
    # LoRA adapters (not merged weights).  Megatron merges by default
    # (merge_lora=True), so the inference engine must NOT have LoRA wrapping.
    if _uses_lora_weight_sync(cfg):
        args.enable_lora = True
        args.max_lora_rank = cfg.trainer.policy.model.lora.rank
        args.max_loras = 1
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

    # Add any extra engine_init_kwargs
    engine_kwargs = get_config_as_dict(ie_cfg.engine_init_kwargs)
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
