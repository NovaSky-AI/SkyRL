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
    for key, value in engine_kwargs.items():
        setattr(args, key, value)

    return args


# P2P transfer connectors supported for PD disaggregation. NIXL is the
# pull-based default; Mooncake is push-based (bootstrap-server handshake,
# vllm-router kv_connector=mooncake mode).
SUPPORTED_PD_P2P_CONNECTORS = ("NixlConnector", "MooncakeConnector")


def get_pd_p2p_connector_name(kv_config: Dict[str, Any]) -> str:
    """Return the P2P transfer connector name from a PD ``kv_transfer_config``.

    Accepts either a bare P2P connector (``NixlConnector`` / ``MooncakeConnector``)
    or a ``MultiConnector`` wrapping exactly one P2P connector plus optional
    store connectors (e.g. ``MooncakeStoreConnector`` for KV offloading).

    Raises:
        ValueError: if no (or more than one) supported P2P connector is found.
    """
    connector = kv_config.get("kv_connector")
    if connector == "MultiConnector":
        children = (kv_config.get("kv_connector_extra_config") or {}).get("connectors", [])
        p2p = [c.get("kv_connector") for c in children if c.get("kv_connector") in SUPPORTED_PD_P2P_CONNECTORS]
        if len(p2p) != 1:
            raise ValueError(
                f"MultiConnector for PD must contain exactly one P2P transfer connector "
                f"out of {SUPPORTED_PD_P2P_CONNECTORS}, got children="
                f"{[c.get('kv_connector') for c in children]}"
            )
        return p2p[0]
    if connector in SUPPORTED_PD_P2P_CONNECTORS:
        return connector
    raise ValueError(
        f"Unsupported kv_connector for PD: {connector!r}. Supported: "
        f"{SUPPORTED_PD_P2P_CONNECTORS} (optionally wrapped in a MultiConnector "
        f"together with store connectors such as MooncakeStoreConnector)."
    )


def get_pd_cli_args(
    cli_args: Namespace,
    role: str = "prefill",
    role_init_kwargs: Optional[Dict[str, Any]] = None,
) -> Namespace:
    """Build PD-specific CLI args by injecting ``kv_role=kv_both``.

    Applies *role_init_kwargs* (``prefill_init_kwargs`` / ``decode_init_kwargs``
    pass-through, mutually exclusive with ``engine_init_kwargs``) on top of the
    base args, then reads ``kv_transfer_config`` from the resulting namespace and
    injects ``kv_role=kv_both``. ``VLLMServerActor._setup_nixl_side_channel``
    later enriches the dict with ``engine_id``.

    Args:
        cli_args: Base CLI args from :func:`build_vllm_cli_args`.
        role: Currently only used for error messages (kv_role is always
            ``kv_both``).
        role_init_kwargs: Role-specific vLLM engine kwargs to apply on top of the
            base args (e.g. a different ``all2all_backend`` for prefill vs decode).

    Returns:
        A deep copy of *cli_args* with *role_init_kwargs* applied and
        ``kv_transfer_config`` as a dict containing ``kv_role=kv_both``.
    """
    args = copy.deepcopy(cli_args)

    if role_init_kwargs:
        for key, value in role_init_kwargs.items():
            setattr(args, key, value)

    kv_config = getattr(args, "kv_transfer_config", None)
    if kv_config is None:
        raise ValueError(
            f"kv_transfer_config must be set when enable_pd=True (via engine_init_kwargs, or via "
            f"prefill_init_kwargs/decode_init_kwargs when using role-specific overrides; missing for "
            f"role={role!r}). E.g. engine_init_kwargs.kv_transfer_config.kv_connector=NixlConnector"
        )

    # kv_transfer_config arrives as a dict from Hydra's nested key resolution
    if isinstance(kv_config, str):
        kv_config = json.loads(kv_config)

    if "kv_connector" not in kv_config:
        raise ValueError("kv_transfer_config.kv_connector must be set when enable_pd=True")

    # Validates the connector (bare NIXL/Mooncake or MultiConnector wrapping one).
    get_pd_p2p_connector_name(kv_config)

    # NIXL runs both roles as kv_both; Mooncake's push-based flow needs explicit
    # kv_producer/kv_consumer roles, so only default when the user did not set one.
    kv_config.setdefault("kv_role", "kv_both")
    args.kv_transfer_config = kv_config

    return args


def build_router_args(
    ie_cfg: InferenceEngineConfig,
    server_urls: Optional[List[str]] = None,
    prefill_urls: Optional[List[str]] = None,
    decode_urls: Optional[List[str]] = None,
    prefill_bootstrap_ports: Optional[List[int]] = None,
    pd_kv_connector: Optional[str] = None,
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
        prefill_bootstrap_ports: Per-prefill-server Mooncake bootstrap ports
            (parallel to *prefill_urls*). Required when *pd_kv_connector* is
            ``"mooncake"``; the router queries each prefill's bootstrap server
            for its engine_id and injects transfer ids per request.
        pd_kv_connector: vllm-router PD transfer flavor: ``"nixl"``
            (pull-based, default) or ``"mooncake"`` (push-based).

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
        # prefill_urls in RouterArgs expects List[Tuple[str, Optional[int]]],
        # where the second element is the Mooncake bootstrap port (None for NIXL).
        if prefill_bootstrap_ports is not None:
            assert len(prefill_bootstrap_ports) == len(prefill_urls), (
                f"prefill_bootstrap_ports ({len(prefill_bootstrap_ports)}) must parallel "
                f"prefill_urls ({len(prefill_urls)})"
            )
            kwargs["prefill_urls"] = list(zip(prefill_urls, prefill_bootstrap_ports))
        else:
            kwargs["prefill_urls"] = [(url, None) for url in prefill_urls]
        kwargs["decode_urls"] = decode_urls
        kwargs["vllm_pd_disaggregation"] = True
        if pd_kv_connector is not None:
            kwargs["kv_connector"] = pd_kv_connector
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
