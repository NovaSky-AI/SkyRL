import logging
from argparse import Namespace

from skyrl.backends.skyrl_train.weight_sync import get_transfer_strategy
from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict

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
        weight_transfer_config=WeightTransferConfig(
            backend=get_transfer_strategy(ie_cfg.weight_sync_backend, cfg.trainer.placement.colocate_all),
        ),
        # NOTE (sumanthrh): We set generation config to be vLLM so that the generation behaviour of the server is same as using the vLLM Engine APIs directly
        generation_config="vllm",
        # NOTE: vllm expects a list entry for served_model_name
        served_model_name=(
            [cfg.generator.inference_engine.served_model_name]
            if cfg.generator.inference_engine.served_model_name
            else None
        ),
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
