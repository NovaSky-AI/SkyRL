from argparse import Namespace
from typing import Union
from omegaconf import DictConfig
from skyrl_train.config import SkyRLConfig, get_config_as_dict


# TODO: Add a test for validation
def build_vllm_cli_args(cfg: "Union[SkyRLConfig, DictConfig]") -> Namespace:
    """Build CLI args for vLLM server from config."""
    from vllm.entrypoints.openai.cli_args import FrontendArgs
    from vllm import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # Create common CLI args namespace
    parser = FlexibleArgumentParser()
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)
    # parse args without any command line arguments
    args: Namespace = parser.parse_args(args=[])

    overrides = dict(
        model=cfg.trainer.policy.model.path,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        dtype=cfg.generator.model_dtype,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        seed=cfg.trainer.seed,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        enable_prefix_caching=cfg.generator.enable_prefix_caching,
        enforce_eager=cfg.generator.enforce_eager,
        max_num_batched_tokens=cfg.generator.max_num_batched_tokens,
        max_num_seqs=cfg.generator.max_num_seqs,
        enable_sleep_mode=cfg.trainer.placement.colocate_all,
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    # Add LoRA params if enabled
    if cfg.trainer.policy.model.lora.rank > 0:
        args.enable_lora = True
        args.max_lora_rank = cfg.trainer.policy.model.lora.rank
        args.max_loras = 1
        args.fully_sharded_loras = cfg.generator.fully_sharded_loras

    # Add any extra engine_init_kwargs
    engine_kwargs = get_config_as_dict(cfg.generator.engine_init_kwargs)
    for key, value in engine_kwargs.items():
        setattr(args, key, value)

    return args
