from argparse import Namespace
from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict


# TODO: Add a test for validation
def build_vllm_cli_args(cfg: SkyRLTrainConfig) -> Namespace:
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
        max_num_seqs=ie_cfg.max_num_seqs,
        enable_sleep_mode=cfg.trainer.placement.colocate_all,
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

    # Add LoRA params if enabled
    if cfg.trainer.policy.model.lora.rank > 0:
        args.enable_lora = True
        args.max_lora_rank = cfg.trainer.policy.model.lora.rank
        args.max_loras = 1
        args.fully_sharded_loras = ie_cfg.fully_sharded_loras

    # Add any extra engine_init_kwargs
    engine_kwargs = get_config_as_dict(ie_cfg.engine_init_kwargs)
    for key, value in engine_kwargs.items():
        setattr(args, key, value)

    return args
