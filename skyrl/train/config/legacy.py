"""
Legacy configuration translation for backward compatibility.

This module provides translation from the old YAML-based configuration format
to the new dataclass-based configuration format.
"""

import copy
import warnings
from typing import Any, Dict

# Fields that moved from generator.* to generator.inference_engine.*
# Maps old field name -> new field name (None means same name)
GENERATOR_TO_INFERENCE_ENGINE_FIELDS: Dict[str, str | None] = {
    "model_dtype": None,
    "run_engines_locally": None,
    "num_inference_engines": "num_engines",
    "backend": None,
    "weight_sync_backend": None,
    "weight_transfer_threshold_cuda_ipc_GB": None,
    "inference_engine_tensor_parallel_size": "tensor_parallel_size",
    "inference_engine_pipeline_parallel_size": "pipeline_parallel_size",
    "inference_engine_expert_parallel_size": "expert_parallel_size",
    "inference_engine_data_parallel_size": "data_parallel_size",
    "async_engine": None,
    "vllm_v1_disable_multiproc": None,
    "enable_prefix_caching": None,
    "enable_chunked_prefill": None,
    "max_num_batched_tokens": None,
    "enforce_eager": None,
    "fully_sharded_loras": None,
    "enable_ray_prometheus_stats": None,
    "gpu_memory_utilization": None,
    "max_num_seqs": None,
    "remote_inference_engine_urls": "remote_urls",
    "enable_http_endpoint": None,
    "http_endpoint_host": None,
    "http_endpoint_port": None,
    "served_model_name": None,
    "engine_init_kwargs": None,
    "override_existing_update_group": None,
    "external_proxy_url": None,
    "external_server_urls": None,
}

# Fields that should be removed (deprecated or derived)
DEPRECATED_FIELDS = {
    "generator": {"model_name"},  # Now derived from trainer.policy.model.path
}


def is_legacy_config(config: Dict[str, Any]) -> bool:
    """Check if the config uses the legacy format.

    Legacy format has inference engine fields directly under 'generator'
    instead of nested under 'generator.inference_engine'.
    """
    generator = config.get("generator", {})
    if not isinstance(generator, dict):
        return False

    # Check if any legacy field exists at generator level
    for old_field in GENERATOR_TO_INFERENCE_ENGINE_FIELDS:
        if old_field in generator:
            return True

    return False


def translate_legacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a legacy config dict to the new format.

    Args:
        config: Configuration dict in legacy format.

    Returns:
        Configuration dict in new format.
    """
    config = copy.deepcopy(config)

    # Translate generator fields
    if "generator" in config and isinstance(config["generator"], dict):
        generator = config["generator"]

        # Initialize inference_engine if not present
        if "inference_engine" not in generator:
            generator["inference_engine"] = {}

        inference_engine = generator["inference_engine"]

        # Move fields from generator to inference_engine
        for old_field, new_field in GENERATOR_TO_INFERENCE_ENGINE_FIELDS.items():
            if old_field in generator:
                target_field = new_field if new_field else old_field
                inference_engine[target_field] = generator.pop(old_field)

        # Remove deprecated fields
        for field in DEPRECATED_FIELDS.get("generator", set()):
            generator.pop(field, None)

    return config


def warn_legacy_config():
    """Issue a warning about using legacy configuration format."""
    warnings.warn(
        "You are using the legacy YAML configuration format. "
        "Please migrate to the new pythonic configuration format. "
        "See the migration guide for details. "
        "Legacy configuration support may be removed in a future release.",
        DeprecationWarning,
        stacklevel=4,
    )
