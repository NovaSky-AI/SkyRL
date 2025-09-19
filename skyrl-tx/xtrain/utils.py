from flax import nnx
from transformers import PretrainedConfig

from xtrain import models


def get_model_class(config: PretrainedConfig) -> nnx.Module:
    "Get the correct model class based on the config."

    for architecture in config.architectures or []:
        if hasattr(models, architecture):
            return getattr(models, architecture)

    raise ValueError(
        f"None of the architectures {config.architectures} is currently supported."
    )


def get_param_mapping(config: PretrainedConfig, model: nnx.Module) -> dict[tuple, str]:
    "Get the mapping from model parameter paths to safetensors keys."

    def get_key(path: tuple) -> str:
        if path[-1] in {"embedding", "kernel"}:
            path = (*path[:-1], "weight")
        return ".".join(map(str, path))

    param_mapping = {}
    model_params = nnx.to_flat_state(nnx.state(model))
    for path, _ in model_params:
        key = get_key(path)
        if "lm_head" in path and config.tie_word_embeddings:
            key = next((get_key(p) for p, _ in model_params if "embed_tokens" in p), key)
        param_mapping[path] = key
    return param_mapping
