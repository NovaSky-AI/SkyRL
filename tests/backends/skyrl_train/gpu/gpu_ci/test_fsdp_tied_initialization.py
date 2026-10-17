"""Check parameter sharing after real FSDP sharding and full-state loading."""

from datetime import timedelta

import pytest
import ray
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.distributed.utils import get_free_port
from skyrl.train.config import FSDPConfig, ModelConfig


@ray.remote(num_gpus=1)
def check_parameter_sharing(port, tied, dtype):
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
    )
    try:
        strategy = FSDPStrategy(FSDPConfig(), model_config=ModelConfig())
        strategy.setup_distributed()
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                tie_word_embeddings=tied,
            )
        ).to(dtype=dtype)
        assert (model.get_input_embeddings().weight is model.get_output_embeddings().weight) is tied
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        count = len(list(model.parameters()))
        previous_swap = torch.__future__.get_swap_module_params_on_conversion()

        result = strategy._fsdp_init_model(model)

        assert isinstance(result, FSDPModule)
        assert all(isinstance(parameter, DTensor) for parameter in result.parameters())
        assert (result.get_input_embeddings().weight is result.get_output_embeddings().weight) is tied
        assert len(list(result.parameters())) == count
        actual = result.state_dict()
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            if isinstance(value, DTensor):
                value = value.full_tensor()
            torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
        assert torch.__future__.get_swap_module_params_on_conversion() == previous_swap
    finally:
        dist.destroy_process_group()


@pytest.mark.usefixtures("ray_init_fixture")
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_fsdp_initialization_preserves_parameter_sharing(tied, dtype):
    ray.get(check_parameter_sharing.remote(get_free_port(), tied, dtype))
