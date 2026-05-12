import tempfile

import pytest
import safetensors.torch
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from tx.models import Qwen3Config
from tx.torch.models.qwen3 import Qwen3ForCausalLM
from tx.torch.layers.lora import LoRAMixin

pytestmark = pytest.mark.torch  # Mark all tests in this file as torch tests


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lora_weights(
    module: LoRAMixin,
    adapter_idx: int,
    lora_A_weights: torch.Tensor,
    lora_B_weights: torch.Tensor,
    scaling: float,
    rank: int,
) -> None:
    """Load LoRA weights from tensors to a module with LoRA support.

    This is a generic helper that works with any LoRAMixin module.

    Args:
        module: Module with LoRA support (LoRAMixin)
        adapter_idx: Index of the adapter to load weights into
        lora_A_weights: Weights for lora_A matrix [in_features, rank]
        lora_B_weights: Weights for lora_B matrix [rank, out_features]
        scaling: Scaling factor (typically lora_alpha / rank)
        rank: Rank of the LoRA adapter
    """
    assert module.lora_A is not None and module.lora_B is not None
    assert module.lora_scaling is not None and module.lora_ranks is not None

    with torch.no_grad():
        # Copy lora_A and lora_B weights
        module.lora_A[adapter_idx, :, :rank].copy_(lora_A_weights)
        module.lora_B[adapter_idx, :rank, :].copy_(lora_B_weights)

        # Set scaling and rank
        module.lora_scaling[adapter_idx] = scaling
        module.lora_ranks[adapter_idx] = rank


def load_model_from_hf_checkpoint(
    checkpoint_dir: str, config: Qwen3Config, device, dtype: torch.dtype = torch.float32
) -> Qwen3ForCausalLM:
    """Load our model from a HuggingFace checkpoint directory."""
    model = Qwen3ForCausalLM(config, dtype=dtype, device=device)

    # Load all safetensors files
    state_dict = {}
    from pathlib import Path

    for file in Path(checkpoint_dir).glob("*.safetensors"):
        state_dict.update(safetensors.torch.load_file(file))

    # Load into model (strict=False because we may have LoRA params that HF doesn't have)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def load_lora_adapter_from_hf(our_model: Qwen3ForCausalLM, hf_peft_model, adapter_idx: int, lora_config: LoraConfig):
    """Load LoRA adapter weights from HuggingFace PEFT model to our model.

    This iterates through all layers and uses the generic load_lora_weights helper
    to load weights from the HF PEFT model structure.
    """
    scaling = lora_config.lora_alpha / lora_config.r
    rank = lora_config.r

    for i, layer in enumerate(our_model.model.layers):
        hf_layer = hf_peft_model.base_model.model.model.layers[i]

        # Attention projections
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            hf_proj = getattr(hf_layer.self_attn, proj_name)
            our_proj = getattr(layer.self_attn, proj_name)
            load_lora_weights(
                our_proj,
                adapter_idx=adapter_idx,
                lora_A_weights=hf_proj.lora_A["default"].weight.T,
                lora_B_weights=hf_proj.lora_B["default"].weight.T,
                scaling=scaling,
                rank=rank,
            )

        # MLP projections
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            hf_proj = getattr(hf_layer.mlp, proj_name)
            our_proj = getattr(layer.mlp, proj_name)
            load_lora_weights(
                our_proj,
                adapter_idx=adapter_idx,
                lora_A_weights=hf_proj.lora_A["default"].weight.T,
                lora_B_weights=hf_proj.lora_B["default"].weight.T,
                scaling=scaling,
                rank=rank,
            )


def test_qwen3_basic_shapes(device):
    """Test that the model initializes and produces correct output shapes."""
    base_config = PretrainedConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=False)

    model = Qwen3ForCausalLM(config, dtype=torch.float32, device=device).to(device)

    # Create dummy input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Check shapes
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert outputs.kv_cache is not None
    assert len(outputs.kv_cache.keys) == config.num_hidden_layers


def test_qwen3_vs_hf(device):
    """Test that our PyTorch implementation matches HuggingFace outputs."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    with tempfile.TemporaryDirectory() as tmp:
        # Load and save HF model
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
        hf_model.save_pretrained(tmp, safe_serialization=True)
        hf_model = hf_model.to(device)
        hf_model.eval()

        # Get HF outputs
        with torch.no_grad():
            hf_outputs = hf_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

        # Load our model from saved checkpoint
        base_config = PretrainedConfig.from_pretrained(model_name)
        config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=False)
        model = load_model_from_hf_checkpoint(tmp, config, device)
        model.eval()

    # Get our outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # Compare outputs
    assert outputs.hidden_states is not None
    hf_hidden_states = hf_outputs.hidden_states
    our_hidden_states = outputs.hidden_states

    # Check first layer (after embedding)
    assert torch.allclose(
        hf_hidden_states[0], our_hidden_states[0], rtol=1e-4, atol=1e-4
    ), f"First hidden state mismatch: max diff = {(hf_hidden_states[0] - our_hidden_states[0]).abs().max()}"

    # Check middle layer
    mid_idx = len(hf_hidden_states) // 2
    assert torch.allclose(
        hf_hidden_states[mid_idx], our_hidden_states[mid_idx], rtol=1e-3, atol=1e-3
    ), f"Middle hidden state mismatch: max diff = {(hf_hidden_states[mid_idx] - our_hidden_states[mid_idx]).abs().max()}"

    # Check final layer
    assert torch.allclose(
        hf_hidden_states[-1], our_hidden_states[-1], rtol=1e-3, atol=1e-3
    ), f"Final hidden state mismatch: max diff = {(hf_hidden_states[-1] - our_hidden_states[-1]).abs().max()}"

    # Check logits
    assert torch.allclose(
        hf_outputs.logits, outputs.logits, rtol=1e-3, atol=1e-3
    ), f"Logits mismatch: max diff = {(hf_outputs.logits - outputs.logits).abs().max()}"


def test_qwen3_lora_adapters(device):
    """Test multiple LoRA adapters by comparing with HuggingFace PEFT models using two different adapters."""
    base_model_name = "Qwen/Qwen3-0.6B"
    lora_adapters = ["pcmoritz/qwen3-0.6b-lora-random", "pcmoritz/qwen3-0.6b-lora-random2"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Use two different inputs to test with different adapters
    inputs = ["The capital of France is", "My name is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    with tempfile.TemporaryDirectory() as base_tmp:
        # Save base model checkpoint
        base_hf_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, attn_implementation="eager", use_safetensors=True
        )
        base_hf_model.save_pretrained(base_tmp, safe_serialization=True)

        # Create HF PEFT models with different adapters
        hf_lora_models = []
        lora_configs = []
        for adapter_name in lora_adapters:
            lora_config = LoraConfig.from_pretrained(adapter_name)
            lora_config.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_configs.append(lora_config)

            hf_base = AutoModelForCausalLM.from_pretrained(
                base_model_name, attn_implementation="eager", use_safetensors=True
            )
            hf_model = get_peft_model(hf_base, lora_config)
            hf_model.load_adapter(adapter_name, adapter_name="default")
            hf_model.to(device)
            hf_model.eval()
            hf_lora_models.append(hf_model)

        # Get outputs from all HF models
        hf_outputs_list = []
        with torch.no_grad():
            for idx in range(len(lora_adapters)):
                hf_output = hf_lora_models[idx](
                    input_ids[idx : idx + 1],
                    attention_mask=attention_mask[idx : idx + 1],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hf_outputs_list.append(hf_output)

        # Create our model with LoRA support and load base weights from checkpoint
        base_config = PretrainedConfig.from_pretrained(base_model_name)
        config = Qwen3Config(
            base_config,
            max_lora_adapters=len(lora_adapters),
            max_lora_rank=max(cfg.r for cfg in lora_configs),
            shard_attention_heads=False,
        )
        model = load_model_from_hf_checkpoint(base_tmp, config, device)

        # Load LoRA adapter weights from all adapters
        for adapter_idx, (hf_model, lora_config) in enumerate(zip(hf_lora_models, lora_configs)):
            load_lora_adapter_from_hf(model, hf_model, adapter_idx, lora_config)

    model.eval()

    # Use different adapter indices for each input
    adapter_indices = torch.arange(len(lora_adapters), dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            adapter_indices=adapter_indices,
            output_hidden_states=True,
        )

    # Compare outputs with corresponding adapters
    for idx in range(len(lora_adapters)):
        max_diff = (hf_outputs_list[idx].logits[0] - outputs.logits[idx]).abs().max().item()
        assert torch.allclose(
            hf_outputs_list[idx].logits[0], outputs.logits[idx], rtol=1e-3, atol=1e-3
        ), f"Adapter {idx} logits mismatch: max diff = {max_diff}"


def test_qwen3_kv_cache(device):
    """Test that KV cache works correctly for generation."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    input_text = "The capital of France is"
    batch = tokenizer([input_text], return_tensors="pt")
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    with tempfile.TemporaryDirectory() as tmp:
        # Save HF model checkpoint
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
        hf_model.save_pretrained(tmp, safe_serialization=True)

        # Load our model from checkpoint
        base_config = PretrainedConfig.from_pretrained(model_name)
        config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=False)
        model = load_model_from_hf_checkpoint(tmp, config, device)
        model.eval()

    # Test 1: Prefill phase (no cache)
    with torch.no_grad():
        output_no_cache = model(input_ids, attention_mask=attention_mask)

    # Test 2: Using cache for next token

    with torch.no_grad():
        # Prefill
        prefill_output = model(input_ids, attention_mask=attention_mask)
        kv_cache = prefill_output.kv_cache

        # Pad cache to accommodate more tokens (e.g., 20 tokens total)
        max_length = 20
        kv_cache = kv_cache.pad_to_length(max_length)

        # Next token (simulate getting next token)
        next_token = output_no_cache.logits[:, -1:, :].argmax(dim=-1)

        # Build attention mask for the full sequence (actual tokens + new token + padding)
        extended_attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=device)], dim=1)

        # Pad attention mask to match cache size
        mask_padding = max_length - extended_attention_mask.shape[1]
        if mask_padding > 0:
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros(1, mask_padding, device=device)], dim=1
            )

        # Compute position for the new token explicitly (matching JAX implementation)
        # The new token is at position cache_position (5 in this case)
        next_position = torch.tensor([[kv_cache.cache_position]], device=device)

        # Generate with cache
        cache_output = model(
            next_token, attention_mask=extended_attention_mask, positions=next_position, kv_cache=kv_cache
        )

    # The cache output should be valid (no NaNs)
    assert not torch.isnan(cache_output.logits).any(), "KV cache produced NaN values"
    assert (
        cache_output.kv_cache.cache_position == input_ids.shape[1] + 1
    ), f"Cache position should be {input_ids.shape[1] + 1}, got {cache_output.kv_cache.cache_position}"
