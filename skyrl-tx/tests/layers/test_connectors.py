import jax
import jax.numpy as jnp
import numpy as np
import pytest
import safetensors.numpy
from flax import nnx

from tx.layers.connectors import is_connector_path
from tx.tinker.types import LoraConfig
from tx.utils.models import (
    extract_adapter_state,
    insert_adapter_state,
    load_lora_checkpoint,
    load_safetensors,
    save_lora_checkpoint,
    save_safetensors,
)
from tx.utils.storage import download_and_unpack


@pytest.fixture(scope="module")
def mesh():
    return jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)


@pytest.mark.parametrize("expansion_rate", [1, 2, 4])
def test_connector_shapes(mesh, expansion_rate: int):
    """Test that LoRAConnector produces correct output shapes."""
    with jax.set_mesh(mesh):
        from tx.layers.connectors import LoRAConnector

        hidden_dim = 64
        batch, seq = 2, 8

        conn = LoRAConnector(
            hidden_dim,
            expansion_rate,
            max_lora_adapters=4,
            dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        adapter_indices = jnp.array([1, 2], dtype=jnp.int32)

        x = jnp.ones((batch, seq, expansion_rate, hidden_dim))
        pre_out, residual_norm = conn.pre(x, lambda y: y, adapter_indices)
        post_out = conn.post(x, pre_out, residual_norm, adapter_indices)

        assert pre_out.shape == (batch, seq, hidden_dim)
        assert residual_norm.shape == (batch, seq, expansion_rate * hidden_dim)
        assert post_out.shape == (batch, seq, expansion_rate, hidden_dim)


@pytest.mark.parametrize("expansion_rate", [1, 2, 4])
def test_connector_identity_initialization(mesh, expansion_rate: int):
    """Test that LoRAConnector identity initialization behaves like residual connection per adapter slot."""
    with jax.set_mesh(mesh):
        from tx.layers.connectors import LoRAConnector

        hidden_dim = 64
        n = expansion_rate

        conn = LoRAConnector(
            hidden_dim,
            n,
            max_lora_adapters=3,
            dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        adapter_idx = 0
        adapter_indices = jnp.array([adapter_idx], dtype=jnp.int32)

        # Verify H_pre = 1/n
        b_pre = conn.b_pre[adapter_indices]
        h_pre = jax.nn.sigmoid(b_pre[0])
        assert np.allclose(h_pre, 1.0 / n, atol=1e-5)

        # Verify H_post = 1
        b_post = conn.b_post[adapter_indices]
        h_post = 2.0 * jax.nn.sigmoid(b_post[0])
        assert np.allclose(h_post, 1.0, atol=1e-6)

        # Verify M = I
        b_res = conn.b_res[adapter_indices]
        M = conn._sinkhorn_knopp(b_res[0])
        assert np.allclose(M, jnp.eye(n), atol=1e-3)


def test_deepseek_connector_identity_expansion_rate():
    """Initial connector behavior should keep logits unchanged across expansion rates."""
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config as HFDeepseekV3Config

    from tx.models.configs import DeepseekV3Config
    from tx.models.deepseekv3 import DeepseekV3ForCausalLM

    base_config = HFDeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        q_lora_rank=None,
        kv_lora_rank=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
    )
    config_e1 = DeepseekV3Config(base_config, max_lora_adapters=4, max_lora_rank=8, shard_attention_heads=True)
    config_e4 = DeepseekV3Config(base_config, max_lora_adapters=4, max_lora_rank=8, shard_attention_heads=True)
    config_e1.expansion_rate = 1
    config_e4.expansion_rate = 4

    input_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
    attention_mask = np.ones_like(input_ids, dtype=np.int32)

    mesh = jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 3)
    with jax.set_mesh(mesh):
        model_e1 = DeepseekV3ForCausalLM(config_e1, dtype=jnp.float32, rngs=nnx.Rngs(0))
        model_e4 = DeepseekV3ForCausalLM(config_e4, dtype=jnp.float32, rngs=nnx.Rngs(0))

        # Align all non-connector weights exactly so only connector expansion differs.
        state_e1 = nnx.state(model_e1)
        state_e4 = nnx.state(model_e4)

        def copy_non_connector(path, v1, v4):
            normalized_path = tuple(p.key if hasattr(p, "key") else p.name for p in path)
            if any(name in normalized_path for name in ("attn_connector", "mlp_connector")):
                return v1
            return v4

        nnx.update(model_e1, jax.tree.map_with_path(copy_non_connector, state_e1, state_e4))

        outputs_e1 = model_e1(input_ids, attention_mask=attention_mask)
        outputs_e4 = model_e4(input_ids, attention_mask=attention_mask)
        logits_e1 = np.asarray(model_e1.compute_logits(outputs_e1.last_hidden_state))
        logits_e4 = np.asarray(model_e4.compute_logits(outputs_e4.last_hidden_state))

    np.testing.assert_allclose(logits_e1, logits_e4, rtol=5e-2, atol=5e-2)


class _TinyConfig:
    def __init__(self, train_connectors: bool):
        self.train_connectors = train_connectors

    def get_num_experts(self):
        return 0


class _TinyLoRA(nnx.Module):
    def __init__(self, max_adapters: int, max_rank: int):
        self.lora_A = nnx.Param(jnp.zeros((max_adapters, 5, max_rank), dtype=jnp.float32))
        self.lora_B = nnx.Param(jnp.zeros((max_adapters, max_rank, 6), dtype=jnp.float32))


class _TinyConnector(nnx.Module):
    def __init__(self, max_adapters: int):
        self.alpha_pre = nnx.Param(jnp.zeros((max_adapters, 4), dtype=jnp.float32))
        self.phi_pre = nnx.Param(jnp.zeros((max_adapters, 4, 2), dtype=jnp.float32))


class _TinyModel(nnx.Module):
    def __init__(self, train_connectors: bool, max_adapters: int = 3, max_rank: int = 4):
        self.config = _TinyConfig(train_connectors=train_connectors)
        self.self_attn = _TinyLoRA(max_adapters=max_adapters, max_rank=max_rank)
        self.attn_connector = _TinyConnector(max_adapters=max_adapters)

    def is_lora_param(self, path: tuple, _value) -> bool:
        is_lora = "lora_A" in path or "lora_B" in path
        is_connector = self.config.train_connectors and is_connector_path(path)
        return is_lora or is_connector


def _fill_tiny_model(model: _TinyModel) -> None:
    model.self_attn.lora_A[...] = jnp.arange(np.prod(model.self_attn.lora_A[...].shape), dtype=jnp.float32).reshape(
        model.self_attn.lora_A[...].shape
    )
    model.self_attn.lora_B[...] = (
        jnp.arange(np.prod(model.self_attn.lora_B[...].shape), dtype=jnp.float32).reshape(
            model.self_attn.lora_B[...].shape
        )
        + 1000
    )
    model.attn_connector.alpha_pre[...] = (
        jnp.arange(np.prod(model.attn_connector.alpha_pre[...].shape), dtype=jnp.float32).reshape(
            model.attn_connector.alpha_pre[...].shape
        )
        + 2000
    )
    model.attn_connector.phi_pre[...] = (
        jnp.arange(np.prod(model.attn_connector.phi_pre[...].shape), dtype=jnp.float32).reshape(
            model.attn_connector.phi_pre[...].shape
        )
        + 3000
    )


def _zero_tiny_model(model: _TinyModel) -> None:
    model.self_attn.lora_A[...] = 0
    model.self_attn.lora_B[...] = 0
    model.attn_connector.alpha_pre[...] = 0
    model.attn_connector.phi_pre[...] = 0


def _adapter_filter(path: tuple) -> bool:
    return ("lora_A" in path or "lora_B" in path) or is_connector_path(path)


def test_connector_adapter_slice_save_load_safetensors(tmp_path):
    src = _TinyModel(train_connectors=True)
    dst = _TinyModel(train_connectors=True)
    _fill_tiny_model(src)
    _zero_tiny_model(dst)

    save_safetensors(
        src.config,
        src,
        tmp_path / "adapter.safetensors",
        filter_fn=_adapter_filter,
        adapter_index=1,
        rank=2,
    )
    load_safetensors(
        tmp_path,
        dst.config,
        dst,
        skip_lora=False,
        filter_fn=_adapter_filter,
        adapter_index=1,
        rank=2,
    )

    np.testing.assert_allclose(
        np.asarray(dst.self_attn.lora_A[...][1, :, :2]), np.asarray(src.self_attn.lora_A[...][1, :, :2])
    )
    np.testing.assert_allclose(
        np.asarray(dst.self_attn.lora_B[...][1, :2, :]), np.asarray(src.self_attn.lora_B[...][1, :2, :])
    )
    np.testing.assert_allclose(
        np.asarray(dst.attn_connector.alpha_pre[...][1]), np.asarray(src.attn_connector.alpha_pre[...][1])
    )
    np.testing.assert_allclose(
        np.asarray(dst.attn_connector.phi_pre[...][1]), np.asarray(src.attn_connector.phi_pre[...][1])
    )
    assert np.allclose(np.asarray(dst.attn_connector.alpha_pre[...][0]), 0)
    assert np.allclose(np.asarray(dst.attn_connector.alpha_pre[...][2]), 0)


def test_connector_extract_insert_adapter_state_roundtrip():
    rank = 2
    adapter_index = 1

    src = _TinyModel(train_connectors=True)
    _fill_tiny_model(src)
    _, src_lora_params, _ = nnx.split(src, src.is_lora_param, ...)
    extracted = extract_adapter_state(adapter_index, src_lora_params, rank)

    dst = _TinyModel(train_connectors=True)
    _zero_tiny_model(dst)
    _, dst_lora_params, _ = nnx.split(dst, dst.is_lora_param, ...)
    insert_adapter_state(adapter_index, dst_lora_params, extracted, rank)

    for path, leaf in jax.tree.leaves_with_path(dst_lora_params):
        key = path[-2].key if hasattr(path[-2], "key") else str(path[-2])
        arr = leaf.value if hasattr(leaf, "value") else leaf
        if key == "alpha_pre":
            np.testing.assert_allclose(
                np.asarray(arr[adapter_index]), np.asarray(src.attn_connector.alpha_pre[...][adapter_index])
            )
        elif key == "phi_pre":
            np.testing.assert_allclose(
                np.asarray(arr[adapter_index]), np.asarray(src.attn_connector.phi_pre[...][adapter_index])
            )
        elif key == "lora_A":
            np.testing.assert_allclose(
                np.asarray(arr[adapter_index, :, :rank]),
                np.asarray(src.self_attn.lora_A[...][adapter_index, :, :rank]),
            )
        elif key == "lora_B":
            np.testing.assert_allclose(
                np.asarray(arr[adapter_index, :rank, :]),
                np.asarray(src.self_attn.lora_B[...][adapter_index, :rank, :]),
            )


def test_lora_checkpoint_includes_connectors_when_trainable(tmp_path):
    adapter_cfg = LoraConfig(rank=2, alpha=4, seed=0)

    model_true = _TinyModel(train_connectors=True)
    _fill_tiny_model(model_true)
    ckpt_true = tmp_path / "with_connectors.tar.gz"
    save_lora_checkpoint(model_true, "dummy/base", adapter_cfg, adapter_index=1, output_path=ckpt_true)

    with download_and_unpack(ckpt_true) as extracted_dir:
        tensors = safetensors.numpy.load_file(extracted_dir / "adapter_model.safetensors")
        assert any("attn_connector" in key for key in tensors.keys())

    loaded_true = _TinyModel(train_connectors=True)
    _zero_tiny_model(loaded_true)
    load_lora_checkpoint(loaded_true, adapter_cfg, adapter_index=1, checkpoint_path=ckpt_true)
    np.testing.assert_allclose(
        np.asarray(loaded_true.attn_connector.alpha_pre[...][1]),
        np.asarray(model_true.attn_connector.alpha_pre[...][1]),
    )

    model_false = _TinyModel(train_connectors=False)
    _fill_tiny_model(model_false)
    ckpt_false = tmp_path / "without_connectors.tar.gz"
    save_lora_checkpoint(model_false, "dummy/base", adapter_cfg, adapter_index=1, output_path=ckpt_false)

    with download_and_unpack(ckpt_false) as extracted_dir:
        tensors = safetensors.numpy.load_file(extracted_dir / "adapter_model.safetensors")
        assert not any("attn_connector" in key for key in tensors.keys())
