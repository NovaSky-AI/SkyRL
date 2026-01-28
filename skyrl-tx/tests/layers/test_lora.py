import jax
import jax.numpy as jnp
from flax import nnx

from tx.layers.lora import LoRAEmbed


def test_lora_embed_transposed():
    """Test that LoRAEmbed.T correctly applies LoRA adapters."""
    vocab_size = 100
    features = 32
    max_lora_adapters = 2
    max_lora_rank = 4
    batch_size = 2
    seq_len = 5

    mesh = jax.make_mesh((1,), ("dp",))
    with jax.set_mesh(mesh):
        embed = LoRAEmbed(
            num_embeddings=vocab_size,
            features=features,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=jnp.float32,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(0.02), (None, None)),
            rngs=nnx.Rngs(0),
        )

        # Set known LoRA weights for testing
        # lora_A: (adapters, vocab_size, rank)
        # lora_B: (adapters, rank, features)
        lora_A_val = jnp.ones((max_lora_adapters, vocab_size, max_lora_rank)) * 0.1
        lora_B_val = jnp.ones((max_lora_adapters, max_lora_rank, features)) * 0.2
        embed.lora_A[...] = lora_A_val
        embed.lora_B[...] = lora_B_val

        # Test input
        hidden_states = jax.random.normal(jax.random.key(42), (batch_size, seq_len, features))
        adapter_indices = jnp.array([0, 1], dtype=jnp.int32)

        # Get the transposed projection callable
        project = embed.T

        # Output without LoRA (adapter_indices=None)
        base_output = project(hidden_states, adapter_indices=None)
        expected_base = hidden_states @ embed.embedding[...].T
        assert jnp.allclose(base_output, expected_base), "Base output without LoRA should match"

        # Output with LoRA
        lora_output = project(hidden_states, adapter_indices=adapter_indices)

        # Verify the math: lora_contribution = hidden_states @ B.T @ A.T
        # For each sample in batch, use its adapter's weights
        for i in range(batch_size):
            adapter_idx = adapter_indices[i]
            h = hidden_states[i]  # (seq_len, features)
            lora_B_T = lora_B_val[adapter_idx].T  # (features, rank)
            lora_A_T = lora_A_val[adapter_idx].T  # (rank, vocab_size)
            expected_lora_contribution = h @ lora_B_T @ lora_A_T  # (seq_len, vocab_size)
            expected_total = expected_base[i] + expected_lora_contribution

            assert jnp.allclose(lora_output[i], expected_total, atol=1e-5), (
                f"LoRA math incorrect for batch {i}"
            )


def test_lora_embed_forward_and_transposed_consistency():
    """Test that forward and transposed LoRA use the same weights correctly."""
    vocab_size = 50
    features = 16
    max_lora_adapters = 1
    max_lora_rank = 4
    batch_size = 1
    seq_len = 3

    mesh = jax.make_mesh((1,), ("dp",))
    with jax.set_mesh(mesh):
        embed = LoRAEmbed(
            num_embeddings=vocab_size,
            features=features,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=jnp.float32,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(0.02), (None, None)),
            rngs=nnx.Rngs(0),
        )

        # Set LoRA weights
        lora_A_val = jax.random.normal(jax.random.key(1), (max_lora_adapters, vocab_size, max_lora_rank)) * 0.1
        lora_B_val = jax.random.normal(jax.random.key(2), (max_lora_adapters, max_lora_rank, features)) * 0.1
        embed.lora_A[...] = lora_A_val
        embed.lora_B[...] = lora_B_val

        adapter_indices = jnp.array([0], dtype=jnp.int32)

        # Forward pass: token_ids -> embeddings
        token_ids = jnp.array([[5, 10, 15]], dtype=jnp.int32)
        forward_output = embed(token_ids, adapter_indices=adapter_indices)

        # Expected forward: base_embed + A[token_ids] @ B
        base_embed = embed.embedding[...][token_ids]  # (1, 3, features)
        lora_A_lookup = lora_A_val[0, token_ids[0], :]  # (3, rank)
        forward_lora_contribution = lora_A_lookup @ lora_B_val[0]  # (3, features)
        expected_forward = base_embed + forward_lora_contribution

        assert jnp.allclose(forward_output, expected_forward, atol=1e-5), "Forward LoRA incorrect"

        # Transposed pass: hidden_states -> logits
        hidden_states = jax.random.normal(jax.random.key(3), (batch_size, seq_len, features))
        transposed_output = embed.T(hidden_states, adapter_indices=adapter_indices)

        # Expected transposed: hidden @ embed.T + hidden @ B.T @ A.T
        base_transposed = hidden_states @ embed.embedding[...].T
        lora_B_T = lora_B_val[0].T  # (features, rank)
        lora_A_T = lora_A_val[0].T  # (rank, vocab_size)
        transposed_lora_contribution = hidden_states @ lora_B_T @ lora_A_T
        expected_transposed = base_transposed + transposed_lora_contribution

        assert jnp.allclose(transposed_output, expected_transposed, atol=1e-5), "Transposed LoRA incorrect"
