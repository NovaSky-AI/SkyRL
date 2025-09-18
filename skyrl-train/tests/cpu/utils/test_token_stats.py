import pytest
import torch
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from skyrl_train.utils.token_stats import (
    TokenStats,
    CorpusDataset,
    collate_corpus_batch,
    compute_token_entropy,
    compute_corpus_entropy,
    save_token_stats,
    load_token_stats,
    select_minority_tokens,
    save_minority_tokens,
    load_minority_tokens,
    create_sample_corpus
)


class TestTokenStats:
    """Test the TokenStats dataclass."""
    
    def test_token_stats_initialization(self):
        """Test TokenStats initialization."""
        stats = TokenStats(token_id=42, frequency=0, entropy_sum=0.0, entropy_count=0)
        assert stats.token_id == 42
        assert stats.frequency == 0
        assert stats.entropy_sum == 0.0
        assert stats.entropy_count == 0
        assert stats.avg_entropy == 0.0
    
    def test_token_stats_update(self):
        """Test TokenStats update method."""
        stats = TokenStats(token_id=42, frequency=0, entropy_sum=0.0, entropy_count=0)
        
        # Update with first entropy value
        stats.update(1.5)
        assert stats.frequency == 1
        assert stats.entropy_sum == 1.5
        assert stats.entropy_count == 1
        assert stats.avg_entropy == 1.5
        
        # Update with second entropy value
        stats.update(2.5)
        assert stats.frequency == 2
        assert stats.entropy_sum == 4.0
        assert stats.entropy_count == 2
        assert stats.avg_entropy == 2.0
    
    def test_token_stats_to_dict(self):
        """Test TokenStats to_dict method."""
        stats = TokenStats(token_id=42, frequency=10, entropy_sum=15.0, entropy_count=10)
        result = stats.to_dict()
        
        expected = {
            "freq": 10,
            "avg_entropy": 1.5,
            "entropy_sum": 15.0,
            "entropy_count": 10
        }
        assert result == expected
    
    def test_token_stats_from_dict(self):
        """Test TokenStats from_dict method."""
        data = {
            "freq": 10,
            "avg_entropy": 1.5,
            "entropy_sum": 15.0,
            "entropy_count": 10
        }
        stats = TokenStats.from_dict(token_id=42, data=data)
        
        assert stats.token_id == 42
        assert stats.frequency == 10
        assert stats.entropy_sum == 15.0
        assert stats.entropy_count == 10
        assert stats.avg_entropy == 1.5


class TestCorpusDataset:
    """Test the CorpusDataset class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        return tokenizer
    
    def test_corpus_dataset_initialization(self, mock_tokenizer):
        """Test CorpusDataset initialization."""
        texts = ["Hello world", "Test sentence"]
        dataset = CorpusDataset(texts, mock_tokenizer, max_length=512)
        
        assert len(dataset) == 2
        assert dataset.texts == texts
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 512
    
    def test_corpus_dataset_getitem(self, mock_tokenizer):
        """Test CorpusDataset __getitem__ method."""
        texts = ["Hello world"]
        dataset = CorpusDataset(texts, mock_tokenizer, max_length=512, include_labels=True)
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert torch.equal(item["input_ids"], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(item["attention_mask"], torch.tensor([1, 1, 1, 1]))
        assert torch.equal(item["labels"], torch.tensor([1, 2, 3, 4]))


class TestCollateBatch:
    """Test the collate_corpus_batch function."""
    
    def test_collate_corpus_batch_basic(self):
        """Test basic collate functionality."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "pad_token_id": 0
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "pad_token_id": 0
            }
        ]
        
        result = collate_corpus_batch(batch)
        
        expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        
        assert torch.equal(result["input_ids"], expected_input_ids)
        assert torch.equal(result["attention_mask"], expected_attention_mask)
    
    def test_collate_corpus_batch_with_labels(self):
        """Test collate with labels."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
                "pad_token_id": 0
            },
            {
                "input_ids": torch.tensor([3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([3, 4, 5]),
                "pad_token_id": 0
            }
        ]
        
        result = collate_corpus_batch(batch)
        
        expected_labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
        assert torch.equal(result["labels"], expected_labels)


class TestComputeTokenEntropy:
    """Test the compute_token_entropy function."""
    
    def test_compute_token_entropy_basic(self):
        """Test basic entropy computation."""
        # Create logits that should produce known entropy
        # Uniform distribution over 4 tokens should give entropy = log(4) ≈ 1.386
        logits = torch.zeros(2, 3, 4)  # batch=2, seq_len=3, vocab=4
        entropy = compute_token_entropy(logits)
        
        assert entropy.shape == (2, 3)
        # Uniform distribution entropy = log(vocab_size)
        expected_entropy = torch.log(torch.tensor(4.0))
        assert torch.allclose(entropy, expected_entropy, atol=1e-6)
    
    def test_compute_token_entropy_deterministic(self):
        """Test entropy with deterministic distribution."""
        # Create logits where one token has very high probability
        logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])  # shape: (1, 1, 4)
        entropy = compute_token_entropy(logits)
        
        # Nearly deterministic distribution should have very low entropy
        assert entropy.shape == (1, 1)
        assert entropy.item() < 0.1  # Very low entropy


class TestSelectMinorityTokens:
    """Test the select_minority_tokens function."""
    
    def test_select_minority_tokens_basic(self):
        """Test basic minority token selection."""
        # Create token stats with known distribution
        token_stats = {
            1: TokenStats(1, frequency=100, entropy_sum=10.0, entropy_count=10),  # High freq, low entropy
            2: TokenStats(2, frequency=50, entropy_sum=15.0, entropy_count=10),   # Med freq, med entropy  
            3: TokenStats(3, frequency=10, entropy_sum=25.0, entropy_count=10),   # Low freq, high entropy
            4: TokenStats(4, frequency=5, entropy_sum=30.0, entropy_count=10),    # Very low freq, very high entropy
            5: TokenStats(5, frequency=1, entropy_sum=2.0, entropy_count=2),      # Only few observations
        }
        
        selected_tokens, metadata = select_minority_tokens(
            token_stats=token_stats,
            freq_percentile=50.0,  # Should select tokens with freq <= 10
            top_k=2,
            min_observations=5
        )
        
        # Should select tokens 4 and 3 (highest entropy among minority tokens with enough observations)
        assert len(selected_tokens) == 2
        assert selected_tokens[0] == 4  # Highest entropy minority token
        assert selected_tokens[1] == 3  # Second highest entropy minority token
        
        assert metadata["num_selected"] == 2
        assert metadata["freq_percentile"] == 50.0
        assert metadata["top_k"] == 2
    
    def test_select_minority_tokens_empty_result(self):
        """Test minority token selection with no valid tokens."""
        token_stats = {
            1: TokenStats(1, frequency=100, entropy_sum=10.0, entropy_count=2),  # Only few observations
        }
        
        selected_tokens, metadata = select_minority_tokens(
            token_stats=token_stats,
            min_observations=5
        )
        
        assert len(selected_tokens) == 0
        assert "error" in metadata


class TestSaveLoadFunctions:
    """Test save/load functionality."""
    
    def test_save_load_token_stats(self):
        """Test saving and loading token statistics."""
        token_stats = {
            1: TokenStats(1, frequency=10, entropy_sum=15.0, entropy_count=10),
            2: TokenStats(2, frequency=20, entropy_sum=25.0, entropy_count=15),
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_stats.json")
            
            # Save token stats
            save_token_stats(token_stats, save_path, processed_count=100)
            
            # Verify file exists
            assert os.path.exists(save_path)
            
            # Load token stats
            loaded_stats = load_token_stats(save_path)
            
            # Verify loaded stats match original
            assert len(loaded_stats) == len(token_stats)
            for token_id in token_stats:
                original = token_stats[token_id]
                loaded = loaded_stats[token_id]
                assert loaded.token_id == original.token_id
                assert loaded.frequency == original.frequency
                assert loaded.entropy_sum == original.entropy_sum
                assert loaded.entropy_count == original.entropy_count
    
    def test_save_load_minority_tokens(self):
        """Test saving and loading minority tokens."""
        selected_tokens = [1, 2, 3]
        metadata = {"test": "data"}
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_minority.json")
            
            # Save minority tokens
            save_minority_tokens(selected_tokens, metadata, mock_tokenizer, save_path)
            
            # Verify file exists
            assert os.path.exists(save_path)
            
            # Load minority tokens
            loaded_tokens, loaded_metadata = load_minority_tokens(save_path)
            
            # Verify loaded data matches original
            assert loaded_tokens == selected_tokens
            assert loaded_metadata["test"] == "data"


class TestCreateSampleCorpus:
    """Test the create_sample_corpus function."""
    
    def test_create_sample_corpus_default(self):
        """Test creating sample corpus with default parameters."""
        corpus = create_sample_corpus()
        
        assert len(corpus) == 100
        assert all(text.startswith("Problem ") for text in corpus)
        assert "winter of my twelfth year" in corpus[0]  # First Kite Runner sample
    
    def test_create_sample_corpus_custom_size(self):
        """Test creating sample corpus with custom size."""
        corpus = create_sample_corpus(num_texts=50)
        
        assert len(corpus) == 50
        assert all(text.startswith("Problem ") for text in corpus)
    
    def test_create_sample_corpus_variety(self):
        """Test that sample corpus has variety in content."""
        corpus = create_sample_corpus(num_texts=20)
        
        # Should have variety due to cycling through different base texts
        unique_content = set(text.split(": ", 1)[1] for text in corpus[:8])  # First 8 should be unique
        assert len(unique_content) == 8  # All base texts should appear


# Integration-style tests
class TestTokenStatsIntegration:
    """Integration tests for token statistics pipeline."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer for integration tests."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
        
        # Mock tokenizer call
        def tokenize_side_effect(text, **kwargs):
            # Simple tokenization: convert to list of token IDs
            words = text.split()
            token_ids = [hash(word) % 100 + 1 for word in words]  # Simple hash-based tokenization
            return {
                "input_ids": torch.tensor([token_ids]),
                "attention_mask": torch.tensor([[1] * len(token_ids)])
            }
        mock_tokenizer.side_effect = tokenize_side_effect
        
        # Mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        # Mock model output - create logits that will produce varied entropy
        def model_forward(**kwargs):
            input_ids = kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            vocab_size = 100
            
            # Create varied logits for testing
            logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0
            mock_output = Mock()
            mock_output.logits = logits
            return mock_output
        
        mock_model.side_effect = model_forward
        
        return mock_model, mock_tokenizer
    
    @patch('skyrl_train.utils.token_stats.AutoModelForCausalLM')
    @patch('skyrl_train.utils.token_stats.AutoTokenizer')
    def test_compute_corpus_entropy_integration(self, mock_tokenizer_class, mock_model_class, mock_model_and_tokenizer):
        """Test corpus entropy computation integration."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Test data
        corpus_texts = [
            "Hello world this is a test",
            "Another test sentence here",
            "Final test example text"
        ]
        
        # Run computation
        with patch('torch.no_grad'):  # Skip actual torch.no_grad for testing
            token_stats = compute_corpus_entropy(
                model=mock_model,
                tokenizer=mock_tokenizer,
                corpus_texts=corpus_texts,
                batch_size=2,
                max_length=512,
                device="cpu"
            )
        
        # Verify results
        assert isinstance(token_stats, dict)
        assert len(token_stats) > 0  # Should have collected some token statistics
        
        # Check that all values are TokenStats instances
        for token_id, stats in token_stats.items():
            assert isinstance(token_id, int)
            assert isinstance(stats, TokenStats)
            assert stats.frequency > 0
            assert stats.entropy_count > 0


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestFileOperations:
    """Test file operations and edge cases."""
    
    def test_save_token_stats_creates_directory(self, temp_output_dir):
        """Test that save_token_stats creates necessary directories."""
        token_stats = {
            1: TokenStats(1, frequency=10, entropy_sum=15.0, entropy_count=10)
        }
        
        nested_path = os.path.join(temp_output_dir, "nested", "path", "stats.json")
        save_token_stats(token_stats, nested_path, processed_count=100)
        
        assert os.path.exists(nested_path)
        assert os.path.isfile(nested_path)
    
    def test_load_token_stats_missing_file(self):
        """Test loading token stats from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_token_stats("nonexistent_file.json")
    
    def test_load_token_stats_invalid_json(self, temp_output_dir):
        """Test loading token stats from invalid JSON file."""
        invalid_json_path = os.path.join(temp_output_dir, "invalid.json")
        with open(invalid_json_path, 'w') as f:
            f.write("invalid json content {")
        
        with pytest.raises(json.JSONDecodeError):
            load_token_stats(invalid_json_path)


if __name__ == "__main__":
    pytest.main([__file__])