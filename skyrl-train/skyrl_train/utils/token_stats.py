import os
import time
import sys
import logging
import math
import json
import numpy as np
import ray
import torch
import torch.nn.functional as F

from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy, PlacementGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

from .constants import SKYRL_LD_LIBRARY_PATH_EXPORT, SKYRL_RAY_PG_TIMEOUT_IN_S
from .utils import Timer

@dataclass
class TokenStats:
    """Statistics for a single token across the corpus."""
    token_id: int
    frequency: int
    entropy_sum: float
    entropy_count: int
    
    @property
    def avg_entropy(self) -> float:
        """Average entropy for this token."""
        return self.entropy_sum / self.entropy_count if self.entropy_count > 0 else 0.0
    
    def update(self, entropy: float) -> None:
        """Update token statistics with new entropy observation."""
        self.frequency += 1
        self.entropy_sum += entropy
        self.entropy_count += 1
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "freq": self.frequency,
            "avg_entropy": self.avg_entropy,
            "entropy_sum": self.entropy_sum,
            "entropy_count": self.entropy_count
        }
    
    @classmethod
    def from_dict(cls, token_id: int, data: Dict[str, float]) -> "TokenStats":
        """Creating TokenStats from dictionary."""
        return cls(
            token_id=token_id,
            frequency=int(data["freq"]),
            entropy_sum=data["entropy_sum"],
            entropy_count=int(data["entropy_count"])
        )


class CorpusDataset(Dataset):
    """Dataset wrapper for corpus entropy computation."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: AutoTokenizer, 
        max_length: int = 2048, #not sure if I should do 4086 or 8192
        include_labels: bool = True
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # tokenize with truncation
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # labels are shifted input_ids
        if self.include_labels:
            result["labels"] = input_ids.clone()
        
        return result


def collate_corpus_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for corpus processing with proper padding."""
    
    # Get maximum sequence length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_size = len(batch)
    pad_token_id = batch[0].get("pad_token_id", 0)  # Fallback to 0 if not provided
    
    # Initialize tensors
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if "labels" in batch[0]:
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 is ignored in loss
        result["labels"] = labels
    
    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        
        if "labels" in item:
            labels[i, :seq_len] = item["labels"]
    
    return result


def compute_token_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy H(p) = -∑ p_i log p_i for token distributions.
    
    Args:
        logits: Raw model logits [..., vocab_size]
        dim: Dimension to compute entropy over (default: last dimension)
        
    Returns:
        entropy: Entropy values [...] (same shape as logits except dim is removed)
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=dim)
    
    # Compute entropy: H = -∑ p log p
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    
    return entropy


def compute_corpus_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    corpus_texts: List[str],
    batch_size: int = 8,
    max_length: int = 2048,
    device: str = "cuda",
    save_path: Optional[str] = None,
    resume_from: Optional[str] = None
) -> Dict[int, TokenStats]:
    """
    Compute token-level entropy statistics across a corpus.
    
    Args:
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        corpus_texts: List of text strings to process
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run computation on
        save_path: Path to save intermediate results
        resume_from: Path to resume computation from
        
    Returns:
        Dictionary mapping token_id to TokenStats
    """
    logger.info(f"Computing corpus entropy for {len(corpus_texts)} texts")
    
    # Initialize or resume token statistics
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from {resume_from}")
        with open(resume_from, 'r') as f:
            saved_stats = json.load(f)
        token_stats = {
            int(token_id): TokenStats.from_dict(int(token_id), stats)
            for token_id, stats in saved_stats.items()
        }
        start_idx = len(saved_stats.get("_processed_texts", []))
    else:
        token_stats = {}
        start_idx = 0
    
    # Setup model
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = CorpusDataset(
        texts=corpus_texts[start_idx:],
        tokenizer=tokenizer,
        max_length=max_length,
        include_labels=False  # We don't need labels for entropy computation
    )
    
    # Add pad_token_id to dataset items for collate function
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def collate_with_pad_token(batch):
        # Add pad_token_id to each item
        for item in batch:
            item["pad_token_id"] = pad_token_id
        return collate_corpus_batch(batch)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_pad_token,
        num_workers=0  # Keep at 0 to avoid multiprocessing issues with model on GPU
    )
    
    processed_count = start_idx
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Compute per-token entropy
            token_entropy = compute_token_entropy(logits)  # [batch_size, seq_len]
            
            # Process each sequence in the batch
            for seq_idx in range(input_ids.size(0)):
                seq_input_ids = input_ids[seq_idx]
                seq_attention_mask = attention_mask[seq_idx]
                seq_entropy = token_entropy[seq_idx]
                
                # Only process tokens that are not padding
                valid_positions = seq_attention_mask.bool()
                valid_tokens = seq_input_ids[valid_positions]
                valid_entropies = seq_entropy[valid_positions]
                
                # Skip first token (no prediction for it) for language modeling
                if len(valid_tokens) > 1:
                    pred_tokens = valid_tokens[1:]  # Tokens being predicted
                    pred_entropies = valid_entropies[:-1]  # Entropy when predicting each token
                    
                    # Update statistics for each token
                    for token_id, entropy in zip(pred_tokens.cpu().numpy(), pred_entropies.cpu().numpy()):
                        token_id = int(token_id)
                        entropy = float(entropy)
                        
                        if token_id not in token_stats:
                            token_stats[token_id] = TokenStats(
                                token_id=token_id,
                                frequency=0,
                                entropy_sum=0.0,
                                entropy_count=0
                            )
                        
                        token_stats[token_id].update(entropy)
            
            processed_count += input_ids.size(0)
            
            # Periodic saving
            if save_path and batch_idx % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(corpus_texts)} texts")
                save_token_stats(token_stats, save_path, processed_count)
    
    logger.info(f"Completed entropy computation for {processed_count} texts")
    
    # Final save
    if save_path:
        save_token_stats(token_stats, save_path, processed_count)
    
    return token_stats


def save_token_stats(
    token_stats: Dict[int, TokenStats], 
    save_path: str, 
    processed_count: int
) -> None:
    """Save token statistics to a JSON file."""
    
    stats_dict = {
        str(token_id): stats.to_dict() 
        for token_id, stats in token_stats.items()
    }
    stats_dict["_metadata"] = {
        "processed_texts": processed_count,
        "total_tokens": len(token_stats),
        "timestamp": time.time()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)


def load_token_stats(load_path: str) -> Dict[int, TokenStats]:
    """Load token statistics from JSON file."""
    
    with open(load_path, 'r') as f:
        saved_stats = json.load(f)
    
    # Remove metadata
    if "_metadata" in saved_stats:
        del saved_stats["_metadata"]
    
    token_stats = {
        int(token_id): TokenStats.from_dict(int(token_id), stats)
        for token_id, stats in saved_stats.items()
    }
    
    return token_stats


def select_minority_tokens(
    token_stats: Dict[int, TokenStats],
    freq_percentile: float = 20.0,
    top_k: int = 1000,
    min_observations: int = 5
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select high-entropy minority tokens based on frequency and entropy.
    
    Args:
        token_stats: Dictionary of token statistics
        freq_percentile: Percentile threshold for considering tokens "minority" (default: 20th percentile)
        top_k: Number of top entropy minority tokens to select
        min_observations: Minimum number of observations required for a token
        
    Returns:
        Tuple of (selected_token_ids, selection_metadata)
    """
    logger.info(f"Selecting minority tokens from {len(token_stats)} tokens")
    
    # Filter tokens with minimum observations
    valid_tokens = {
        token_id: stats for token_id, stats in token_stats.items()
        if stats.entropy_count >= min_observations
    }
    
    logger.info(f"Filtered to {len(valid_tokens)} tokens with >= {min_observations} observations")
    
    # Extract frequencies for percentile calculation
    frequencies = [stats.frequency for stats in valid_tokens.values()]
    freq_threshold = np.percentile(frequencies, freq_percentile)
    
    # Select minority tokens (below frequency threshold)
    minority_tokens = {
        token_id: stats for token_id, stats in valid_tokens.items()
        if stats.frequency <= freq_threshold
    }
    
    logger.info(f"Found {len(minority_tokens)} minority tokens (freq <= {freq_threshold:.1f})")
    
    # Sort by average entropy (descending)
    sorted_minority = sorted(
        minority_tokens.items(),
        key=lambda x: x[1].avg_entropy,
        reverse=True
    )
    
    # Select top-k
    selected_tokens = [token_id for token_id, _ in sorted_minority[:top_k]]
    
    # Compute selection metadata
    if selected_tokens:
        selected_stats = [minority_tokens[token_id] for token_id in selected_tokens]
        metadata = {
            "num_selected": len(selected_tokens),
            "freq_percentile": freq_percentile,
            "freq_threshold": freq_threshold,
            "top_k": top_k,
            "min_observations": min_observations,
            "avg_entropy_range": {
                "min": min(stats.avg_entropy for stats in selected_stats),
                "max": max(stats.avg_entropy for stats in selected_stats),
                "mean": np.mean([stats.avg_entropy for stats in selected_stats])
            },
            "freq_range": {
                "min": min(stats.frequency for stats in selected_stats),
                "max": max(stats.frequency for stats in selected_stats),
                "mean": np.mean([stats.frequency for stats in selected_stats])
            },
            "total_corpus_tokens": len(token_stats),
            "valid_tokens_after_filtering": len(valid_tokens),
            "minority_tokens_found": len(minority_tokens)
        }
    else:
        metadata = {
            "num_selected": 0,
            "error": "No tokens selected"
        }
    
    logger.info(f"Selected {len(selected_tokens)} high-entropy minority tokens")
    if selected_tokens:
        logger.info(f"Entropy range: {metadata['avg_entropy_range']['min']:.3f} - {metadata['avg_entropy_range']['max']:.3f}")
        logger.info(f"Frequency range: {metadata['freq_range']['min']} - {metadata['freq_range']['max']}")
    
    return selected_tokens, metadata


def save_minority_tokens(
    selected_tokens: List[int],
    metadata: Dict[str, Any],
    tokenizer: AutoTokenizer,
    save_path: str
) -> None:
    """Save selected minority tokens with metadata."""
    
    # Create human-readable token information
    token_info = []
    for token_id in selected_tokens:
        try:
            token_str = tokenizer.decode([token_id])
            token_info.append({
                "token_id": token_id,
                "token_str": repr(token_str),  # Use repr to handle special characters
                "token_bytes": token_str.encode('utf-8').hex() if token_str else ""
            })
        except Exception as e:
            token_info.append({
                "token_id": token_id,
                "token_str": f"<decode_error: {str(e)}>",
                "token_bytes": ""
            })
    
    output = {
        "selected_tokens": selected_tokens,
        "token_info": token_info,
        "metadata": metadata,
        "timestamp": time.time()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved {len(selected_tokens)} minority tokens to {save_path}")


def load_minority_tokens(load_path: str) -> Tuple[List[int], Dict[str, Any]]:
    """Load selected minority tokens."""
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    return data["selected_tokens"], data.get("metadata", {})


def run_corpus_entropy_pipeline(
    model_path: str,
    corpus_texts: List[str],
    output_dir: str,
    batch_size: int = 8,
    max_length: int = 2048,
    device: str = "cuda",
    freq_percentile: float = 20.0,
    top_k: int = 1000,
    min_observations: int = 5,
    resume: bool = True
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Run complete pipeline for computing corpus entropy and selecting minority tokens.
    
    Args:
        model_path: Path to model (Hugging Face model name or local path)
        corpus_texts: List of corpus texts
        output_dir: Directory to save outputs
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device for computation
        freq_percentile: Percentile for minority token selection
        top_k: Number of tokens to select
        min_observations: Minimum observations per token
        resume: Whether to resume from existing computation
        
    Returns:
        Tuple of (selected_token_ids, metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    token_stats_path = os.path.join(output_dir, "token_stats.json")
    minority_tokens_path = os.path.join(output_dir, "minority_tokens.json")
    
    logger.info(f"Starting corpus entropy pipeline with {len(corpus_texts)} texts")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load model and tokenizer
    with Timer("Loading model and tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None  # We'll handle device placement manually
        )
    
    # Compute token statistics
    with Timer("Computing corpus entropy"):
        token_stats = compute_corpus_entropy(
            model=model,
            tokenizer=tokenizer,
            corpus_texts=corpus_texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            save_path=token_stats_path,
            resume_from=token_stats_path if resume else None
        )
    
    # Select minority tokens
    with Timer("Selecting minority tokens"):
        selected_tokens, metadata = select_minority_tokens(
            token_stats=token_stats,
            freq_percentile=freq_percentile,
            top_k=top_k,
            min_observations=min_observations
        )
    
    # Saving results
    save_minority_tokens(
        selected_tokens=selected_tokens,
        metadata=metadata,
        tokenizer=tokenizer,
        save_path=minority_tokens_path
    )
    
    logger.info("Corpus entropy pipeline completed successfully")
    return selected_tokens, metadata


# Example usage and testing functions
def create_sample_corpus(num_texts: int = 100) -> List[str]:
    """Create a sample corpus for testing."""
    
    # Kite runner text - my favorite novel
    sample_texts = [
        "The winter of my twelfth year brought unexpected changes to our neighborhood, where children played in dusty streets and mothers watched from windows with worried eyes.",
        "In the marketplace, vendors called out prices for pomegranates and almonds while the scent of freshly baked naan drifted through the crowded alleyways filled with searching customers.",
        "My father often spoke of honor and redemption, his voice carrying the weight of memories that seemed to dance behind his tired but determined eyes.",
        "The compound walls were high and painted white, protecting gardens where fruit trees grew alongside roses that my mother had planted many seasons ago.",
        "Stories of friendship and betrayal were whispered among the adults, tales that children were not supposed to hear but somehow always managed to understand.",
        "The mountain roads wound through villages where time moved slowly and traditions passed from one generation to the next like precious family heirlooms.",
        "Letters arrived sporadically from distant relatives, bearing news of marriages, births, and the endless cycles of joy and sorrow that mark human existence.",
        "In the evening light, the call to prayer echoed across rooftops while families gathered for meals shared in comfortable silence or animated conversation."
    ]
    
    # Repeat and vary the sample texts
    corpus = []
    for i in range(num_texts):
        base_text = sample_texts[i % len(sample_texts)]
        # Add some variation
        corpus.append(f"Problem {i+1}: {base_text}")
    
    return corpus
