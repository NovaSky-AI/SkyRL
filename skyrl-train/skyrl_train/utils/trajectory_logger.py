"""
Trajectory logging utilities for debugging and analysis during training.

This module provides a flexible framework for logging complete trajectories
including model outputs and environment observations/feedback.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd
from omegaconf import DictConfig
from skyrl_train.inference_engines.base import ConversationType, MessageType


@dataclass
class Trajectory:
    """
    Complete trajectory data including prompts, responses, and rewards.
    
    Attributes:
        prompt: Original prompt (can be ConversationType or string)
        chat_history: Complete conversation history
        response: Final generated response
        reward: Reward signal from environment
        stop_reason: Reason for trajectory termination
        env_class: Environment class used
        env_extras: Additional environment metadata
        prompt_tokens: Tokenized prompt (optional)
        response_tokens: Tokenized response (optional)
        loss_mask: Loss mask for training (optional)
        metadata: Additional metadata (optional)
    """
    prompt: Union[ConversationType, str]
    chat_history: ConversationType
    response: str
    reward: float
    stop_reason: str
    env_class: str
    env_extras: Dict[str, Any]
    prompt_tokens: Optional[List[int]] = None
    response_tokens: Optional[List[int]] = None
    loss_mask: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


class TrajectoryLogger(ABC):
    """
    Abstract base class for trajectory logging.
    
    Subclasses should implement the log method to handle trajectory
    data in their specific format (e.g., wandb tables, CSV files, etc.)
    """
    
    def __init__(self, tokenizer=None, max_trajectories: Optional[int] = None):
        """
        Initialize the trajectory logger.
        
        Args:
            tokenizer: Optional tokenizer for converting between tokens and text
            max_trajectories: Maximum number of trajectories to log per batch
        """
        self.tokenizer = tokenizer
        self.max_trajectories = max_trajectories
        self.logged_count = 0
    
    @abstractmethod
    def log(
        self, 
        trajectories: List[Trajectory], 
        step: int,
        prefix: str = "train"
    ) -> None:
        """
        Log a batch of trajectories.
        
        Args:
            trajectories: List of Trajectory objects to log
            step: Current training step
            prefix: Prefix for logging (e.g., "train", "eval")
        """
        raise NotImplementedError()
    
    def _format_conversation(self, conversation: ConversationType) -> str:
        """
        Format a conversation history into a readable string.
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Formatted conversation string
        """
        formatted = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"[{role}]: {content}")
        return "\n".join(formatted)
    
    def _truncate_text(self, text: str, max_length: int = 1000) -> str:
        """
        Truncate text to a maximum length with ellipsis.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def _detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs to text using the tokenizer.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            return f"<{len(token_ids)} tokens>"
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def to_dataframe(self, trajectories: List[Trajectory]) -> pd.DataFrame:
        """
        Convert trajectories to a pandas DataFrame for analysis.
        
        Args:
            trajectories: List of Trajectory objects
            
        Returns:
            DataFrame with trajectory data
        """
        data = []
        for traj in trajectories:
            # Format prompt based on type
            if isinstance(traj.prompt, list):
                prompt_str = self._format_conversation(traj.prompt)
            else:
                prompt_str = str(traj.prompt)
            
            # Detokenize if needed
            if traj.response_tokens and self.tokenizer:
                response_str = self._detokenize(traj.response_tokens)
            else:
                response_str = traj.response
            
            data.append({
                "prompt": prompt_str,
                "response": response_str,
                "reward": traj.reward,
                "stop_reason": traj.stop_reason,
                "env_class": traj.env_class,
                "chat_turns": len(traj.chat_history),
                "response_length": len(traj.response_tokens) if traj.response_tokens else len(self.tokenizer.encode(traj.response, add_special_tokens=False)) if self.tokenizer else len(traj.response),
            })
        
        return pd.DataFrame(data)


class WandbTableTrajectoryLogger(TrajectoryLogger):
    """
    Trajectory logger that uploads trajectories as tables to Weights & Biases.
    """
    
    def __init__(
        self, 
        tokenizer=None, 
        max_trajectories: Optional[int] = 10,
        max_text_length: int = 2000,
        log_full_history: bool = False
    ):
        """
        Initialize the WandB table trajectory logger.
        
        Args:
            tokenizer: Optional tokenizer for converting between tokens and text
            max_trajectories: Maximum number of trajectories to log per batch
            max_text_length: Maximum length of text fields in the table
            log_full_history: Whether to log full chat history or just prompt/response
        """
        super().__init__(tokenizer, max_trajectories)
        self.max_text_length = max_text_length
        self.log_full_history = log_full_history
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb is required for WandbTableTrajectoryLogger. Install with: pip install wandb")
    
    def log(
        self, 
        trajectories: List[Trajectory], 
        step: int,
        prefix: str = "train"
    ) -> None:
        """
        Log trajectories to wandb as a table.
        
        Args:
            trajectories: List of Trajectory objects to log
            step: Current training step
            prefix: Prefix for logging (e.g., "train", "eval")
        """
        # Limit number of trajectories if specified
        if self.max_trajectories and len(trajectories) > self.max_trajectories:
            trajectories = trajectories[:self.max_trajectories]
        
        # Define columns based on logging configuration
        if self.log_full_history:
            columns = ["step", "env_class", "full_conversation", "reward", "stop_reason", "response_length"]
        else:
            columns = ["step", "env_class", "prompt", "response", "reward", "stop_reason", "response_length"]
        
        # Create table
        table = self.wandb.Table(columns=columns)
        
        # Add rows for each trajectory
        for traj in trajectories:
            # Format prompt
            if isinstance(traj.prompt, list):
                prompt_str = self._format_conversation(traj.prompt)
            else:
                prompt_str = str(traj.prompt)
            
            # Detokenize response if needed
            if traj.response_tokens and self.tokenizer:
                response_str = self._detokenize(traj.response_tokens)
            else:
                response_str = traj.response
            
            # Truncate texts
            prompt_str = self._truncate_text(prompt_str, self.max_text_length)
            response_str = self._truncate_text(response_str, self.max_text_length)
            
            # Calculate response length (always use token count for consistency)
            response_length = len(traj.response_tokens) if traj.response_tokens else len(self.tokenizer.encode(traj.response, add_special_tokens=False)) if self.tokenizer else len(traj.response)
            
            if self.log_full_history:
                # Format full conversation
                full_conv = self._format_conversation(traj.chat_history)
                full_conv = self._truncate_text(full_conv, self.max_text_length * 2)
                row_data = [step, traj.env_class, full_conv, traj.reward, traj.stop_reason, response_length]
            else:
                row_data = [step, traj.env_class, prompt_str, response_str, traj.reward, 
                           traj.stop_reason, response_length]
            
            table.add_data(*row_data)
        
        # Log to wandb
        self.wandb.log({f"{prefix}/trajectories": table}, step=step)
        self.logged_count += len(trajectories)


class CSVTrajectoryLogger(TrajectoryLogger):
    """
    Trajectory logger that saves trajectories to CSV files for offline analysis.
    """
    
    def __init__(
        self,
        output_dir: str,
        tokenizer=None,
        max_trajectories: Optional[int] = None
    ):
        """
        Initialize the CSV trajectory logger.
        
        Args:
            output_dir: Directory to save CSV files
            tokenizer: Optional tokenizer for converting between tokens and text
            max_trajectories: Maximum number of trajectories to log per batch
        """
        super().__init__(tokenizer, max_trajectories)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def log(
        self,
        trajectories: List[Trajectory],
        step: int,
        prefix: str = "train"
    ) -> None:
        """
        Save trajectories to a CSV file.
        
        Args:
            trajectories: List of Trajectory objects to log
            step: Current training step
            prefix: Prefix for logging (e.g., "train", "eval")
        """
        # Limit number of trajectories if specified
        if self.max_trajectories and len(trajectories) > self.max_trajectories:
            trajectories = trajectories[:self.max_trajectories]
        
        # Convert to dataframe
        df = self.to_dataframe(trajectories)
        df["step"] = step
        df["prefix"] = prefix
        
        # Save to CSV
        filename = os.path.join(self.output_dir, f"{prefix}_trajectories_step_{step}.csv")
        df.to_csv(filename, index=False)
        
        self.logged_count += len(trajectories)



def create_trajectory_logger_from_config(
    logging_cfg: DictConfig,
    tokenizer
) -> Optional[TrajectoryLogger]:
    """
    Create trajectory logger from configuration following repo factory pattern.
    
    Args:
        logging_cfg: Trajectory logging configuration
        tokenizer: Tokenizer for text processing
        
    Returns:
        TrajectoryLogger instance or None if disabled
    """
    if not logging_cfg.get('enabled', False):
        return None
        
    logger_type = logging_cfg.get('type', 'wandb')
    
    if logger_type == 'wandb':
        return WandbTableTrajectoryLogger(
            tokenizer=tokenizer,
            max_trajectories=logging_cfg.get('max_trajectories', 10),
            max_text_length=logging_cfg.get('max_text_length', 2000),
            log_full_history=logging_cfg.get('log_full_history', False)
        )
    elif logger_type == 'csv':
        output_dir = logging_cfg.get('output_dir', './trajectory_logs')
        return CSVTrajectoryLogger(
            output_dir=output_dir,
            tokenizer=tokenizer,
            max_trajectories=logging_cfg.get('max_trajectories', None)
        )
    else:
        raise ValueError(f"Unknown trajectory logger type: {logger_type}")