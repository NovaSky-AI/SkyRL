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
from skyrl_train.inference_engines.base import ConversationType


@dataclass
class Trajectory:
    """
    Complete trajectory data including prompts, responses, and rewards.
    
    Attributes:
        prompt: Original prompt (can be ConversationType or string)
        chat_history: Complete conversation history
        response: Final generated response text
        reward: Reward signal from environment
        stop_reason: Reason for trajectory termination
        env_class: Environment class used
        env_extras: Additional environment metadata
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
    loss_mask: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


class TrajectoryLogger(ABC):
    """
    Abstract base class for trajectory logging.
    
    Subclasses should implement the log method to handle trajectory
    data in their specific format (e.g., wandb tables, CSV files, etc.)
    """
    
    def __init__(self, max_trajectories: int = -1):
        """
        Initialize the trajectory logger.
        
        Args:
            max_trajectories: Maximum number of trajectories to log per batch (-1 for unlimited)
        """
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
            
            data.append({
                "prompt": prompt_str,
                "response": traj.response,
                "reward": traj.reward,
                "stop_reason": traj.stop_reason,
                "env_class": traj.env_class,
                "chat_turns": len(traj.chat_history),
            })
        
        return pd.DataFrame(data)


class WandbTableTrajectoryLogger(TrajectoryLogger):
    """
    Trajectory logger that uploads trajectories as tables to Weights & Biases.
    """
    
    def __init__(
        self, 
        max_trajectories: int = -1,
        log_full_history: bool = True
    ):
        """
        Initialize the WandB table trajectory logger.
        
        Args:
            max_trajectories: Maximum number of trajectories to log per batch (-1 for unlimited)
            log_full_history: Whether to log full chat history or just prompt/response
        """
        super().__init__(max_trajectories)
        self.log_full_history = log_full_history
        
        import wandb
        self.wandb = wandb
    
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
        if self.max_trajectories > 0 and len(trajectories) > self.max_trajectories:
            trajectories = trajectories[:self.max_trajectories]
        
        # Define columns based on logging configuration
        if self.log_full_history:
            columns = ["step", "env_class", "full_conversation", "reward", "stop_reason"]
        else:
            columns = ["step", "env_class", "prompt", "response", "reward", "stop_reason"]
        
        # Create table
        table = self.wandb.Table(columns=columns)
        
        # Add rows for each trajectory
        for traj in trajectories:
            # Format prompt
            if isinstance(traj.prompt, list):
                prompt_str = self._format_conversation(traj.prompt)
            else:
                prompt_str = str(traj.prompt)
            
            if self.log_full_history:
                # Format full conversation
                full_conv = self._format_conversation(traj.chat_history)
                row_data = [step, traj.env_class, full_conv, traj.reward, traj.stop_reason]
            else:
                row_data = [step, traj.env_class, prompt_str, traj.response, traj.reward, 
                           traj.stop_reason]
            
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
        max_trajectories: int = -1
    ):
        """
        Initialize the CSV trajectory logger.
        
        Args:
            output_dir: Directory to save CSV files
            max_trajectories: Maximum number of trajectories to log per batch (-1 for unlimited)
        """
        super().__init__(max_trajectories)
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
        if self.max_trajectories > 0 and len(trajectories) > self.max_trajectories:
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
) -> Optional[TrajectoryLogger]:
    """
    Create trajectory logger from configuration following repo factory pattern.
    
    Args:
        logging_cfg: Trajectory logging configuration
        
    Returns:
        TrajectoryLogger instance or None if disabled
    """
    if not logging_cfg.get('enabled', False):
        return None
        
    logger_type = logging_cfg.get('type', 'wandb')
    
    if logger_type == 'wandb':
        return WandbTableTrajectoryLogger(
            max_trajectories=logging_cfg.get('max_trajectories', -1),
            log_full_history=logging_cfg.get('log_full_history', True)
        )
    elif logger_type == 'csv':
        output_dir = logging_cfg.get('output_dir', './trajectory_logs')
        return CSVTrajectoryLogger(
            output_dir=output_dir,
            max_trajectories=logging_cfg.get('max_trajectories', -1)
        )
    else:
        raise ValueError(f"Unknown trajectory logger type: {logger_type}")