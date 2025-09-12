"""
Trajectory logging utilities for debugging and analysis during training.

This module provides a simple framework for logging prompts, responses, and rewards.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from omegaconf import DictConfig


class TrajectoryLogger(ABC):
    """
    Abstract base class for trajectory logging.
    
    TODO: Allow users to bring a custom trajectory logger. They should be able to
    define their own class (outside of the skyrl-train package) and add it to
    a registry (see AdvantageEstimatorRegistry for an example) so that it can
    be referenced by name in the config.
    """
    
    def __init__(self, max_trajectories: int = -1):
        self.max_trajectories = max_trajectories
    
    @abstractmethod
    def log(
        self, 
        prompts: List[str],
        responses: List[str], 
        rewards: List[float]
    ) -> None:
        pass


class WandbTableTrajectoryLogger(TrajectoryLogger):
    
    def __init__(self, max_trajectories: int = -1):
        super().__init__(max_trajectories)
        import wandb
        self.wandb = wandb
    
    def log(
        self, 
        prompts: List[str],
        responses: List[str], 
        rewards: List[float]
    ) -> None:
        if self.max_trajectories > 0:
            prompts = prompts[:self.max_trajectories]
            responses = responses[:self.max_trajectories]
            rewards = rewards[:self.max_trajectories]
        
        data = [[i, prompt, response, reward] for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards))]
        table = self.wandb.Table(columns=["step", "prompt", "response", "reward"], data=data)
        
        self.wandb.log({"trajectories": table})


class CSVTrajectoryLogger(TrajectoryLogger):
    
    def __init__(self, output_dir: str, max_trajectories: int = -1):
        super().__init__(max_trajectories)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def log(
        self,
        prompts: List[str],
        responses: List[str], 
        rewards: List[float]
    ) -> None:
        if self.max_trajectories > 0:
            prompts = prompts[:self.max_trajectories]
            responses = responses[:self.max_trajectories]
            rewards = rewards[:self.max_trajectories]
        
        df = pd.DataFrame({
            "step": list(range(len(prompts))),
            "prompt": prompts,
            "response": responses,
            "reward": rewards
        })
        
        filename = os.path.join(self.output_dir, "trajectories.csv")
        df.to_csv(filename, index=False)


def create_trajectory_logger_from_config(logging_cfg: DictConfig) -> TrajectoryLogger:
    assert logging_cfg.enabled
    
    if logging_cfg.type == 'wandb':
        return WandbTableTrajectoryLogger(max_trajectories=logging_cfg.max_trajectories)
    elif logging_cfg.type == 'csv':
        return CSVTrajectoryLogger(
            output_dir=logging_cfg.output_dir,
            max_trajectories=logging_cfg.max_trajectories
        )
    else:
        raise ValueError(f"Unknown trajectory logger type: {logging_cfg.type}")