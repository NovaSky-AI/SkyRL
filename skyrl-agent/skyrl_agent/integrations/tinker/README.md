# Tinker Integration for SkyRL Agent

This directory contains the integration between SkyRL Agent and Tinker's RL training framework.

## Overview

The Tinker integration enables you to:
1. Use **AgentRunner** to collect trajectories from your agents
2. Train models using **Tinker's LoRA training** with importance sampling or PPO
3. Track metrics with **Weights & Biases**
4. Save and resume from checkpoints

## Architecture

### Components

1. **`tinker_backend.py`**: Backend interface for Tinker's sampling client
   - Handles async generation from token IDs
   - Decodes outputs with proper special token handling
   - Detects finish reasons (EOS vs length limit)

2. **`tinker_train.py`**: Main training script
   - Loads data from Parquet files
   - Runs AgentRunner for trajectory collection
   - Converts AgentRunner output to Tinker's format
   - Performs RL training with advantage normalization
   - Saves checkpoints periodically

3. **`__init__.py`**: Backend registration
   - Registers Tinker backend in the SkyRL Agent registry

## Data Flow

```
Input Batch (from Parquet)
  ↓
AgentRunner.run() → Trajectories
  ↓
Extract: {prompt_token_ids, response_ids, logprobs, rewards, loss_masks}
  ↓
Compute: Advantages (with optional normalization)
  ↓
Prepare: Tinker Datums (with masks and advantages)
  ↓
Tinker Training: forward_backward() + optim_step()
  ↓
Metrics & Checkpointing
```
