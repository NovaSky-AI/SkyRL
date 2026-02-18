# Clip-Cov and KL-Cov Policy Loss Examples

This directory contains examples for using **Clip-Cov** and **KL-Cov** policy loss functions, based on the implementation from [PRIME-RL/Entropy-Mechanism-of-RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL).

## Overview

Both methods improve training stability by using covariance-based token selection:

- **Clip-Cov**: Combines standard PPO clipping with covariance-based correction masking
- **KL-Cov**: Applies KL regularization to tokens selected based on covariance values

## Usage

### Prerequisites

1. Prepare GSM8K data:
```bash
uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

2. Set up Weights & Biases (optional):
```bash
export WANDB_API_KEY=<your_key_here>
```

### Running Clip-Cov

```bash
bash examples/algorithms/clip_cov_kl_cov/run_clip_cov.sh
```

**Key parameters:**
- `trainer.algorithm.policy_loss_type="clip_cov"`
- `trainer.algorithm.clip_cov.clip_ratio=0.0002` - fraction of tokens to clip based on covariance
- `trainer.algorithm.clip_cov.clip_cov_lb=1.0` - lower bound for covariance clipping
- `trainer.algorithm.clip_cov.clip_cov_ub=5.0` - upper bound for covariance clipping

### Running KL-Cov

```bash
bash examples/algorithms/clip_cov_kl_cov/run_kl_cov.sh
```

**Key parameters:**
- `trainer.algorithm.policy_loss_type="kl_cov"`
- `trainer.algorithm.kl_cov.kl_cov_frac=0.2` - percentage of tokens to apply KL regularization to (20%)
- `trainer.algorithm.kl_cov.ppo_kl_coef=1.0` - coefficient for KL regularization term

## Configuration

Both methods are configured through the algorithm section of your config:

```yaml
trainer:
  algorithm:
    policy_loss_type: "clip_cov"  # or "kl_cov"
    
    # Clip-Cov specific parameters
    clip_cov:
      clip_ratio: 0.0002
      clip_cov_lb: 1.0
      clip_cov_ub: 5.0
    
    # KL-Cov specific parameters  
    kl_cov:
      kl_cov_frac: 0.2
      ppo_kl_coef: 1.0
```


## Reference

- Paper: https://arxiv.org/abs/2505.22617
- Code: https://github.com/PRIME-RL/Entropy-Mechanism-of-RL
