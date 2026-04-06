# Atropos-SkyRL SHM Integration

This integration provides a high-performance **Zero-Copy Shared Memory (SHM)** transport backplane between the Atropos environment orchestrator and the SkyRL training core.

## Overview

Reasoning-dense RL workloads (e.g., PPO on long-context thinking traces) generate massive rollout data. Standard HTTP/JSON transport creates a significant "serialization tax" that often caps collection at ~2,000 trajectories/sec.

This SHM-based integration eliminates that bottleneck, enabling **16.5k+ trajectories/sec** on local-cluster hardware (e.g., RTX 3090).

## Components

* **`generator.py`**: Implements the `AtroposSHMGenerator` which polls the circular SHM buffer.
* **`utils.py`**: Low-level binary protocol and atomic index management for zero-copy read/write.
* **`test_generator.py`**: Logic-verification suite for out-of-order trajectory stashing.

## How to Run

Use the dedicated Atropos launcher:

```bash
python skyrl/train/entrypoints/main_atropos.py --group training_run --batch_size 128
```

This will automatically orchestrate both the Atropos reasoning server and the SkyRL trainer.

## CI/CD Verification

This integration is verified by the SkyRL GPU CI suite. You can run the local integration tests manually with:

```bash
pytest skyrl/train/integrations/atropos/test_generator.py
```
