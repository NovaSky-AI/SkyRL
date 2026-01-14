---
title: "SkyRL - Full-Stack RL for LLMs"
---

<header class="site-header">
  <h1>SkyRL</h1>
  <p class="subheadline">
    A modular full-stack reinforcement learning library for training real-world, long-horizon agents with LLMs
  </p>
</header>

## Why SkyRL?

<div class="value-props">
  <div class="value-prop">
    <h3>🏗️ Modular</h3>
    <p>Composable training framework, environment library, and agent pipelines. Use what you need, extend what you want.</p>
  </div>
  <div class="value-prop">
    <h3>⚡ Performant</h3>
    <p>Optimized for distributed RL training at scale. Built for real-world production workloads.</p>
  </div>
  <div class="value-prop">
    <h3>🌍 Real-World Tasks</h3>
    <p>Designed for long-horizon, multi-turn tool use on complex environments like SWE-Bench and Text-to-SQL.</p>
  </div>
</div>

## Get Started

<div class="code-section">
  <h3>Quick Start</h3>

```python
from skyrl.gym import make_env
from skyrl.train import RLTrainer

# Create environment and configure trainer
env = make_env("math-solver")
trainer = RLTrainer(
    model="meta-llama/Llama-3-8B",
    env=env,
    algorithm="ppo"
)

# Train your agent
trainer.train(num_iterations=1000)
```

</div>

<div class="traction">
  <h3>Trusted By</h3>
  <div class="companies">
    <div class="company">Anyscale</div>
    <div class="company">Scale AI</div>
    <div class="company">Datadog</div>
  </div>
</div>

<div class="cta-section">
  <a href="https://github.com/NovaSky-AI/SkyRL" class="button">View on GitHub</a>
  <a href="https://skyrl.readthedocs.io/" class="button button-secondary">Documentation</a>
</div>

<footer>
  <p>Built at <a href="https://sky.cs.berkeley.edu/">Berkeley Sky Computing Lab</a></p>
</footer>
