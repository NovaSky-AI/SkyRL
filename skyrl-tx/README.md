# SkyRL tx: Unified API for training and inference

> ⚠️  The project is currently very early with lots of missing features
> (e.g. currently LoRA is only supported for the MLP layer, model sharding
> is in a very early state). Many of these are easy to implement and we
> welcome contributions! ⚠️


SkyRL tx is an open-source cross-platform library that allows users to
set up their own service exposing a
[Tinker](https://tinker-docs.thinkingmachines.ai/) like REST API for
neural network forward and backward passes. It unifies inference and
training into a single, common API, abstracting away the
infrastructure challenges of managing GPUs.

The `t` in `tx` stands for transformers, training, or tinker, and the `x`
stands for "cross-platform".

## Getting Started
See the following SkyRL tx blog posts for more info and examples:
- [Initial blog post](https://novasky-ai.notion.site/skyrl-tx) with the piglatin training example
- [v0.0.2 release](https://novasky-ai.notion.site/skyrl-tx-v002) with a `Qwen/Qwen3-4B` training example
- [v0.0.3 release](https://novasky-ai.notion.site/skyrl-tx-003) with a `Qwen/Qwen3-Coder-30B-A3B` training and a sampling example
- [v0.1.0 release](https://novasky-ai.notion.site/skyrl-tx-v010) with an end-to-end RL example

See also our talk [SkyRL tx: A unified training and inference engine](https://docs.google.com/presentation/d/1g-u8zxz7FsnlQXXShBVoqjUJhS48c6rxkJJJn0sj78A/view) at Ray Summit 2025.

## Features

### ✅ Implemented
- **Training**: MultiLoRA fine-tuning with gradient accumulation
- **Inference**: Text generation with
  - Temperature sampling
  - Stop token support
- **API**: REST API compatible with Tinker specification

### 🚧 In Progress
- Model sharding improvements
- Additional LoRA layer support

## Project Status

This is a very early release of SkyRL tx. While the project is
functional end-to-end, there is still a lot of work to be done. We are
sharing it with the community to invite feedback, testing, and
contributions.
