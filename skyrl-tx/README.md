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
See the SkyRL tx [blog post](https://www.notion.so/SkyRL-tx-An-open-source-project-to-implement-the-Tinker-API-2848f0016b9d80fe9873eea1e38815ca?source=copy_link#2848f0016b9d80bc8a4bebeedac69f6e) for a quickstart example. 

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
