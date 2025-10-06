# SkyRL tx: Unified API for training and inference

SkyRL tx is an open-source cross-platform library that allows users to
set up their own service exposing a Tinker API like REST API for
neural network forward and backward passes. It unifies inference and
training into a single, common API, abstracting away the
infrastructure challenges of managing GPUs.

The `t` in `tx` stands for transformers, training, or tinker, and the `x`
stands for "cross-platform".

## Key Features

**Unified Engine**: A single engine for both inference and training
  eliminates numerical discrepancies and the need for expensive
  checkpoint transfers.

**Seamless Online Learning**: Models can be updated in real-time.

**Reduced Operational Complexity**: Manage one API and one system for
  both training and serving, simplifying deployment strategies.

**Cost-Effective Multi-Tenancy**: Using techniques like LoRA, a single
  base model can serve thousands of users with personalized adapters.

## Project Status

This is a very early release of SkyRL tx. While the project is
functional end-to-end, there is still a lot of work to be done. We are
sharing it with the community to invite feedback, testing, and
contributions.
