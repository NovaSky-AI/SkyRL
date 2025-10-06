# SkyRL tx: Unified API for training and inference

> ⚠️  The project is currently very early with lots of missing features
> (e.g. currently LoRA is only supported for the MLP layer, pure inference
> is not supported, model sharding is in a very early state). Many of
> these are easy to implement and we welcome contributions! ⚠️


SkyRL tx is an open-source cross-platform library that allows users to
set up their own service exposing a
[Tinker](https://tinker-docs.thinkingmachines.ai/) like REST API for
neural network forward and backward passes. It unifies inference and
training into a single, common API, abstracting away the
infrastructure challenges of managing GPUs.

The `t` in `tx` stands for transformers, training, or tinker, and the `x`
stands for "cross-platform".

## Project Status

This is a very early release of SkyRL tx. While the project is
functional end-to-end, there is still a lot of work to be done. We are
sharing it with the community to invite feedback, testing, and
contributions.
