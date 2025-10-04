# tx: Cross-Platform Transformer Training

tx (**t**ransformers **x**-platform) is a JAX/OpenXLA-based library
designed for training transformers and other neural networks. Since it
is based on OpenXLA, tx enables you to run the same code across
diverse hardware platforms like GPUs, TPUs, AWS Trainium, and
Tenstorrent accelerators, without the complexity of adapting to
platform-specific APIs or execution models like you would need if you
switch from PyTorch to PyTorch/XLA.

We try to keep the code simple but powerful and write the library in a
way that feels intuitive to developers in the PyTorch and HuggingFace
ecosystems.

Key Benefits:
- **Write once, run anywhere**: Single codebase that works across all major AI accelerators
- **Familiar conventions**: Designed with PyTorch and Hugging Face developers in mind
- **Clean and maintainable**: Simple, powerful code that doesn't compromise on capability

The code is very early, features we want to support going forward:
- More flexible dataset support
- More model architectures like MoE models
- More optimizations

Contributions are welcome! Please keep the code as simple as possible :)
