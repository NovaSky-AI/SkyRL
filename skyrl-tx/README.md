<div align="center">

# 🧸 SkyRL tx: Unifying LLM training and inference

<p align="center">
  <a href="https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx">GitHub</a> •
  <a href="https://tinker-docs.thinkingmachines.ai/">Tinker Docs</a> •
  <a href="https://github.com/thinking-machines-lab/tinker-cookbook">Tinker Cookbook</a> •
  <a href="https://skyrl.slack.com/archives/C09K1JGNPJS">Slack</a>
</p>

</div>

SkyRL tx is an open-source library that implements a backend for the [Tinker API](https://thinkingmachines.ai/tinker/), allowing you to set up your own Tinker-like service running on your own hardware. It provides a unified interface for both training and inference, enabling seamless online learning, cost-effective multi-tenancy through LoRA, and simplified ML infrastructure.

## ✨ Key Features

- **Unified Training & Inference** — Single engine for forward passes, backward passes, and sampling
- **Multi-User LoRA Support** — Efficient GPU sharing across users with individual adapters
- **SFT & RL Support** — Supervised fine-tuning and reinforcement learning with PPO and custom loss functions
- **Multi-Node Training** — FSDP and tensor parallelism for distributed training
- **Multiple Model Architectures** — Support for Qwen3 (dense & MoE), Llama 3, and DeepSeek V3
- **External Inference Engine** — Optional vLLM integration for optimized inference
- **Production Ready** — PostgreSQL support, cloud storage checkpoints, and database migrations

## 🏗️ Architecture

SkyRL tx consists of four main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        REST API Server                          │
│                    (FastAPI - handles requests)                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           Database                              │
│         (SQLite/PostgreSQL - metadata, job queue)               │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                            Engine                               │
│        (Scheduling & batching across users/adapters)            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                            Worker                               │
│       (Model execution, forward/backward, optimizer)            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL/skyrl-tx

# For GPU
uv run --extra gpu --extra tinker -m tx.tinker.api --base-model <model>

# For TPU
uv run --extra tpu --extra tinker -m tx.tinker.api --base-model <model>
```

### Basic Training Example (Pig Latin)

Start the server:

```bash
uv run --extra gpu --extra tinker -m tx.tinker.api --base-model "Qwen/Qwen3-0.6B"
```

Run a simple training loop:

```python
import tinker
import numpy as np
from tinker import types

# Connect to the local server
service_client = tinker.ServiceClient(base_url="http://localhost:8000", api_key="tml-dummy")
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B")
tokenizer = training_client.get_tokenizer()

# Training examples
examples = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]

def process_example(example, tokenizer):
    prompt = f"English: {example['input']}\nPig Latin:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=dict(weights=weights[1:], target_tokens=tokens[1:])
    )

processed = [process_example(ex, tokenizer) for ex in examples]

# Training loop
for _ in range(6):
    fwdbwd = training_client.forward_backward(processed, "cross_entropy").result()
    training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()

    logprobs = np.concatenate([o['logprobs'].tolist() for o in fwdbwd.loss_fn_outputs])
    weights = np.concatenate([e.loss_fn_inputs['weights'].tolist() for e in processed])
    print(f"Loss: {-np.dot(logprobs, weights) / weights.sum():.4f}")
```

### Sampling

```python
# After training, create a sampling client
sampling_client = training_client.save_weights_and_get_sampling_client(name='my-model')

# Sample from the model
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(max_tokens=20, temperature=0.0)
result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8).result()

for i, seq in enumerate(result.sequences):
    print(f"{i}: {tokenizer.decode(seq.tokens)}")
```

## 📖 Usage Examples

### Dense Model Training (Qwen3-8B on 8×H100)

```bash
# Start the server
uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model Qwen/Qwen3-8B \
    --backend-config '{"max_lora_adapters": 2, "max_lora_rank": 1, "tensor_parallel_size": 8, "train_micro_batch_size": 1}'

# Run training (using tinker-cookbook)
export TINKER_API_KEY="tml-dummy"
uv run --with wandb --with tinker sl_loop.py \
    base_url=http://localhost:8000 \
    model_name=Qwen/Qwen3-8B lora_rank=1
```

### MoE Model Training (Qwen/Qwen3-30B-A3B)

```bash
# Start the server
uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model Qwen/Qwen3-30B-A3B \
    --backend-config '{"max_lora_adapters": 2, "max_lora_rank": 1, "tensor_parallel_size": 8, "train_micro_batch_size": 1, "shard_attention_heads": false}'

# Run training (using tinker-cookbook)
export TINKER_API_KEY="tml-dummy"
uv run --with wandb --with tinker sl_loop.py \
    base_url=http://localhost:8000 \
    model_name=Qwen/Qwen3-30B-A3B lora_rank=1 max_length=512
```

### Reinforcement Learning (Qwen/Qwen3-8B)

```bash
# Start server
uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model Qwen/Qwen3-8B \
    --backend-config '{"max_lora_adapters": 3, "max_lora_rank": 1, "tensor_parallel_size": 8, "train_micro_batch_size": 8, "sample_max_num_sequences": 256}' > out.log

# Run RL loop
uv run --with wandb --with tinker rl_loop.py \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-8B" \
    lora_rank=1 max_length=1024
```

### Multi-Node Training

```bash
# Node 0 (coordinator + API server)
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model Qwen/Qwen3-8B \
    --backend-config '{
        "max_lora_adapters": 3,
        "max_lora_rank": 1,
        "tensor_parallel_size": 4,
        "fully_sharded_data_parallel_size": 2,
        "train_micro_batch_size": 8,
        "sample_max_num_sequences": 256,
        "coordinator_address": "node0:7777",
        "num_processes": 2
    }' > out.log

# Node 1 (worker)
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --extra gpu --extra tinker -m tx.tinker.backends.jax \
    --coordinator-address "node0:7777" \
    --num-processes 2 \
    --process-id 1
```

### With External vLLM Inference

```bash
# Start vLLM
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_PLUGINS=lora_filesystem_resolver \
VLLM_LORA_RESOLVER_CACHE_DIR=/tmp/lora_models/ \
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --with vllm vllm serve Qwen/Qwen3-4B \
    --tensor-parallel-size 4 --port 7999 --enable-lora

# Start SkyRL tx with external inference
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model Qwen/Qwen3-4B \
    --external-inference-url "http://0.0.0.0:7999" \
    --backend-config '{"max_lora_adapters": 3, "max_lora_rank": 1, "tensor_parallel_size": 4, "train_micro_batch_size": 8}' > out.log
```

## 🎯 Supported Features

| Feature | Status |
|---------|--------|
| Qwen3 Dense Models | ✅ |
| Qwen3 MoE Models | ✅ |
| Llama 3 Models | ✅ |
| DeepSeek V3 Models | ✅ |
| Multi-User LoRA | ✅ |
| LoRA (all layers) | ✅ |
| Forward/Backward | ✅ |
| Sampling | ✅ |
| Gradient Accumulation | ✅ |
| Gradient Checkpointing | ✅ |
| JIT Compilation | ✅ |
| Tensor Parallelism | ✅ |
| FSDP | ✅ |
| Multi-Node | ✅ |
| PostgreSQL | ✅ |
| Cloud Storage Checkpoints | ✅ |
| Custom Loss Functions | ✅ |
| External Inference (vLLM) | ✅ |
| Local Model Loading | ✅ |

## 🗺️ Roadmap

- **Performance** — Expert parallelism, context parallelism, optimized kernels
- **Models** — More architectures, PyTorch model definitions via torchax
- **API Coverage** — Full Tinker API compatibility
- **Operations** — Dashboard/frontend, improved logging and metrics
- **Integration** — SkyRL-train Tinkerification

## 🤝 Contributing

We welcome contributions! The project is early and hackable — now is a great time to get involved.

**Ways to contribute:**
- Try examples from the [Tinker documentation](https://tinker-docs.thinkingmachines.ai/) or [cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- Fix issues or implement features from our [issue tracker](https://github.com/NovaSky-AI/SkyRL/issues?q=is%3Aissue%20state%3Aopen%20label%3Atx)
- Improve documentation
- Add support for more models
- Performance optimizations

## 📚 Resources

- **[Ray Summit Talk](https://www.youtube.com/watch?v=_JLnESEu2gw)** — SkyRL tx: A unified training and inference engine
- **[Slides](https://docs.google.com/presentation/d/1g-u8zxz7FsnlQXXShBVoqjUJhS48c6rxkJJJn0sj78A/)** — Presentation slides
- **[Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)** — Official Tinker API docs
- **[Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)** — Example recipes

## 📝 Blog Posts

- **[Introducing SkyRL tx](https://novasky-ai.notion.site/skyrl-tx)**
- **[SkyRL tx v0.0.2](https://novasky-ai.notion.site/skyrl-tx-v002)**
- **[SkyRL tx v0.0.3](https://novasky-ai.notion.site/skyrl-tx-003)**
- **[SkyRL tx v0.1.0](https://novasky-ai.notion.site/skyrl-tx-v010)**
- **[SkyRL tx v0.2.0](https://novasky-ai.notion.site/skyrl-tx-v02)**
- **[SkyRL tx v0.2.1](https://novasky-ai.notion.site/skyrl-tx-v021)**

## 📬 Contact

- **Slack**: [#skyrl-tx](https://skyrl.slack.com/archives/C09K1JGNPJS)
- **GitHub**: [NovaSky-AI/SkyRL/skyrl-tx](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx)
- **Twitter/X**: [@NovaSkyAI](https://x.com/NovaSkyAI)

## 📄 License

See [LICENSE](LICENSE) for details.
