# LogitsProcessor Design

## Overview

This document proposes a design for `LogitsProcessor` - a utility for computing logits and log probabilities from model hidden states.

## Background

In causal language models, the forward pass produces hidden states `[B, T, H]` which must be projected to vocabulary logits `[B, T, V]` via the `lm_head` layer. Different scenarios have different requirements:

### Training

Compute logprobs for all positions to calculate loss.

```
hidden_states [B, T, H] → logprobs [B, T] → loss
```

Full logits `[B, T, V]` are not needed - we only need logprobs of target tokens. This enables **chunked computation**: process tokens in chunks, compute logits and extract logprobs per chunk, avoiding full `[B*T, V]` materialization.

### Inference: Prefill

Process the prompt. Return logits for the last position (to start decoding). Optionally return logprobs of prompt tokens.

```
hidden_states [B, T, H] → logits [B, 1, V]  (last position, for sampling)
                        → logprobs [B, T-1]  (optional, for prompt logprobs)
```

For prompt logprobs, same as training - full logits not needed, can use chunked computation.

### Inference: Decode

Generate one token at a time.

1. **Compute logits:** `hidden_states [B, 1, H] → logits [B, 1, V]`
2. **Apply sampling transforms:** temperature scaling, top_k filtering, top_p filtering on logits
3. **Sample:** draw next_token from the transformed distribution
4. **Extract logprob:** get log probability of the sampled token from original logits

**Full logits required** because step 2 operates on the full vocabulary distribution.

## Existing Designs

### SGLang

**Pattern:** LogitsProcessor as a model attribute, called inside `model.forward()`.

**Key files:**
- [LogitsProcessor class](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/logits_processor.py#L235)
- [LlamaForCausalLM.forward()](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py#L499) calls [logits_processor](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py#L522)

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, ...):
        self.logits_processor = LogitsProcessor(config)

    def forward(self, input_ids, positions, forward_batch, ...) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, ...)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, ...)
```

**Problems:**

1. **Wrapper pattern:** `forward()` just returns `logits_processor(...)` output. No encapsulation benefit.

2. **Inconsistent return types:** `forward()` returns [different types](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py#L520-L532) based on runtime conditions (LogitsProcessorOutput, PoolerOutput, or Tensor).

3. **God object:** [LogitsProcessor.forward()](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/logits_processor.py#L379) is 500+ lines handling many modes through complex branching.

### vLLM

**Pattern:** LogitsProcessor as a model attribute, called via separate `compute_logits()` method.

**Key files:**
- [LogitsProcessor class](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py#L18)
- [LlamaForCausalLM.compute_logits()](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py#L640)
- [model_runner calls compute_logits()](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L3336)

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, ...):
        self.logits_processor = LogitsProcessor(vocab_size, scale=logit_scale)

    def forward(self, input_ids, positions, ...) -> Tensor:
        return self.model(input_ids, positions, ...)  # returns hidden_states

    def compute_logits(self, hidden_states) -> Tensor:
        return self.logits_processor(self.lm_head, hidden_states)
```

**Improvements over SGLang:**
- `forward()` has single responsibility (returns hidden_states)
- Logits computation is explicit via separate method

**Remaining Problems:**

1. **Still a wrapper:** `compute_logits()` just wraps `self.logits_processor(...)`.

2. **Unnecessary model attribute:** `logits_processor` stores minimal state. Could be a static utility.

3. **No logprobs support:** Only computes logits. Logprobs computation happens elsewhere.

## Proposed Design

### Principles

1. **Standalone utility** - Not a model attribute
2. **Model returns hidden_states** - Single responsibility, consistent return type
3. **Caller decides what to compute** - Logits for sampling, logprobs for training
4. **Unified logprobs API** - Same method for training and prompt logprobs

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Caller                                │
│         (JaxBackend for training, Generator for sampling)       │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    │ model(input_ids, ...)     │ LogitsProcessor.*()
                    ▼                           ▼
┌───────────────────────────┐     ┌───────────────────────────────┐
│     CausalLM Model        │     │      LogitsProcessor          │
│                           │     │                               │
│  forward() → hidden_states│     │  compute_logits()             │
│  lm_head property         │     │  compute_logprobs()           │
└───────────────────────────┘     │  logits_to_logprobs()         │
                                  └───────────────────────────────┘
```

### API

```python
class LogitsProcessor:
    """Utility for computing logits and logprobs from hidden states."""

    @staticmethod
    def compute_logits(hidden_states, lm_head, adapter_indices=None) -> jax.Array:
        """Compute logits from hidden states. For sampling."""

    @staticmethod
    def compute_logprobs(hidden_states, lm_head, target_ids, adapter_indices=None,
                         chunk_size=0, gradient_checkpointing=False) -> jax.Array:
        """Compute logprobs from hidden states. For training and prompt logprobs.

        Supports chunked computation to avoid materializing full [B*T, V] logits.
        """

    @staticmethod
    def logits_to_logprobs(logits, target_ids) -> jax.Array:
        """Convert logits to logprobs. For decode logprobs when logits already computed."""
```

### Usage

**Training:**
```python
output = model(input_ids, attention_mask=attention_mask, ...)
logprobs = LogitsProcessor.compute_logprobs(
    output.last_hidden_state, model.lm_head, target_ids,
    chunk_size=1024, gradient_checkpointing=True
)
loss = compute_loss(logprobs, ...)
```

**Sampling (prompt logprobs):**
```python
output = model(input_ids, attention_mask=attention_mask, ...)
prompt_logprobs = LogitsProcessor.compute_logprobs(
    output.last_hidden_state, model.lm_head, input_ids[:, 1:],
    chunk_size=1024
)
```

**Sampling (decode):**
```python
output = model(next_token, kv_cache=kv_cache, ...)
logits = LogitsProcessor.compute_logits(output.last_hidden_state, model.lm_head)
next_token = sample(logits, temperature, top_k, top_p)
logprob = LogitsProcessor.logits_to_logprobs(logits, next_token)
```

### Benefits

1. **Separation of concerns** - Model produces hidden states, LogitsProcessor transforms them
2. **Consistent model interface** - forward() always returns hidden_states
3. **Unified logprobs** - Same API for training and prompt logprobs
4. **Reduced code duplication** - Currently, logprobs computation is duplicated in `generator.py` (`compute_prompt_logprobs`) and `jax.py` backend (chunked loss). This design consolidates both into `LogitsProcessor.compute_logprobs()`
5. **Testable** - Easy to unit test with mock inputs

### Migration Path

1. Update `LogitsProcessor` to standalone utility with three methods
2. Update model to return hidden_states only (remove `skip_logits`, `skip_prompt_logits` flags)
3. Update generator to use `LogitsProcessor.compute_logits()` and `compute_logprobs()`
4. Update backend to use `LogitsProcessor.compute_logprobs()`
5. Remove `logits_processor` attribute from model classes
6. Simplify `CausalLMOutput` (remove `logits`, `lm_head` fields)
