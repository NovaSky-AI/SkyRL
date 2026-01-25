# SkyRL Training Mechanics

## Key Configuration Flags

### `step_wise_trajectories` (default: false)

**Location**: `generator.step_wise_trajectories`

**What it does**: Controls how multi-turn trajectories become training samples.

| Setting | Sample Structure | Batch Weighting |
|---------|------------------|-----------------|
| `false` | 1 trajectory = 1 sample (all turns concatenated) | Each trajectory contributes equally |
| `true` | 1 trajectory = N samples (one per turn) | Longer trajectories contribute more samples |

**With `step_wise=true`**:
- Each turn becomes a separate training sample
- Advantage is computed on FINAL step only, then propagated to all earlier steps
- All steps in same trajectory get the same advantage
- Loss mask per turn is trivially correct (just that turn's response)

**Important**: There is NO double normalization. With `step_wise=true`, a 10-turn trajectory contributes 10 samples to the batch, giving it 10x the gradient contribution of a 1-turn trajectory.

**Code locations**:
- Generator: `skyrl_train/generators/skyrl_gym_generator.py:258`
- Advantage computation: `skyrl_train/trainer.py:787-818`
- Sample assembly: `skyrl_train/generators/skyrl_gym_generator.py:700-720`

---

### `retokenize_chat_history`

**Not a config flag** - computed as:
```python
retokenize_chat_history = self.use_conversation_multi_turn and self.custom_chat_template
```

**What it does**: When true, re-tokenizes the entire conversation every turn instead of appending tokens incrementally.

**Why it exists**: To get accurate loss masks via `return_assistant_tokens_mask=True` from the tokenizer.

**Status**: Being deprecated in favor of `step_wise_trajectories`. See comment at `skyrl_gym_generator.py:218-220`:
```python
# NOTE: `custom_chat_template` was mainly for getting accurate loss masks for thinking models.
# This is no longer needed now given that step wise training is supported
```

---

### `loss_reduction` modes

**Location**: `trainer.algorithm.loss_reduction`

| Mode | How it works |
|------|--------------|
| `token_mean` | Mean over all valid tokens in batch |
| `sequence_mean` | Per-sample mean, then batch mean (each sample weighted equally) |
| `seq_mean_token_sum_norm` | Sum/max_len per sample, then batch mean (Dr. GRPO) |

**Code**: `skyrl_train/utils/ppo_utils.py:881-906`

---

## Chat Templates

### What is a chat template?

A Jinja2 template that converts a list of messages to a tokenizable string:

```python
# Input
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<think>math</think>4"},
]

# Output (after template)
"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>math</think>4<|im_end|>\n"
```

### `qwen3_acc_thinking.jinja2`

**Location**: `skyrl_train/utils/templates/qwen3_acc_thinking.jinja2`

**Purpose**: Outputs assistant content verbatim WITHOUT reformatting `<think>` tags.

**Problem it solves**: The official Qwen3 template parses `<think>` tags and adds newlines:
```
# Model generates:
<think>reasoning</think>answer

# Official template reformats to:
<think>
reasoning
</think>

answer
```

This reformatting produces different tokens, causing off-policy training.

**Primary use case**: Pass to vLLM inference engine to prevent reformatting:
```bash
+generator.engine_init_kwargs.chat_template=/path/to/qwen3_acc_thinking.jinja2
```

---

## Where Templates Are Applied

1. **L239** - Initial prompt tokenization
2. **L281** - Each turn (if `step_wise` or `retokenize_chat_history`)
3. **L519** - Adding observations to sequence
4. **L401** - Final loss mask computation (if `retokenize_chat_history`)

**Code**: `skyrl_train/generators/skyrl_gym_generator.py`

---

## Training Path Options for Thinking Models

### Option A: `step_wise_trajectories=true`
- Each turn is separate sample
- Loss masks trivially correct
- Training dynamics change: longer trajectories = more samples

### Option B: `step_wise=false` + `qwen3_acc_thinking` template
- Whole trajectory is one sample
- Use template in vLLM so context isn't reformatted
- Incremental token appending stays on-policy
- Training dynamics unchanged

---

## Agent Loop Flow

```
skyrl_gym_generator.py:agent_loop()

1. init(): Get initial prompt from env
2. apply_chat_template() → initial_input_ids
3. while not done:
   a. Generate output tokens
   b. env.step(output) → observation, reward, done
   c. Update state (append tokens or retokenize)
4. Build final response_ids and loss_mask
```

**Main loop**: `skyrl_gym_generator.py:272` - `while not agent_loop_state.done:`

---

## Thinking Tokens: When They Get Stripped

### The Problem

The official Qwen3 chat template **strips `<think>` content from non-last assistant messages**. This is by design for inference (users don't want to see old reasoning), but breaks RL training.

**Example - Multi-turn with official template:**
```
Turn 1: Model generates "<think>step 1</think>call_tool()"
Turn 2: Model sees context WITHOUT the thinking: "call_tool()"  ← thinking stripped!
Turn 3: Loss computed on tokens that don't match what model saw
```

This causes **off-policy training** - the model is trained on token sequences it never actually generated in that context.

### Why It Matters for RL

1. **On-policy requirement**: PPO/GRPO assume the policy that generated the data is the same as the one being trained
2. **Context mismatch**: If thinking is stripped from context, the model's next-turn generation was conditioned on different tokens than what we compute loss on
3. **Credit assignment**: The model can't learn from its reasoning if that reasoning disappears from context

### Solutions

| Approach | How | Trade-off |
|----------|-----|-----------|
| **`step_wise_trajectories=true`** | Each turn is separate sample, no cross-turn context needed | Longer trajectories get more weight in batch |
| **`qwen3_acc_thinking.jinja2`** | Custom template that preserves thinking verbatim | Need to pass to vLLM |
| **Don't use custom template** (Fleet default) | Append raw tokens without re-applying template | Works if not using `retokenize_chat_history` |

### Fleet Tool-Use Config (Current)

```yaml
step_wise_trajectories=false      # Whole trajectory = 1 sample
custom_chat_template=null         # No template reformatting
use_conversation_multi_turn=true  # Multi-turn enabled
```

**Why this works**: Without a custom template, SkyRL appends tokens directly without re-applying any template. The model sees its own thinking in context, and loss is computed on the same tokens. Fully on-policy.

### When Thinking DOES Get Stripped

1. Using official Qwen3 template with `retokenize_chat_history=true`
2. Passing messages through any template that parses `<think>` tags
3. Using `step_wise=true` with default template (each turn re-tokenized, thinking reformatted)

### Key Insight

The confusion arises because there are TWO places templates matter:
1. **vLLM inference** - How the engine formats the prompt for generation
2. **Training tokenization** - How we build `response_ids` for loss computation

For on-policy training, both must produce the same tokens.

---

## Key Files

| File | Purpose |
|------|---------|
| `skyrl_train/generators/skyrl_gym_generator.py` | Agent loop, token management |
| `skyrl_train/trainer.py` | Training loop, advantage computation |
| `skyrl_train/utils/ppo_utils.py` | Loss functions, advantage estimators |
| `skyrl_train/utils/templates/` | Chat templates for thinking models |
| `skyrl_train/config/ppo_base_config.yaml` | Default configuration |
