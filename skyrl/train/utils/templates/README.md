This folder contains customized chat templates.

### `qwen3_acc_thinking.jinja2`

Exactly the same as [the official Qwen3 jinja template](https://huggingface.co/Qwen/Qwen3-8B/blob/895c8d171bc03c30e113cd7a28c02494b5e068b7/tokenizer_config.json#L230)
except that we do not read `reasoning_content` at all and always do `{{- '<|im_start|>' + message.role + '\n' + content }}` for assistant messages. Therefore, you
should not parse the thinking content and pass in all generated text as part of `content`.

The motivation is to not strip thinking tokens and keep the chat history in multi-turn training strictly appending, making the training on-policy without performing step-wise training.

Besides, unlike the official Qwen3, this template does not add an empty thinking block when no thinking tokens are present:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
]

# Default Qwen3 template: adds empty thinking block
default_output = tokenizer.apply_chat_template(messages, tokenize=False)
# Result: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n"

# Custom qwen3_acc_thinking template: no thinking block added
with open("qwen3_acc_thinking.jinja2") as f:
    custom_template = f.read()
custom_output = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=custom_template)
# Result: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"
```

Specifically for the chat template, we change from

```jinja2
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
        ...
```

to

```jinja2
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- if message.tool_calls %}
        ...
```

When using `/chat/completions` with vllm via the HTTP endpoint for rollout, you can pass this in with 

```bash
+generator.engine_init_kwargs.chat_template=/path/to/templates/qwen3_acc_thinking.jinja2
```
