"""
Expected templated strings used for
skyrl-train/tests/cpu/generators/test_skyrl_gym_generator_chat_templating.py::test_skyrl_gym_generator_chat_templating_exact
"""

from datetime import date


# Produced by expected_str = tokenizer.apply_chat_template(expected_chat_history, tokenize=False)
# where expected_chat_history is:
def get_expected_chat_history(mock_response_text: str):
    return [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": mock_response_text},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": mock_response_text},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": mock_response_text},
    ]


QWEN2_5_EXPECTED_STR = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
a<|im_end|>
<|im_start|>assistant
b<|im_end|>
<|im_start|>user
1<|im_end|>
<|im_start|>assistant
b<|im_end|>
<|im_start|>user
2<|im_end|>
<|im_start|>assistant
b<|im_end|>
"""

LLAMA3_2_EXPECTED_STR = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {date.today().strftime("%d %b %Y")}

<|eot_id|><|start_header_id|>user<|end_header_id|>

a<|eot_id|><|start_header_id|>assistant<|end_header_id|>

b<|eot_id|><|start_header_id|>user<|end_header_id|>

1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

b<|eot_id|><|start_header_id|>user<|end_header_id|>

2<|eot_id|><|start_header_id|>assistant<|end_header_id|>

b<|eot_id|>"""

QWEN3_WITHOUT_THINKING_EXPECTED_STR = """<|im_start|>user
a<|im_end|>
<|im_start|>assistant
b<|im_end|>
<|im_start|>user
1<|im_end|>
<|im_start|>assistant
b<|im_end|>
<|im_start|>user
2<|im_end|>
<|im_start|>assistant
<think>
mock thinking
</think>

b<|im_end|>
"""

# Produced by the following code:
# keep_thinking_chat_template = get_custom_chat_template(
#     {"source": "name", "name_or_path": "qwen3_with_thinking"}
# )
# expected_str = tokenizer.apply_chat_template(
#     expected_chat_history, tokenize=False, chat_template=keep_thinking_chat_template
# )
QWEN3_TITO_EXPECTED_STR = """<|im_start|>user
a<|im_end|>
<|im_start|>assistant
<think>
mock thinking
</think>

b<|im_end|>
<|im_start|>user
1<|im_end|>
<|im_start|>assistant
<think>
mock thinking
</think>

b<|im_end|>
<|im_start|>user
2<|im_end|>
<|im_start|>assistant
<think>
mock thinking
</think>

b<|im_end|>
"""
