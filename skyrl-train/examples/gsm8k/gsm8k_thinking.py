import sys
sys.path.append('.')

from transformers import AutoTokenizer
from skyrl_train.generators.utils import get_custom_chat_template

test_conversation = [
    {"role": "user", "content": "What is 5 + 3?"},
    {"role": "assistant", "content": "<think>\nI need to add 5 + 3.\n5 + 3 = 8\n</think>\n\nThe answer is 8."}
]

model_name = "Qwen/Qwen3-0.6B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
except:
    print("Using fallback tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

print("="*60)
print("TESTING SINGLE QUESTION")
print("="*60)

# Test WITH thinking tokens
print("\n1. WITH THINKING TOKENS:")
template_with = get_custom_chat_template(model_name, thinking_mode=True)
if template_with:
    result_with = tokenizer.apply_chat_template(test_conversation, chat_template=template_with, tokenize=False)
    tokens_with = tokenizer.apply_chat_template(test_conversation, chat_template=template_with, tokenize=True)
    print(f"Output: {repr(result_with)}")
    print(f"Tokens: {len(tokens_with)}")
else:
    print("No template found")

# Test WITHOUT thinking tokens  
print("\n2. WITHOUT THINKING TOKENS:")
template_without = get_custom_chat_template(model_name, thinking_mode=False)
if template_without:
    result_without = tokenizer.apply_chat_template(test_conversation, chat_template=template_without, tokenize=False)
    tokens_without = tokenizer.apply_chat_template(test_conversation, chat_template=template_without, tokenize=True)
    print(f"Output: {repr(result_without)}")
    print(f"Tokens: {len(tokens_without)}")
    
    if template_with:
        print(f"\nDifference: {len(tokens_with) - len(tokens_without)} tokens")
else:
    print("No template found")

print("\n" + "="*60)
print("✅ THINKING TOKEN TEST COMPLETE")
