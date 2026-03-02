import asyncio
import os
from typing import List, Dict, Any
import pytest
from omegaconf import DictConfig
from transformers import AutoTokenizer
from skyrl_train.generators.context_folding import ContextFolder

# Mock inference engine client that mimics the real inference engine's generate() method
class MockInferenceEngineClient:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.call_count = 0
    
    async def generate(self, engine_input):
        """Mock the inference engine's generate method for summarization"""
        self.call_count += 1
        
        # engine_input is a dict, not an object
        prompt_ids = engine_input["prompt_token_ids"][0]
        prompt_text = self.tokenizer.decode(prompt_ids)
        
        print(f"\n📞 Inference Engine Called (call #{self.call_count})")
        print(f"📝 Prompt length: {len(prompt_ids)} tokens")
        print(f"📝 Session ID: {engine_input['session_ids'][0]}")
        print(f"📝 Sampling params: {engine_input['sampling_params']}")
        print(f"📋 Prompt preview (first 200 chars):\n{prompt_text[:200]}...")
        
        # Generate a mock summary response wrapped in <summary> tags
        summary_text = (
            "<summary>"
            "The conversation covered machine learning fundamentals including supervised, "
            "unsupervised, and reinforcement learning. Discussed neural networks with "
            "backpropagation and deep learning. Explained transformer architecture with "
            "attention mechanisms and their effectiveness in NLP tasks. User now asking "
            "about implementation details."
            "</summary>"
        )
        
        # Encode the summary
        summary_ids = self.tokenizer.encode(summary_text, add_special_tokens=False)
        
        # Return in the format expected by ContextFolder
        return {
            "responses": [summary_text],
            "response_ids": [summary_ids],
            "stop_reasons": ["stop"],
            "response_logprobs": [[0.0] * len(summary_ids)]  # Mock logprobs
        }

@pytest.mark.asyncio
async def test_context_folding():
    print("🧪 Testing Context Folding...")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup mock inference client
    mock_client = MockInferenceEngineClient(tokenizer)
    
    # Context folding configuration
    folding_cfg = DictConfig({
        "enabled": True,
        "trigger_ratio": 0.7,
        "min_tokens": 50,
        "max_folds": 3,
        "summary_max_tokens": 200,
        "summary_prompt": "Your context window is full. Summarize the conversation so far. Wrap your summary in <summary></summary> tags.",
        "summary_prefix": "[Previous conversation summary]\n{summary}\n\nPlease continue.",
        "summary_role": "user",
        "keep_initial_prompt_tokens": 1,  # Keep system message
        "keep_last_messages": 2,  # Keep last 2 messages
        "include_summary_in_training": False
    })
    
    # Create context folder with complete sampling params for vLLM
    base_sampling_params = DictConfig({
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "min_p": 0.0,
        "logprobs": None,
        "max_generate_length": 100
    })
    
    context_folder = ContextFolder(
        cfg=folding_cfg,
        tokenizer=tokenizer,
        inference_engine_client=mock_client,
        backend="vllm",
        base_sampling_params=base_sampling_params,
        chat_template_kwargs={}
    )
    
    # Create mock conversation that's long enough to trigger folding
    mock_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about machine learning algorithms."},
        {"role": "assistant", "content": "Machine learning algorithms are computational methods that enable computers to learn patterns from data without being explicitly programmed. There are several main categories: supervised learning (like linear regression and decision trees), unsupervised learning (like clustering and dimensionality reduction), and reinforcement learning (where agents learn through trial and error). Each type has different use cases and strengths."},
        {"role": "user", "content": "Can you explain neural networks in more detail?"},
        {"role": "assistant", "content": "Neural networks are inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that determines its strength. During training, these weights are adjusted using backpropagation to minimize prediction errors. Deep neural networks with multiple hidden layers can learn complex patterns and representations, making them powerful for tasks like image recognition, natural language processing, and game playing."},
        {"role": "user", "content": "What about transformers?"},
        {"role": "assistant", "content": "Transformers revolutionized natural language processing and are the architecture behind models like GPT and BERT. Key innovations include the attention mechanism, which allows the model to focus on relevant parts of the input sequence, and parallel processing capabilities. The self-attention mechanism computes relationships between all positions in a sequence simultaneously, making transformers very effective for understanding context and long-range dependencies in text."},
        {"role": "user", "content": "How do I implement a simple transformer?"},
    ]
    
    # Calculate token length to see if it triggers folding
    full_text = tokenizer.apply_chat_template(mock_conversation, tokenize=False)
    input_ids = tokenizer.encode(full_text)
    current_length = len(input_ids)
    max_length = 300  # Small max length to force folding
    
    print(f"📊 Current conversation length: {current_length} tokens")
    print(f"📊 Max allowed length: {max_length} tokens")
    print(f"📊 Trigger ratio: {folding_cfg.trigger_ratio}")
    print(f"📊 Trigger threshold: {int(max_length * folding_cfg.trigger_ratio)} tokens")
    
    # Test fold trigger
    should_fold = context_folder.fold_trigger(
        current_input_length=current_length,
        max_input_length=max_length,
        fold_count=0
    )
    
    print(f"🤔 Should fold? {should_fold}")
    
    if should_fold:
        print("\n🔄 Attempting to fold context...")
        
        # Test the actual folding
        fold_result = await context_folder.fold(
            chat_history=mock_conversation,
            current_input_length=current_length,
            max_input_length=max_length,
            initial_chat_history_length=len(mock_conversation),
            session_id="test_session",
            fold_count=0
        )
        
        if fold_result.folded:
            print("✅ Context folding successful!")
            print(f"📝 Original messages: {len(mock_conversation)}")
            print(f"📝 After folding: {len(fold_result.new_chat_history)}")
            print(f"🎯 Summary tokens: {len(fold_result.summary_output_ids) if fold_result.summary_output_ids else 0}")
            print(f"🔍 Extracted summary text: {fold_result.summary_text}")
            
            print("\n📋 New conversation structure:")
            for i, msg in enumerate(fold_result.new_chat_history):
                role = msg["role"]
                content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
                print(f"  {i}: [{role}] {content}")
            
            print(f"\n📊 Token counts:")
            print(f"  Before: {current_length} tokens")
            new_length = len(fold_result.new_input_ids)
            print(f"  After: {new_length} tokens")
            print(f"  Saved: {current_length - new_length} tokens ({100 * (current_length - new_length) / current_length:.1f}%)")
            
            print(f"\n🔧 Inference engine stats:")
            print(f"  Total calls: {mock_client.call_count}")
        else:
            print("❌ Context folding did not occur")
    else:
        print("ℹ️  Context folding not triggered (conversation too short)")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_context_folding())