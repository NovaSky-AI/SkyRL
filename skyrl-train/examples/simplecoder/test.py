import asyncio
from simplecoder_generator import SimpleCoderGenerator
from skyrl_train.generators.base import GeneratorInput

async def test_simple_coder_generator():
    # Create a SimpleCoderGenerator instance
    generator = SimpleCoderGenerator()
    
    # Create an input batch with 2 prompts
    input_batch: GeneratorInput = {
        "prompts": [
            [{"role": "user", "content": "Test prompt 1"}],
            [{"role": "user", "content": "Test prompt 2"}]
        ],
        "env_classes": ["default", "default"],
        "env_extras": [{"test": "extra1"}, {"test": "extra2"}],
        "sampling_params": {"temperature": 0.7}
    }
    
    # Call generate() method
    result = await generator.generate(input_batch)
    
    print("Generated results: todo")
    # print(f"Number of prompts processed: {len(input_batch['prompts'])}")
    # print(f"Result keys: {list(result.keys())}")
    
    return result

# Run the test
if __name__ == "__main__":
    asyncio.run(test_simple_coder_generator())