## uv run --active python -m skyrl_gym.envs.rlm.test

import asyncio
import os
from omegaconf import DictConfig
from typing import Optional

import ray
from loguru import logger

from skyrl_gym.envs.rlm.env import RLMEnvironment


async def simple_generation(
    prompt: str,
    openai_api_key: Optional[str] = None,
):
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(examples_dir, "rlm_system_prompt.txt"), "r") as f:
        init_prompt = f.read()
    env_cfg = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5-mini",
        "init_prompt": init_prompt,
        "openai_api_key": openai_api_key,
    }

    env = RLMEnvironment()
    with open(os.path.join(examples_dir, "context.txt"), "r") as f:
        context_data = f.read()
    env.load_context(context_data)
    env._engine_setup(DictConfig(env_cfg))
    out = env.step(prompt)
    env.close()
    return out


@ray.remote(num_cpus=1)
def run_simple_test(prompt: str):

    import dotenv

    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    return asyncio.run(
        simple_generation(
            prompt=prompt,
            openai_api_key=openai_api_key,
        )
    )


def main():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized")

    test_prompt = (
        "In the above data, which of the labels is the most common? "
        "Give your final answer in the form 'Label: answer' where answer is one of the labels: ham, spam."
    )
    logger.info(f"Testing with prompt: {test_prompt}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        logger.warning("OPENAI_API_KEY not set in environment. Make sure to set it before running.")

    result = ray.get(
        run_simple_test.remote(
            prompt=test_prompt,
        )
    )
    logger.success("Test completed successfully!")
    return result


if __name__ == "__main__":
    main()
