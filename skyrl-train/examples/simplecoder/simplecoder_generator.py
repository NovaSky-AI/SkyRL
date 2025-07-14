import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import simplecoder

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput

TASK = """
    I'm running missing_colon.py as follows:

division(23, 0)
but I get the following error:

  File "/Users/fuchur/Documents/24/git_sync/swe-agent-test-repo/tests/./missing_colon.py", line 4
    def division(a: float, b: float) -> float
                                             ^
SyntaxError: invalid syntax
"""

class SimpleCoderGenerator(GeneratorInterface):
    def __init__(self):
        pass

    def _run_single_task(self, task, manifest, working_dir):
        """Run a single task - this will be executed in the thread pool"""
        executor = simplecoder.GuixExecutor(working_dir, manifest)
        coder = simplecoder.SimpleCoder(os.environ["OPENAI_API_KEY"], "o4-mini", executor)
        return coder.run(task)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        tasks = [TASK] * len(input_batch["prompts"])
        
        manifest = os.path.abspath("manifest.scm")
        working_dir = os.path.abspath("test-repo")
        
        # Run all tasks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    executor, 
                    self._run_single_task, 
                    task, 
                    manifest, 
                    working_dir
                )
                for task in tasks
            ]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*futures)
        
        return {
            "prompt_token_ids": [],
            "response_ids": [],
            "rewards": [],
            "loss_masks": [],
            "stop_reasons": [],
            "rollout_metrics": {}
        }