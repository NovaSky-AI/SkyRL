import asyncio
import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
import time
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.generators.utils import get_custom_chat_template, get_generation_prompt_ids, apply_overlong_filtering
import threading
import terminal_bench
from terminal_bench import Harness
from terminal_bench.agents import AgentName
from pathlib import Path
from datetime import datetime
import logging
from skyrl_train.inference_engines.launch_inference_engine_http_server import (
    serve,
    wait_for_server_ready,
    shutdown_server,
    handle_chat_completion,
)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"
from litellm import completion

from fastapi import Request
from skyrl_train.inference_engines.openai_api_protocol import ChatCompletionRequest

class TBenchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)
        
        print("ETASH WUZ HERE")
        print(type(self.inference_engine_client))
        print(self.inference_engine_client)

        def run_server():
            serve(self.inference_engine_client, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
        print('starting server')
        # run_server()
        # print('server started')
        server_thread = threading.Thread(target=run_server, daemon=False)
        server_thread.start()
        print('server started')
        # Wait for server to be ready using the helper method
        t1 = time.time()

        request = ChatCompletionRequest(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [(b"host", b"testserver")],
            "query_string": b"",
        }
        
        raw_request = Request(scope)
        wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=30)
        t2 = time.time()
        print(f"time taken: {t2 - t1}")
        print("server ready")

        r = handle_chat_completion(request, raw_request)
        print("RESPONSE", r)
        print(r.choices[0].message.content)
        
        chat_history = [
            {"role": "user", "content": "Hello, how are you?"},
        ]
        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

        from openai import OpenAI

        openai_api_key = "EMPTY"  # vLLM doesn't require a key
        openai_api_base = f"http://{SERVER_HOST}:{SERVER_PORT}/v1" # Adjust if your server is on a different port/host

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model=inference_engine_client.model_name, # Replace with the model served by vLLM
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short story."},
            ]
        )
        print("Chat response:", chat_response.choices[0].message.content)
        response = completion(
            model=f"openai/{MODEL}",  # Add openai/ prefix for custom endpoints
            messages=chat_history,
            api_base=base_url,
            temperature=0.7,
            max_tokens=100,
        )
        print("AFTER RESPONSE")
        print("response", response)
        print(response.choices[0].message.content)
        # Initialize TBench harness

        # Start up basic vLLM server
        # import subprocess
        # import time
        
        # # Launch vLLM server as a subprocess
        # vllm_cmd = [
        #     "python", "-m", "vllm.entrypoints.openai.api_server",
        #     "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        #     "--host", SERVER_HOST,
        #     "--port", str(SERVER_PORT),
        #     "--served-model-name", "Qwen/Qwen2.5-0.5B-Instruct"
        # ]
        
        # print(f"Starting vLLM server with command: {' '.join(vllm_cmd)}")
        # self.vllm_process = subprocess.Popen(vllm_cmd)
        
        # # # Wait for server to be ready
        # # wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=60)
        # print(f"vLLM server ready at {SERVER_HOST}:{SERVER_PORT}")
        self.tbench_harness = Harness(
            dataset_name="terminal-bench-core",
            dataset_version="0.1.1",
            dataset_path=None,
            dataset_config=None,
            registry_url=None,
            local_registry_path=None,
            output_path=Path("runs"),
            run_id=datetime.now().strftime("%Y-%m-%d__%H-%M-%S"),
            agent_name=AgentName.TERMINUS,
            agent_import_path=None,
            model_name="hosted_vllm/Qwen2.5-0.5B-Instruct",
            agent_kwargs={"url": f"http://{SERVER_HOST}:{SERVER_PORT}/v1", "key": "fake_key"},
            no_rebuild=False,
            cleanup=False,
            log_level=logging.INFO,
            task_ids=['hello-world'],
            n_tasks=None,
            livestream=False,
            upload_results=False,
            n_concurrent_trials=16,
            exclude_task_ids=None,
            n_attempts=1,
            global_timeout_multiplier=1.0,
            global_agent_timeout_sec=None,
            global_test_timeout_sec=None,
        )

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], float, str, List[int], List[int]]:
        """
        Multi-turn generation loop that executes a single trajectory.

        Args:
            prompt: ConversationType
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: float
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
        """        
        
        task_path = self.tbench_harness._dataset._tasks[0]
        print(task_path)


        attempt = 0  # Define the attempt variable
        trial_name = self.tbench_harness._get_trial_name(task_path, attempt)
        print("ETASH 110")
        results = self.tbench_harness._execute_single_trial(
            trial_name=trial_name,
            task_path=task_path,
        )

        
        
        # Define the return variables properly
        # You'll need to extract these from the results
        response_ids = []  # Extract from results
        reward = 0.0  # Extract from results
        stop_reason = "completed"  # Extract from results
        loss_mask = []  # Extract from results
        prompt_ids = []  # Extract from results
        
        return response_ids, reward, stop_reason, loss_mask, prompt_ids


    async def generate_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int --> Currently unused as we assume batched is used only for single-turn.
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            GeneratorOutput
        """
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.agent_loop(
                    prompts[i],
                    env_classes[i],
                    env_extras[i],
                    1024,
                    1024,
                    sampling_params=sampling_params,
                )
            )
        all_outputs = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
        )

        rewards = sum([[output.is_resolved] for output in all_outputs], [])

        responses = sum([[output[0]] for output in all_outputs], [])
        stop_reasons = sum([[output[2]] for output in all_outputs], [])
        loss_masks = sum([[output[3]] for output in all_outputs], [])
        prompt_token_ids = sum([[output[4]] for output in all_outputs], [])

        rollout_metrics = self._rollout_metrics(responses, rewards)

        if self.generator_cfg.zero_reward_on_non_stop:
            # set reward to 0 if the stop reason is not "stop"
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self.tokenizer.eos_token_id)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
        }

        return generator_output