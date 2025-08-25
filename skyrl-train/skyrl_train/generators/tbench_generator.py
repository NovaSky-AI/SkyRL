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
from transformers import AutoTokenizer
from sandbox.models.trial.config import TrialConfig, AgentConfig, LocalTaskConfig
from pathlib import Path
from sandbox.models.task.id import GitTaskId, LocalTaskId
from sandbox.models.agent.name import AgentName
from sandbox.trial.trial import Trial, TrialEvent
import os
import hashlib

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1
SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"

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
        

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        # [Marianna] set trial dir as environment var for testing (permission denied)
        trials_dir = os.environ.get("TRIALS_DIR", "/p/project/laionize/marianna/terminal_bench/sandboxes/runs")

        self.trial_config = TrialConfig(
            task=LocalTaskConfig(id=LocalTaskId(path="/p/project/laionize/marianna/terminal_bench/sandboxes/examples/tasks/hello-world")),
            trials_dir=Path(trials_dir),
            agent=AgentConfig(
                name=AgentName.TERMINUS.value,
                model_name=self.model_name,
                kwargs={"api_base": f"{self.base_url}/v1", "model_name": f"hosted_vllm/{MODEL}", "key": "fake_key"},
            )
        )

        # self.trial_config = TrialConfig(
        #     task=LocalTaskConfig(id=LocalTaskId(path="/p/project/laionize/marianna/terminal_bench/sandboxes/examples/tasks/hello-world")),
        #     trials_dir=Path(trials_dir),
        #     agent=AgentConfig(
        #         name=AgentName.ORACLE.value,
        #         model_name=self.model_name,
        #         # kwargs={"api_base": f"{self.base_url}/v1", "model_name": f"hosted_vllm/{MODEL}", "key": "fake_key"},
        #     )
        # )

        self.tokenizer = tokenizer

        

    async def tbench_agent_loop(
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

        trial = Trial(self.trial_config)
        # Run the trial
        while True:
            results = await trial.run()
            
            reward = results.verifier_result.rewards
            chat_history = results.agent_result.all_messages
            print("REWARD:", reward)
            print("Results:", results)
            
            if len(chat_history) > 0:
                break
        
        # Use the first message as the prompt
        prompt = [chat_history[0]]
        initial_input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,  # Always add generation prompt for multi-turn
            tokenize=True,
        )
        initial_prompt_length = len(initial_input_ids)
        
        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]
        
        response_ids = []
        loss_mask = []

        for message in response_messages:
            # Apply chat template and tokenize each message
            msg_encoding = self.tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=False,
                tokenize=True
            )
            
            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)
            
            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] == "user":
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))

        print("RESPONSE_MESSAGES:", response_messages)
        print("RESPONSE_IDS:", response_ids)
        print("LOSS_MASK:", loss_mask)
        # Extract prompt ids
        prompt_ids = initial_input_ids
        
        # No need to add EOS token for multi-turn conversations
        
        # Calculate maximum response tokens allowed
        if hasattr(self, 'max_turns') and self.max_turns > 1:
            max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        else:
            max_response_tokens = max_tokens
        
        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        
        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
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
                self.tbench_agent_loop(
                    "Hello, how are you?",
                    "hello-world",
                    [],
                    1024,
                    1024,
                    sampling_params=sampling_params,
                )
            )
 
        all_outputs = await asyncio.gather(*tasks)

        responses = [output[0] for output in all_outputs]
        rewards = [output[1] for output in all_outputs]
        stop_reasons = [output[2] for output in all_outputs]
        loss_masks = [output[3] for output in all_outputs]
        prompt_token_ids = [output[4] for output in all_outputs]
        rollout_metrics = self._rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
        }

        return generator_output