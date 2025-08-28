import asyncio
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import yaml
import traceback
import os
import json
import threading
from pathlib import Path

from loguru import logger
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import (
    DATASET_MAPPING,
    get_sb_environment,
)
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj
from minisweagent.config import builtin_config_dir, get_config_path

from skyrl_train.generators.skyrl_gym_generator import AgentLoopOutput, SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from .eval import evaluate_result

_OUTPUT_FILE_LOCK = threading.Lock()

def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))

class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
	self,
	generator_cfg: DictConfig,
	skyrl_gym_cfg: DictConfig,
	inference_engine_client: InferenceEngineClient,
	tokenizer,
	model_name: str,
    ):
        # Call parent constructor first                                                                                                                                                                       
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)
        
        self.http_server_inference_engine_client_host = generator_cfg.get(
                "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
                "http_server_inference_engine_client_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        
        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        print("env extras", list(env_extras.keys()))
        instance: Dict[str, Dict[str, Any]] = env_extras["instance"]
        messages, reward = await asyncio.to_thread(self._init_and_run, sweagent_config, instance)
        if not len(messages):
            return None, None, None, None, None, None
        response_messages = messages[2:]

        initial_input_ids = self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=True)
        initial_prompt_length = len(initial_input_ids)

        response_ids: List[int] = []
        loss_mask: List[int] = []

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
        # Extract prompt ids
        prompt_ids = initial_input_ids
        
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

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)
    
    def _init_and_run(self, sweagent_config, instance):
        model_name = "hosted_vllm/" + self.model_name
        model = get_model(model_name, sweagent_config.get("model", {}))
        print("model: ", model, flush=True)
        agent = None
        env = None
        extra_info = None
        result = None
        reward = 0
        try:
            env = get_sb_environment(sweagent_config, instance)
            agent =	DefaultAgent(model, env, **sweagent_config.get("agent", {}))
            print("agent: ", agent, flush=True)
            exit_status, result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            path = Path(self.generator_cfg.miniswe_traj_dir)
            path.mkdir(parents=True, exist_ok=True)
            path = path / (str(instance["instance_id"]) + ".json")
            print("save path", path, flush=True)
            if agent is not None:
                save_traj(agent, path, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
                update_preds_file(path, instance_id=instance['instance_id'], model_name=model_name, result=result)
                
                try: 
                    reward, error = evaluate_result(instance, result, instance['instance_id'], "swebench", sweagent_config)
                    if error:
                        print("error during evaluation: ", error)
                except Exception as e:
                    print("Error during evaluation", e)

        return (agent.messages if agent is not None else [], reward)
    
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
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    self.generator_cfg.sampling_params.max_generate_length,
                    max_input_length = self.generator_cfg.max_input_length,
                    sampling_params=sampling_params,
                )
            )
 
        all_outputs = await asyncio.gather(*tasks)

        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[1] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[2] is not None]
        loss_masks = [output[3] for output in all_outputs if output[3] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[4] is not None]
        rollout_metrics = self._rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output