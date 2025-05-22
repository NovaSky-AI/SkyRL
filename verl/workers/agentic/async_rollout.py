
# Modified from: https://github.com/volcengine/verl/pull/917 
import logging
import os
from functools import partial
from json import JSONDecodeError

import os
import torch
import traceback
import sglang as sgl
import torch.distributed
from omegaconf import DictConfig
from sglang.srt.function_call_parser import FunctionCallParser
from sglang.srt.openai_api.protocol import Tool
from torch.distributed import DeviceMesh
from tensordict import TensorDict
# NOTE (sumanthrh): Ideally, we can guard this import and run it only for the swe bench task. 
# However, in openhands' long chain of dependencies, the import for `scantree` fails when this is done.
from verl.workers.agentic.swe_agent.codeact import CodeActAgentGroup

from verl import DataProto
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def _pre_process_inputs(pad_token_id, token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = token_ids[non_pad_index:].tolist()
    return token_ids


class AsyncRollout(BaseRollout):

    def __init__(self, model_path, config: DictConfig, device_mesh: DeviceMesh):
        super().__init__()
        torch.distributed.barrier()
        # print(f"nodedup in AsyncRollout: {torch.distributed.is_initialized() = } {torch.distributed.get_rank() = }")
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.tp_rank = device_mesh.get_local_rank(1)
        self.device_mesh = device_mesh
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"]
        visible_devices: list[str | None] = [None] * device_mesh.size(1)
        torch.distributed.all_gather_object(visible_devices, cuda_visible_device, group=device_mesh.get_group(1))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        print(
            f"nodedup in async rollout {os.environ['CUDA_VISIBLE_DEVICES']=} @ {torch.distributed.get_rank()=} {self.tp_rank=}"
        )
        self.total_len = config.prompt_length + config.response_length
        print(f"async rollout {config.gpu_memory_utilization=}")
        torch.distributed.barrier()
        # print(f"nodedup in async rollout {os.environ['CUDA_VISIBLE_DEVICES']=} @ {torch.distributed.get_rank()=} {self.tp_rank=}")
        if self.tp_rank == 0:
            self.engine = sgl.Engine(
                model_path=model_path,
                port=40000,
                dtype=config.dtype,
                max_total_tokens=60*self.total_len,
                max_prefill_tokens=2*self.total_len,
                enable_memory_saver=config.enable_memory_saver,
                mem_fraction_static=config.gpu_memory_utilization,
                tp_size=device_mesh.size(1),
                log_level="INFO",
                # enable_metrics=True,
            )
            print(f"nodedup {torch.distributed.get_rank() = } releasing memory occupation")
            self.engine.release_memory_occupation()
            print(f"nodedup {torch.distributed.get_rank() = } engine initialized")
        else:
            self.engine = None
        self.engine: sgl.srt.entrypoints.engine.Engine | None
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        torch.distributed.barrier()
        self.config = config
        self.task_type = config.task_type
        # print("Config sampling params:", config.sampling_params)
        self.sampling_params = dict(config.sampling_params)
        self.sampling_params.update({
            "skip_special_tokens": False,
        })
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto | None:
        # print(f"nodedup in generate seq {torch.distributed.get_rank()=} {self.tp_rank=} {prompts.non_tensor_batch=}")
        logger.info(f"nodedup in generate seq {torch.distributed.get_rank()=} {self.tp_rank=}")
        if self.tp_rank != 0:
            return None
        
        sampling_params = self.sampling_params.copy()
        # print("kwargs: ", kwargs)
        sampling_params.update(kwargs)
        print("final sampling params:", sampling_params)
        device = torch.cuda.current_device()
        if sampling_params.get("n", 1) > 1: 
            raise ValueError("Sampling parameter `n` is not supported for multi-turn agentic tasks. For generating multiple trajectories per instance, please use `rollout.n_trajectories` instead.")
        
        if self.config.task_type == "swegym":
            codeact_agent_group = CodeActAgentGroup(
                batch=prompts,
                num_trajectories=self.config.n_trajectories,
                infer_engine=self.engine,
                max_prompt_length=self.config.prompt_length,
                max_response_length=self.config.response_length,
                max_starting_message_length=self.config.max_starting_message_length,
                max_parallel_agents=max(self.config.max_parallel_agents // self.device_mesh.size(0), 1),
                max_eval_parallel_agents=max(self.config.max_eval_parallel_agents // self.device_mesh.size(0), 1),
                max_iterations = self.config.max_iterations,
                tokenizer=self.engine.tokenizer_manager.tokenizer,
                sampling_params=sampling_params,
                device=device,
                log_messages_dir=self.config.log_messages_dir,
                remove_think_tokens=self.config.remove_think_tokens,
                qwen3_enable_thinking=self.config.qwen3_enable_thinking,
            )

            results = codeact_agent_group.run()
        elif self.config.task_type == "sql":
            from verl.workers.agentic.llm_sql_agent.sqlact import SQLActAgentGroup
            from verl.workers.agentic.llm_sql_agent.generation import GenerationConfig
            total_world_size = torch.distributed.get_world_size()
            gen_config = GenerationConfig(
                max_turns=self.config.max_iterations,
                max_start_length=self.config.sql.max_start_length,
                max_prompt_length=self.config.sql.max_prompt_length,
                max_response_length=self.config.sql.max_response_length,
                max_obs_length=self.config.sql.max_obs_length,
                num_gpus= total_world_size // self.device_mesh.size(0),
                db_path=self.config.sql.db_path,
                no_think_rl=False,
            )
            agent_group = SQLActAgentGroup(
                batch=prompts,
                infer_engine=self.engine,
                num_trajectories=self.config.n_trajectories,
                gen_config=gen_config,
                tokenizer=self.engine.tokenizer_manager.tokenizer, 
                sampling_params=self.sampling_params,
            )
            results = agent_group.run()
        else:
            raise NotImplementedError(f"Task type {self.task_type} is not supported.")
        logger.info(f"nodedup finish generate seq {torch.distributed.get_rank()=} {self.tp_rank=}")
        return results
