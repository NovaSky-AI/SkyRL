# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List, Dict, Any
from contextlib import contextmanager
import os
import logging
import asyncio
import json5
import json 
import copy
import torch
import torch.distributed
from tensordict import TensorDict
from verl.utils.model import compute_position_id_with_mask
import requests
from typing import Any, Union, List, Tuple
from verl import DataProto
from openhands.utils.async_utils import call_sync_from_async
from typing import Tuple
import asyncio
import json
import copy
from verl.utils.reward_score.math_torl import compute_score
from .qwen_agent.tools.python_executor import PythonExecutor
import time


logger = logging.getLogger(__name__)

# Global counter for code execution time
total_code_execution_time = 0.0

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


OBS_START = '```output'
OBS_END = '\n```\n'
def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ''
    start = False
    for line in result.split('\n'):
        if line.startswith('```python') or line.endswith('```python'):
            if last_only:
                program = ''  # only extract the last program
            else:
                program += '\n# ========\n'
            start = True
        elif line.startswith('```'):
            start = False
        elif start:
            program += line + '\n'
    if start:
        # the code is incomplete
        program = ''
    return program

def _detect_tool(text: str) -> Tuple[bool, str, str, str]:
    program = extract_program(text)
    if program:
        program = json.dumps({'code': program}, ensure_ascii=False)
    return (program != ''), PythonExecutor.name, program, text

# From codeact
def convert_right_padding_to_left(tokenizer, input_ids, attention_mask, device, max_len=None):
    """
    Converts right-padded tensors to left-padded tensors with optional custom length.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        device: The device to place the new tensors on
        max_len (int, optional): The desired maximum length of the returned tensors.
                                If None, uses the original sequence length.
    
    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, orig_seq_length = input_ids.size()
    
    # Use original length if max_len is not specified
    seq_length = max_len if max_len is not None else orig_seq_length
    
    # Create new tensors with the desired size
    left_padded_input_ids = torch.full((batch_size, seq_length), 
                                     tokenizer.pad_token_id, 
                                     dtype=input_ids.dtype, 
                                     device=device)
    left_padded_attention_mask = torch.zeros((batch_size, seq_length), 
                                           dtype=attention_mask.dtype, 
                                           device=device)
    
    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()
        
        # Trim sequence if it's longer than max_len
        if seq_len > seq_length:
            logger.warning(f"Trimming sequence length from {seq_len} to {seq_length}")
            seq_len = seq_length
        
        # Calculate the offset for left padding
        offset = seq_length - seq_len
        
        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = 1  # Set attention mask for non-padding tokens
    
    return left_padded_input_ids, left_padded_attention_mask

def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    """
    Pads tokenizer outputs to a specific maximum length with configurable padding side.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        encodings (dict): Dictionary containing 'input_ids', 'attention_mask', and optionally 'assistant_masks'
        max_length (int): The desired maximum length to pad to
        device: The device to place the tensors on
        
    Returns:
        dict: Dictionary with padded tensors for 'input_ids', 'attention_mask', and 'assistant_masks' if present
    """
    batch_size = len(encodings['input_ids'])
    
    # Initialize output tensors
    padded_input_ids = torch.full((batch_size, max_length), 
                                tokenizer.pad_token_id, 
                                dtype=torch.long, 
                                device=device)
    padded_attention_mask = torch.zeros((batch_size, max_length), 
                                      dtype=torch.long, 
           
           
                                      device=device)
    padded_assistant_mask = torch.zeros((batch_size, max_length), 
                                          dtype=torch.long, 
                                          device=device)
    
    # Fill tensors with actual values
    num_trimmed = 0
    for i in range(batch_size):
        seq_len = encodings["attention_mask"][i].sum().item() if isinstance(encodings["attention_mask"][i], torch.Tensor) else sum(encodings["attention_mask"][i])
        # Trim if longer than max_length
        actual_len = min(seq_len, max_length)
        if seq_len > max_length:
            logger.warning(
                f"Trimming sequence length from {seq_len} to {actual_len} for batch item {i}"
            )
            num_trimmed += 1
        
        # Right padding - copy sequence data to the beginning
        padded_input_ids[i, :actual_len] = torch.tensor(encodings['input_ids'][i][:actual_len], device=device)
        padded_attention_mask[i, :actual_len] = torch.tensor(encodings['attention_mask'][i][:actual_len], device=device)
        padded_assistant_mask[i, :actual_len] = torch.tensor(encodings['assistant_masks'][i][:actual_len], device=device)
    
    logger.info(f"Trimmed {num_trimmed*100 / max(batch_size, 1)}% of samples in the batch of size {batch_size}")
    return padded_input_ids, padded_attention_mask, padded_assistant_mask

chat_template = (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    )

class OnlineToRLAgent():

    def __init__(
        self,
        infer_engine: Any,
        tokenizer: Any,
        sampling_params: Any,
        config: Any,
        max_prompt_length: int,
        max_response_length: int,
        python_executor: Any = None,
    ) -> None:
        self.inference_engine = infer_engine  # Use the shared inference engine
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.config = config
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
    def _get_prompts_and_indices(self, samples_info):
        prompts, indices=[], []
        for index, info in enumerate(samples_info):
            if not info['stop']:
                prompts.append(info['sequence'])
                indices.append(info['index'])
        return prompts, indices

    def _tokenize_and_find_mask_token_indices(self, sample_info):
        response=sample_info['response']
        mask_str_ranges=sample_info['mask_info']
        # print(response, "Line 166")
        # print(mask_str_ranges, "Line 167")

        encoding=self.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        
        response_token_ids=encoding['input_ids']

        offset_mapping_tensor=torch.tensor(encoding['offset_mapping'], dtype=torch.long)
        token_starts = offset_mapping_tensor[:,0]
        token_ends = offset_mapping_tensor[:,1]

        mask_tensor=torch.ones(len(response_token_ids))
        for mask_str_range in mask_str_ranges:
            start_index, end_index=mask_str_range[0], mask_str_range[1]
            mask = (token_starts < end_index) & (token_ends > start_index) & (token_starts >= start_index)
            mask_tensor[mask]=0 
        # print("response_token_ids", response_token_ids, "Line 180")
        # print("mask_tensor", mask_tensor, "Line 180")
        return response_token_ids, mask_tensor


    async def _tir_generate(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=False):
        sampling_params=copy.deepcopy(sampling_params)
        prompts=[self.tokenizer.decode(prompt['prompt_token_ids'], skip_special_tokens=False) for prompt in prompts]

        system_prompt = prompts[0].split("system\n")[1].split("<|im_end|>")[0].strip()
        user_prompt = prompts[0].split("user\n")[1].split("<|im_end|>")[0].strip()
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Handle both dict and object-style sampling_params
        n_value = sampling_params.get('n', 1) if isinstance(sampling_params, dict) else getattr(sampling_params, 'n', 1)
        prompts=[prompt for prompt in prompts for _ in range(n_value)]
        
        if isinstance(sampling_params, dict):
            sampling_params['n'] = 1
            sampling_params['stop'] = ["```output"]
        else:
            sampling_params.n = 1
            sampling_params.stop = ["```output"]
        
        samples_info=[{"prompt": prompt, "sequence": prompt, "response": "", "stop": False, "finish_reason": None,"index": index, "mask_info": [], "execution_pass": 0} for index, prompt in enumerate(prompts)]
        num_llm_calls_available=copy.deepcopy(self.config.max_iterations)
        while num_llm_calls_available >= 0:
            if num_llm_calls_available==0: 
                if isinstance(sampling_params, dict):
                    sampling_params['stop'] = None
                else:
                    sampling_params.stop = None
            num_llm_calls_available-=1
            # llm generate response, stop at eos token or ```output
            input_prompts, indices=self._get_prompts_and_indices(samples_info)
            input_prompts = [self.tokenizer.encode(x, add_special_tokens=False)[:self.config.prompt_length+self.config.response_length] for x in input_prompts]
            
            outputs = await self.inference_engine.async_generate(input_ids=input_prompts, sampling_params=sampling_params)
            sorted_outputs = sorted(outputs, key=lambda output: output["meta_info"]["id"])
            assert len(sorted_outputs) == 1, "Dacheng: only one output should be here."
            responses=[x["text"] for x in sorted_outputs]
            finish_reason=[x["meta_info"]["finish_reason"]["type"] for x in sorted_outputs]
            stop_reason=[x["meta_info"]["finish_reason"].get("matched", "") for x in sorted_outputs]
            if num_llm_calls_available==-1:
                for i ,index in enumerate(indices):
                    samples_info[index]['response']+=responses[i]
                    samples_info[index]['sequence']+=responses[i]
                    samples_info[index]['stop']=True
                    samples_info[index]['finish_reason']=finish_reason[i]
                break

            def _python_execution(finish_reason, stop_reason):
                if finish_reason=='stop' and stop_reason==None: return False
                if finish_reason=='stop' and stop_reason=='```output': return True
                if finish_reason=='length': False
                return False
            is_execution=[_python_execution(finish_reason[i], stop_reason[i]) for i in range(len(finish_reason))]
            # check if all samples are finished
            message.append(
                {
                    "role": "assistant",
                    "content": responses[0]
                }
            )
            if all([not x for x in is_execution]): break

            # prepare for python execution
            tool_infos=[ _detect_tool(response) for response in responses]
            # print("tool_infos", tool_infos, "Line 251")
            tool_indices=[]
            tool_inputs=[]
            for i, tool_info in enumerate(tool_infos):
                if tool_info[0] and is_execution[i]:
                    tool_indices.append(i)
                    tool_inputs.append(tool_info[2])
            # print("tool_inputs", tool_inputs, "Line 258")
            def postproc_observation(observation):
                execution_pass=0
                try:
                    observation_list=observation
                    if observation_list[-1] == 'Done':
                        observation = observation_list[0]
                        execution_pass=1
                    else:
                        observation = observation_list[-1]
                except Exception:
                    observation="Error"
                if "Error" in observation: observation=observation.strip().split("\n")[-1]
                if len(observation.strip())==0: observation="timeout_decorator.timeout_decorator.TimeoutError: 'Timed Out'"
                observation = observation.strip()
                if len(observation)>=256:
                    observation = observation[:128]+"..."+observation[-128:]
                observation = f'{OBS_START}\n{observation}{OBS_END}'
                return observation, execution_pass

            # execute python code

            # observations=self.executor.batch_apply([json5.loads(x)['code'] for x in tool_inputs])
            observations= await call_sync_from_async(self.code_interpreter_batch_call, [json5.loads(x)['code'] for x in tool_inputs])
            
            # construction responses from observations
            responses=[response+"\n" if not response.endswith('\n') else response for response in responses]
            responses_w_res=copy.deepcopy(responses)
            execution_passes=[0 for _ in range(len(responses))]
            observation_list = ["" for _ in range(len(responses))]
            for i, index in enumerate(tool_indices):
                processed_observation=postproc_observation(observations[i])
                # print(f"processed_observation: {processed_observation[0]} in line 305")
                observation_list[index] += processed_observation[0] #self.tokenizer.decode(processed_observation[0], skip_special_tokens=True)
                responses_w_res[index]+=processed_observation[0]
                execution_passes[index]=processed_observation[1]
            message.append(
                {
                    "role": "user",
                    "content": observation_list[0]
                }
            )
            for i ,index in enumerate(indices):
                mask=[ len(responses[i]) + len('```output'), len(responses_w_res[i]) ]
                samples_info[index]['mask_info'].append(mask)
                samples_info[index]['response']+=responses_w_res[i]
                samples_info[index]['sequence']+=responses_w_res[i]
                samples_info[index]['stop']=not is_execution[i]
                samples_info[index]['finish_reason']=finish_reason[i]
                samples_info[index]['execution_pass']=execution_passes[i]

            # print("samples_info", samples_info, "Line 305")
        
        for i, line in enumerate(samples_info):
            if samples_info[i]['finish_reason']!='length': samples_info[i]['response']+=self.tokenizer.eos_token
        
        responses_ids=[]
        tool_output_masks=[]
        execution_passes=[]
        for idx, sample_info in enumerate(samples_info):
            # print("sample_info", sample_info, "Line 309")
            response_id, tool_output_mask = self._tokenize_and_find_mask_token_indices(sample_info)
            responses_ids.append(response_id[:self.config.response_length])
            tool_output_masks.append(tool_output_mask[:self.config.response_length])
            execution_passes.append(sample_info['execution_pass'])
        return message
        
        # return responses_ids, tool_output_masks, torch.tensor(execution_passes, dtype=torch.long)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                # Handle both dict and object-style sampling_params
                if isinstance(self.sampling_params, dict):
                    if key in self.sampling_params:
                        old_value = self.sampling_params[key]
                        old_sampling_params_args[key] = old_value
                        self.sampling_params[key] = value
                else:
                    if hasattr(self.sampling_params, key):
                        old_value = getattr(self.sampling_params, key)
                        old_sampling_params_args[key] = old_value
                        setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            if isinstance(self.sampling_params, dict):
                self.sampling_params[key] = value
            else:
                setattr(self.sampling_params, key, value)

    @torch.no_grad()
    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # print(prompts)
        # assert False
        idx = prompts.batch['input_ids']  # (bs, prompt_length)

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # response, tool_output_masks, execution_passes = await self._tir_generate(
            message = await self._tir_generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False)
        
        # convert to a message
        # print(f"message: {message} in line 606")
        return message # DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def send_request(self, tool_input):
        try:
            url = self.config.torl.sandbox_url # Dacheng: added this
            response = requests.post(url, json=tool_input, timeout=10)
            return response.json()  # 返回响应的 JSON 数据
        except:
            print("sanbox timeout")

    # Dacheng: This function can only take batched inputs, so have to put it here.
    def code_interpreter_batch_call(self, tool_inputs, timeout=20):
        global total_code_execution_time
        start_time = time.time()
        
        tool_inputs = [{'code': tool_input, 'language': 'python'} for tool_input in tool_inputs]
        results = []
        
        for tool_input in tool_inputs:
            try:
                result = self.send_request(tool_input)
                results.append(result)
            except:
                results.append({"run_result": {"stdout": "Error", "stderr": "TimeoutError"}})
        
        def postproc(output):
            try:
                if str(output['run_result']['return_code']) == '0' or len(str(output['run_result']['stdout'])) != 0:
                    return output['run_result']['stdout'], "Done"
                else:
                    return output['run_result']['stdout'], output['run_result']['stderr'].strip()
            except Exception:
                return "Error", "UnknownError"
        
        results = [postproc(result) for result in results]
        
        # Update total execution time
        execution_time = time.time() - start_time
        total_code_execution_time += execution_time
        
        # Log the current total time spent in code execution
        logger.info(f"Total time spent in code execution: {total_code_execution_time:.2f} seconds")
        
        return results
    
class ToRLActAgentGroup:
    """
    A class that manages multiple ToRLActAgent instances to generate trajectories in parallel.
    """
    
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine: Any,
        max_prompt_length: int,
        max_response_length: int,
        max_starting_message_length: int,
        max_parallel_agents: int,
        max_iterations: int,
        tokenizer: Any,
        sampling_params: Any,
        device: Any,
        config: Any,
    ) -> None:
        """
        Initialize the ToRLActAgentGroup to manage multiple agent instances.
        
        Args:
            batch: DataProto containing the batch of data
            num_trajectories: Number of trajectories to generate per instance
            infer_engine: The infer engine for generation
            max_prompt_length: Maximum prompt length
            max_response_length: Maximum response length
            max_starting_message_length: Maximum starting message length
            max_parallel_agents: Maximum number of agents to run in parallel
            max_iterations: Maximum number of iterations per agent
            tokenizer: Tokenizer to use for text encoding/decoding
            sampling_params: Sampling parameters for generation
            device: Device to use for computation
        """
        self.batch = batch
        self.infer_engine = infer_engine
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_starting_message_length = max_starting_message_length
        self.max_parallel_agents = max_parallel_agents
        self.max_iterations = max_iterations
        self.num_trajectories = num_trajectories
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.device = device
        self.config = config
        # Calculate total length
        self.total_len = max_prompt_length + max_response_length
        # self.tool_batch_size =
        
        # Map of instance ID to agent instance
        self.agents = {}
        
        # Map of instance ID to agent results
        self.results = []
    
    def _convert_results_to_dataproto(self) -> DataProto:
        """
        Convert the results dictionary to a single DataProto by concatenating all individual results.
        
        Args:
            results: Dictionary mapping batch_idx -> trajectory_id -> result_data
            
        Returns:
            DataProto containing all concatenated results
        """
        # Get batch of messages
        #print(f"Reached line 677")
        all_messages = []
        all_prompts = []
        all_responses = []
        eval_score = []
        for result in self.results:
            # messages = result.get('messages', [])
            messages = result['messages']
            all_messages.append(messages)
            answer_str = ""
            # get the response: starting from the first assistant message
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    starting_index = i
                    break
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    answer_str += msg["content"]
            # print(f"answer_str: {answer_str} in line 695")
            if starting_index == 0:
                # If we don't find an assistant, all messages are prompts and there are no responses
                # print(f'ERROR: Found no assistant message. len(messages) == {len(messages)} and roles {[msg["role"] for msg in messages]}')
                starting_index = len(messages)
            prompt = messages[:starting_index]
            all_prompts.append(prompt)
            response = messages[starting_index:]
            all_responses.append(response)
            eval_score.append(compute_score(answer_str, result['ground_truth']))
        # Encode messages, get assitant mask and position ids
        prompt_encodings = self.tokenizer.apply_chat_template(
            all_prompts, 
            # return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        prompt_input_ids = torch.tensor(prompt_encodings['input_ids'], device=self.device)
        prompt_attention_mask = torch.tensor(prompt_encodings['attention_mask'], device=self.device)
        prompt_input_ids, prompt_attention_mask = convert_right_padding_to_left(self.tokenizer, prompt_input_ids, prompt_attention_mask, self.device, self.max_starting_message_length)

        response_encodings = self.tokenizer.apply_chat_template(
            all_responses,
            chat_template=chat_template,
            # return_tensors="pt",
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        
        response_ids, response_attention_mask, response_assistant_mask = pad_to_max_length_right(
            self.tokenizer, response_encodings, self.total_len, self.device)
            
        
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create tensor dictionary
        logger.info(f"input_ids shape: {input_ids.shape}, response_ids shape: {response_ids.shape}, max_starting_message_length: {self.max_starting_message_length}, max_response_length: {self.total_len}")
        assert input_ids.shape[1] == attention_mask.shape[1] == position_ids.shape[1], f"input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}, position_ids shape {position_ids.shape} do not match"
        assert response_ids.shape[1] == response_assistant_mask.shape[1], f"response_ids shape {response_ids.shape}, response_assistant_mask shape {response_assistant_mask.shape} do not match"
        tensor_dict = {
            'input_ids': input_ids,
            'responses': response_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': response_assistant_mask,
        }

        # Create non-tensor dictionary
        non_tensor_dict = {
            'eval_score': eval_score
        }
        
        # Create and return DataProto
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        # print(f"self.results: {self.results} in line 557")
        logger.info(f"result_dataproto......")
        return result_dataproto
    
    async def generate_trajectories(self) -> DataProto:
        """
        Generate trajectories using OnlineToRLAgent agents.
        Uses producer-consumer pattern like codeact.py.
        """
        total_instances = len(self.batch)
        logger.info(f"Total instances: {total_instances}")
        python_executor = PythonExecutor()
        
        # Initialize results tracking like codeact.py
        self.results = []
        
        # Create asyncio queue for running agents with maxsize
        # run_queue = asyncio.Queue(maxsize=self.max_parallel_agents)
        run_queue = asyncio.Queue()
        
        # Track active tasks
        active_run_tasks = set()
        needed_run_tasks = self.num_trajectories * total_instances

        # Initialize and enqueue agents (producer)
        for trajectory_id in range(self.num_trajectories):
            for batch_idx in range(total_instances):
                import time
                start_time = time.time()
                # Create OnlineToRLAgent
                agent = OnlineToRLAgent(
                    infer_engine=self.infer_engine,
                    tokenizer=self.tokenizer,
                    sampling_params=self.sampling_params,
                    config=self.config,
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                    python_executor=python_executor,
                )
                if batch_idx not in self.agents:
                    self.agents[batch_idx] = {}
                self.agents[batch_idx][trajectory_id] = agent
                elpased_time = time.time() - start_time
                # logger.info(f"time taken to create agent: {elpased_time} seconds")
                
                agent_info = {
                    'trajectory_id': trajectory_id,
                    'batch_idx': batch_idx,
                    # "tool_function": self.code_interpreter_batch_call,
                }
                # logger.info(f"adding to run queue: {agent_info}")
                
                # Add to run queue (this will block if queue is full, providing backpressure)
                await run_queue.put(agent_info)
        
        # Helper function to run one agent (following codeact.py pattern)
        async def run_one_agent():
            # nonlocal needed_run_tasks
            # while needed_run_tasks > 0:
            logger.info(f"in run_one_agent.......")
            agent_info = await run_queue.get()
                
            batch_idx = agent_info['batch_idx']
            trajectory_id = agent_info['trajectory_id']
            agent = self.agents[batch_idx][trajectory_id]
            
            logger.info(f"Running agent for batch {batch_idx}, trajectory {trajectory_id}")
            
            # Get the specific batch item for this agent using DataProto's indexing
            agent_batch = self.batch[batch_idx:batch_idx+1]
            # print(f"agent_batch: {agent_batch}")
            try:
                # Call the agent's generate_sequences method
                messages = await agent.generate_sequences(agent_batch)
                
                # print(f"{self.batch.non_tensor_batch} in line 795")
                # print(f"{self.batch.non_tensor_batch['reward_model']} in line 796")
                # Store the result
                self.results.append({
                    'batch_idx': batch_idx,
                    'trajectory_id': trajectory_id,
                    'messages': messages,
                    "ground_truth": self.batch.non_tensor_batch['reward_model'][batch_idx]["ground_truth"]
                })
                
                logger.info(f"Successfully completed batch {batch_idx}, trajectory {trajectory_id}")
                
            except Exception as e:
                logger.error(f"Error running agent for batch {batch_idx}, trajectory {trajectory_id}: {str(e)}")
                # Store error result
                self.results.append({
                    'batch_idx': batch_idx,
                    'trajectory_id': trajectory_id,
                    'messages': [],
                    "ground_truth": self.batch.non_tensor_batch['reward_model'][batch_idx]["ground_truth"]
                })
            finally:
                run_queue.task_done()
                # Start another run task if available
                nonlocal needed_run_tasks
                if needed_run_tasks > 0:
                    needed_run_tasks -= 1
                    task = asyncio.create_task(run_one_agent())
                    active_run_tasks.add(task)
                    task.add_done_callback(lambda t: active_run_tasks.discard(t))
        
        # Start initial batch of run tasks (they'll wait on the run_queue)
        for _ in range(self.max_parallel_agents):
            needed_run_tasks -= 1
            task = asyncio.create_task(run_one_agent())
            active_run_tasks.add(task)
            task.add_done_callback(lambda t: active_run_tasks.discard(t))
        
        # Wait for all run tasks to complete
        await run_queue.join()
        
        logger.info(f"Generated trajectories for {total_instances} instances")
        
        # Print total code execution time at the end
        global total_code_execution_time
        logger.info(f"=== Total time spent in code execution across all trajectories: {total_code_execution_time:.2f} seconds ===")
        
        results = self._convert_results_to_dataproto()
        logger.info(f"Finish convert results to dataproto")
        return results

    def run(self) -> DataProto:
        """
        Run the agent group synchronously by creating a new event loop if necessary.
        
        Returns:
            Dict mapping instance ID to a dict of trajectory ID to results
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the generate_trajectories coroutine in the event loop
        
        return loop.run_until_complete(self.generate_trajectories())