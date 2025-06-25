import json
import asyncio
import uuid
from collections import deque
from pathlib import Path
from typing import Any, List, Dict, Optional, Set, Callable, Tuple
import os
import pandas as pd
import tempfile
import time

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F
from transformers import AutoTokenizer

import openhands
import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State, AgentState
from openhands.core.config import LLMConfig, AgentConfig, SandboxConfig, AppConfig
from openhands.core.main import create_runtime, run_controller
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.core.message_utils import (
    events_to_messages,
)
from openhands.core.exceptions import (
    AgentStuckInLoopError,
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)
from openhands.events.event import EventSource
from openhands.memory.condenser import Condenser
from openhands.core.config.condenser_config import (
    NoOpCondenserConfig,
)
from openhands.llm.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.llm.llm import LLM
from openhands.utils.prompt import PromptManager
from openhands.utils.async_utils import call_sync_from_async, call_async_from_sync

from swegym.harness.test_spec import make_test_spec
from swegym.harness.run_evaluation import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
)
from swegym.harness.grading import get_eval_report
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from .utils import process_git_patch

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

# this is for the tokenizer.apply_chat_template to be able to generate assistant masks directly
# todo: this is a hack, we should find a better way to do this
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

# chat template for qwen3 thinking mode to remove think tokens similar to generation phase
chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

    
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


def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'If you think you have solved the task, please first send your answer to user through message and then finish the interaction.\n'
        f'{encaps_str}'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()

# Helper function for sandbox config
def get_default_sandbox_config_for_eval():
    return SandboxConfig(
        use_host_network=False,
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
    )

class OnlineCodeActAgent(Agent):
    """
    An online implementation of CodeActAgent that leverages infer's asynchronous capabilities
    for a single agent instance.
    """
    
    def __init__(
        self,
        instance_id: int,
        trajectory_id: int,
        max_prompt_length: int = 1024,
        infer_engine=None,
        tokenizer=None,
        sampling_params=None,
        qwen3_enable_thinking: bool = True,
    ) -> None:
        """
        Initialize a single OnlineCodeActAgent instance.
        """
        # dummy value to let openhands tracks the name
        llm = LLM(LLMConfig(model="dummy"))

        super().__init__(llm, AgentConfig())
        
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.reset()
        self.step_count = 0
        self.infer_engine = infer_engine
        self.sampling_params = sampling_params
        
        # Store instance and trajectory IDs separately
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id
        
        # Initialize tools
        self.tools = codeact_function_calling.get_tools(
            codeact_enable_browsing=False,
            codeact_enable_jupyter=False,
            codeact_enable_llm_editor=False,
        )
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(
            microagent_dir=os.path.join(
                os.path.dirname(os.path.dirname(openhands.__file__)),
                'microagents',
            ),
            prompt_dir=os.path.join(os.path.dirname(openhands.agenthub.codeact_agent.__file__), 'prompts'),
            disabled_microagents=None,
        )
        
        # Initialize condenser
        self.condenser = Condenser.from_config(NoOpCondenserConfig())
        
        # Initialize state
        self.pending_actions = deque()
        
        # will be set in _initialize_runtime_for_agent
        self.runtime = None
        self.instruction = None
        self.config = None

        self.qwen3_enable_thinking = qwen3_enable_thinking

    def close(self):
        """Close the agent runtime."""
        if self.runtime:
            # remove all threads in event stream
            self.runtime.event_stream.close()
            
            self.runtime.close()


        
    def _initial_messages(self) -> list[Message]:
        """Creates the initial messages (including the system prompt) for the LLM conversation."""
        return [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.prompt_manager.get_system_message(),
                        cache_prompt=False,  # Assuming caching is active
                    )
                ],
            )
        ]
        
    def _enhance_messages(self, messages: list[Message]) -> list[Message]:
        """Enhances the user message with additional context based on keywords matched."""
        results: list[Message] = []
        is_first_message_handled = False

        for msg in messages:
            if msg.role == 'user' and not is_first_message_handled:
                is_first_message_handled = True
                # Compose the first user message with examples
                self.prompt_manager.add_examples_to_initial_message(msg)

                # Add repo/runtime info if enabled
                if self.config.get_agent_config().enable_prompt_extensions:
                    self.prompt_manager.add_info_to_initial_message(msg)

            # Enhance the user message with additional context based on keywords matched
            if msg.role == 'user':
                self.prompt_manager.enhance_message(msg)

            results.append(msg)

        return results
        
    def _get_messages(self, state: State) -> List[Message]:
        """Get the message history for this agent."""
        # Start with initial messages (system prompt)
        messages = self._initial_messages()
        
        # If using a condenser, condense the history
        events = self.condenser.condensed_history(state)
        
        # Convert history events to messages
        messages += events_to_messages(
            events,
            max_message_chars=32768,  # Default value, adjust as needed
            vision_is_active=False,  # Assuming vision is not active
            enable_som_visual_browsing=False,  # Assuming SOM visual browsing is not enabled
        )
        
        messages = self._enhance_messages(messages)
        
        return messages

    
    # Conversion utility function
    def convert_str_to_completion_format(self, fn_call_messages):
        # from types import SimpleNamespace
        from litellm import ModelResponse

        role = fn_call_messages[0]['role']
        response_str = fn_call_messages[0]['content']
        tool_calls = fn_call_messages[0].get('tool_calls', None)
        
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "content": response_str,
                        "role": role,
                        "tool_calls": tool_calls,
                        "function_calling": None
                    }
                }
            ]
        )
    
    async def generate(self, input_ids, sampling_params):
        res = await self.infer_engine.async_generate(input_ids=input_ids, sampling_params=self.sampling_params)
        response_str = res["text"]
        return response_str


    async def step(self, state: State) -> Action:
        """Generate a response using batched infer."""
        self.step_count += 1
        print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, step {self.step_count}")
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        messages = self.llm.format_messages_for_llm(messages)
        messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, self.tools
                )
        
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=self.qwen3_enable_thinking
            )
            if len(input_ids) >= self.max_prompt_length:
                return AgentFinishAction(thought="The context is too long. Exit now.")

            response_str = call_async_from_sync(self.generate, input_ids=input_ids, sampling_params=self.sampling_params)
            
            if not response_str:
                # If we got an empty response (possible error), return a message action
                self.pending_actions.append(
                    MessageAction(
                        content="I encountered an error processing your request. Let's try again.",
                    )
                )
            else:
                # Convert to actions
                message = [
                    {
                        'role': 'assistant',
                        'content': response_str,
                    }
                ]
                fn_call_messages = convert_non_fncall_messages_to_fncall_messages(
                    message, self.tools
                )
                actions = codeact_function_calling.response_to_actions(
                    self.convert_str_to_completion_format(fn_call_messages)
                )
                print(f"Take action: {[type(action) for action in actions]}")
                
                for action in actions:
                    self.pending_actions.append(action)
        
        except (
            LLMMalformedActionError,
            LLMNoActionError,
            LLMResponseError,
            FunctionCallValidationError,
            FunctionCallNotExistsError,
        ):
            raise

        except Exception as e:
            logger.error(f"Error in agent step: {str(e)}")
            # Handle errors gracefully by creating a message action
            self.pending_actions.append(
                MessageAction(
                    content=f"An error: {str(e)} encountered. Please try a different approach.",
                )
            )
        
        # Return the first pending action
        if not self.pending_actions:
            # Fallback in case of empty actions
            return AgentFinishAction()
            
        return self.pending_actions.popleft()
    
    def get_final_messages(self, state: State) -> List[Message]:
        """Get the final messages for this agent."""
        messages = self._get_messages(state)
        messages = self.llm.format_messages_for_llm(messages)
        messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, self.tools
                )
        return messages
    
    def _is_last_action_finish(self, state: State) -> bool:
        if state and state.history:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            if isinstance(last_action, AgentFinishAction):
                return True
        return False
    

Agent.register('OnlineCodeActAgent', OnlineCodeActAgent) 



class CodeActAgentGroup:
    """
    A class that manages multiple CodeActAgent instances to generate trajectories in parallel.
    """
    
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine: Any,
        max_prompt_length: int = 1024,
        max_response_length: int = 1024,
        max_starting_message_length: int = 12000,
        max_parallel_agents: int = 1,
        max_eval_parallel_agents: int = 1,
        max_iterations: int = 10,
        tokenizer: Any = None,
        sampling_params: Any = None,
        device: Any = None,
        log_messages_dir: str = None,
        remove_think_tokens: bool = False,
        qwen3_enable_thinking: bool = True
    ) -> None:
        """
        Initialize the CodeActAgentGroup to manage multiple agent instances.
        
        Args:
            batch: DataProto containing the batch of data
            num_trajectories: Number of trajectories to generate per instance
            infer_engine: The infer engine for generation
            max_prompt_length: Maximum prompt length
            max_parallel_agents: Maximum number of agents to run in parallel
            max_iterations: Maximum number of iterations per agent
            tokenizer: Tokenizer to use for text encoding/decoding
            max_batch_size: Maximum batch size for LLM generation
        """
        self.batch = batch
        self.infer_engine = infer_engine
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.total_len = self.max_prompt_length + self.max_response_length
        # todo: make it a config
        self.max_starting_message_length = max_starting_message_length
        self.max_parallel_agents = max_parallel_agents
        self.max_eval_parallel_agents = max_eval_parallel_agents
        print("max eval parallel agents: ", self.max_eval_parallel_agents)
        if max_eval_parallel_agents <= 0: 
            print(f"`max_eval_parallel_agents` has not been set. Setting it to `max_parallel_agents` i.e {max_parallel_agents}")
            self.max_eval_parallel_agents = max_parallel_agents
        self.max_iterations = max_iterations
        self.num_trajectories = num_trajectories
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.device = device
        
        # Map of instance ID to agent instance
        self.agents = {}
        
        # Map of instance ID to agent results
        self.results = {}
        
        self.qwen3_enable_thinking = qwen3_enable_thinking
        self.log_messages_dir = None
        if log_messages_dir:
            self.log_messages_dir = Path(log_messages_dir)
            logger.info(f"Logging all messages to {self.log_messages_dir}")
        
        # Initialize agents for each instance
        self._initialize_agents()
        
        self.remove_think_tokens = remove_think_tokens
        if self.remove_think_tokens:
            logger.info("Removing think tokens....")
    

    def _convert_results_to_dataproto(self) -> DataProto:
        """
        Convert results to DataProto format for training.
        
        Args:
            results: Dictionary of results, with structure {instance_id: {trajectory_id: result_dict}}
            input_dataproto: The input DataProto that contains the original batch data
            tokenizer: The tokenizer to use for encoding messages
            
        Returns:
            DataProto: A DataProto object with the converted results
        """

        # Non-tensor data
        git_patch_list = []
        success_list = []
        error_list = []
        resolved_list = []
        has_finish_action_list = []
        
        # Create a mapping of instance_id -> list of trajectories
        instance_trajectories = {}
        for instance_id, trajectories in self.results.items():
            instance_trajectories[instance_id] = []
            for trajectory_id, result in trajectories.items():
                instance_trajectories[instance_id].append(result)

        # Create the final results in the same order as the batch
        matched_results = []
        instance_list = []
        for batch_item in self.batch:
            instance_id = batch_item.non_tensor_batch['instance']['instance_id']
            instance = batch_item.non_tensor_batch['instance']
            if instance_id in instance_trajectories:
                # Add all trajectories for this instance
                traj_results = instance_trajectories[instance_id]
                matched_results.extend(traj_results)
                instance_list.extend([instance] * len(traj_results))
        
        assert len(matched_results) == self.num_trajectories * len(self.batch), f"Expected number of results {self.num_trajectories * len(self.batch)}, got {len(matched_results)}"
        
        # Group results by instance_id for message handling
        results_by_instance = {}
        for i, result in enumerate(matched_results):
            instance_id = instance_list[i]['instance_id']
            if instance_id not in results_by_instance:
                results_by_instance[instance_id] = []
            results_by_instance[instance_id].append((i, result))
        
        # Handle empty messages by copying from another trajectory of the same instance
        for instance_id, results in results_by_instance.items():
            # Find a valid messages list to use as fallback
            valid_messages = None
            valid_patch = None
            for _, result in results:
                messages = result.get('messages', [])
                if messages and len(messages) > 0:
                    valid_messages = messages
                    valid_patch = result.get('git_patch', None)
                    valid_resolved = result.get('resolved', False)
                    valid_finish = result.get('finish', False)
                    valid_error = result.get('error', None)
                    break
            
            # If we found valid messages, use them for trajectories with empty messages
            if valid_messages:
                for idx, result in results:
                    if not result.get('messages') or len(result.get('messages', [])) == 0:
                        print(f"Got empty messages for instance_id {instance_id}, trajectory {idx}. Copying messages array from a valid trajectory. ")
                        # Copy messages from the valid trajectory
                        matched_results[idx]['messages'] = valid_messages.copy()
                        matched_results[idx]['git_patch'] = valid_patch
                        matched_results[idx]['resolved'] = valid_resolved
                        matched_results[idx]['error'] = valid_error
                        matched_results[idx]['finish'] = valid_finish
        
        # Get batch of messages
        all_messages = []
        all_prompts = []
        all_responses = []
        for result in matched_results:
            messages = result.get('messages', [])
            all_messages.append(messages)
            # get the response: starting from the first assistant message
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    starting_index = i
                    break
            if starting_index == 0:
                # If we don't find an assistant, all messages are prompts and there are no responses
                print(f'ERROR: Found no assistant message. len(messages) == {len(messages)} and roles are {[msg["role"] for msg in messages]}')
                starting_index = len(messages)
            prompt = messages[:starting_index]
            all_prompts.append(prompt)
            response = messages[starting_index:]
            all_responses.append(response)

            # Also add non-tensor data
            git_patch_list.append(result.get('git_patch', None))
            success_list.append(result.get('success', False))
            error_list.append(result.get('error', None))
            resolved_list.append(result.get('resolved', False))
            has_finish_action_list.append(result.get('finish', False))


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
            chat_template=chat_template_qwen3_thinking if self.remove_think_tokens else chat_template,
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
            'git_patch': git_patch_list,
            'success': success_list,
            'error': error_list,
            'instance': instance_list,
            'resolved': resolved_list,
            'finish': has_finish_action_list,
        }
        
        # Create and return DataProto
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        
        return result_dataproto
        
    def close(self):
        """Clean up resources"""
            
        # Close all agent instances
        for instance_id in self.agents:
            for trajectory_id in self.agents[instance_id]:
                self._cleanup_agent(instance_id, trajectory_id)
    
    def _cleanup_agent(self, instance_id, trajectory_id):
        try:
            self.agents[instance_id][trajectory_id].close()
        except Exception as e:
            logger.warning(f"Error closing agent {instance_id}, trajectory {trajectory_id}: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        self.close()
    
    def _initialize_agents(self) -> None:
        """Initialize agent instances for each task."""
        for data_item in self.batch:
            instance_id = data_item.non_tensor_batch['instance']['instance_id']
            self.agents[instance_id] = {}
            for n in range(self.num_trajectories):
                self.agents[instance_id][n] = OnlineCodeActAgent(
                    instance_id=instance_id,
                    trajectory_id=n,
                    max_prompt_length=self.max_prompt_length,
                    tokenizer=self.tokenizer,
                    infer_engine=self.infer_engine,
                    sampling_params=self.sampling_params,
                    qwen3_enable_thinking=self.qwen3_enable_thinking
                )
                # Set the instance data for each agent
                self.agents[instance_id][n].instance_data = data_item.non_tensor_batch['instance']
                self.agents[instance_id][n].max_iterations = self.max_iterations
    
    async def _initialize_runtime_for_agent(self, batch_id: int, trajectory_id: int) -> None:
        """Initialize the runtime for a specific agent."""
        instance_id = self.batch[batch_id].non_tensor_batch['instance']['instance_id']
        instance = pd.Series(self.batch[batch_id].non_tensor_batch['instance'])
        agent = self.agents[instance_id][trajectory_id]
        
        try:
            # Configure sandbox
            RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
            SWE_BENCH_CONTAINER_IMAGE = 'ghcr.io/opendevin/eval-swe-bench:full-v1.2.1'
            
            if os.environ.get('USE_INSTANCE_IMAGE', 'true').lower() == 'true':
                # Use a different instance image for each instance of swe-bench eval
                base_container_image = get_instance_docker_image(instance_id)
                logger.info(
                    f'Using instance container image: {base_container_image}. '
                    f'Please make sure this image exists. '
                    f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
                )
            else:
                base_container_image = SWE_BENCH_CONTAINER_IMAGE
                logger.info(f'Using swe-bench container image: {base_container_image}')
            
            sandbox_config = get_default_sandbox_config_for_eval()
            sandbox_config.base_container_image = base_container_image
            sandbox_config.enable_auto_lint = True
            sandbox_config.use_host_network = False
            sandbox_config.platform = 'linux/amd64'
            
            app_config = AppConfig(
                default_agent='OnlineCodeActAgent',
                run_as_openhands=False,
                max_iterations=self.max_iterations,
                runtime='remote',
                sandbox=sandbox_config,
                workspace_base=None,
                workspace_mount_path=None,
            )
            agent_config = AgentConfig(
                codeact_enable_jupyter=False,
                codeact_enable_browsing=False,
                codeact_enable_llm_editor=False,
                condenser=NoOpCondenserConfig(),
                enable_prompt_extensions=False,
            )
            app_config.set_agent_config(agent_config)
            agent.config = app_config
            
            # Create runtime
            runtime = create_runtime(app_config)
            
            # Connect runtime
            await runtime.connect()
            
            # Initialize runtime
            from .utils import initialize_runtime, get_instruction
            # initialize_runtime(runtime, instance)
            await call_sync_from_async(initialize_runtime, runtime, instance)
            
            # Store the runtime and instruction
            agent.runtime = runtime
            agent.instruction = get_instruction(instance)
            
            logger.info(f"Successfully initialized runtime for instance {instance_id}")
        except Exception as e:
            logger.error(f"Failed to initialize runtime for instance {instance_id}: {str(e)}")
            if 'runtime' in locals() and runtime:
                runtime.event_stream.close()
                runtime.close()
            
            # Update agent state to reflect error
            agent.error = str(e)
            agent.agent_state = AgentState.ERROR
            raise
    
    async def _run_agent(self, batch_id: int, trajectory_id: int, pos_id: int) -> Dict[str, Any]:
        instance_id = self.batch[batch_id].non_tensor_batch['instance']['instance_id']
        """Run a single agent to completion and return the results."""
        agent = self.agents[instance_id][trajectory_id]
        assert agent is not None
        instance = pd.Series(self.batch[batch_id].non_tensor_batch['instance'])
        runtime = agent.runtime
        
        try:
            # Run the agent controller
            state = await run_controller(
                config=agent.config,
                initial_user_action=MessageAction(content=agent.instruction),
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
            )

            if state:
                print(state.last_error)
            
            from .utils import complete_runtime, is_fatal_evaluation_error
            # Check for fatal errors
            if state and is_fatal_evaluation_error(state.last_error):
                logger.error(f"Fatal error in agent {instance_id}: {state.last_error}")
                raise Exception('Fatal error detected: ' + state.last_error)
            
            final_messages = agent.get_final_messages(state)
            # Complete the runtime and get the git patch
            return_val = await call_sync_from_async(complete_runtime, runtime, instance)
            # return_val = complete_runtime(runtime, instance)
            # print patch
            if return_val.get('git_patch', None):
                print(f"Git patch for instance {instance_id}, traj {trajectory_id}:")
                print('-' * 80)
                print(return_val['git_patch'])
                print('-' * 80)
                
            return_val =  {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'state': state,
                'git_patch': return_val.get('git_patch', None),
                'messages': final_messages,
                'success': not bool(state.last_error if state else True),
                'error': state.last_error if state and state.last_error else None,
                'finish': agent._is_last_action_finish(state)
            }
        except Exception as e:
            logger.error(f"Error running agent {instance_id}: {str(e)}")
            # Update agent state to reflect error
            agent.error = str(e)
            agent.agent_state = AgentState.ERROR
            
            if state:
                final_messages = agent.get_final_messages(state)
            else:
                print(f'No final message state for instance {instance_id}, trajectory {trajectory_id}')
                final_messages = []

            if not final_messages or len(final_messages) == 0:
                print(f'1095: Final messages are non-existent (or empty) for instance {instance_id}, trajectory {trajectory_id}')
            
            return_val =  {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'messages': final_messages,
                'state': state,
                'git_patch': None,
                'success': False,
                'error': str(e),
                'finish': agent._is_last_action_finish(state)
            }
        finally:
            # cleanup agent resources
            self._cleanup_agent(instance_id, trajectory_id)

        return return_val
    
    def _apply_patch_and_evaluate(self, runtime, model_patch, instance_id, trajectory_id, test_spec):
        """Apply patch and evaluate the solution."""
        model_patch = process_git_patch(model_patch)
        # Get patch and save it to /tmp/patch.diff
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch file
            patch_file_path = os.path.join(temp_dir, 'patch.diff')
            with open(patch_file_path, 'w') as f:
                f.write(model_patch)
            runtime.copy_to(patch_file_path, '/tmp')
            # Eval script
            eval_script_path = os.path.join(temp_dir, 'eval.sh')
            with open(eval_script_path, 'w') as f:
                f.write(test_spec.eval_script)
            runtime.copy_to(eval_script_path, '/tmp')

        # Set +x
        action = CmdRunAction(command='chmod +x /tmp/eval.sh')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Apply patch
        exec_command = (
            'cd /testbed && '
            "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
            "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "echo 'APPLY_PATCH_FAIL')))"
        )
        action = CmdRunAction(command=exec_command)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        apply_patch_output = obs.content
        assert isinstance(apply_patch_output, str)
        # instance['test_result']['apply_patch_output'] = apply_patch_output

        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            raise Exception(f"Instance {instance_id}, trajectory {trajectory_id} {APPLY_PATCH_FAIL}:\n{apply_patch_output}")
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}, {trajectory_id}] {APPLY_PATCH_PASS}:\n{apply_patch_output}')

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
            action.set_hard_timeout(300)  # Short timeout just to get the process ID
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(
                    f'[{instance_id}, {trajectory_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1200  # 20 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        raise Exception(
                            f'[{instance_id}, {trajectory_id}] Evaluation timed out after {timeout} seconds'
                        )
                    check_action = CmdRunAction(
                        command=f'ps -p {pid} > /dev/null; echo $?'
                    )
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)
                    if (
                        isinstance(check_obs, CmdOutputObservation)
                        and check_obs.content.split()[-1].strip() == '1'
                    ):
                        logger.info(
                            f'[{instance_id}, {trajectory_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}, {trajectory_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again

                # Read the log file
                cat_action = CmdRunAction(command=f'cat {log_file}')
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                # Grade answer
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    assert isinstance(test_output, str)
                    # instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}, {trajectory_id}] Grading answer...')
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a directory structure that matches the expected format
                        # NOTE: this is a hack to make the eval report format consistent
                        # with the original SWE-Bench eval script
                        log_dir = os.path.join(temp_dir, 'logs', instance_id.lower())
                        os.makedirs(log_dir, exist_ok=True)
                        test_output_path = os.path.join(log_dir, 'test_output.txt')
                        with open(test_output_path, 'w') as f:
                            f.write(test_output)
                        try:
                            _report = get_eval_report(
                                test_spec=test_spec,
                                prediction={
                                    'model_patch': model_patch,
                                    'instance_id': instance_id,
                                },
                                log_path=test_output_path,
                                include_tests_status=True,
                            )
                            report = _report[instance_id]
                            logger.info(
                                f"[{instance_id}, {trajectory_id}] report: {report}\nResult for [{instance_id}, {trajectory_id}]: resolved: {report['resolved']}"
                            )
                            self.results[instance_id][trajectory_id]['resolved'] = report[
                                'resolved'
                            ]
                        except Exception as e:
                            logger.error(
                                f'[{instance_id}, {trajectory_id}] Error when getting eval report: {e}'
                            )
                            self.results[instance_id][trajectory_id]['resolved'] = False
            else:
                raise Exception(f'[{instance_id}, {trajectory_id}] Error when starting eval:\n{obs.content}')
        else:
            raise Exception(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )
    
    async def _evaluate_agent(self, batch_id: int, trajectory_id: int) -> None:
        """Initialize the runtime for a specific agent."""
        instance_id = self.batch[batch_id].non_tensor_batch['instance']['instance_id']
        instance = pd.Series(self.batch[batch_id].non_tensor_batch['instance'])
        test_spec = make_test_spec(instance=instance)
        
        try:
            # Configure sandbox
            # We use a different instance image for the each instance of swe-bench eval
            base_container_image = get_instance_docker_image(instance_id)
            logger.info(
                f'Using instance container image: {base_container_image}. '
                f'Please make sure this image exists. '
                f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
            )
            sandbox_config = get_default_sandbox_config_for_eval()
            sandbox_config.base_container_image = base_container_image
            sandbox_config.remote_runtime_resource_factor = 1
            config = AppConfig(
                run_as_openhands=False,
                runtime="remote",
                sandbox=sandbox_config,
                # do not mount workspace
                workspace_base=None,
                workspace_mount_path=None,
            )
            
            # Create runtime
            runtime = create_runtime(config)
            
            # Connect runtime
            await runtime.connect()

            assert instance_id in self.results and trajectory_id in self.results[instance_id], \
            f"Instance {instance_id} or trajectory {trajectory_id} not found in results"
            
            model_patch = self.results[instance_id][trajectory_id].get('git_patch', None)
            if not model_patch:
                raise Exception(f"No git patch found for instance {instance_id}, trajectory {trajectory_id}")
            
            
            await call_sync_from_async(self._apply_patch_and_evaluate, runtime, model_patch, instance_id, trajectory_id, test_spec)
                
        except Exception as e:
            logger.error(f"Failed to evaluate traj {trajectory_id} for instance {instance_id}: {str(e)}")
            self.results[instance_id][trajectory_id]['resolved'] = False
            self.results[instance_id][trajectory_id]['eval_error'] = str(e)
        finally:
            if 'runtime' in locals() and runtime:
                runtime.event_stream.close()
                runtime.close()

        if self.log_messages_dir:
            result = self.results[instance_id][trajectory_id]
            instance_dir  = self.log_messages_dir / str(instance_id) 
            instance_dir.mkdir(exist_ok=True, parents=True)
            with open(instance_dir / f"{trajectory_id}.json", "w") as f: 
                result_json = json.dumps(result, default=lambda x: str(x))
                f.write(result_json)
    
    async def generate_trajectories_pipeline(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Generate trajectories with pipelined runtime initialization to improve efficiency.
        """
        total_instances = len(self.batch)
        print("Total instances:", total_instances)
        
        # Create two queues: one for initialization and one for running
        init_queue = asyncio.Queue()
        run_queue = asyncio.Queue(maxsize=self.max_parallel_agents)
        eval_queue = asyncio.Queue(maxsize=self.max_parallel_agents)
        
        # Fill the initialization queue
        for trajectory_id in range(self.num_trajectories):
            for batch_idx in range(total_instances):
                await init_queue.put((batch_idx, trajectory_id))
        
        # Track active tasks
        active_init_tasks = set()
        active_run_tasks = set()
        active_eval_tasks = set()
        need_init_tasks = self.num_trajectories * total_instances
        needed_run_tasks = self.num_trajectories * total_instances  # Total tasks we'll eventually need   
        needed_eval_tasks = self.num_trajectories * total_instances
        
        # Helper function to initialize runtime
        import time
        async def initialize_one_runtime():
            start_time = time.time()
            batch_idx, trajectory_id = await init_queue.get()
            instance_id = self.batch[batch_idx].non_tensor_batch['instance']['instance_id']
            try:
                logger.info(f"Initializing runtime for instance {instance_id}, trajectory {trajectory_id}")
                await self._initialize_runtime_for_agent(batch_idx, trajectory_id)
                # Add to run queue after successful initialization
                await run_queue.put((batch_idx, trajectory_id))
                elpased_time = time.time() - start_time
                print(f"Successfully initialized runtime for instance {instance_id}, trajectory {trajectory_id} in {elpased_time:.2f} seconds")
            except Exception as e:
                nonlocal needed_run_tasks
                nonlocal needed_eval_tasks
                needed_run_tasks -= 1
                needed_eval_tasks -= 1
                logger.error(f"Error initializing runtime for {instance_id}, trajectory {trajectory_id}: {str(e)}")
                # Handle initialization error
                if instance_id not in self.results:
                    self.results[instance_id] = {}
                self.results[instance_id][trajectory_id] = {
                    'instance_id': instance_id,
                    'trajectory_id': trajectory_id,
                    'messages': [],
                    'state': None,
                    'git_patch': None,
                    'success': False,
                    'error': str(e),
                    'finish': False,
                    'resolved': False
                }
            finally:
                init_queue.task_done()
                # Start another initialization task if available
                nonlocal need_init_tasks
                if not init_queue.empty() and need_init_tasks > 0:
                    need_init_tasks -= 1
                    task = asyncio.create_task(initialize_one_runtime())
                    active_init_tasks.add(task)
                    task.add_done_callback(lambda t: active_init_tasks.discard(t))
        
        # Helper function to run one agent
        async def run_one_agent(pos_id: int):
            batch_idx, trajectory_id = await run_queue.get()
            instance_id = self.batch[batch_idx].non_tensor_batch['instance']['instance_id']
            start_time = time.time()
            try:
                logger.info(f"Running agent for instance {instance_id}, trajectory {trajectory_id}")
                result = await self._run_agent(batch_idx, trajectory_id, pos_id)
                elapsed = time.time() - start_time
                
                # Store the result
                if instance_id not in self.results:
                    self.results[instance_id] = {}
                self.results[instance_id][trajectory_id] = result

                await eval_queue.put((batch_idx, trajectory_id))
                
                print(f"Successfully completed instance {instance_id}, trajectory {trajectory_id} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"[This line should not be reached!!] Error running agent for {instance_id}, trajectory {trajectory_id}: {str(e)}")
                nonlocal needed_eval_tasks
                needed_eval_tasks -= 1
                # Store error result
                if instance_id not in self.results:
                    self.results[instance_id] = {}
                self.results[instance_id][trajectory_id] = {
                    'instance_id': instance_id,
                    'trajectory_id': trajectory_id,
                    'messages': [],
                    'state': None,
                    'git_patch': None,
                    'success': False,
                    'error': str(e),
                    'finish': False,
                    'resolved': False
                }
            finally:
                run_queue.task_done()
                nonlocal needed_run_tasks
                # Start another run task if available
                if needed_run_tasks > 0:
                    needed_run_tasks -= 1
                    task = asyncio.create_task(run_one_agent(pos_id))
                    active_run_tasks.add(task)
                    task.add_done_callback(lambda t: active_run_tasks.discard(t))
        
        # Helper function to eval one trajectory
        async def eval_one_agent():
            batch_idx, trajectory_id = await eval_queue.get()
            instance_id = self.batch[batch_idx].non_tensor_batch['instance']['instance_id']
            start_time = time.time()
            try:
                logger.info(f"Evaluating agent for instance {instance_id}, trajectory {trajectory_id}")
                await self._evaluate_agent(batch_idx, trajectory_id)
                elapsed = time.time() - start_time
                
                print(f"Successfully completed evaluating instance {instance_id}, trajectory {trajectory_id} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Error evaluating agent for {instance_id}, trajectory {trajectory_id}: {str(e)}")
                # Store error result
                self.results[instance_id][trajectory_id]['resolved'] = False
            finally:
                eval_queue.task_done()
                nonlocal needed_eval_tasks
                # Start another run task if available
                if needed_eval_tasks > 0:
                    needed_eval_tasks -= 1
                    task = asyncio.create_task(eval_one_agent())
                    active_eval_tasks.add(task)
                    task.add_done_callback(lambda t: active_eval_tasks.discard(t))
        
        # Start initial batch of initialization tasks
        max_parallel_init = self.max_parallel_agents  # Use some parallel initialization tasks
        for _ in range(min(max_parallel_init, init_queue.qsize())):
            need_init_tasks -= 1
            task = asyncio.create_task(initialize_one_runtime())
            active_init_tasks.add(task)
            task.add_done_callback(lambda t: active_init_tasks.discard(t))
        
        # Start a few agent run tasks (they'll wait on the run_queue)
        for pos_id in range(self.max_parallel_agents):
            needed_run_tasks -= 1
            task = asyncio.create_task(run_one_agent(pos_id))
            active_run_tasks.add(task)
            task.add_done_callback(lambda t: active_run_tasks.discard(t))
        
        for _ in range(self.max_eval_parallel_agents):
            needed_eval_tasks -= 1
            task = asyncio.create_task(eval_one_agent())
            active_eval_tasks.add(task)
            task.add_done_callback(lambda t: active_eval_tasks.discard(t))
        
        # Wait for all initialization tasks to complete
        if init_queue.qsize() > 0:
            await init_queue.join()
        
        # Wait for all run tasks to complete
        if run_queue.qsize() > 0:
            await run_queue.join()
        
        # Wait for all eval tasks to complete
        if eval_queue.qsize() > 0:
            await eval_queue.join()
        
        # Wait for any remaining active tasks
        all_tasks = active_init_tasks.union(active_run_tasks)
        all_tasks = all_tasks.union(active_eval_tasks)
        if all_tasks:
            logger.info(f"Waiting for {len(all_tasks)} (init: {len(active_init_tasks)}, run: {len(active_run_tasks)}, eval: {len(active_eval_tasks)}) remaining tasks to complete")
            await asyncio.wait(all_tasks)
        
        results_dataproto = self._convert_results_to_dataproto()
        return results_dataproto
    
    def run(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
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
        try:
            return loop.run_until_complete(self.generate_trajectories_pipeline())
        finally:
            # Close the batch manager to ensure cleanup
            self.close()
            # loop.close()