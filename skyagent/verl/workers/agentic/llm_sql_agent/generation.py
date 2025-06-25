import torch
import re
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from .utils import execute_sqls_parallel
import pandas as pd 
from time import perf_counter
import logging

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    db_path: str 
    no_think_rl: bool = False
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        generator,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.generator = generator 
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at sql operation or solution operation."""
        orig_device = responses.device
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</sql>')[0] + '</sql>'
                 if '</sql>' in resp 
                 else resp.split('</solution>')[0] + '</solution>'
                 if '</solution>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<solution>{envs[idx].ACTION_LOOKUP[action]}</solution>" for idx, action in enumerate(actions)]

        responses = self._batch_tokenize(responses_str).to(orig_device)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids'].type(torch.int64)

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding 
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.generator.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.generator.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.generator.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        # print(f"[DEBUGGING] Padded active batch size: {padded_active_batch.batch['input_ids'].shape}")
        padded_output = self.generator.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, db_files, initial_input_ids: torch.Tensor) -> DataProto:
        """Run main LLM generation loop."""   
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_db_files = db_files.copy()
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            logging.info(f"run_llm_loop::STEP]: Step {step}, gen_batch size: {gen_batch.batch['input_ids'].shape[0]}")
            start_step = perf_counter()
            
            active_db_files = [db_file for i, db_file in enumerate(active_db_files) if active_mask[i]]
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            start = perf_counter()
            gen_output = self._generate_with_gpu_padding(rollings_active)
            end = perf_counter()
            logging.info(f"run_llm_loop::generate_with_gpu_padding]: vLLM generation in {end - start:.2f} seconds")
            
            start = perf_counter()
            meta_info = gen_output.meta_info            
            logging.info(f"device after gen -> {gen_output.batch['responses'].device}")
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids = responses_ids.to(rollings_active.batch["input_ids"].device)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            end = perf_counter()
            logging.info(f"run_llm_loop::_postprocess_responses]: Execution in {end - start:.2f} seconds")

            # Execute in environment and process observations
            # NOTE(shu): execute predictions here, where to truncate only first response? ^ postprogess?
            start = perf_counter()
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, db_files, step, self.tokenizer.pad_token, active_mask
            )
            end = perf_counter()
            
            logging.info(f"run_llm_loop::execute_predictions]: Execution in {end - start:.2f} seconds")
        
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            start = perf_counter()
            next_obs_ids = self._process_next_obs(next_obs).to(rollings_active.batch["input_ids"].device)
            end = perf_counter()
            logging.info(f"run_llm_loop::process_next_obs]: Execution in {end - start:.2f} seconds")
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            end = perf_counter()
            logging.info(f"run_llm_loop::STEP]: STEP finishes in {end - start_step:.2f} seconds")
            
        # final LLM rollout
        if active_mask.sum():
            active_db_files = [db_file for i, db_file in enumerate(active_db_files) if active_mask[i]]
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, db_files, -1, self.tokenizer.pad_token, active_mask, do_sql=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], db_files: List[str], step: int, pad_token: str, active_mask=None, do_sql=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_sql = [], [], [], []
        
        sql_queries = [content for action, content in zip(cur_actions, contents) if action == 'sql']
        db_files_to_execute = [db_file for action, db_file in zip(cur_actions, db_files) if action == 'sql']
        if step == -1:
            reminder_text = ""
        else:
            reminder_text = f'<reminder>You have {self.config.max_turns-step} turns left to complete the task.</reminder>'
        
        if do_sql:
            sql_results = self.batch_execution(sql_queries, db_files_to_execute)
            assert len(sql_results) == sum([1 for action in cur_actions if action == 'sql'])
            
        else:
            print(f"[DEBUG-Agent] Skip SQL execution")
            sql_results = [''] * sum([1 for action in cur_actions if action == 'sql'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_sql.append(0)
            else:
                if action == 'solution':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_sql.append(0)
                elif action == 'sql':
                    res = sql_results.pop(0)
                    
                    if isinstance(res, frozenset):
                        # Make this a pandas DF v.s. list above 
                        df = pd.DataFrame(res)
                        
                        # NOTE(shu): to make a fast training....
                        # df = df.head(5)
                        res_str = df.to_string(index=False)
                    else:
                        res_str = str(res)

                    append_obs_str = f'\n\n<observation>{res_str}\n{reminder_text}</observation>\n\n'
                    
                    # NOTE: observation too long, just truncate 
                    if len(append_obs_str) > 9000:
                        # print(f"[DEBUG-WARNING] OBSERVATION TOO LONG BEFORE TOKENIZATION — LEN = {len(append_obs_str)} chars, EST TOKENS ~ {len(append_obs_str)//4}")
                        
                        # just truncate
                        truncated_df = df.head(50)
                        res_str = truncated_df.to_string(index=False)  # or index=True if you want row numbers
                        
                        append_obs_str = f'\n\n<observation>Truncated to 50 lines since returned response too long: {res_str}\n{reminder_text}</observation>\n\n'
                        
                    next_obs.append(append_obs_str)
                    dones.append(0)
                    valid_action.append(1)
                    is_sql.append(1)
                else:
                    next_obs.append(f'\n<observation>Your previous action is invalid. \
                    Follow the format of outputing thinking process and sql tool, and try again.\n{reminder_text}</observation>\n\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_sql.append(0)
            
        assert len(sql_results) == 0            
        return next_obs, dones, valid_action, is_sql

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(sql|solution)>(.*?)</\1>'

                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_execution(self, queries: List[str] = None, db_files: List[str] = None) -> str:
        """
        Batchified execution for SQL queries.
        Args:
            queries: queries to execute in intermediate steps 
        Returns:
            execution results which is concatenated into a string
        """
        assert(len(queries) == len(db_files)), f"Number of queries ({len(queries)}) and db_files ({len(db_files)}) must be the same"
        results = self._batch_execution(queries, db_files)
        
        execution_res_list = [execution_res 
                      for (_, _, _, execution_res, _) in results]
            
        assert len(execution_res_list) == len(queries)
        return execution_res_list

    def _batch_execution(self, queries, db_files):
        return execute_sqls_parallel(
            db_files,
            queries,
        )