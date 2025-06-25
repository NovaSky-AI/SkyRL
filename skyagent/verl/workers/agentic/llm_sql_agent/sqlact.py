from __future__ import annotations
import os
import logging 
logger = logging.getLogger(__file__)
from verl import DataProto
from verl.workers.agentic.llm_sql_agent.generation import LLMGenerationManager, GenerationConfig
import torch 

import os
from typing import Any, List, Dict
import torch.nn.functional as F
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from sglang import Engine as SglangEngine
from sglang.srt.sampling.sampling_params import SamplingParams
from torch.nn.utils.rnn import pad_sequence
from verl.utils.model import compute_position_id_with_mask


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

# NOTE(linjunrong): adhoc
def _post_process_outputs(tokenizer, output):

    def _map_each_response(l):
        # output_token_ids = torch.tensor(l['token_ids'])
        log_probs = []
        output_token_ids = []
        for log_prob, token_ids, _ in l["meta_info"]["output_token_logprobs"]:
            log_probs.append(log_prob)
            output_token_ids.append(token_ids)
        log_probs = torch.tensor(log_probs)
        output_token_ids = torch.tensor(output_token_ids)
        return output_token_ids, log_probs

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs

# Modified from offline Sglang rollout. 
# TODO: Use async_generate
class Generator:
    def __init__(
        self,
        infer_engine: SglangEngine,
        tokenizer: Any,
        sampling_params: Dict[str, Any],
    ):
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        assert "max_new_tokens" in self.sampling_params
        
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # if self.config.free_cache_engine:

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = idx.size(0)
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.tokenizer.pad_token_id, idx[i]))

        print(f"{self.sampling_params=}")
        print(f"DEBUG: Number of inputs: {len(idx_list)}, max size: {max(len(i) for i in idx_list)}")
        output = self.infer_engine.generate(
            prompt=None,  # because we have already convert it to prompt token id
            sampling_params=self.sampling_params,
            return_logprob=True,
            input_ids=idx_list,
        )

        out = _post_process_outputs(self.tokenizer, output)

        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        if response.shape[1] < self.sampling_params["max_new_tokens"]:
            response = pad_sequence_to_length(response, self.sampling_params["max_new_tokens"], self.tokenizer.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.sampling_params["max_new_tokens"], self.tokenizer.pad_token_id)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask.to(idx.device),
                "position_ids": position_ids.to(idx.device),
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch)
    
    
class SQLActAgentGroup:
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine: Any,
        gen_config: GenerationConfig,
        sampling_params: Dict[str, Any], 
        tokenizer: Any = None,
        device: Any = None,
    ) -> None:
        
        # Repeat the batch to be n x n trajectories 
        self.num_trajectories = num_trajectories
        
        batch: DataProto = batch.repeat(repeat_times=self.num_trajectories, interleave=True)
        print(f"DEBUg: batch size: {batch.batch.batch_size}")
        self.batch = batch 
        
        # Initialize other paramsver 
        self.infer_engine = infer_engine
        self.gen_config = gen_config
            
        self.device = device
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        # Initialize generator and generator manager 
        self.generator = Generator(
            infer_engine=infer_engine,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )
        self.generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            generator=self.generator,
            config=gen_config,
            is_validation=True,
        )
        # Accessed DB files
        self.db_files = self._get_db_files(batch, db_path=self.gen_config.db_path)
    
    def _get_db_files(self, data, db_path):
        db_ids = data.non_tensor_batch['db_id']
        data_srcs = data.non_tensor_batch['data_source']
        
        db_files = []
        for db_id, data_src in zip(db_ids, data_srcs):
            if data_src == 'synsql':
                db_files.append(os.path.join(
                    db_path,
                    "SynSQL-2.5M/databases",
                    db_id,
                    db_id + ".sqlite"
                ))
            elif data_src == 'spider':
                db_files.append(os.path.join(
                    db_path, 
                    "spider/database",
                    db_id, 
                    db_id + ".sqlite"
                ))
            elif data_src == 'bird':
                db_files.append(os.path.join(
                    db_path, 
                    "bird/train/train_databases",
                    db_id,
                    db_id + ".sqlite"
                ))
            else:
                raise NotImplementedError
            
        return db_files
    
    def _create_loss_mask(self, batch):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        # response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask
        
        return batch
    
    def _pad_to_max_length_right(self, data: DataProto):
        # assert set(data.batch.keys()) == set(["prompts", "attention_mask", "info_mask", 'responses'])
        padding = (0, self.gen_config.max_prompt_length + self.gen_config.max_response_length - data.batch["input_ids"].shape[1])
        padded_responses = F.pad(data.batch["responses"], padding, mode="constant", value=self.tokenizer.pad_token_id)
        padded_attention_mask = F.pad(data.batch["attention_mask"], padding, mode="constant", value=0)
        padded_info_mask = F.pad(data.batch["info_mask"], padding, mode="constant", value=0)
        
        data.batch["responses"] = padded_responses
        data.batch["attention_mask"] = padded_attention_mask
        data.batch["info_mask"] = padded_info_mask
        return data
        
    def run(self) -> DataProto:
        # Go into the run LLM loop
        first_input_ids = self.batch.batch['input_ids'][:, -self.gen_config.max_start_length:].clone().long()
        final_gen_batch_output = self.generation_manager.run_llm_loop(
                                    gen_batch=self.batch,
                                    db_files=self.db_files,
                                    initial_input_ids=first_input_ids,
                                )
        
        # Do some processing...
        for key in final_gen_batch_output.batch.keys():
            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

        # prepare final DataProto object
        final_gen_batch_output = self._pad_to_max_length_right(final_gen_batch_output)
        # reform input ids
        final_gen_batch_output.batch["input_ids"] = torch.cat([final_gen_batch_output.batch["prompts"], final_gen_batch_output.batch["responses"]], dim=1)

        # Create loss mask 
        final_gen_batch_output = self._create_loss_mask(batch=final_gen_batch_output)
        # Create position ids
        final_gen_batch_output.batch["position_ids"] = compute_position_id_with_mask(final_gen_batch_output.batch["attention_mask"])

        final_output = final_gen_batch_output.pop(batch_keys=["input_ids", "prompts", "responses", "attention_mask", "loss_mask", "position_ids"])
        # prepare for broadcasting to other workers
        final_output.batch = final_output.batch.contiguous()

        # Copy the original non tensor batch over
        final_output.non_tensor_batch = self.batch.non_tensor_batch
        final_output.meta_info = self.batch.meta_info

        # Return DataProto
        return final_output