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

import statistics
from typing import List
import re

from ray.util import scheduling_strategies
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import ray
from ray.util import ActorPool

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# From PRIME team
async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:100]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_info=None,
                                       num_processes=64):
    # TODO: use spawn in ray application code: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html  
    ctx = multiprocessing.get_context("spawn") 
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes, mp_context=ctx) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows
        """
        results = []
        for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info):
            print(f"Using eval func: {evaluation_func}")
            cur_score = evaluation_func(task, completion, reference, task_extra_info)
            results.append([cur_score])
            print(f"Get score: {cur_score}")
        """
            # results.append(single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)) 
        # """
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise
        # """

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        # print(f"Looping result: {result}")
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class SWEBenchRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, compute_score=None) -> None:
        self.data_source = "SWE-Gym/SWE-Gym"

        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.verifier_func = compute_score or _default_compute_score
        self.config = config
            
    
    def verify(self, data):
        resolved = data.non_tensor_batch['resolved']
        error = data.non_tensor_batch['error']
        print(error)
        has_finish_action = data.non_tensor_batch['finish']
        print("has_finish_action:", has_finish_action)
        score = [0. for _ in range(len(resolved))]
        for i, r in enumerate(resolved):
            if r:
                score[i] = 1.0

        print("scores:", score)
        reward_metrics = {}
        reward_metrics['max_turn_ratio'] = sum("RuntimeError: Agent reached maximum iteration in headless mode" in e for e in error if e) / len(error)
        reward_metrics['finish_action_ratio'] = sum(has_finish_action) / len(has_finish_action)
        reward_metrics['stuck_ratio'] = sum("stuck in a loop" in e for e in error if e) / len(error)

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        for ability in list(set(data.non_tensor_batch['ability'])):
            score_ = [data.batch['acc'][i].item() for i in range(len(data.batch['acc'])) if
                      data.non_tensor_batch['ability'][i] == ability]
            reward_metrics[f'{ability}'] = statistics.mean(score_)
        reward_metrics['all'] = data.batch['acc'].mean().item()
        
        return score, reward_metrics
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        verifier_reward=torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        response_ids = data.batch['responses']
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, -response_length:].sum(-1)
        
        # if the batch already contains evaluation results, the verification is skipped here.
        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            # verifier_score, verifier_metrics = self.verify(data)
            # Use ray based concurrency
            verifier_score, verifier_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i, valid_response_length[i] - 1] = verifier_score[i]

        reward_tensor_dict['gt_scores'] = verifier_reward
        
        if 'rm_scores' in data.batch.keys():
            reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
            reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
            if self.config.reward_model.rm_coef!=0:
                reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if self.config.verifier.reward_coef!=0:
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics

