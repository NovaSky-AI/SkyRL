import multiprocessing as mp
from typing import Optional
from pathlib import Path

from datasets.utils.py_utils import multiprocess 
from .utils import execute_sql_wrapper
from time import perf_counter
import re
import random 
import logging
import json 

THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"

def verify_format_and_extract(output: str):
    if output.count(SOLUTION_START) != 1 or output.count(SOLUTION_END) != 1:
        return False, None, None, None

    pre_solution, tail = output.split(SOLUTION_START, 1)
    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None, None, None

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return False, None, None, None

    for m in re.finditer(r"</observation>", pre_solution, re.I):
        rest = pre_solution[m.end():].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None, None

    return True, thoughts, solution_text.strip(), None

    
def calculate_reward_parallel(db_files, completions, references, questions, num_cpus=32, timeout=30, n_agent: Optional[int] = None, log_dir: Optional[str] = None):
    """
    Calculate rewards in parallel for SynSQL.
    
    Args:
        db_files: List of database files to execute the SQL queries.
        completions: List of model outputs containing the SQL solutions.
        references: List of ground truth SQL queries.
        num_cpus: Number of CPU cores to use for parallel processing.
        timeout: Timeout for each SQL execution.

    Returns:
        List of rewards for each completion.
    """
    if log_dir:
        assert n_agent is not None, "n_agent must be provided for logging"
    start = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: calculating {len(completions)} rewards", flush=True)

    # serial filter for format reward
    rewards = [0.0] * len(completions)
    num_comparisons = 0
    to_execute = []
    
    # serially filter for format reward
    for i, output in enumerate(completions):
        is_valid, _, pred_sql, _ = verify_format_and_extract(output)
        if not is_valid:
            rewards[i] = -1.0
        else:
            num_comparisons += 1
            to_execute.append((i, db_files[i], pred_sql, timeout, output))
            to_execute.append((i, db_files[i], references[i], timeout, output))
    
    if len(to_execute) == 0:
        print(f"[DEBUG]: All format wrong, completions: {completions}")
        
    # parallely execute for correctness reward
    exec_start = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: executing {len(to_execute)} SQL statements in parallel", flush=True)
    # NOTE (sumanthrh): We use mp context instead of the global context to avoid changing the default start method
    # this can affect dataloading code since PyTorch uses fork by default.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_cpus) as pool:
        results = pool.starmap(execute_sql_wrapper, to_execute)
    exec_end = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: executed {len(to_execute)} SQL statements in {exec_end - exec_start:.2f} seconds", flush=True)

    # evaluate the results
    # NOTE(shu): for printing purpose 
    correct_examples = []
    wrong_examples = [] 
    for i in range(num_comparisons):
        idx, _, p_sql, pred_results, _, pred_completion = results[i * 2]
        _, _, g_sql, gt_results, _, _ = results[i * 2 + 1]
        
        if pred_results is not None and gt_results is not None and pred_results == gt_results:
            rewards[idx] = 1.0
            correct_examples.append((idx, p_sql, g_sql, pred_completion))
            # print(f"[DEBUG-SHU-EXECUTE]: CORRECT, SQL is {p_sql}", flush=True)
        else:
            rewards[idx] = 0.0
            wrong_examples.append((idx, p_sql, g_sql, pred_completion))
            # print(f"[DEBUG-SHU-EXECUTE]: WRONG, SQL is {p_sql}, \nGOLD SQL is {g_sql}", flush=True)
    
    # log to directory
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(completions)):
            traj_file = log_dir / f"traj_{index}.json"
            traj_data = {"completion": completions[index], "db_file": db_files[index], "reference": references[index], "reward": rewards[index], "question": questions[index]}
            with open(traj_file, "w") as f:
                json.dump(traj_data, f, default=lambda x: str(x))

    end = perf_counter()
    print(f"synsql_utils::calculate_reward_parallel: calculated {len(completions)} rewards in {end - start:.2f} seconds", flush=True)
    return rewards