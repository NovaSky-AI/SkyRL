from typing import Optional
from .synsql import calculate_reward_parallel
from time import perf_counter

def sql_compute_score(completions, references, db_files, tasks, questions, n_agent: Optional[int] = None, log_dir: Optional[str] = None):
    """

    Args:
        completions: completions of entire LLM responses 
        references: ground-truth SQL queries
        db_files: database files to execute the SQL queries
        tasks: default to be 'synsql', do not accept anything else 

    Returns:
        List of scores for each completion [default to be just one, no batch]
    """
    assert len(completions) == len(references) == len(db_files) == len(tasks), "Length of completions, references, db_files, and tasks must match"
    assert all(task == "synsql" for task in tasks), "Only 'synsql' task is supported"
    print(f"Compute Score for SynSQL: {len(completions)} queries")
    try:
        start = perf_counter()
        res = calculate_reward_parallel(db_files, completions, references, questions, num_cpus=64, n_agent=n_agent, log_dir=log_dir)
        end = perf_counter()
        print(f"Total processing time: {end - start:.2f} seconds")
        return res
    except Exception as e:
        print(f"Unexpected error: {e}; Setting reward as 0")
        return [0 for _ in range(len(completions))]