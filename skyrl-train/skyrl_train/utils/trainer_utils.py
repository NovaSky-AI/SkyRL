from typing import List, Dict, Any, Union, Callable, Optional
from enum import Enum
import ray
from skyrl_train.workers.worker import PPORayActorGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import os
import shutil
from loguru import logger
import glob
import json
from skyrl_train.generators.utils import get_metrics_from_generator_output
from skyrl_train.generators.base import GeneratorOutput
from transformers import AutoTokenizer
from pathlib import Path

BasicType = Union[int, float, str, bool, type(None)]

GLOBAL_STEP_PREFIX = "global_step_"


class ResumeMode(Enum):
    NONE = "none"
    LATEST = "latest"
    FROM_PATH = "from_path"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        return super()._missing_(value)


def get_node_ids(
    policy_model: PPORayActorGroup, critic_model: Optional[PPORayActorGroup], ref_model: Optional[PPORayActorGroup]
) -> List[str]:
    """Get the node ids of the policy, critic, and ref models.

    Args:
        policy_model: Policy model actor group
        critic_model: Critic model actor group (Optional)
        ref_model: Ref model actor group (Optional)
    """
    policy_node_ids: List[str] = ray.get(policy_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    if critic_model is not None:
        critic_node_ids: List[str] = ray.get(critic_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    else:
        critic_node_ids = []
    if ref_model is not None:
        ref_node_ids: List[str] = ray.get(ref_model.async_run_ray_method("pass_through", "get_ray_node_id"))
    else:
        ref_node_ids = []

    unique_node_ids = list(set(policy_node_ids + critic_node_ids + ref_node_ids))
    return unique_node_ids


def run_on_each_node(node_ids: List[str], fn: Callable, *args, **kwargs):
    """Simple helper to run a function on each node.

    Args:
        node_ids: List of node ids to run the function on
        fn: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    node_ids = list(set(node_ids))
    task = ray.remote(num_cpus=0.25)(fn)
    refs = []

    for node_id in node_ids:
        node_task = task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        )
        refs.append(node_task.remote(*args, **kwargs))

    return ray.get(refs)


def extract_step_from_path(path: str) -> int:
    basename = os.path.basename(path)
    if basename.startswith(GLOBAL_STEP_PREFIX):
        return int(basename.split(GLOBAL_STEP_PREFIX)[1])
    return -1


def cleanup_old_checkpoints(ckpt_path: str, max_ckpts_to_keep: int, current_global_step: int):
    """Remove old global_step directories, keeping only the most recent max_ckpts_to_keep"""

    if max_ckpts_to_keep < 0:
        return

    # Find all global_step directories
    pattern = os.path.join(ckpt_path, f"{GLOBAL_STEP_PREFIX}*")
    checkpoint_dirs = glob.glob(pattern)

    # track only valid checkpoints - id <= current_global_step
    checkpoint_dirs = [dir for dir in checkpoint_dirs if extract_step_from_path(dir) <= current_global_step]

    if len(checkpoint_dirs) <= max_ckpts_to_keep:
        logger.info(f"Only {len(checkpoint_dirs)} checkpoints found for the current run, no need to cleanup")
        return

    checkpoint_dirs.sort(key=extract_step_from_path, reverse=True)

    logger.info(
        f"Found {len(checkpoint_dirs)} checkpoints for the current run, keeping only the most recent {max_ckpts_to_keep}"
    )

    # Remove old checkpoints
    for old_dir in checkpoint_dirs[max_ckpts_to_keep:]:
        try:
            shutil.rmtree(old_dir)
            logger.info(f"Removed old checkpoint: {old_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")


def validate_consistency_for_latest_checkpoint(
    root_ckpt_folder: str, ckpt_iteration: int, checkpoint_path: str, latest_checkpoint_file: str, save_interval: int
):
    """Validate that the checkpoint folder is consistent with the latest checkpoint file.

    Asserts that the folder with the highest global step is the latest checkpoint tracked by `latest_checkpoint_file`.
    Otherwise, the folder state is inconsistent and the user should delete other checkpoints.
    """
    global_step_values = [
        extract_step_from_path(p) for p in os.listdir(root_ckpt_folder) if p.startswith(GLOBAL_STEP_PREFIX)
    ]
    max_global_step_in_folder = max(global_step_values)
    # NOTE (sumanthrh): We allow a checkpoint folder to be `save_interval` steps ahead of the latest checkpoint in `latest_checkpoint_file`. This is because the last checkpoint can be an incomplete checkpoint.
    if max_global_step_in_folder - ckpt_iteration > save_interval:
        max_global_step_in_folder_path = os.path.join(
            root_ckpt_folder, f"{GLOBAL_STEP_PREFIX}{max_global_step_in_folder}"
        )
        raise ValueError(
            f"Inconsistent checkpoint folder. Latest checkpoint file {latest_checkpoint_file} points to {ckpt_iteration}, but the folder has checkpoints with higher global step - Found global steps {max_global_step_in_folder_path}. This is likely because checkpoint {max_global_step_in_folder_path} was created in a previous run while the latest run is at {checkpoint_path}. Please delete/move checkpoints from older runs and try again."
        )


def sanitize_data_source(data_source: str) -> str:
    """Sanitize data source name for use in file paths."""
    if data_source is None:
        return "unknown"
    return data_source.replace("/", "_")


def calculate_per_dataset_metrics(
    concat_generator_outputs: GeneratorOutput,
    concat_uids: List[str],
    concat_data_sources: List[str],
    n_samples_per_prompt: int,
) -> Dict[str, float]:
    """Calculate metrics per data source."""
    eval_metrics = {}

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Calculate metrics for each data source
    for data_source, indices in data_source_indices.items():
        # Extract subset for this data source
        subset_generator_output = {key: [value[i] for i in indices] for key, value in concat_generator_outputs.items()}
        subset_uids = [concat_uids[i] for i in indices]

        # Calculate metrics for this subset
        avg_score, pass_at_n = get_metrics_from_generator_output(subset_generator_output, subset_uids)

        # Add to eval metrics with proper naming
        sanitized_data_source = sanitize_data_source(data_source)
        eval_metrics[f"eval/{sanitized_data_source}/avg_score"] = avg_score
        eval_metrics[f"eval/{sanitized_data_source}/pass_at_{n_samples_per_prompt}"] = pass_at_n

    return eval_metrics


def dump_per_dataset_eval_results(
    dump_dir_path: Path,
    tokenizer: AutoTokenizer,
    concat_generator_outputs: GeneratorOutput,
    concat_data_sources: List[str],
    concat_all_envs: List[str],
    concat_env_extras: List[Dict[str, Any]],
    eval_metrics: Dict[str, float],
):
    """Dump evaluation results per dataset and overall aggregated results."""

    # Prepare common data
    input_prompts = [tokenizer.decode(prompt) for prompt in concat_generator_outputs["prompt_token_ids"]]
    output_responses = [tokenizer.decode(response) for response in concat_generator_outputs["response_ids"]]

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Dump per-dataset files
    for data_source, indices in data_source_indices.items():
        sanitized_data_source = sanitize_data_source(data_source)
        filename = dump_dir_path / f"{sanitized_data_source}.jsonl"

        with open(filename, "w") as f:
            for i in indices:
                entry = {
                    "input_prompt": input_prompts[i],
                    "output_response": output_responses[i],
                    "score": concat_generator_outputs["rewards"][i],
                    "stop_reason": concat_generator_outputs.get("stop_reasons", [None] * len(input_prompts))[i],
                    "env_class": concat_all_envs[i],
                    "env_extras": concat_env_extras[i],
                    "data_source": data_source,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped eval data for {data_source} to {filename}")

    # Dump aggregated results file
    aggregated_filename = dump_dir_path / "aggregated_results.jsonl"
    with open(aggregated_filename, "w") as f:
        f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    print(f"Dumped aggregated eval metrics to {aggregated_filename}")
