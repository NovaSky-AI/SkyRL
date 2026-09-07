import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from skyrl.train.utils.tracking import Tracking

import torch
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
)
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    prepare_generator_input,
)
from skyrl.train.utils import Timer
from skyrl.train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output,
)
from skyrl.train.utils.trajectory_logging import TrajectoryLogger, pretty_print_example

if TYPE_CHECKING:
    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper


@dataclass
class _EvalRows:
    """Per-row bookkeeping, aligned index-for-index with the rows of the concatenated ``GeneratorOutput``.

    Not step-wise: one row per input prompt. Step-wise: one row per step of each trajectory.
    """

    env_classes: List[str] = field(default_factory=list)
    env_extras: List[Dict[str, Any]] = field(default_factory=list)
    uids: List[str] = field(default_factory=list)
    prompts: List[Any] = field(default_factory=list)

    @property
    def data_sources(self) -> List[Optional[str]]:
        return [env_extra.get("data_source") for env_extra in self.env_extras]

    def extend(self, other: "_EvalRows") -> None:
        self.env_classes.extend(other.env_classes)
        self.env_extras.extend(other.env_extras)
        self.uids.extend(other.uids)
        self.prompts.extend(other.prompts)

    def select(self, indices: List[int]) -> "_EvalRows":
        return _EvalRows(
            env_classes=[self.env_classes[i] for i in indices],
            env_extras=[self.env_extras[i] for i in indices],
            uids=[self.uids[i] for i in indices],
            prompts=[self.prompts[i] for i in indices],
        )


def _rows_for_output(
    generator_input: GeneratorInput,
    uids: List[str],
    generator_output: GeneratorOutput,
    step_wise: bool,
) -> _EvalRows:
    """Map each output row back to the input prompt that produced it.

    Not step-wise, the generator returns one row per input prompt in input order, so the input
    columns are the rows. Step-wise, each trajectory expands to one row per step, so rows are
    resolved through ``generator_output["trajectory_ids"]``.
    """
    if not step_wise:
        return _EvalRows(
            env_classes=list(generator_input["env_classes"]),
            env_extras=list(generator_input["env_extras"]),
            uids=list(uids),
            prompts=list(generator_input["prompts"]),
        )
    by_instance = {
        traj_id.instance_id: (env_class, env_extra, prompt)
        for traj_id, env_class, env_extra, prompt in zip(
            generator_input["trajectory_ids"],
            generator_input["env_classes"],
            generator_input["env_extras"],
            generator_input["prompts"],
        )
    }
    rows = _EvalRows()
    for traj_id in generator_output["trajectory_ids"]:
        assert traj_id.instance_id in by_instance, f"Trajectory ID {traj_id.instance_id} not found in input"
        env_class, env_extra, prompt = by_instance[traj_id.instance_id]
        rows.env_classes.append(env_class)
        rows.env_extras.append(env_extra)
        rows.uids.append(traj_id.instance_id)
        rows.prompts.append(prompt)
    return rows


def _scored_view(
    concat_generator_outputs: GeneratorOutput,
    rows: _EvalRows,
    step_wise: bool,
) -> Tuple[Dict[str, list], _EvalRows, Optional[List[int]]]:
    """Select the rows that metrics are computed over: one per trajectory.

    Not step-wise, every row is a trajectory. Step-wise, only the last step of each trajectory
    carries the trajectory's reward, so the view keeps ``is_last_step`` rows only.

    Returns the output view, the matching rows, and the per-trajectory step counts to report as
    ``num_turns`` to the trajectory logger (``None`` lets the logger derive turns from loss masks).
    """
    if not step_wise:
        return concat_generator_outputs, rows, None

    is_last_step = concat_generator_outputs["is_last_step"]
    keep = [i for i, last in enumerate(is_last_step) if last]
    view: Dict[str, list] = {}
    for key, value in concat_generator_outputs.items():
        if isinstance(value, list):
            assert len(value) == len(
                is_last_step
            ), f"Length mismatch: {len(value)} != {len(is_last_step)} for key {key}"
            view[key] = [value[i] for i in keep]

    # TODO (kyuds): this count is likely wrong when `eval_n_samples_per_prompt > 1`. `rows.uids`
    # holds `TrajectoryID.instance_id`, i.e. the *prompt* id, which every repetition of that prompt
    # shares; counting by it merges repetitions, so if repetition 0 of a prompt took 3 steps and
    # repetition 1 took 2, both of their last-step rows report 5 turns. Only the wandb trajectory
    # table's turn column is affected. Keying the count and the lookup on
    # `TrajectoryID.to_string()` (instance + repetition) would fix it; preserved as-is here because
    # this consolidation must not change behaviour.
    step_counts = Counter(rows.uids)  # counted BEFORE the filter, as today
    scored_rows = rows.select(keep)
    return view, scored_rows, [step_counts[uid] for uid in scored_rows.uids]


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    tracker: Optional["Tracking"] = None,
    vllm_metrics_scraper: Optional["VLLMMetricsScraper"] = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Handles both plain and step-wise generation (``cfg.generator.step_wise_trajectories``). Step-wise
    output has one row per step; metrics are computed from the last step of each trajectory, which
    is where the reward is assigned, while the dump keeps every step.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (SkyRLTrainConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        vllm_metrics_scraper: when set, the open ``vllm/eval`` window is resumed
            around each generation and paused after, so only generation time
            counts toward eval throughput.

    Returns:
        Dict[str, float]: evaluation metrics
    """
    step_wise = cfg.generator.step_wise_trajectories

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    rows = _EvalRows()
    sampling_params = cfg.generator.eval_sampling_params
    eval_generate_time = 0.0
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.inference_engine.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        gen_start = time.monotonic()
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.resume()
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.pause()
        eval_generate_time += time.monotonic() - gen_start
        validate_generator_output(len(generator_input["prompts"]), generator_output, step_wise=step_wise)
        generator_outputs.append(generator_output)
        rows.extend(_rows_for_output(generator_input, uids, generator_output, step_wise))
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs, step_wise=step_wise)

    if cfg.trainer.print_example_interval > 0:
        vis = tokenizer.decode(generator_output["response_ids"][0])
        pretty_print_example(
            logger,
            prompt=generator_input["prompts"][0],
            response=vis,
            reward=generator_output["rewards"][0],
        )

    # Metrics score one row per trajectory: every row when not step-wise, the last step of each
    # trajectory when step-wise.
    scored_outputs, scored_rows, num_turns_list = _scored_view(concat_generator_outputs, rows, step_wise)

    # Optionally upload up to `num_logger_eval_samples` samples to tracker (wandb)
    if trajectory_logger is not None:
        with Timer("log_eval_results"):
            trajectory_logger.log(
                tracker=tracker,
                num_samples=cfg.trainer.num_logger_eval_samples,
                prompts=scored_rows.prompts,
                generator_output=scored_outputs,
                tokenizer=tokenizer,
                global_step=global_step,
                num_turns_list=num_turns_list,
                wandb_key="trajectories/eval",
            )

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        scored_outputs, scored_rows.uids, scored_rows.data_sources, cfg.generator.eval_n_samples_per_prompt
    )

    # 3. Calculate overall metrics across all datasets
    overall_metrics = get_metrics_from_generator_output(scored_outputs, scored_rows.uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_metrics["avg_score"],
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
    )

    for key, value in concat_generator_outputs["rollout_metrics"].items():
        eval_metrics[f"eval/all/{key}"] = value

    # 4. Prepare dumping data. The dump keeps every row (every step when step-wise).
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                rows.data_sources,
                rows.env_classes,
                rows.env_extras,
                eval_metrics,
            )

    eval_metrics["timing/eval_generate"] = eval_generate_time
    return eval_metrics
