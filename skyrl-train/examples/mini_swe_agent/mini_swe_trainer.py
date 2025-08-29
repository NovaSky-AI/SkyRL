
import asyncio
import math
import os
import shutil
from typing import Any, List, Optional, Dict, Tuple, Union
from jaxtyping import Float
from pathlib import Path
import ray
import uuid
import torch
from loguru import logger
from omegaconf import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl_train.dataset import PromptDataset
from skyrl_train.utils.tracking import Tracking
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorOutput,
    GeneratorInterface,
)
from skyrl_train.generators.utils import concatenate_generator_outputs, get_metrics_from_generator_output
from skyrl_train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
)
from skyrl_train.utils import ppo_utils
from skyrl_train.utils import trainer_utils
from skyrl_train.utils import Timer, get_ray_pg_ready_with_timeout
from skyrl_train.utils.ppo_utils import (
    compute_approx_kl,
    masked_mean,
    get_kl_controller,
    FixedKLController,
    AdaptiveKLController,
    normalize_advantages_dict,
)
from skyrl_train.distributed.dispatch import MeshRank, concatenate_outputs_after_mesh_dispatch, ActorInfo
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.weights_manager import InferenceWeightsManager, ConditionalWeightsManager
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.trainer_utils import (
    cleanup_old_checkpoints,
    run_on_each_node,
    get_node_ids,
    extract_step_from_path,
    validate_consistency_for_latest_checkpoint,
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output,
    GLOBAL_STEP_PREFIX,
    ResumeMode,
    DynamicSamplingState,
)
from skyrl_train.trainer import RayPPOTrainer

class MiniSWEPPOTrainer(RayPPOTrainer):
    @torch.no_grad()
    async def generate(
        self,
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate rollouts.

        If colocate_all is enabled:
        - before calling this method, the policy model should be on CPU and inference engine should
            be awake (i.e. on GPU).
        - after calling this method, the same model placement still holds.
        """
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        # validate_generator_output(input_batch, generator_output)

        return generator_output