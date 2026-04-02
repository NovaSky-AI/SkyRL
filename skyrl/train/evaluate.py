from typing import Dict

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.evaluator import StandaloneEvaluator
from skyrl.train.generators.base import GeneratorInterface


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Compatibility wrapper for standalone evaluation."""
    evaluator = StandaloneEvaluator(cfg=cfg, generator=generator, tokenizer=tokenizer)
    return await evaluator.evaluate(eval_dataloader=eval_dataloader, global_step=global_step)


@torch.no_grad()
async def evaluate_step_wise(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Compatibility wrapper for standalone step-wise evaluation."""
    evaluator = StandaloneEvaluator(cfg=cfg, generator=generator, tokenizer=tokenizer)
    return await evaluator.evaluate_step_wise(eval_dataloader=eval_dataloader, global_step=global_step)
