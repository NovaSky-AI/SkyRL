"""Check Nemotron-120B on 8 trainer + 8 inference GPUs (BF16), without an optimizer.

uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs
uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs --full-ft

Start Ray across both nodes and provide a shared /shared mount for LoRA export.
Pass a config JSON to change placement or the model; the default is nemotron_120b.json.
"""

import argparse
import asyncio
import json
from pathlib import Path

from examples.model_checks.check_logprobs import check_logprobs
from examples.model_checks.megatron_lora import open_runtime
from skyrl.train.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_name("nemotron_120b.json"))
    parser.add_argument("--full-ft", action="store_true")
    args = parser.parse_args()
    cfg = SkyRLTrainConfig.from_cli_overrides(json.loads(args.config.read_text()))
    if args.full_ft:
        cfg.trainer.policy.model.lora.rank = 0
    if cfg.trainer.policy.model.lora.rank > 0:
        assert not cfg.trainer.policy.megatron_config.lora_config.merge_lora
    assert cfg.trainer.strategy == "megatron"
    assert not cfg.trainer.placement.colocate_all
    assert cfg.trainer.algorithm.temperature == 1.0
    cfg.trainer.policy.inference_only_init = True
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    async with open_runtime(cfg, tokenizer) as (policy, client):
        await check_logprobs(policy, client, cfg, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
