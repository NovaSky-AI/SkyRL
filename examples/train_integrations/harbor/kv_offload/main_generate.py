"""Run a fixed Harbor task subset and save per-engine KV cache metrics."""

import asyncio
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import aiohttp
import ray
import yaml

from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborSkyRLConfig,
    _deep_merge,
)
from examples.train_integrations.harbor.entrypoints.main_harbor_generate import HarborGenerateExp
from skyrl.train.generators.base import GeneratorInput, TrajectoryID
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


class KVOffloadGenerateExp(HarborGenerateExp):
    """Evaluate identical task IDs with configurable concurrency and cache capacity."""

    def run(self):
        client = self.get_inference_client()
        asyncio.run(client.wake_up())
        generator = self.get_generator(self.cfg, self.tokenizer, client)
        count = int(os.environ.get("SKYRL_HARBOR_NUM_SAMPLES", "500"))
        tasks = sorted(self.train_dataset, key=lambda item: item["prompt"])
        if not 1 <= count <= len(tasks):
            raise ValueError(f"Sample count must be between 1 and {len(tasks)}, got {count}.")
        tasks = random.Random(42).sample(tasks, count)
        artifact_dir = Path(os.environ["SKYRL_HARBOR_ARTIFACT_DIR"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "tasks.json").write_text(json.dumps([t["prompt"] for t in tasks], indent=2))
        (artifact_dir / "config.json").write_text(
            json.dumps(
                {
                    "inference_engine": dataclasses.asdict(self.cfg.generator.inference_engine),
                    "rate_limit": dataclasses.asdict(self.cfg.generator.rate_limit),
                    "versions": {
                        package: importlib.metadata.version(package)
                        for package in ("skyrl", "vllm", "ray", "torch", "harbor")
                    },
                },
                indent=2,
            )
        )
        inputs = GeneratorInput(
            prompts=[t["prompt"] for t in tasks],
            trajectory_ids=[TrajectoryID(instance_id=t["uid"], repetition_id=0) for t in tasks],
            env_classes=None,
            env_extras=None,
            sampling_params=None,
        )
        asyncio.run(self._generate(generator, inputs, client.server_urls, artifact_dir))

    async def _generate(self, generator, inputs, server_urls, artifact_dir):
        """Record raw metrics through the final logger flush after generation."""
        stop = asyncio.Event()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:

            async def scrape():
                snapshots = []
                for index, url in enumerate(server_urls):
                    async with session.get(f"{url}/metrics") as response:
                        response.raise_for_status()
                        body = await response.text()
                    (artifact_dir / f"engine-{index}-latest.prom").write_text(body)
                    snapshots.append({"url": url, "metrics": body})
                with (artifact_dir / "metrics.jsonl").open("a") as output:
                    output.write(json.dumps({"time": time.time(), "engines": snapshots}) + "\n")

            async def monitor():
                while not stop.is_set():
                    await scrape()
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        pass

            await scrape()
            monitoring = asyncio.create_task(monitor())
            start = time.time()
            try:
                result = await generator.generate(inputs)
                end = time.time()
                metrics = result.get("rollout_metrics") or {}
                valid = not metrics.get("generate/num_masked_instances", 0)
                (artifact_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "start": start,
                            "end": end,
                            "elapsed_seconds": end - start,
                            "num_tasks": len(inputs["prompts"]),
                            "server_urls": server_urls,
                            "rollout_metrics": metrics,
                            "valid": valid,
                        },
                        indent=2,
                    )
                )
                await asyncio.sleep(10)
                await scrape()
                if not valid:
                    raise RuntimeError("Evaluation contains masked tasks; inspect the trial errors.")
            finally:
                stop.set()
                await monitoring


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    KVOffloadGenerateExp(cfg).run()


def main():
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as source:
        cfg.harbor_trial_config = _deep_merge(yaml.safe_load(source), cfg.harbor_trial_config)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
