"""On-policy distillation (OPD) entrypoint.

The student samples its own rollouts; a frozen teacher served by an inference engine scores every
response token; the per-token reverse KL to the teacher, ``log pi_student - log pi_teacher``,
becomes a dense advantage. Pure distillation is the default (the environment reward is only
logged); ``trainer.algorithm.opd.use_task_reward=true`` adds the teacher term on top of the
reward's advantages instead.

On top of ``main_base``: ``OPDTrainer.generate`` runs one ``generator.generate`` call per prompt
group concurrently and scores each group under the teacher as soon as its rollouts are back, so
teacher latency overlaps with the rest of the batch's generation and any generator works; the
trainer's other overrides consume the resulting ``GeneratorOutput["teacher_logprobs"]``; and this
experiment class builds the teacher client and runs a determinism self-test against it before any
model is loaded. See ``skyrl.train.opd``.

Usage (Fireworks teacher):

    export FIREWORKS_API_KEY=...
    uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \\
        trainer.policy.model.path=Qwen/Qwen3-4B-Base \\
        trainer.teacher.model=accounts/<account>/deployments/<teacher-deployment> \\
        data.train_data="['$HOME/data/dapo/dapo-math-17k-cleaned.parquet']" \\
        environment.env_class=aime ...

Run scripts: ``examples/train/on_policy_distillation/``.
"""

import asyncio
import os
import sys
from typing import List, Tuple

import ray
from loguru import logger

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.opd.config import OPDExpConfig, validate_opd_cfg
from skyrl.train.opd.teacher_client import FireworksTeacherClient, TeacherLogprobClient
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg

# Text the teacher self-test scores; any text works, it only has to be reproducible.
SELF_TEST_PROMPT = "The quick brown fox jumps over the lazy dog. Then it"
SELF_TEST_RESPONSE = " went home, curled up in its den and slept until the sun came up the next morning."


class OPDExp(BasePPOExp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._teacher_client: TeacherLogprobClient = None

    def get_teacher_client(self) -> TeacherLogprobClient:
        """Build the teacher client from ``trainer.teacher``. Override for other backends."""
        teacher = self.cfg.trainer.teacher
        if teacher.backend == "fireworks":
            return FireworksTeacherClient(
                teacher.model,
                api_key=os.environ[teacher.api_key_var],
                base_url=teacher.base_url,
                max_concurrency=teacher.max_concurrency,
                request_timeout_s=teacher.request_timeout_s,
                max_retries=teacher.max_retries,
            )
        raise ValueError(f"unknown trainer.teacher.backend {teacher.backend!r}")

    def get_trainer(self, *args, **kwargs) -> RayPPOTrainer:
        return OPDTrainer(*args, teacher_client=self._teacher_client, **kwargs)

    def _self_test_sequence(self) -> Tuple[List[int], List[int]]:
        prompt_ids = self.tokenizer.encode(SELF_TEST_PROMPT)
        response_ids = self.tokenizer.encode(SELF_TEST_RESPONSE, add_special_tokens=False)
        return list(prompt_ids), list(response_ids)

    def run_teacher_self_test(self) -> None:
        """Test to see if the teacher must answer, and answer reproducibly."""
        opd = self.cfg.trainer.algorithm.opd
        prompt_ids, response_ids = self._self_test_sequence()

        async def _run() -> float:
            try:
                return await self._teacher_client.self_test(
                    prompt_ids, response_ids, n=opd.self_test_samples, max_abs_diff=opd.self_test_max_abs_diff
                )
            finally:
                # Sessions are bound to this temporary loop; the training loop opens its own.
                await self._teacher_client.aclose()

        worst = asyncio.run(_run())
        logger.info(
            f"Teacher self-test passed: {opd.self_test_samples} identical requests agree within {worst:.4f} nats "
            f"(limit {opd.self_test_max_abs_diff})"
        )

    def _setup_trainer(self) -> RayPPOTrainer:
        self._teacher_client = self.get_teacher_client()
        self.run_teacher_self_test()
        return super()._setup_trainer()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = OPDExp(cfg)
    exp.run()


def main() -> None:
    cfg = OPDExpConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    validate_opd_cfg(cfg)

    initialize_ray(cfg)
    # A local Ray cluster inherits the driver's environment; a `ray job submit` cluster does not.
    # Forward the teacher API key to the task explicitly so both work.
    teacher = cfg.trainer.teacher
    env_vars = {}
    if teacher.backend == "fireworks" and os.environ.get(teacher.api_key_var):
        env_vars[teacher.api_key_var] = os.environ[teacher.api_key_var]
    entrypoint = skyrl_entrypoint.options(runtime_env={"env_vars": env_vars}) if env_vars else skyrl_entrypoint
    ray.get(entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
