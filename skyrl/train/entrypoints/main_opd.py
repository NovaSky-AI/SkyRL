"""On-policy distillation (OPD) entrypoint.

Usage (teacher launched by the job on its own GPUs):

    uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \\
        trainer.policy.model.path=Qwen/Qwen3-4B-Base \\
        trainer.teacher.model=Qwen/Qwen3-32B \\
        trainer.teacher.inference_engine.num_engines=1 \\
        trainer.teacher.inference_engine.tensor_parallel_size=4 \\
        data.train_data="['$HOME/data/dapo/dapo-math-17k-cleaned.parquet']" \\
        environment.env_class=aime ...

For more usecases and example scripts, look at
``examples/train/on_policy_distillation/``.
"""

import sys

import ray

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.opd.config import OPDExpConfig, validate_opd_cfg
from skyrl.train.opd.teacher_client import (
    SkyRLTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
)
from skyrl.train.opd.teacher_launch import launch_teacher
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg


class OPDExp(BasePPOExp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._teacher_client: TeacherLogprobClient = None

    def get_teacher_client(self) -> TeacherLogprobClient:
        """Build the teacher client from ``trainer.teacher``."""
        teacher = self.cfg.trainer.teacher
        if teacher.backend == "skyrl":
            client, server_setup = launch_teacher(self.cfg)
            return SkyRLTeacherClient(client, server_setup=server_setup, max_concurrency=teacher.max_concurrency)
        if teacher.backend == "vllm":
            return VLLMTeacherClient(
                teacher.model,
                server_url=teacher.server_url,
                max_concurrency=teacher.max_concurrency,
                request_timeout_s=teacher.request_timeout_s,
                max_retries=teacher.max_retries,
            )
        raise ValueError(f"unknown trainer.teacher.backend {teacher.backend!r}")

    def get_trainer(self, *args, **kwargs) -> RayPPOTrainer:
        return OPDTrainer(*args, teacher_client=self._teacher_client, **kwargs)

    def _setup_trainer(self) -> RayPPOTrainer:
        self._teacher_client = self.get_teacher_client()
        # TODO (kyuds): It might be good to consider whether we want to have
        # preflight checks on the teacher before any model is loaded.
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
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
