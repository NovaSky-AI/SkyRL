"""On-policy distillation (OPD) entrypoint.

Usage (teacher launched by the job on its own GPUs):

    uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \\
        trainer.policy.model.path=Qwen/Qwen3-4B-Base \\
        trainer.teacher.model=Qwen/Qwen3-32B \\
        trainer.teacher.inference_engine.num_engines=1 \\
        trainer.teacher.inference_engine.tensor_parallel_size=4 \\
        data.train_data="['$HOME/data/dapo/dapo-math-17k-cleaned.parquet']" \\
        environment.env_class=aime ...

Several teachers (MOPD) replace ``trainer.teacher`` with ``trainer.teachers``, each named after the route label of
its rows: ``trainer.teachers.math.model=...`` and ``trainer.teachers.code.model=...``, with each row's
``skyrl_mopd_domain`` column naming its teacher.

For more usecases and example scripts, look at
``examples/train/on_policy_distillation/``.
"""

import asyncio
import sys
from typing import Dict, Optional

import ray

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.opd.config import (
    OPDExpConfig,
    TeacherConfig,
    teacher_logprob_width,
    teacher_path,
    teacher_topk,
    validate_opd_cfg,
)
from skyrl.train.opd.teacher_client import (
    SkyRLTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
)
from skyrl.train.opd.teacher_launch import launch_teacher, teacher_start_ports
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg


class OPDExp(BasePPOExp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._teacher_clients: Optional[Dict[str, TeacherLogprobClient]] = None

    def get_teacher_client(self, name: str, teacher: TeacherConfig, start_port: Optional[int]) -> TeacherLogprobClient:
        """Build one teacher's client; a launched one starts at ``start_port``."""
        if teacher.backend == "skyrl":
            client, server_setup = launch_teacher(self.cfg, teacher, start_port)
            return SkyRLTeacherClient(client, server_setup=server_setup, max_concurrency=teacher.max_concurrency)
        if teacher.backend == "vllm":
            return VLLMTeacherClient(
                teacher.model,
                server_url=teacher.server_url,
                max_concurrency=teacher.max_concurrency,
                request_timeout_s=teacher.request_timeout_s,
                max_retries=teacher.max_retries,
            )
        raise ValueError(f"unknown {teacher_path(name)}.backend {teacher.backend!r}")

    def get_teacher_clients(self) -> Dict[str, TeacherLogprobClient]:
        """Every teacher's client, by its ``trainer.teachers`` name, in config order. Launched teachers take
        consecutive port windows."""
        ports = teacher_start_ports(self.cfg)
        return {
            name: self.get_teacher_client(name, teacher, ports.get(name))
            for name, teacher in self.cfg.trainer.teachers.items()
        }

    def get_trainer(self, *args, **kwargs) -> RayPPOTrainer:
        return OPDTrainer(*args, teacher_clients=self._teacher_clients, **kwargs)

    def _setup_trainer(self) -> RayPPOTrainer:
        self._teacher_clients = self.get_teacher_clients()
        teacher_clients = self._teacher_clients

        async def preflight() -> None:
            # TODO (kyuds): consider checking each teacher's tokenizer against the student's here.
            # The probe currently does not catch a teacher with a different vocabulary.
            try:
                results = await asyncio.gather(
                    *(
                        client.preflight(teacher_logprob_width(self.cfg), teacher_topk(self.cfg))
                        for client in teacher_clients.values()
                    ),
                    return_exceptions=True,
                )
                failures = {
                    name: result for name, result in zip(teacher_clients, results) if isinstance(result, BaseException)
                }
                if failures:
                    raise RuntimeError(
                        "; ".join(f"{teacher_path(name)}: {error}" for name, error in failures.items())
                    ) from next(iter(failures.values()))
            finally:
                # Their sessions are bound to this short-lived loop.
                await asyncio.gather(*(client.aclose() for client in teacher_clients.values()))

        asyncio.run(preflight())
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
