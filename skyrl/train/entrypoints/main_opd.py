"""On-policy distillation (OPD) entrypoint.

The student samples its own rollouts; a frozen teacher served by an inference engine scores every
response token; the negative per-token reverse KL to the teacher,
``log pi_teacher - log pi_student``, becomes a dense advantage. Pure distillation is the default
(the environment reward is only logged); ``trainer.algorithm.opd.use_task_reward=true`` adds the
teacher term on top of the reward's advantages instead.

On top of ``main_base``: ``OPDTrainer.generate`` runs one ``generator.generate`` call per prompt
group concurrently and scores each group under the teacher as soon as its rollouts are back, so
teacher latency overlaps with the rest of the batch's generation and any generator works; the
trainer's other overrides consume the resulting ``GeneratorOutput["teacher_logprobs"]``; and this
experiment class builds the teacher client. Nothing verifies the teacher before training yet; see
the TODO in ``_setup_trainer``. See ``skyrl.train.opd``.

Usage (Fireworks teacher):

    export FIREWORKS_API_KEY=...
    uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \\
        trainer.policy.model.path=Qwen/Qwen3-4B-Base \\
        trainer.teacher.model=accounts/<account>/deployments/<teacher-deployment> \\
        data.train_data="['$HOME/data/dapo/dapo-math-17k-cleaned.parquet']" \\
        environment.env_class=aime ...

Run scripts: ``examples/train/on_policy_distillation/``.
"""

import os
import sys

import ray

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.opd.config import OPDExpConfig, validate_opd_cfg
from skyrl.train.opd.teacher_client import (
    FireworksTeacherClient,
    TeacherLogprobClient,
    VLLMTeacherClient,
)
from skyrl.train.opd.trainer import OPDTrainer
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg


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
        if teacher.backend == "vllm":
            return VLLMTeacherClient(
                teacher.model,
                server_urls=list(teacher.server_urls),
                max_concurrency=teacher.max_concurrency,
                request_timeout_s=teacher.request_timeout_s,
                max_retries=teacher.max_retries,
            )
        raise ValueError(f"unknown trainer.teacher.backend {teacher.backend!r}")

    def get_trainer(self, *args, **kwargs) -> RayPPOTrainer:
        return OPDTrainer(*args, teacher_client=self._teacher_client, **kwargs)

    def _setup_trainer(self) -> RayPPOTrainer:
        self._teacher_client = self.get_teacher_client()
        # TODO (kyuds): preflight checks on the teacher before any model is loaded. Nothing verifies the
        # teacher today, so a wrong setup surfaces minutes into the run, or never: a teacher with a
        # different vocabulary accepts the student's token ids, echoes them and scores them
        # deterministically. What other frameworks do (surveyed 2026-09-22):
        #   - NeMo-RL (nemo_rl/algorithms/distillation.py, check_vocab_equality): loads the teacher
        #     tokenizer and asserts get_vocab(), len() and config.vocab_size equal the student's;
        #     skippable with an env var. The only one of these with a tokenizer check.
        #   - verl (verl/workers/config/distillation.py, validate_and_prepare_for_distillation): the
        #     teacher's max_model_len must cover prompt_length + response_length + 1; teacher config
        #     completeness (model_path, key, num_replicas, no duplicate keys).
        #   - prime-rl (orchestrator/clients.py, wait_for_ready / maybe_check_has_model): polls /health
        #     on every teacher server until ready, then requires the configured model in /v1/models.
        #   - Miles (utils/arguments.py, rollout/on_policy_distillation.py): argument validation only
        #     (teacher URL syntax, duplicates, a default entry, checkpoint path exists); no probe.
        #   - tinker-cookbook: nothing; both sides share the student's tokenizer by construction.
        # Candidates here, to be decided: NeMo-RL's vocabulary equality (needs the teacher's HF
        # tokenizer path; Fireworks model ids are not HF paths), prime-rl's /v1/models listing plus
        # verl's context bound for vLLM teachers (max_input_length + max_generate_length + 1), and a
        # reproducibility probe that scores one sequence n times and refuses replica-dependent teachers
        # (serverless Fireworks replicas disagreed by ~0.25 nats mean / 2.7 nats max on identical
        # requests, more than the 0.01-0.09 nat distillation signal).
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
