"""Launching the teacher's vLLM deployment inside the job (``trainer.teacher.backend="skyrl"``).

The deployment follows the rollout inference engine's launch path, ``create_inference_servers``
(server groups plus a router), with a frozen-model argument set from ``build_frozen_vllm_cli_args``.
The teacher model has no weight transfer, no sleep mode, no LoRA. It gets its own placement group
(never the colocate group), a port window past the student's, and is driven through a
``RemoteInferenceClient``.
"""

from argparse import Namespace
from typing import TYPE_CHECKING, Tuple

from skyrl.backends.skyrl_train.inference_servers.common import (
    SERVER_PORT_STRIDE,
    VLLM_START_PORT,
)
from skyrl.train.opd.config import TeacherConfig, teacher_logprob_width, teacher_topk

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.setup import InferenceServerSetup
    from skyrl.train.config import SkyRLTrainConfig


def _teacher_start_port(cfg: "SkyRLTrainConfig") -> int:
    """The first port window past the student's launched servers.

    Each server actor owns ``SERVER_PORT_STRIDE`` ports from its base; the student's servers take
    ``num_engines * data_parallel_size`` windows from ``VLLM_START_PORT``. With an external student
    nothing is launched and the teacher starts there.
    """
    student = cfg.generator.inference_engine
    student_launched = (
        student.run_engines_locally and student.external_server_urls is None and student.external_proxy_url is None
    )
    windows = student.num_engines * student.data_parallel_size if student_launched else 0
    return VLLM_START_PORT + windows * SERVER_PORT_STRIDE


def launch_teacher(cfg: "SkyRLTrainConfig") -> Tuple["RemoteInferenceClient", "InferenceServerSetup"]:
    """Launch the teacher deployment; return the client that drives it and the setup that owns its actors.

    The setup has to be referenced during the entire run. Ray terminates a non-detached actor once every
    handle to it is gone, and only the setup's server groups hold the handles.

    Blocks until every server answers ``/health``. The deployment creates its own placement group;
    a GPU budget the cluster cannot satisfy surfaces as that group's timeout
    (``SKYRL_RAY_PG_TIMEOUT_IN_S``).
    """
    # TODO (kyuds): check the GPU budget (teacher + student engines + training workers when not
    # colocated) against ray.cluster_resources() here and fail fast with the three terms named,
    # instead of waiting out the placement-group timeout. Left out for now because a snapshot
    # under-reports an autoscaling cluster; decide warn-vs-error with the team.
    from skyrl.backends.skyrl_train.inference_servers.setup import (
        build_new_inference_client_impl,
    )
    from skyrl.backends.skyrl_train.inference_servers.utils import (
        build_frozen_vllm_cli_args,
    )

    teacher: TeacherConfig = cfg.trainer.teacher

    def make_cli_args() -> Namespace:
        args = build_frozen_vllm_cli_args(teacher.model, teacher.inference_engine, cfg.trainer.seed)
        # vLLM refuses a request above --max-logprobs (-1 is no cap) by the ids it scores per position and,
        # separately, by the top-k it returns. Raised on the args, so engine_init_kwargs is never mutated.
        if args.max_logprobs != -1:
            args.max_logprobs = max(args.max_logprobs, teacher_logprob_width(cfg), teacher_topk(cfg))
        return args

    return build_new_inference_client_impl(
        teacher.inference_engine,
        model_path=teacher.model,
        log_path=cfg.trainer.log_path,
        make_cli_args=make_cli_args,
        start_port=_teacher_start_port(cfg),
    )
