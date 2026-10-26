"""Launching the teacher's vLLM deployment inside the job (``trainer.teacher.backend="skyrl"``).

The deployment follows the rollout inference engine's launch path, ``create_inference_servers``
(server groups plus a router), with a frozen-model argument set from ``build_frozen_vllm_cli_args``.
The teacher model has no weight transfer, no sleep mode, no LoRA. It gets its own placement group
(never the colocate group), a port window past the student's, and is driven through a
``RemoteInferenceClient``.
"""

from typing import TYPE_CHECKING, Tuple

from skyrl.backends.skyrl_train.inference_servers.common import (
    SERVER_PORT_STRIDE,
    VLLM_START_PORT,
)
from skyrl.train.opd.config import TeacherConfig

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.setup import InferenceServerSetup
    from skyrl.train.config import SkyRLTrainConfig


def _teacher_start_port(cfg: "SkyRLTrainConfig") -> int:
    """The first port window past the student's launched servers.

    Each server actor owns ``SERVER_PORT_STRIDE`` ports (HTTP port plus its DP TCPStore probe range)
    from its base; the student's servers take ``num_engines * data_parallel_size`` windows from
    ``VLLM_START_PORT``. With an external student nothing is launched and the teacher starts there.
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
        launch_remote_inference_client,
    )
    from skyrl.backends.skyrl_train.inference_servers.utils import (
        build_frozen_vllm_cli_args,
    )

    teacher: TeacherConfig = cfg.trainer.teacher
    return launch_remote_inference_client(
        teacher.inference_engine,
        build_frozen_vllm_cli_args(teacher.model, teacher.inference_engine, cfg.trainer.seed),
        # The name the launched servers know the teacher by, and the `model` scoring requests send.
        model_name=teacher.inference_engine.served_model_name or teacher.model,
        log_path=cfg.trainer.log_path,
        start_port=_teacher_start_port(cfg),
    )
