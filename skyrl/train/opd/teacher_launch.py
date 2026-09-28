"""Launching the teacher's vLLM deployment inside the job (``trainer.teacher.backend="skyrl"``).

The deployment is the student's own launch path, ``create_inference_servers`` (server groups plus a
router), with a frozen-model argument set from ``build_frozen_vllm_cli_args``: no weight transfer, no
sleep mode, no LoRA. It gets its own placement group (never the colocate group), a port window past
the student's, and is driven through a ``RemoteInferenceClient`` exactly as the student's engines are.
The rules below (which fields are frozen, the context bound, the port offset) are OPD's; core only
learns "launch a deployment from these args at this port and hand back its client". Design notes:
``plan/opd-entrypoint/teacher-launcher.md`` in the workspace.
"""

import copy
from argparse import Namespace
from typing import TYPE_CHECKING, Tuple

from skyrl.backends.skyrl_train.inference_servers.common import (
    SERVER_PORT_STRIDE,
    VLLM_START_PORT,
)
from skyrl.train.opd.config import TeacherConfig, teacher_max_model_len

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.setup import InferenceServerSetup


def served_teacher_name(teacher: TeacherConfig) -> str:
    """The name the launched servers know the teacher by, and the ``model`` scoring requests send."""
    return teacher.inference_engine.served_model_name or teacher.model


def teacher_cli_args(cfg) -> Namespace:
    """vLLM server args for the teacher: the frozen role, with ``max_model_len`` defaulted to what scoring needs."""
    from skyrl.backends.skyrl_train.inference_servers.utils import (
        build_frozen_vllm_cli_args,
    )

    teacher: TeacherConfig = cfg.trainer.teacher
    ie_cfg = copy.deepcopy(teacher.inference_engine)
    # vLLM otherwise takes the model's native context, which is more KV budget than a prefill server
    # needs and, on a small teacher GPU, more than the KV pool can hold at startup.
    ie_cfg.engine_init_kwargs.setdefault("max_model_len", teacher_max_model_len(cfg))
    return build_frozen_vllm_cli_args(teacher.model, ie_cfg, cfg.trainer.seed)


def teacher_start_port(cfg) -> int:
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


def launch_teacher(cfg) -> Tuple["RemoteInferenceClient", "InferenceServerSetup"]:
    """Launch the teacher deployment; return the client that drives it and the setup that owns its actors.

    Keep the setup referenced for the run: Ray terminates a non-detached actor once every handle to
    it is gone, and the setup's server groups hold the only handles.

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

    teacher: TeacherConfig = cfg.trainer.teacher
    return launch_remote_inference_client(
        teacher.inference_engine,
        teacher_cli_args(cfg),
        model_name=served_teacher_name(teacher),
        log_path=cfg.trainer.log_path,
        start_port=teacher_start_port(cfg),
    )
