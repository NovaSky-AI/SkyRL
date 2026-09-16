"""Train on Harbor tasks with inference-capture recording the exact tokens.

The inference setup hook brings capture up beside the engine and registers the
router as a ``skyrl`` target. Everything after that is the sibling Harbor
entrypoint: the generator is the only thing swapped.

Runnable as it stands, the same way as the sibling generate entrypoint:

    python -m examples.train_integrations.harbor.icap.entrypoints.main_harbor_icap \\
        trainer.policy.model.path=... data.train_data="['/path/to/harbor/tasks']"

Capture comes up inside this process by default, so nothing has to be started
first. `ICAP_INPROCESS=0` with `CAPTURE_ENDPOINT` uses a separate `icap serve`.
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import ray
import yaml

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...entrypoints.main_harbor import HARBOR_DEFAULT_CONFIG, HarborSkyRLConfig, _deep_merge
from ...entrypoints.main_harbor_generate import HarborGenerateExp

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("./icap-data")


def start_capture(
    *,
    engine_url: str,
    model_name: str,
    tokenizer_name: str,
    max_model_len: int,
    data_dir: Any = None,
    port: int | None = None,
    num_workers: int = 4,
    target_name: str = "policy",
) -> Any:
    """Bring capture up in this process and register the engine.

    Called once per run, beside the inference engine. Nothing has to exist
    first: with no ``DATABASE_URL`` the service starts its own PostgreSQL and
    keeps the database, the queue and the payloads under one directory.

    ``engine_url`` is the **router's root**, not a generate endpoint. The
    ``skyrl`` target type knows the path, the singular request shape, the
    ``X-Session-ID`` affinity header and vLLM's sampling-parameter rules -- all
    of which a `tokens` target gets wrong against this router, which is why a
    translating proxy used to be needed here.

    In-process is possible because capture's ``transformers<5`` cap, which
    conflicted with SkyRL's ``>=5.6.1``, turned out to be unnecessary and was
    lifted. Before that the service had to run in a second environment.
    """
    import asyncio
    import os

    from inference_capture.service import CaptureService

    # Registers `type="skyrl"`. capture speaks protocols, not engines, so this
    # wire is ours; importing is what puts it in capture's registry. A separate
    # `icap serve` needs ICAP_UPSTREAM_MODULES pointed at the same module.
    from .. import upstream as _skyrl_upstream  # noqa: F401

    # `initdb` refuses an ungenerated locale, which several base images have.
    os.environ.setdefault("LANG", "C.utf8")
    os.environ.setdefault("LC_ALL", "C.utf8")
    # `CaptureService` reads CONTROL_KEY; the SDK reads CAPTURE_CONTROL_KEY.
    # In one process they have to agree.
    if "CAPTURE_CONTROL_KEY" in os.environ:
        os.environ.setdefault("CONTROL_KEY", os.environ["CAPTURE_CONTROL_KEY"])

    service = CaptureService(
        data_dir=data_dir or os.environ.get("ICAP_DATA_DIR") or DEFAULT_DATA_DIR,
        port=port if port is not None else int(os.environ.get("ICAP_PORT", 8080)),
        num_workers=num_workers,
    )
    # Returns once /healthz answers, so trajectory URLs handed out on the next
    # line are usable rather than a race the harness loses.
    base_url = service.start()
    logger.info("inference-capture serving at %s", base_url)

    asyncio.run(
        service.ensure_target(
            name=target_name,
            type="skyrl",
            url=engine_url,
            model=model_name,
            tokenizer=tokenizer_name,
            config={"max_model_len": max_model_len},
        )
    )
    return service


def build_generator(cfg: Any, harbor_trial_config: Dict[str, Any], engine_client: Any, service: Any):
    """The generator, pointed at a capture service rather than the engine."""
    from ..harbor_generator import ICapHarborGenerator

    return ICapHarborGenerator(
        generator_cfg=cfg.generator,
        harbor_trial_config=harbor_trial_config,
        inference_engine_client=engine_client,
        capture_service=service,
        project=getattr(cfg, "experiment_name", None) or "harbor-icap",
        target_name="policy",
    )


def capture_for_run(
    cfg: Any,
    engine_url: str,
    *,
    tokenizer_name: str | None = None,
    num_workers: int = 4,
) -> Any:
    """Capture for this run: in this process, or one already serving.

    In-process is the default and is what most jobs want -- nothing has to be
    started first, and it dies with the job. ``ICAP_INPROCESS=0`` with
    ``CAPTURE_ENDPOINT`` points at a separate ``icap serve`` instead, which is
    right when several jobs share one capture, or when the UI should outlive
    the run.
    """
    import os

    if os.environ.get("ICAP_INPROCESS", "1").strip().lower() in ("0", "false", "no", "off"):
        endpoint = os.environ.get("CAPTURE_ENDPOINT", "")
        if not endpoint:
            raise RuntimeError("ICAP_INPROCESS=0 needs CAPTURE_ENDPOINT to point at `icap serve`")
        logger.info("inference-capture out-of-process at %s", endpoint)
        # That process resolves the target type, not this one, so it needs the
        # `skyrl` wire registered too:
        #   icap serve -u examples.train_integrations.harbor.icap.upstream
        return RemoteCaptureService(endpoint)

    engine_init = cfg.generator.inference_engine.engine_init_kwargs
    engine_init = engine_init if isinstance(engine_init, dict) else dict(engine_init)
    return start_capture(
        engine_url=engine_url,
        model_name=cfg.generator.inference_engine.served_model_name,
        # The tokenizer capture renders with. It need not be the one the
        # trainer renders with -- when they differ, that difference is the
        # thing worth measuring.
        tokenizer_name=tokenizer_name
        or os.environ.get("ICAP_TOKENIZER", cfg.trainer.policy.model.path),
        max_model_len=int(engine_init["max_model_len"]),
        num_workers=num_workers,
    )


class RemoteCaptureService:
    """The part of ``CaptureService`` a client needs when it is running elsewhere.

    ``base_url`` and ``ensure_target`` are the whole surface the generator uses,
    which is why swapping the two is a substitution rather than an abstraction.
    """

    def __init__(self, endpoint: str) -> None:
        self.base_url = endpoint.rstrip("/")

    def stop(self) -> None:
        """Not ours to stop."""

    async def ensure_target(self, *, name: str, **fields: Any) -> Dict[str, Any]:
        """Create the target, or update it to match -- one idempotent call.

        Deliberately not delete-then-create: capture never resurrects a deleted
        target, so one wrong bootstrap burns the name permanently.
        """
        from inference_capture.sdk import CaptureClient

        return CaptureClient(self.base_url).put(f"/v1/targets/{name}", fields)


class ICapHarborGenerateExp(HarborGenerateExp):
    """`HarborGenerateExp` with capture in front of the engine.

    The generator is the only thing swapped. Harbor runs unmodified in text
    space against a per-trajectory route; the proxy renders the prompt, calls
    the engine with token IDs, and keeps a message graph -- so a rewritten
    history is a branch rather than a hole.
    """

    capture: Any = None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        engine_url = inference_engine_client.get_endpoint_url()
        logger.info("inference engine at %s", engine_url)
        self.capture = capture_for_run(cfg, engine_url)

        harbor_config = deepcopy(cfg.harbor_trial_config)
        harbor_config.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{cfg.generator.inference_engine.served_model_name}"

        return build_generator(cfg, harbor_config, inference_engine_client, self.capture)

    def stop_capture(self) -> None:
        """Idempotent. In-process this stops the embedded PostgreSQL and
        flushes the spool; skipping it leaves a database running after the job
        and the last records unwritten."""
        service, self.capture = self.capture, None
        if service is not None:
            logger.info("stopping inference-capture")
            service.stop()

    def run(self):
        try:
            super().run()
        finally:
            self.stop_capture()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    ICapHarborGenerateExp(cfg).run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as handle:
        defaults = yaml.safe_load(handle)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
