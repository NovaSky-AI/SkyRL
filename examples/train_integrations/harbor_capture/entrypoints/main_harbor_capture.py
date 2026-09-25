"""Train on Harbor tasks with skyrl-capture recording the exact tokens.

The inference setup hook brings capture up beside the engine, configured with
the router as its one upstream. Everything after that is the sibling Harbor
entrypoint: the generator is the only thing swapped.

Runnable as it stands, the same way as the sibling generate entrypoint:

    python -m examples.train_integrations.harbor_capture.entrypoints.main_harbor_capture \\
        trainer.policy.model.path=... data.train_data="['/path/to/harbor/tasks']"

Capture comes up inside this process by default, so nothing has to be started
first -- it writes a record directory and there is no database. Setting
`CAPTURE_ENDPOINT` uses a separate `skyrl-capture serve` instead.
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

from ...harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborSkyRLConfig,
    _deep_merge,
)
from ...harbor.entrypoints.main_harbor_generate import HarborGenerateExp

logger = logging.getLogger(__name__)

#: The module whose import registers `type="skyrl"` with capture.
UPSTREAM_MODULE = "examples.train_integrations.harbor_capture.upstream"

DEFAULT_RECORD_DIR = Path("./harbor-capture-record")
DEFAULT_PORT = 8080


def start_capture(
    *,
    engine_url: str,
    model_name: str,
    tokenizer_name: str,
    max_model_len: int,
    record_dir: Any = DEFAULT_RECORD_DIR,
    port: int = DEFAULT_PORT,
) -> Any:
    """Bring capture up in this process, in front of the engine.

    Called once per run. One capture process owns one configured upstream --
    there is nothing to register afterwards and no target to name, because the
    upstream is part of the configuration the service is built with. Several
    policy endpoints mean several capture processes.

    ``engine_url`` is the **router's root**, not a generate endpoint. The
    ``skyrl`` protocol knows the path, the singular request shape, the
    ``X-Session-ID`` affinity header, vLLM's sampling-parameter rules and
    ``cache_salt`` -- all of which a `tokens` upstream gets wrong against this
    router.

    Nothing has to exist first: capture writes a record directory, and creates
    it if it is new. There is no database.
    """
    from skyrl_capture.config import Config, TitoUpstream
    from skyrl_capture.service import CaptureService

    config = Config(
        record_dir=Path(record_dir),
        upstream=TitoUpstream(
            type="skyrl",
            url=engine_url,
            tokenizer=tokenizer_name,
            model=model_name,
            max_model_len=max_model_len,
        ),
        # Imported by the service before it serves, which is what puts
        # `type="skyrl"` in capture's registry. Named rather than imported
        # here so the same configuration works for a service in another
        # process, where this module's import would not have happened.
        upstream_modules=(UPSTREAM_MODULE,),
    )
    service = CaptureService(config=config, port=port)
    # Returns once the service is serving, so trajectory URLs handed out on
    # the next line are usable rather than a race the harness loses.
    service.start(blocking=False)
    logger.info("skyrl-capture serving at %s, recording to %s", service.base_url, config.record_dir)
    return service


def build_generator(cfg: Any, harbor_trial_config: Dict[str, Any], engine_client: Any, service: Any):
    """The generator, pointed at a capture service rather than the engine."""
    from ..harbor_generator import HarborCaptureGenerator

    return HarborCaptureGenerator(
        generator_cfg=cfg.generator,
        harbor_trial_config=harbor_trial_config,
        inference_engine_client=engine_client,
        capture_endpoint=service.base_url,
        project=getattr(cfg, "experiment_name", None) or "harbor-capture",
        run_id=getattr(cfg, "run_id", None) or getattr(cfg, "experiment_name", None),
    )


def capture_for_run(
    cfg: Any,
    engine_url: str,
    *,
    tokenizer_name: str | None = None,
) -> Any:
    """Capture for this run: in this process, or one already serving.

    In-process is the default and is what most jobs want -- nothing has to be
    started first, and it dies with the job. Setting ``CAPTURE_ENDPOINT``
    points at a separate ``skyrl-capture serve`` instead, which is right when
    several jobs share one capture, or when the viewer should outlive the run.

    Naming an endpoint is the whole intent, so there is no second variable
    saying whether to use it. ``CAPTURE_ENDPOINT`` is also what the capture
    SDK reads, so this is not a setting invented here.
    """
    import os

    endpoint = os.environ.get("CAPTURE_ENDPOINT", "").strip()
    if endpoint:
        logger.info("skyrl-capture out-of-process at %s", endpoint)
        # That process is configured with its own upstream, and resolves the
        # protocol by name, so it needs this wire registered too:
        #   skyrl-capture serve --upstream-module <UPSTREAM_MODULE> ...
        return RemoteCaptureService(endpoint)

    engine_init = cfg.generator.inference_engine.engine_init_kwargs
    engine_init = engine_init if isinstance(engine_init, dict) else dict(engine_init)
    return start_capture(
        engine_url=engine_url,
        model_name=cfg.generator.inference_engine.served_model_name,
        # The tokenizer capture renders with. It need not be the one the
        # trainer renders with -- when they differ, that difference is the
        # thing worth measuring.
        tokenizer_name=tokenizer_name or cfg.trainer.policy.model.path,
        max_model_len=int(engine_init["max_model_len"]),
    )


class RemoteCaptureService:
    """The part of ``CaptureService`` a caller needs when it runs elsewhere.

    ``base_url`` is now the whole surface, which is why swapping the two is a
    substitution rather than an abstraction. A service in another process
    carries its own upstream configuration; there is nothing to create or
    update from here.
    """

    def __init__(self, endpoint: str) -> None:
        self.base_url = endpoint.rstrip("/")

    def stop(self, **_kwargs: Any) -> None:
        """Not ours to stop."""


class HarborCaptureGenerateExp(HarborGenerateExp):
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
        """Idempotent. Draining is the point: stopping waits for trajectories
        still committing, so skipping it leaves the last records unwritten."""
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
    HarborCaptureGenerateExp(cfg).run()


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
