"""Train on Harbor tasks with inference-capture recording the exact tokens.

The inference setup hook brings capture up beside the engine and registers the
router as a ``skyrl`` target. Everything after that is the sibling Harbor
entrypoint: the generator is the only thing swapped.

Run it with capture resolved from a checkout, which is what development wants:

    uv run --with-editable /path/to/anyscale-capture \\
        -m examples.train_integrations.harbor.icap.entrypoints.main_harbor_icap

or from a release once there is one:

    uv run --with inference-capture -m ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

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
    from ..harbor_generator import ICapHarborGenerator

    return ICapHarborGenerator(
        generator_cfg=cfg.generator,
        harbor_trial_config=harbor_trial_config,
        inference_engine_client=engine_client,
        capture_service=service,
        project=getattr(cfg, "experiment_name", "harbor-icap"),
        target_name="policy",
    )


def main() -> None:  # pragma: no cover - needs a cluster
    raise SystemExit(
        "Copy the sibling entrypoint (examples/train_integrations/harbor/entrypoints/"
        "main_harbor.py), then call start_capture() in the inference setup hook and "
        "build_generator() where HarborGenerator is constructed. Those two calls are "
        "the whole integration."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
