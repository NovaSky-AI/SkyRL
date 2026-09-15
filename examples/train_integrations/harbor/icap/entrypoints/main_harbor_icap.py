"""Train on Harbor tasks with inference-capture recording the exact tokens.

The inference setup hook brings capture up beside the engine and registers the
engine as a ``tokens`` target. Everything after that is the sibling Harbor
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


def start_capture(cfg: Any, engine_url: str, model_name: str, tokenizer_name: str) -> Any:
    """Bring capture up and register the engine. Returns the running service.

    Called once per run, beside the inference engine. Nothing has to exist
    first: with no ``DATABASE_URL`` the service starts its own PostgreSQL and
    keeps the database, the queue and the payloads under one directory.
    """
    from inference_capture.service import CaptureService

    proxy_cfg = getattr(getattr(cfg, "inference", None), "proxy", None)
    service = CaptureService(
        data_dir=getattr(proxy_cfg, "data_dir", DEFAULT_DATA_DIR),
        port=getattr(proxy_cfg, "port", 8080),
        num_workers=getattr(proxy_cfg, "num_workers", 4),
    )
    # Returns once /healthz answers, so trajectory URLs handed out on the next
    # line are usable rather than a race the harness loses.
    base_url = service.start()
    logger.info("inference-capture serving at %s", base_url)

    import asyncio

    asyncio.run(
        service.ensure_target(
            name="policy",
            type="tokens",
            url=f"{engine_url.rstrip('/')}/generate",
            model=model_name,
            tokenizer=tokenizer_name,
            config={"max_model_len": cfg.generator.max_input_length},
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
