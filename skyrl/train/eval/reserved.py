"""``ReservedEvalBackend``: an eval-only vLLM engine group that loads each evaluated step's HF export.

The group is built by the entrypoint (``BasePPOExp.get_eval_backend``) after the training fleet,
in its own placement group, and is never handed to ``init_weight_sync_state``: that single omission
keeps it out of weight sync. Each eval loads its step's export -- from a shared filesystem, or a
cloud URI fetched once per node -- through ``RemoteInferenceClient.load_weights_from_path``, then
invalidates the engines' prefix cache and the client's weight version so the next eval cannot read
this one's cache.

Only the config helpers are importable without the vLLM-side modules; ``create`` imports those
lazily, the way the entrypoint imports ``build_new_inference_client``.
"""

import asyncio
import copy
import hashlib
import os
import tempfile
import uuid
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.eval.backend import EvalBackend
from skyrl.train.eval.types import EvalLease, EvalRequest, EvalSkip

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.inference_servers.setup import InferenceServerSetup
    from skyrl.train.config.config import InferenceEngineConfig, SkyRLTrainConfig
    from skyrl.train.generators.base import GeneratorInterface


def build_reserved_engine_cfg(cfg: "SkyRLTrainConfig") -> "InferenceEngineConfig":
    """The reserved group's engine config: a copy of the training one plus ``engine_overrides``,
    sized by ``eval_dispatch.num_engines``, with everything that only makes sense for the training
    fleet switched off."""
    ie = copy.deepcopy(cfg.generator.inference_engine)
    for key, value in cfg.trainer.eval_dispatch.engine_overrides.items():
        setattr(ie, key, value)
    ie.num_engines = cfg.trainer.eval_dispatch.num_engines
    ie.enable_pd = False
    ie.num_prefill = 0
    ie.external_proxy_url = None
    ie.external_server_urls = None
    ie.offload_kv_for_weight_sync = False  # keeps enable_sleep_mode off: the group is never slept
    ie.speculative_config = None  # the load-from-path route does not reload a drafter
    return ie


def reserved_train_cfg(cfg: "SkyRLTrainConfig") -> "SkyRLTrainConfig":
    """A deep copy of ``cfg`` describing the reserved group: its engine config swapped in and
    ``placement.colocate_all`` off whatever the training fleet does -- ``build_vllm_cli_args`` reads
    that flag for ``enable_sleep_mode`` and for the weight-transfer strategy. Shared by ``create``
    and by ``validate_cfg``, so both see the same group."""
    cfg_r = copy.deepcopy(cfg)
    cfg_r.generator.inference_engine = build_reserved_engine_cfg(cfg)
    cfg_r.trainer.placement.colocate_all = False
    return cfg_r


def _default_export_cache_dir(export_path: str) -> str:
    """Per-node cache for exports fetched from a cloud ``export_path``, derived from that path the
    way the delta weight-sync cache is derived from its ``sync_dir``."""
    digest = hashlib.sha1(export_path.encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), "skyrl_eval_exports", digest)


class ReservedEvalBackend(EvalBackend):
    """``EvalBackend`` over a reserved vLLM engine group. Build with ``create``."""

    def __init__(
        self,
        *,
        setup: "InferenceServerSetup",
        client: "RemoteInferenceClient",
        generator: "GeneratorInterface",
        cache_root: str,
    ):
        self._setup = setup
        self._client = client
        self._generator = generator
        self._cache_root = cache_root

    @classmethod
    def create(
        cls,
        cfg: "SkyRLTrainConfig",
        tokenizer: Any,
        *,
        log_path: str,
        make_generator: Callable[["RemoteInferenceClient"], "GeneratorInterface"],
    ) -> "ReservedEvalBackend":
        """Launch the group, build its client and generator, and probe that its nodes can read
        ``trainer.export_path``. Runs outside any event loop, from the entrypoint's setup."""
        from skyrl.backends.skyrl_train.inference_servers.common import (
            SERVER_PORT_STRIDE,
        )
        from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
            RemoteInferenceClient,
        )
        from skyrl.backends.skyrl_train.inference_servers.setup import (
            VLLM_START_PORT,
            create_inference_servers,
        )
        from skyrl.backends.skyrl_train.inference_servers.utils import (
            build_vllm_cli_args,
        )

        cfg_r = reserved_train_cfg(cfg)
        ie = cfg_r.generator.inference_engine
        cli_args = build_vllm_cli_args(cfg_r)
        # Past the training fleet's port span: the DP master port derives from the base, so two
        # groups sharing one would collide under data_parallel_size > 1.
        training_ie = cfg.generator.inference_engine
        start_port = VLLM_START_PORT + training_ie.num_engines * training_ie.data_parallel_size * SERVER_PORT_STRIDE
        setup = create_inference_servers(ie, cli_args, log_path, placement_group=None, start_port=start_port)
        client = RemoteInferenceClient(
            proxy_url=setup.proxy_url,
            server_urls=setup.server_urls,
            model_name=ie.served_model_name or cfg.trainer.policy.model.path,
            enable_return_routed_experts=ie.enable_return_routed_experts,
            uses_lora_weight_sync=False,
            data_parallel_size=ie.data_parallel_size,
            tokenizer=tokenizer,
        )
        cache_root = cfg.trainer.eval_dispatch.local_cache_dir or _default_export_cache_dir(cfg.trainer.export_path)
        backend = cls(setup=setup, client=client, generator=make_generator(client), cache_root=cache_root)
        # Setup runs outside a loop (the entrypoint already does the same to sleep the training
        # engines); the client keeps one session per loop and drops this one's when the loop closes.
        asyncio.run(backend._probe_export_root(cfg.trainer.export_path))
        logger.info(
            f"Reserved eval engine group up: {ie.num_engines} engine(s) at {setup.server_urls}, "
            f"router {setup.proxy_url}, export cache {cache_root}"
        )
        return backend

    async def _probe_export_root(self, export_path: str) -> None:
        """Fail at minute 0, not minute 40: write a sentinel under ``export_path`` and require every
        reserved worker to see it -- a non-shared local path, or missing cloud credentials on an
        engine node."""
        sentinel = os.path.join(export_path, f".reserved_eval_probe_{uuid.uuid4().hex}")
        io.makedirs(export_path)  # no-op for a cloud root
        with io.open_file(sentinel, "wb"):
            pass
        try:
            seen = await self._client.paths_exist(sentinel)
        finally:
            io.remove(sentinel)
            await self._client.aclose()  # the session bound to this throwaway loop
        if not seen or not all(seen):
            raise RuntimeError(
                f"reserved eval engines cannot read trainer.export_path={export_path!r}: it must be a shared "
                "filesystem visible to every engine node, or a cloud URI the engine nodes have credentials for"
            )

    async def sync(self, req: EvalRequest) -> EvalLease:
        # Step 0 is the launch weights, not an export. Every other step must have its export, judged
        # by its ``config.json`` (cloud-aware through ``io``): an FSDP export appears atomically
        # (staging directory + rename), and a Megatron one, written in place, is complete because
        # ``save_models()`` is synchronous and precedes the eval site.
        if req.global_step > 0 and not io.exists(os.path.join(req.export_dir, "config.json")):
            raise EvalSkip("ckpt_missing")
        version = str(req.global_step)
        cache_dir = os.path.join(self._cache_root, f"global_step_{req.global_step}")  # used for a cloud root only
        try:
            versions = await self._client.load_weights_from_path(
                req.export_dir, weight_version=version, cache_dir=cache_dir
            )
            # Two invalidations, or the next eval reads this one's prefix cache: the engines' cache,
            # and the client's weight version, which salts the generator's prefix cache keys.
            await self._client.reset_prefix_cache()
            self._client.increment_weight_version()
        except Exception:
            logger.exception(f"loading {req.export_dir} onto the reserved eval engine group failed")
            raise EvalSkip("sync_failed")
        # Every worker on every server must attest the version, or the router mixes versions.
        if set(versions) != {version}:
            logger.warning(f"reserved eval sync mismatch: wanted {version!r}, got {sorted(set(versions))}")
            raise EvalSkip("sync_mismatch")
        return EvalLease(generator=self._generator)

    async def release(self, lease: EvalLease) -> None:
        return  # nothing borrowed: the group is held for the whole run

    async def close(self) -> None:
        """Tear the group down: router, server groups (and their placement group), client session.
        Each step is attempted even if an earlier one fails."""
        steps = [self._setup.router.shutdown] if self._setup.router is not None else []
        steps += [group.shutdown for group in self._setup.server_groups]
        steps.append(self._client.teardown)
        for step in steps:
            try:
                result = step()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.warning("reserved eval engine group teardown step failed", exc_info=True)
