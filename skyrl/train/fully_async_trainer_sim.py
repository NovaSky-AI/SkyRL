"""Simulated-trainer variant of the fully-async trainer (no trainer components).

``FullyAsyncTrainerSim`` reuses ``FullyAsyncRayPPOTrainer.train()`` as-is and overrides only
the model-dependent seams so that NO policy/critic/ref models are built and NO weights are
broadcast. Used to benchmark the generation/inference side (e.g. router load-balancing) on
large models without paying for trainer GPUs — typically pointed at already-served endpoints
via ``generator.inference_engine.external_proxy_url`` / ``external_server_urls``.

See ``FullyAsyncConfig.simulate_training`` and the
``examples/train/fully_async/main_fully_async_sim.py`` entrypoint.
"""

import asyncio

from loguru import logger

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils.trainer_utils import ResumeMode


class _SimDispatch:
    """Stub for ``FullyAsyncRayPPOTrainer.dispatch`` in simulated-trainer mode.

    Replaces the real model→engine weight-sync dispatch. ``save_weights_for_sampler`` stands in
    for the (skipped) weight broadcast by pausing generation, optionally sleeping
    ``simulate_weight_sync_seconds``, then resuming.
    """

    def __init__(self, inference_engine_client, sync_sleep):
        self._client = inference_engine_client
        self._sync_sleep = float(sync_sleep)

    def init_weight_sync_state(self, inference_engine_client):
        # No real weight-sync process group in sim mode.
        pass

    def get_lcm_dp_size(self) -> int:
        # No trainer models are built in sim mode, so there is no data-parallel
        # group to pad the training batch for. Return 1 so the padding in
        # ``convert_to_training_input`` (pad to a multiple of dp_size) is a no-op.
        return 1

    async def save_weights_for_sampler(self):
        await self._client.pause_generation()
        try:
            if self._sync_sleep > 0:
                await asyncio.sleep(self._sync_sleep)
        finally:
            await self._client.resume_generation()


class FullyAsyncTrainerSim(FullyAsyncRayPPOTrainer):
    """Fully-async trainer with a SIMULATED training step (no trainer components).

    Faithfully reproduces the **generation-side** dynamics of real fully-async training —
    the staleness controller, capacity gating, rate limiting, generation-buffer fill, and the
    pause/resume around weight sync — but **instantiates no trainer components and broadcasts
    no weights**. Each step:
      1. waits for a full mini-batch in the generation buffer (consuming it),
      2. sleeps ``fully_async.simulate_training_step_seconds`` (stands in for fwd/bwd/optim),
      3. issues ``pause_generation`` → optional ``simulate_weight_sync_seconds`` sleep →
         ``resume_generation`` (a real step would broadcast new weights in between),
      4. advances ``global_step`` and notifies the staleness manager (unblocking generation).

    Designed to benchmark inference/router behavior (e.g. session-aware load balancing) on
    large models without paying for trainer GPUs. Typically pointed at **already-served**
    endpoints via ``generator.inference_engine.external_proxy_url`` /
    ``external_server_urls`` (with ``run_engines_locally=false``), so generation hits the live
    deployment directly. No models are built (the entrypoint skips ``build_models``), no
    checkpoints, no eval, no weight sync.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fa = self.cfg.trainer.fully_async
        assert fa.simulate_training, "FullyAsyncTrainerSim requires trainer.fully_async.simulate_training=true."
        # We reuse the inherited FullyAsyncRayPPOTrainer.train() as-is and only override the
        # model-dependent seams (init_weight_sync_state, dispatch.save_weights_for_sampler,
        # _run_training). Everything else in train() that touches trainer models is config-gated,
        # so assert those are off — no models are built in sim mode, so they would otherwise crash.
        t = self.cfg.trainer
        assert t.eval_interval <= 0, "FullyAsyncTrainerSim: set trainer.eval_interval=0 (no models to eval)."
        assert t.ckpt_interval <= 0, "FullyAsyncTrainerSim: set trainer.ckpt_interval=-1 (no models to checkpoint)."
        assert t.hf_save_interval <= 0, "FullyAsyncTrainerSim: set trainer.hf_save_interval=-1 (no models to save)."
        assert not t.update_ref_every_epoch, "FullyAsyncTrainerSim: trainer.update_ref_every_epoch must be false."
        assert self.resume_mode == ResumeMode.NONE, "FullyAsyncTrainerSim: resumption is unsupported (no models)."
        self._step_sleep = float(fa.simulate_training_step_seconds)
        # build_models() is skipped in sim mode, so model handles are never created. train()'s
        # per-epoch epilogue reads self.ref_model — default it to None so that guard is safe.
        self.ref_model = None

    def init_weight_sync_state(self):
        """Sim override: no real model→engine weight-sync handshake.

        Installs a stub ``self.dispatch`` whose ``save_weights_for_sampler`` simulates the
        (skipped) weight broadcast by pausing/resuming generation (see ``_SimDispatch``).
        Called by the inherited ``train()`` before the loop; replaces the base implementation
        that would dispatch to real policy-model actors.
        """
        self.dispatch = _SimDispatch(
            self.inference_engine_client,
            self.cfg.trainer.fully_async.simulate_weight_sync_seconds,
        )
        logger.info("[SIM] init_weight_sync_state stubbed — no models; weight sync simulated via pause/resume.")

    async def _run_training(self, training_input):
        """Sim override: stand in for fwd/bwd/optim with a sleep (no models built)."""
        await asyncio.sleep(self._step_sleep)
        return {"sim_step": self.global_step, "sim_step_seconds": self._step_sleep}
