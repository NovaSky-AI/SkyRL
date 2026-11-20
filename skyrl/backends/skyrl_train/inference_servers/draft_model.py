"""Hand a spec-decode drafter to a weight-transfer engine that syncs it with the target model."""

from typing import Any


def attach_draft_model(worker: Any) -> bool:
    """``collective_rpc`` target, run on every worker once at server start under sharded_rdt, before the
    trainer's init.

    vLLM builds the receive engine with the target model only (``Worker.load_model``), and the drafter
    lives on the model runner, which the engine cannot reach. ``sharded_rdt`` bakes one pull plan covering
    both models at init, so it needs the drafter before then. vLLM's ``set_weight_update_target`` is no
    substitute: it swaps the engine onto the drafter for a single session, one model at a time.

    Only SkyRL's sharded_rdt engine, which ``weight_sync/register.py`` registers in place of vLLM's, defines
    ``skyrl_set_draft_model``; any other engine fails startup here with an ``AttributeError``.
    Returns whether this worker holds a drafter (pipeline stages before the last don't).
    """
    draft = worker.model_runner.get_draft_model()
    if draft is None:
        return False
    worker.weight_transfer_engine.skyrl_set_draft_model(draft, worker.vllm_config.speculative_config.draft_model_config)
    return True
