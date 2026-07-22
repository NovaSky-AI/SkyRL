# Clip the MTP/draft head independently of the policy inside the SHARED optimizer.
#
# The draft loss is autograd-decoupled (its gradient reaches only ``.mtp.`` params), but the head's
# grads sit in the policy's grad buffer and optimizer, so Megatron's single global grad-norm counts
# them. The head's soft-CE grads are ~20-30 per microbatch at loss weight 1.0 (measured, GPU probe
# test_mtp_grad_isolation), so they dominate the global norm and over-clip the policy: measured
# policy/grad_norm 21.6 vs 0.81 for the same step -> the policy update shrank ~20x. Clipping the two
# groups by their OWN norms keeps the policy's clip (and the reported grad_norm) identical to a
# no-MTP run while the head still trains at full strength.

from __future__ import annotations

import contextlib
from typing import List

import torch
from megatron.core.optimizer.clip_grads import (
    clip_grad_by_total_norm_fp32,
)

# Megatron keeps a param's model-side tensor and its optimizer-side "main" (fp32) tensor in
# positionally-matched lists; main params are identified by walking these pairs (the optimizer's
# param_groups hold main params whose identities no longer match the model's).
_MODEL_TO_MAIN_GROUPS = (
    ("model_float16_groups", "shard_fp32_from_float16_groups"),  # DistributedOptimizer
    ("model_fp32_groups", "shard_fp32_groups"),
    ("float16_groups", "fp32_from_float16_groups"),  # Float16OptimizerWithFloat16Params
    ("fp32_from_fp32_groups", "fp32_from_fp32_groups"),  # fp32 params are their own main param
)


def _sub_optimizers(optimizer) -> List:
    """The ChainedOptimizer's members, or the optimizer itself if it is not chained."""
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def is_mtp_param_name(name: str) -> bool:
    """True for parameter names belonging to the MTP / draft head."""
    return ".mtp." in name or name.startswith("mtp.")


def _mtp_model_param_ids(model_chunks) -> set:
    """Ids of the MTP head's model params, matched BY NAME on the chunks the optimizer was built from."""
    ids = set()
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if is_mtp_param_name(name):
                ids.add(id(param))
    return ids


def _mtp_main_params(sub_opt, mtp_model_ids: set) -> List[torch.nn.Parameter]:
    """Optimizer-side main params corresponding to the MTP head's model params.

    May legitimately be EMPTY on this rank: the DistributedOptimizer shards param ownership across
    the DP group, and a rank whose grad-buffer shard contains no head params has none in its groups.
    """
    found, seen = [], set()
    for model_attr, main_attr in _MODEL_TO_MAIN_GROUPS:
        model_groups = getattr(sub_opt, model_attr, None)
        main_groups = getattr(sub_opt, main_attr, None)
        if not model_groups or not main_groups:
            continue
        for model_group, main_group in zip(model_groups, main_groups):
            for model_param, main_param in zip(model_group, main_group):
                if id(model_param) in mtp_model_ids and id(main_param) not in seen:
                    seen.add(id(main_param))
                    found.append(main_param)
    return found


@contextlib.contextmanager
def _restrict_params(subs, subsets):
    """Make each sub-optimizer report only ``subsets[i]`` from ``get_parameters()``.

    Megatron derives the grad-norm from ``get_parameters()`` (via ``get_main_grads_for_grad_norm``),
    so restricting it reuses Megatron's own norm code -- its shared/TP-duplicate/decoupled-grad
    filtering, the cross-optimizer aggregation an MoE model relies on, and the collective reduction
    (which runs even for an empty local subset, keeping DP ranks without head shards in sync).
    """
    saved = [sub.get_parameters for sub in subs]
    for sub, subset in zip(subs, subsets):
        sub.get_parameters = (lambda s: (lambda: s))(subset)
    try:
        yield
    finally:
        for sub, fn in zip(subs, saved):
            sub.get_parameters = fn


def _use_decoupled_grad(sub_opt, params: List[torch.nn.Parameter]) -> bool:
    cfg = sub_opt.config
    return bool(
        cfg.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        # Megatron-FSDP always uses decoupled_grad with FusedAdam.
        or (cfg.use_precision_aware_optimizer and params and getattr(params[0], "__fsdp_param__", False))
    )


def install_mtp_separate_grad_clip(optimizer, model_chunks) -> int:
    """Replace ``optimizer.step`` so the policy and the MTP head are clipped by their own grad norms.

    Mirrors ``ChainedOptimizer.step`` but computes two norms instead of one. Must be installed on
    EVERY rank when MTP is on -- both norms are collective reductions, so a rank without local head
    params (PP stage without the head, or a DP shard not covering it) still participates.

    Reports the POLICY grad norm as the step's grad_norm, so ``policy/grad_norm`` stays comparable
    to a no-MTP run; the head's own (pre-clip) norm is stashed on ``optimizer.mtp_grad_norm``.
    Returns the number of head main params isolated on THIS rank (0 is normal under DP sharding).
    """
    subs = _sub_optimizers(optimizer)
    mtp_model_ids = _mtp_model_param_ids(model_chunks)
    if not mtp_model_ids:
        raise RuntimeError(
            "install_mtp_separate_grad_clip: no parameter name matched is_mtp_param_name (expected "
            "'.mtp.' or 'mtp.'). The head's naming changed -- it would be clipped together with the "
            "policy and throttle the policy update. Update mtp/grad_clip.py."
        )
    policy_subsets, mtp_subsets = [], []
    for sub in subs:
        all_main = sub.get_parameters()
        mtp_main_ids = {id(p) for p in _mtp_main_params(sub, mtp_model_ids)}
        mtp_subsets.append([p for p in all_main if id(p) in mtp_main_ids])
        policy_subsets.append([p for p in all_main if id(p) not in mtp_main_ids])
    n_local = sum(len(s) for s in mtp_subsets)

    # The head must be covered SOMEWHERE across the group (ownership is DP-sharded; a given rank
    # owning none is normal). Zero everywhere means the model->main pairing broke and the head is
    # silently back in the policy clip -- fail loud, collectively.
    if torch.distributed.is_initialized():
        covered = torch.tensor([n_local], device=torch.cuda.current_device())
        torch.distributed.all_reduce(covered, op=torch.distributed.ReduceOp.SUM)
        n_global = int(covered.item())
    else:
        n_global = n_local
    if n_global == 0:
        raise RuntimeError(
            "install_mtp_separate_grad_clip: the model has an MTP head but none of its params mapped "
            "to optimizer main params on any rank. megatron-core's model/main param group lists "
            "likely changed; update _MODEL_TO_MAIN_GROUPS in mtp/grad_clip.py."
        )

    orig_get_grad_norm = optimizer.get_grad_norm

    @torch.no_grad()
    def step():
        if optimizer.prepare_grads():
            return False, None, None

        with _restrict_params(subs, policy_subsets):
            policy_norm = orig_get_grad_norm()
        with _restrict_params(subs, mtp_subsets):
            mtp_norm = orig_get_grad_norm()
        optimizer.mtp_grad_norm = mtp_norm

        should_skip_update = False
        for sub, policy_main, mtp_main in zip(subs, policy_subsets, mtp_subsets):
            clip_grad = sub.config.clip_grad
            if clip_grad > 0.0:
                for params, total_norm in ((policy_main, policy_norm), (mtp_main, mtp_norm)):
                    if params:
                        clip_grad_by_total_norm_fp32(
                            params,
                            max_norm=clip_grad,
                            total_norm=total_norm,
                            use_decoupled_grad=_use_decoupled_grad(sub, params),
                        )
            # Skip on the POLICY norm only: a head-driven spike must not stall the policy.
            if policy_norm > sub.config.grad_norm_skip_threshold:
                should_skip_update = True

        num_zeros_in_grad = optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else None
        update_successful = False if should_skip_update else optimizer.step_with_ready_grads()
        return update_successful, policy_norm, num_zeros_in_grad

    optimizer.step = step
    return n_local
