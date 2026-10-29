"""AdamW step and grad-norm clipping on the local shards of FSDP2 DTensor parameters.

``torch._foreach_*`` on a list of N DTensors runs DTensor sharding propagation per
output over all N inputs (O(N^2) host work). With per-expert LoRA (tens of thousands
of parameter tensors) the stock AdamW foreach step and ``clip_grad_norm_`` stall for
minutes with the GPU idle. Both operations are elementwise on a ``Shard`` placement
except the norm reduction, so they run on ``to_local()`` tensors (plain foreach / fused
kernels) and only the squared norm crosses ranks, as ``_NormPartial`` does for DTensor.
"""

from collections import defaultdict
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.optim.adam import adam as _adam


def _local(tensors):
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensors]


class LocalShardAdamW(torch.optim.AdamW):
    """``torch.optim.AdamW`` whose update runs on the parameters' local shards.

    State initialization, ``state_dict`` and ``load_state_dict`` are AdamW's own (state
    tensors stay DTensors shaped like the parameters); only the functional update gets
    the local tensors, which alias the DTensors' storage. Same math and same kernels as
    AdamW on the DTensors.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = [], [], [], [], [], []
            has_complex = self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps)
            beta1, beta2 = group["betas"]
            _adam(
                _local(params),
                _local(grads),
                _local(exp_avgs),
                _local(exp_avg_sqs),
                _local(max_exp_avg_sqs),
                steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )
        return loss


def _shard_groups(grad: DTensor):
    """Process groups to sum a local squared norm over (the mesh dims the grad is sharded on)."""
    groups = []
    for dim, placement in enumerate(grad.placements):
        if isinstance(placement, Shard):
            groups.append(grad.device_mesh.get_group(dim))
        elif not isinstance(placement, Replicate):
            raise ValueError(f"local-shard grad norm: unsupported placement {placement}")
    return tuple(groups)


def _norms(tensors, norm_type, foreach):
    """Per-tensor p-norms, foreach per (device, dtype) bucket as torch's ``_get_total_norm``."""
    out = [None] * len(tensors)
    buckets = defaultdict(list)
    for i, t in enumerate(tensors):
        buckets[(t.device, t.dtype)].append(i)
    for idx in buckets.values():
        ts = [tensors[i] for i in idx]
        ns = (
            torch._foreach_norm(ts, norm_type)
            if foreach is not False
            else [torch.linalg.vector_norm(t, norm_type) for t in ts]
        )
        for i, n in zip(idx, ns):
            out[i] = n
    return out


@torch.no_grad()
def local_shard_clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    """``clip_grad_norm_`` for FSDP2 parameters without DTensor foreach.

    Same reduction as torch on DTensors: each grad's local norm is raised to
    ``norm_type``, summed over the ranks it is sharded on and raised to ``1/norm_type``
    (DTensor ``_NormPartial``; one all_reduce per placement group for all grads), then
    the total is the norm of the per-grad norms in parameter order. Grads are scaled in
    place with torch's clip coefficient. Returns the total norm before clipping.
    """
    if norm_type == float("inf"):
        raise NotImplementedError("local-shard clipping supports finite p-norms only")
    params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    grads = [p.grad for p in params if p.grad is not None]
    out_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    if not grads:
        return torch.tensor(0.0, device=out_device)
    local = _local(grads)
    norms = _norms(local, norm_type, foreach)
    # Reduce where the collective runs: NCCL needs CUDA tensors (grads may be CPU-offloaded).
    nccl = dist.is_initialized() and dist.get_backend() == "nccl"
    device = out_device if nccl else local[0].device
    norms = [n.to(device) for n in norms]
    by_key = defaultdict(list)
    for i, g in enumerate(grads):
        if isinstance(g, DTensor):
            by_key[_shard_groups(g)].append(i)
    for groups, idx in by_key.items():
        if not groups:
            continue
        powered = torch.stack([norms[i] for i in idx]).pow(norm_type)
        for pg in groups:
            dist.all_reduce(powered, group=pg)
        reduced = powered.pow(1.0 / norm_type)
        for j, i in enumerate(idx):
            norms[i] = reduced[j]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    if out_device is not None:
        total_norm = total_norm.to(out_device)
    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(f"The total norm of order {norm_type} for gradients is non-finite")

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    by_dev = defaultdict(list)
    for g in local:
        by_dev[g.device].append(g)
    for dev, gs in by_dev.items():
        coef = clip_coef_clamped.to(dev)
        if foreach is False:
            for g in gs:
                g.mul_(coef)
        else:
            torch._foreach_mul_(gs, coef)
    return total_norm
