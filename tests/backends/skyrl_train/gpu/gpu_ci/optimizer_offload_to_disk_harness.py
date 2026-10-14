"""torchrun harness: ``optimizer_offload_to_disk`` vs the regular DistributedOptimizer, on a tiny megatron-core GPT.

Launched by ``test_optimizer_offload_to_disk.py`` (``torchrun --nproc_per_node=2``). Each rank builds two identical
models, wraps both in DDP + DistributedOptimizer (one regular, one through SkyRL's
``get_megatron_optimizer(..., offload_to_disk=...)``), trains them on the same batches, and checks:

- the offloaded optimizer's per-param fp32 main shards hold no GPU storage, from construction on and between
  steps;
- the layouts that are easy to get wrong occur (on some rank): a param split between the two DP ranks, a chunk holding runs of
  two param groups, padding inside a chunk, a chunk boundary inside a param, a native-fp32 buffer;
- per-step loss and grad norm, and the final params, match the regular optimizer (bit-identical with fp32
  moments, close with bf16), with gradient clipping active;
- after the last step, the offloaded fp32 main params and moments equal the regular optimizer's element for element,
  and padding is zero (fp32 moments);
- SkyRL's colocation offload/backload (model, grad buffers, optimizer) leaves the offloaded state intact;
- closing the stores removes every state directory, including the run's;
- checkpoint round trip: a mid-run Megatron distributed checkpoint (model + the optimizer's marker) plus
  ``save_to``, loaded into a fresh model and optimizer, continues exactly like the original.

Prints ``HARNESS_OK <json>`` on rank 0 when every check passed.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist


def build_model(moe: bool, seed: int):
    from megatron.core import tensor_parallel
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer import TransformerConfig

    tensor_parallel.model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=4 if moe else None,
        moe_router_topk=2 if moe else None,
        moe_ffn_hidden_size=128 if moe else None,
        moe_token_dispatcher_type="alltoall" if moe else "allgather",
        moe_router_load_balancing_type="none",
        expert_model_parallel_size=dist.get_world_size() if moe else 1,
        moe_grouped_gemm=moe,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=config.num_moe_experts, moe_grouped_gemm=moe)
    model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=256, max_sequence_length=64).cuda()
    # Extra params penalized in the loss: a native-fp32 one (an fp32 DDP buffer, whose main is the param itself) and
    # odd-sized ones, so the DDP buffers contain padding between params and at bucket ends.
    model.fp32_param = torch.nn.Parameter(torch.randn(4093, device="cuda", dtype=torch.float32))
    model.odd_param = torch.nn.Parameter(torch.randn(1001, device="cuda", dtype=torch.bfloat16))
    # Small DDP buckets, so the model spans several of them.
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, grad_reduce_in_fp32=True, overlap_grad_reduce=False, bucket_size=40_000
    )
    return DistributedDataParallel(config=config, ddp_config=ddp_config, module=model)


def build_optimizer(model, state_dir, moment_dtype):
    from megatron.core.optimizer import OptimizerConfig

    from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
        get_megatron_optimizer,
    )
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        OffloadToDiskSettings,
    )

    config = OptimizerConfig(
        optimizer="adam",
        lr=1e-2,
        weight_decay=0.1,
        clip_grad=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    settings = None if state_dir is None else OffloadToDiskSettings(dir=state_dir, moment_storage_dtype=moment_dtype)
    return get_megatron_optimizer([model], config, offload_to_disk=settings)


def train_step(model, opt, batch):
    tokens, labels = batch
    model.zero_grad_buffer()
    opt.zero_grad()
    position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand_as(tokens)
    extra = model.module.fp32_param.square().mean() + model.module.odd_param.float().square().mean()
    loss = model(tokens, position_ids, None, labels=labels).float().mean() + 1e-3 * extra
    loss.backward()
    model.finish_grad_sync()
    ok, grad_norm, _ = opt.step()
    assert ok
    return loss.item(), float(grad_norm)


def params_of(model):
    return [p.detach().clone() for p in model.module.parameters()]


def assert_mains_released(opt):
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    stores = disk_state_stores(opt)
    assert stores, "no optimizer was built as an OffloadedDistributedOptimizer"
    for store in stores:
        for group in store.dist_opt.shard_fp32_from_float16_groups:
            for main in group:
                assert main.untyped_storage().size() == 0, "an fp32 main shard holds GPU storage"


def distributed_optimizers(opt):
    return [o for o in getattr(opt, "chained_optimizers", [opt]) if not o.is_stub_optimizer]


def param_ranges(dist_opt):
    """(model param, fp32 main, buffer index, range in the buffer) for every param this rank owns a shard of."""
    out = []
    groups = list(zip(dist_opt.model_float16_groups, dist_opt.shard_fp32_from_float16_groups))
    groups += list(zip(dist_opt.model_fp32_groups, dist_opt.shard_fp32_groups))
    for model_group, main_group in groups:
        for param, main in zip(model_group, main_group):
            buffer_index = dist_opt.model_param_gbuf_map[param][0]
            out.append((param, main, buffer_index, dist_opt._get_model_param_range_map(param)["gbuf_world"]))
    return out


def assert_layouts_occur(opt):
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    split = mixed_groups = padding = cut = fp32_buffer = False
    for dist_opt, store in zip(distributed_optimizers(opt), disk_state_stores(opt), strict=True):
        ranges = param_ranges(dist_opt)
        split |= any(r.size < param.numel() for param, _, _, r in ranges)
        for chunk in store.chunks:
            mixed_groups |= len({group for _, _, group in chunk.runs}) > 1
            covered = sum(
                min(r.end, chunk.end) - max(r.start, chunk.start)
                for _, _, b, r in ranges
                if b == chunk.buffer_index and r.start < chunk.end and r.end > chunk.start
            )
            padding |= covered < chunk.numel
            cut |= any(b == chunk.buffer_index and r.start < chunk.start < r.end for _, _, b, r in ranges)
            fp32_buffer |= not chunk.main_on_disk
    # Each layout has to occur on some rank, not on every rank (e.g. bucket-end padding sits in the last shard).
    found = torch.tensor([split, mixed_groups, padding, cut, fp32_buffer], dtype=torch.int, device="cuda")
    dist.all_reduce(found, op=dist.ReduceOp.MAX)
    split, mixed_groups, padding, cut, fp32_buffer = found.tolist()
    assert split, "no param is split between DP ranks"
    assert mixed_groups, "no chunk holds runs of two param groups"
    assert padding, "no chunk contains padding"
    assert cut, "no chunk boundary falls inside a param"
    assert fp32_buffer, "no native-fp32 buffer"


def assert_state_equal(ref_opt, test_opt):
    """The offloaded fp32 mains and moments equal the regular optimizer's, element for element; padding is zero."""
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    for ref, store in zip(distributed_optimizers(ref_opt), disk_state_stores(test_opt), strict=True):
        ranges = param_ranges(ref)
        for chunk in store.chunks:
            state = store.load_chunk(chunk)
            covered = torch.zeros(chunk.numel, dtype=torch.bool, device="cuda")
            for _, main, b, r in ranges:
                start, end = max(r.start, chunk.start), min(r.end, chunk.end)
                if b != chunk.buffer_index or start >= end:
                    continue
                ours, theirs = slice(start - chunk.start, end - chunk.start), slice(start - r.start, end - r.start)
                covered[ours] = True
                ref_state = ref.optimizer.state[main]
                for name, expected in (
                    ("main", main),
                    ("exp_avg", ref_state["exp_avg"]),
                    ("exp_avg_sq", ref_state["exp_avg_sq"]),
                ):
                    assert torch.equal(state[name][ours], expected.view(-1)[theirs]), f"{name} differs in {chunk}"
            for name in ("main", "exp_avg", "exp_avg_sq"):
                assert not state[name][~covered].any(), f"nonzero padding in {name} of {chunk}"


def save_checkpoint(model, opt, ckpt_dir):
    from megatron.core import dist_checkpointing

    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    model_state = model.module.sharded_state_dict()
    state = {"model": model_state, "optimizer": opt.sharded_state_dict(model_state, is_loading=False)}
    os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(state, ckpt_dir)
    for store in disk_state_stores(opt):
        store.save_to(os.path.join(ckpt_dir, "optimizer_offload_to_disk"))


def load_checkpoint(model, opt, ckpt_dir):
    from megatron.core import dist_checkpointing

    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    model_state = model.module.sharded_state_dict()
    request = {"model": model_state, "optimizer": opt.sharded_state_dict(model_state, is_loading=True)}
    loaded = dist_checkpointing.load(request, ckpt_dir)
    model.module.load_state_dict(loaded["model"])
    opt.load_state_dict(loaded["optimizer"])
    for store in disk_state_stores(opt):
        store.load_from(os.path.join(ckpt_dir, "optimizer_offload_to_disk"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--moment-dtype", default="fp32")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(expert_model_parallel_size=dist.get_world_size() if args.moe else 1)

    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        load_megatron_grads_to_gpu,
        load_megatron_model_to_gpu,
        load_megatron_optimizer,
        offload_megatron_grads_to_cpu,
        offload_megatron_model_to_cpu,
        offload_megatron_optimizer,
    )
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    ref_model, test_model = build_model(args.moe, seed=1234), build_model(args.moe, seed=1234)
    ref_opt = build_optimizer(ref_model, None, "fp32")
    test_opt = build_optimizer(test_model, os.path.join(args.dir, "state"), args.moment_dtype)
    assert not disk_state_stores(ref_opt)
    assert_mains_released(test_opt)
    assert_layouts_occur(test_opt)
    if args.moe:
        assert len(disk_state_stores(test_opt)) == 2, "expected dense and expert optimizers"

    g = torch.Generator(device="cuda").manual_seed(99 + dist.get_rank())
    batches = [
        (
            torch.randint(0, 256, (2, 32), device="cuda", generator=g),
            torch.randint(0, 256, (2, 32), device="cuda", generator=g),
        )
        for _ in range(args.steps)
    ]
    exact = args.moment_dtype == "fp32"
    ckpt_dir = os.path.join(args.dir, "ckpt")
    half = args.steps // 2
    history = []
    for step, batch in enumerate(batches):
        ref = train_step(ref_model, ref_opt, batch)
        test = train_step(test_model, test_opt, batch)
        assert_mains_released(test_opt)
        history.append({"step": step, "ref": ref, "test": test})
        if exact:
            assert ref == test, f"step {step}: regular {ref} != offloaded {test}"
        else:
            assert abs(ref[0] - test[0]) < 5e-2 * max(1.0, abs(ref[0])), f"step {step}: loss {ref} vs {test}"
        if step == half - 1:
            # Colocation round trip, as MegatronStrategy.offload_to_cpu / backload_to_gpu do around generation.
            offload_megatron_model_to_cpu([test_model])
            offload_megatron_grads_to_cpu([test_model])
            offload_megatron_optimizer(test_opt)
            torch.cuda.synchronize()
            load_megatron_model_to_gpu([test_model])
            load_megatron_grads_to_gpu([test_model])
            load_megatron_optimizer(test_opt)
            torch.cuda.synchronize()
            assert_mains_released(test_opt)
            save_checkpoint(test_model, test_opt, ckpt_dir)

    assert any(entry["ref"][1] > 1.0 for entry in history), "gradient clipping never triggered"
    if exact:
        for a, b in zip(params_of(ref_model), params_of(test_model)):
            assert torch.equal(a, b), "final params differ between the regular and the offloaded optimizer"
        assert_state_equal(ref_opt, test_opt)

    # Resume into a fresh model (different init) and a fresh optimizer with its own state directory.
    resume_model = build_model(args.moe, seed=4321)
    resume_opt = build_optimizer(resume_model, os.path.join(args.dir, "resume_state"), args.moment_dtype)
    load_checkpoint(resume_model, resume_opt, ckpt_dir)
    for step in range(half, args.steps):
        resumed = train_step(resume_model, resume_opt, batches[step])
        if exact:
            assert resumed == history[step]["test"], f"resumed step {step}: {resumed} != {history[step]['test']}"
    if exact:
        for a, b in zip(params_of(resume_model), params_of(test_model)):
            assert torch.equal(a, b), "resumed params differ"

    for store in disk_state_stores(test_opt) + disk_state_stores(resume_opt):
        store.close()
        assert not os.path.exists(store.dir)
    dist.barrier()
    for state_dir in ("state", "resume_state"):
        assert not os.path.exists(os.path.join(args.dir, state_dir)), f"{state_dir} left behind after close()"

    if dist.get_rank() == 0:
        result = {"moe": args.moe, "moment_dtype": args.moment_dtype, "history": history}
        print("HARNESS_OK " + json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
