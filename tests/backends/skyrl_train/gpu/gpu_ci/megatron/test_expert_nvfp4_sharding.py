"""expert_nvfp4 under Megatron TP / EP / CP / DP on 8 GPUs (one node; skipped otherwise), with whole experts
per rank (expert TP 1, the only layout expert_nvfp4 supports). PP's layer assignment is covered in
``test_expert_nvfp4.py``.

Per layout, one Ray actor per GPU builds a small GPT MoE, installs expert_nvfp4 (w4a4) as the Megatron worker does, runs one
forward + backward, and checks, bit for bit:

  weights   each rank's rounded expert weights == FlashInfer's fp4_quantize grid of those weights, dequantized
  fc1 / fc2 in   each rank's rounded expert-GEMM inputs == the per-token round trip of its own raw rows

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_expert_nvfp4_sharding.py
"""

import os
import socket
import time

import pytest
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

LAYOUTS = ["tp1-ep8", "tp2-ep4", "tp4-ep2", "tp2-ep2-cp2", "tp1-ep2-cp2", "tp1-ep4"]


def parse(name):
    d = dict(tp=1, ep=1, pp=1, cp=1)
    for part in name.split("-"):
        k = part.rstrip("0123456789")
        d[k] = int(part[len(k) :])
    return d


@ray.remote(num_gpus=1, num_cpus=4)
class W:
    def run(self, rank, world, addr, port, par, mode, block_scale):
        import torch
        import torch.distributed as dist

        # gpu_ci's ray_init_fixture exports NVTE_FUSED_ATTN=0 job-wide; this model uses attention_backend=auto,
        # whose Megatron check needs the variable unset (it then sets it itself).
        os.environ.pop("NVTE_FUSED_ATTN", None)
        from megatron.core import parallel_state as mpu

        if torch.cuda.get_device_capability()[0] < 10:
            return None
        from megatron.core import tensor_parallel

        os.environ["LOCAL_RANK"] = "0"
        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl", init_method=f"tcp://{addr}:{port}", rank=rank, world_size=world, device_id=torch.device("cuda:0")
        )
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=par["tp"],
            pipeline_model_parallel_size=par["pp"],
            context_parallel_size=par["cp"],
            expert_model_parallel_size=par["ep"],
            expert_tensor_parallel_size=1,
        )
        from megatron.core.distributed import (
            DistributedDataParallel,
            DistributedDataParallelConfig,
        )
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.transformer import TransformerConfig

        from skyrl.backends.skyrl_train.workers.megatron.quantization import (
            expert_nvfp4 as q,
        )

        E, H, F = 8, 256, 256
        tensor_parallel.model_parallel_cuda_manual_seed(0)
        torch.manual_seed(0)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=H,
            ffn_hidden_size=512,
            num_attention_heads=8,
            num_moe_experts=E,
            moe_router_topk=2,
            moe_ffn_hidden_size=F,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            gradient_accumulation_fusion=False,
            tensor_model_parallel_size=par["tp"],
            pipeline_model_parallel_size=par["pp"],
            context_parallel_size=par["cp"],
            expert_model_parallel_size=par["ep"],
            expert_tensor_parallel_size=1,
            sequence_parallel=par["tp"] > 1,
            pipeline_dtype=torch.bfloat16,
        )
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=E, moe_grouped_gemm=True)
        pre, post = mpu.is_pipeline_first_stage(), mpu.is_pipeline_last_stage()
        model = GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=512,
            max_sequence_length=512,
            pre_process=pre,
            post_process=post,
        ).cuda()
        model = DistributedDataParallel(config=config, ddp_config=DistributedDataParallelConfig(), module=model)

        q.install_grouped_gemm_hook(block_scale, fp4_forward=mode == "fp4fwd")
        q.register_routed_expert_weights([model], torch.bfloat16)
        # As the worker: the pre-hook rounds (and with fp4_forward packs).
        q.register_routed_expert_input_rounding([model], block_scale, pack_fp4=mode == "fp4fwd")
        res = {"rank": rank, "host": socket.gethostname(), "mismatch": {}}

        # (1) weights: the hook's round trip == FlashInfer's bytes of these weights, dequantized
        from skyrl.backends.skyrl_train.kernels.nvfp4_quant import fake_experts

        linears = q.routed_expert_grouped_linears([model])
        wm = 0
        for name, mod in linears:
            ws = [w.detach() for w in mod._get_weight_tensors()]
            ours = q.fake_nvfp4_quantize_weights(ws, block_scale)
            ref = fake_experts.reference_round_trip(torch.stack(ws), block_scale)
            wm += sum(int((o.view(torch.int16) != r.view(torch.int16)).sum()) for o, r in zip(ours, ref.unbind(0)))
        res["mismatch"]["weights"] = wm
        res["linears"] = len(linears)

        # (2, 3) expert inputs: capture the raw input before QAT's rounding and the rounded one after
        # the module (pre-hook, in-place fp4 write-back).
        captured = []
        for name, mod in linears:
            is_fc2 = name.endswith("linear_fc2")

            def pre(m, args, is_fc2=is_fc2):
                m._qat_raw = args[0].detach().clone()
                m._qat_in = args[0]

            def post(m, args, out, is_fc2=is_fc2):
                captured.append((is_fc2, m._qat_raw, m._qat_in.detach().clone()))

            mod.register_forward_pre_hook(pre, prepend=True)
            mod.register_forward_hook(post)

        # One microbatch through the pipeline schedule.
        from megatron.core.pipeline_parallel import get_forward_backward_func

        seq = 256
        g = torch.Generator(device="cuda").manual_seed(1)
        tokens = torch.randint(0, 512, (2, seq), device="cuda", generator=g)
        if par["cp"] > 1:
            cp_r, cp_n = mpu.get_context_parallel_rank(), par["cp"]
            # Megatron's load-balanced CP split: chunks r and 2n-1-r of 2n.
            chunks = tokens.chunk(2 * cp_n, dim=1)
            tokens_local = torch.cat([chunks[cp_r], chunks[2 * cp_n - 1 - cp_r]], dim=1)
            pos_chunks = torch.arange(seq, device="cuda").chunk(2 * cp_n)
            pos = torch.cat([pos_chunks[cp_r], pos_chunks[2 * cp_n - 1 - cp_r]])[None].expand_as(tokens_local)
        else:
            tokens_local = tokens
            pos = torch.arange(seq, device="cuda")[None].expand_as(tokens)

        def fwd_step(data_iter, m):
            out = m(input_ids=tokens_local, position_ids=pos, attention_mask=None)

            def loss_fn(o):
                loss = o.float().square().mean()
                return loss, {"loss": loss.detach()}

            return out, loss_fn

        losses = get_forward_backward_func()(
            forward_step_func=fwd_step,
            data_iterator=iter([None]),
            model=[model],
            num_microbatches=1,
            seq_length=seq,
            micro_batch_size=2,
            forward_only=False,
        )
        res["loss_finite"] = all(bool(torch.isfinite(x["loss"])) for x in losses) if losses else None
        res["grads_finite"] = all(bool(torch.isfinite(p.main_grad).all()) for p in model.parameters())

        am = {"fc1": 0, "fc2": 0}
        for is_fc2, raw, rounded in captured:
            ref = q.fake_nvfp4_quantize_activations(raw, block_scale)
            am["fc2" if is_fc2 else "fc1"] += int((rounded != ref).sum())
        res["mismatch"].update(am)
        res["captured"] = len(captured)
        dist.destroy_process_group()
        return res


def _run_layout(name, mode, block_scale):
    pg = placement_group([{"GPU": 8, "CPU": 32}], strategy="PACK")
    ray.get(pg.ready(), timeout=1800)
    try:
        ws = [
            W.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=0)).remote()
            for _ in range(8)
        ]
        addr = ray.get(ws[0].__ray_call__.remote(lambda self: ray.util.get_node_ip_address()))
        port = 20000 + int(time.time() * 1000) % 20000
        return ray.get([w.run.remote(i, 8, addr, port, parse(name), mode, block_scale) for i, w in enumerate(ws)])
    finally:
        remove_placement_group(pg)


@pytest.mark.megatron
@pytest.mark.parametrize("mode", ["prehook", "fp4fwd"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_expert_nvfp4_sharding_matches_unsharded(ray_init_fixture, layout, mode):
    gpus = ray.cluster_resources().get("GPU", 0)
    if gpus < 8:
        pytest.skip("needs 8 GPUs")
    out = _run_layout(layout, mode, "four_over_six")
    if out[0] is None:
        pytest.skip("NVFP4 needs Blackwell (SM100+)")
    for k in ("weights", "fc1", "fc2"):
        assert sum(o["mismatch"][k] for o in out) == 0, (layout, mode, k)
    assert all(o["loss_finite"] in (True, None) for o in out) and all(o["grads_finite"] for o in out)
