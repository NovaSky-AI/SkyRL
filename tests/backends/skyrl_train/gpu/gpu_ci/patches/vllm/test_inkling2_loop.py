"""Looped Inkling2 checkpoints (``loop_config``) on SkyRL's vLLM model (``patches/vllm/inkling2``).

On the looped synthetic checkpoint (``synthetic_checkpoint.write_checkpoint(looped=True)``: 12 layers, core = layers
3-8 run 3 times, controllers scaled so the messages move the output) vLLM's prompt logprobs must match
``reference.py`` at TP 1 and TP 2, and the same comparison against a reference with a single-site loop bug (k / v
message queries swapped; two passes instead of three) must fail by a wide margin. Every expert is selected per token,
so routing cannot flip on rounding noise.

R3: with routed-expert capture on, each token gets ``num_layer_visits`` rows of experts, one per layer run in
execution order (prelude, core pass 0, 1, 2, coda), the order the reference routes in (checked on a copy of the
checkpoint routing to 2 of 8 experts).

Weight sync: new controllers, cross-pass queries and a core layer's attention weights loaded in place into the
running engine (CUDA graphs on) give the reference's logprobs for the new weights, in every pass.

Needs a Blackwell GPU and the Inkling2 vLLM build (``pyproject.toml``); TP 2 needs two GPUs.

    uv run --isolated --extra dev --extra megatron pytest -s \\
        tests/backends/skyrl_train/gpu/gpu_ci/patches/vllm/test_inkling2_loop.py
"""

import importlib.util
import json
import multiprocessing
import os
import shutil

import pytest
import torch

if importlib.util.find_spec("vllm.models.inkling2") is None:
    pytest.skip("needs a vLLM with Inkling2 (vllm.models.inkling2)", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("Inkling2 on vLLM needs an SM100 GPU", allow_module_level=True)

from tests.backends.skyrl_train.gpu.gpu_ci.patches.megatron.inkling2 import (  # noqa: E402
    synthetic_checkpoint as synth,
)

pytestmark = pytest.mark.b200

# vLLM's attention kernels need d_rel >= 16 and a relative extent (local window) that is a multiple of 128; the
# synthetic checkpoint's are 8 and 16, so the test redraws those tensors. Prompts are longer than the window.
PROMPT_LENS = (150, 300)
D_REL, WINDOW = 16, 128
# The special tokens vLLM's Inkling renderer requires of the tokenizer (the stand-in tokenizer lacks them).
INKLING_SPECIAL_TOKENS = [
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|content_thinking|>",
    "<|content_xml|>",
    "<|content_invoke_tool_json|>",
    "<|end_message|>",
]
# Mean / max |delta logprob| over prompt tokens. vLLM's kernels drift from the reference by ~0.009 on the 7-layer
# unlooped checkpoint and ~0.036 over the looped one's 24 layer runs (TP 1); the per-run partial-sum error grows
# steadily from the first layer on, with no step at a pass boundary. The loop bugs below cost >= 0.08.
THRESHOLDS = {"unlooped": (0.015, 0.08), "looped": (0.05, 0.3)}


def _write(path: str, looped: bool = True, topk=None) -> str:
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.configuration import (
        release_config,
    )

    os.makedirs(path, exist_ok=True)
    # vLLM's Inkling renderer needs a tokenizer; prompts are token ids below the synthetic vocabulary's size.
    synth.write_checkpoint(path, tokenizer="Qwen/Qwen3-0.6B", looped=looped)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens({"additional_special_tokens": INKLING_SPECIAL_TOKENS})
    tokenizer.save_pretrained(path)
    _widen_rel(path)
    config_path = os.path.join(path, "config.json")
    config = release_config(json.load(open(config_path)))  # the vLLM build reads only the release names
    config["text_config"].update(d_rel=D_REL, sliding_window_size=WINDOW, rel_extent=WINDOW)
    if topk is not None:
        config["text_config"]["num_experts_per_tok"] = topk
    json.dump(config, open(config_path, "w"))
    return path


def _widen_rel(path: str) -> None:
    """Redraw the relative-position tensors at ``D_REL`` x ``WINDOW``."""
    from safetensors.torch import load_file, save_file

    file = os.path.join(path, "model.safetensors")
    tensors = load_file(file)
    g = torch.Generator().manual_seed(1)
    for name in list(tensors):
        if name.endswith(".attn.wr_du.weight"):
            tensors[name] = (0.02 * torch.randn(synth.HEADS * D_REL, synth.H, generator=g)).bfloat16()
        elif name.endswith(".attn.rel_l2_clip.scale"):
            tensors[name] = (1 + 0.3 * torch.randn(synth.HEADS, D_REL, generator=g)).bfloat16()
        elif name.endswith(".attn.rel_logits_proj.proj"):
            tensors[name] = (0.3 * torch.randn(D_REL, WINDOW, generator=g)).bfloat16()
    save_file(tensors, file, metadata={"format": "pt"})


def _prompts(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, synth.UNPADDED_V, (n,), generator=g).tolist() for n in PROMPT_LENS]


def _reference(path: str, prompts, bug: str | None = None):
    """Per-prompt reference logprobs of the prompt tokens, and expert ids per layer run ``[runs][T, k]``."""
    from skyrl.backends.skyrl_train.patches.megatron.inkling2 import reference as R

    ref = R.load_reference(path, [torch.device("cuda", 0)])
    read = R.MessageBank.read
    if bug == "swap_kv_query":
        R.MessageBank.read = lambda self, l, m: read(self, l, {"k": "v", "v": "k"}.get(m, m))
    if bug == "two_passes":
        ref.config.loop_config = dict(ref.config.loop_config, num_passes=2)
    try:
        out = []
        for ids in prompts:
            routing = []
            with torch.no_grad():
                hidden = ref.hidden_states(torch.tensor(ids), routing=routing)
                lp = torch.log_softmax(ref.logits(hidden, torch.float32), -1)
            tgt = torch.tensor(ids[1:], device=lp.device)[:, None]
            out.append((lp[:-1].gather(-1, tgt)[:, 0].cpu(), routing))
        return out
    finally:
        R.MessageBank.read = read
        del ref
        torch.cuda.empty_cache()


def _engine_job(path, tp, prompts, enforce_eager, routed, reload_from, queue):
    """In a fresh process: vLLM prompt logprobs (and routed experts), optionally again after an in-place reload of
    the tensors in ``reload_from`` (a safetensors file)."""
    try:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        llm = LLM(
            model=path,
            tensor_parallel_size=tp,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.3,
            max_model_len=512,
            max_num_batched_tokens=512,
            # vLLM's routed-expert capture requires prefix caching; otherwise off, so that the prompts after an
            # in-place reload do not reuse KV computed with the old weights.
            enable_prefix_caching=routed,
            limit_mm_per_prompt={"image": 0},
            enable_return_routed_experts=routed,
        )
        params = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False, temperature=0)
        inputs = [TokensPrompt(prompt_token_ids=p) for p in prompts]

        def run():
            res = []
            for prompt, o in zip(prompts, llm.generate(inputs, params, use_tqdm=False)):
                lps = [o.prompt_logprobs[t][prompt[t]].logprob for t in range(1, len(prompt))]
                experts = o.outputs[0].routed_experts
                res.append((torch.tensor(lps), None if experts is None else torch.as_tensor(experts)))
            return res

        if routed:  # record each MoE call's own routing, in call order, next to vLLM's capture

            def install(model):
                from skyrl.backends.skyrl_train.patches.vllm.inkling2.ops import (
                    linear_k_groups,
                    select_experts,
                )

                model._skyrl_calls = []

                def hook(moe, args):
                    gate = moe.gate
                    x = args[0].contiguous()
                    _, ids = select_experts(linear_k_groups(x, gate.weight), gate.bias, gate.topk, gate.route_scale)
                    model._skyrl_calls.append(ids.cpu())

                for layer in model.model.layers:
                    layer.mlp.register_forward_pre_hook(hook)

            llm.apply_model(install)
        results = {"before": run()}
        if routed:
            results["calls"] = llm.apply_model(lambda model: model._skyrl_calls)[0]
        if reload_from is not None:

            def load(model):
                from safetensors.torch import load_file

                model.load_weights(list(load_file(reload_from, device="cuda").items()))

            llm.apply_model(load)
            results["after"] = run()
        queue.put(results)
    except BaseException as e:  # noqa: BLE001
        import traceback

        queue.put(RuntimeError(traceback.format_exc()))
        raise e


def _engine(path, tp=1, prompts=None, enforce_eager=True, routed=False, reload_from=None):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_engine_job, args=(path, tp, prompts, enforce_eager, routed, reload_from, queue))
    proc.start()
    result = queue.get()
    proc.join()
    if isinstance(result, Exception):
        raise result
    return result


def _errors(vllm_lps, ref):
    d = torch.cat([(v - r).abs() for v, (r, _) in zip(vllm_lps, ref)])
    return d.mean().item(), d.max().item()


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("inkling2_loop")
    return {
        "looped": _write(str(root / "looped")),
        "unlooped": _write(str(root / "unlooped"), looped=False),
        "top2": _write(str(root / "top2"), topk=2),
    }


@pytest.mark.parametrize("tp", [1, 2])
def test_loop_matches_reference(ckpts, tp):
    if tp > torch.cuda.device_count():
        pytest.skip(f"needs {tp} GPUs")
    prompts = _prompts()
    stats = {}
    for kind in ("unlooped", "looped"):
        ref = _reference(ckpts[kind], prompts)
        out = _engine(ckpts[kind], tp, prompts)["before"]
        stats[kind] = _errors([lp for lp, _ in out], ref)
        if kind == "looped":
            for bug in ("swap_kv_query", "two_passes"):
                stats[bug] = _errors([lp for lp, _ in out], _reference(ckpts[kind], prompts, bug))
    print(f"tp={tp}", {k: tuple(round(x, 5) for x in v) for k, v in stats.items()})
    for kind, (mean, worst) in THRESHOLDS.items():
        assert stats[kind][0] < mean and stats[kind][1] < worst, (kind, stats[kind])
    # vLLM is much closer to the correct reference than to a reference with a loop bug.
    assert stats["swap_kv_query"][0] > 1.5 * stats["looped"][0], stats
    assert stats["two_passes"][0] > 4 * stats["looped"][0], stats


def test_routed_experts_per_layer_run(ckpts):
    """R3 capture: ``num_layer_visits`` rows per token, in execution order."""
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.configuration import (
        layer_run_order,
    )

    prompts = _prompts(1)
    ref = _reference(ckpts["top2"], prompts)
    result = _engine(ckpts["top2"], 1, prompts, routed=True)
    out, calls = result["before"], result["calls"]
    num_layers, _, _, loop = synth.shape(True)
    runs = layer_run_order(loop, num_layers)
    # One MoE call per layer run and forward (eager: unpadded batches); the scheduler may split the prompts over
    # forwards, whose rows follow each other in prompt order.
    assert len(calls) % len(runs) == 0, len(calls)
    forwards = [calls[f : f + len(runs)] for f in range(0, len(calls), len(runs))]
    calls = torch.cat([torch.stack([c.sort(-1).values for c in fwd], 1) for fwd in forwards])  # [tokens, runs, k]
    assert calls.shape[0] >= sum(len(p) for p in prompts)
    # The fork's cache index order (every layer's first pass, then the later passes): a wrong order.
    cache_order = sorted(range(len(runs)), key=lambda r: (runs[r][1], runs[r][0]))
    start, own, wrong, prelude = 0, [], [], []
    for prompt, (_, experts), (_, routing) in zip(prompts, out, ref):
        assert experts is not None and experts.shape[1:] == (len(runs), 2), tuple(experts.shape)
        n = len(prompt) - 1  # the last prompt token's row may follow the sampled token's
        captured = experts[:n].long().sort(-1).values
        recorded = calls[start : start + n]
        start += len(prompt)
        own.append((captured == recorded).all(-1).float().mean().item())
        wrong.append((captured[:, cache_order] == recorded).all(-1).float().mean().item())
        # Against the reference only the prelude: past it rounding drift flips top-2-of-8 choices of this random model.
        expected = torch.stack([r[:n].sort(-1).values for r in routing[: loop["core_start"]]], 1)
        prelude.append((captured[:, : loop["core_start"]] == expected).all(-1).float().mean().item())
    print("capture vs own routing", own, "in cache order", wrong, "prelude vs reference", prelude)
    assert min(own) == 1.0 and max(wrong) < 0.6 and min(prelude) > 0.95


def test_in_place_reload(ckpts, tmp_path):
    """New loop weights (controllers, cross-pass queries) and a core layer's attention weights, loaded in place."""
    from safetensors.torch import load_file, save_file

    src = ckpts["looped"]
    new = str(tmp_path / "reloaded")
    shutil.copytree(src, new)
    tensors = load_file(os.path.join(src, "model.safetensors"))
    g = torch.Generator().manual_seed(7)
    changed = {}
    for name, t in tensors.items():
        if ".controllers." in name or "cross_pass" in name or (
            name.startswith("model.llm.layers.4.attn.") and name.endswith(("wq_du.weight", "wk_dv.weight", "rel_logits_proj.proj"))
        ):
            changed[name] = (t.float() * (1 + 0.5 * torch.randn(t.shape, generator=g))).to(t.dtype)
    tensors.update(changed)
    save_file(tensors, os.path.join(new, "model.safetensors"), metadata={"format": "pt"})
    update = str(tmp_path / "update.safetensors")
    save_file(changed, update)
    prompts = _prompts(2)
    out = _engine(src, 1, prompts, enforce_eager=False, reload_from=update)
    before = _errors([lp for lp, _ in out["before"]], _reference(src, prompts))
    after = _errors([lp for lp, _ in out["after"]], _reference(new, prompts))
    stale = _errors([lp for lp, _ in out["after"]], _reference(src, prompts))
    print("reload", {"before": before, "after": after, "after_vs_old": stale})
    mean, worst = THRESHOLDS["looped"]
    assert before[0] < mean and after[0] < mean and after[1] < worst
    assert stale[0] > 3 * after[0]
