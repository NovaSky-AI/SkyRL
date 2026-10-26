"""Sample DAPO prompts from a DeepSeek-V4.1 checkpoint with vLLM and report sampled tokens whose logprob is
non-finite or at vLLM's -9999 floor (what SkyRL's rollout-vs-train logprob metric spikes on).

Usage (one 8-GPU node, no weight sync involved):
  python probe_vllm_logprobs_v41.py <model_dir> <dapo.parquet> <out.jsonl> [--prompts 128] [--n 12] [--max-tokens 2048]

The bad tokens seen so far are all the first response token, i.e. a NaN logits row from prefill, so
``--max-tokens 1`` measures the rate on many more requests per minute. Engine toggles bisect the cause:
``--max-num-seqs``, ``--enforce-eager``, ``--max-num-batched-tokens``, ``--no-chunked-prefill``,
``--attention-backend``, ``--engram-gpu``. ``--poison-kv`` fills every KV cache tensor with 0xFF bytes (NaN in fp8
e4m3 and bf16) first: if the rate jumps, attention reads slots that were never written. ``--replay <bad.jsonl>``
resubmits the (prompt, sample) pairs of an earlier run's bad tokens, alone (``--replay-alone``) or in their batch.
"""

import argparse
import json
import math
import time

import pandas as pd
from vllm import LLM, SamplingParams


def _poison_kv_cache(worker) -> int:
    """Fill this worker's KV cache tensors with 0xFF bytes; returns how many tensors were filled."""
    import torch

    runner = getattr(worker, "model_runner", worker)
    caches = getattr(runner, "kv_caches", None)
    seen, count = set(), 0

    def fill(obj):
        nonlocal count
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.data_ptr() not in seen:
                seen.add(obj.data_ptr())
                obj.view(torch.uint8).fill_(255)
                count += 1
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                fill(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                fill(item)

    fill(caches)
    torch.cuda.synchronize()
    return count


def _install_nan_hooks(worker) -> int:
    """Forward hooks on every module: per forward pass, remember the first module (execution order) whose output
    holds NaN, with the NaN row indices. Needs eager mode (hooks do not run inside captured graphs)."""
    import torch

    model = getattr(getattr(worker, "model_runner", worker), "model", None)
    state = {"pass": 0, "seen": None, "events": []}
    worker._nan_probe_state = state

    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from tensors(item)

    def make_hook(name):
        def hook(module, inputs, output):
            if state["seen"] is not None:
                return
            for t in tensors(output):
                if t.is_floating_point() and t.numel() and torch.isnan(t).any():
                    rows = torch.isnan(t.reshape(t.shape[0], -1)).any(-1).nonzero().flatten()
                    in_nan = any(
                        x.is_floating_point() and x.numel() and bool(torch.isnan(x).any()) for x in tensors(inputs)
                    )
                    state["seen"] = name
                    state["events"].append(
                        {
                            "pass": state["pass"],
                            "module": name,
                            "type": type(module).__name__,
                            "shape": list(t.shape),
                            "nan_rows": rows[:16].tolist(),
                            "n_nan_rows": int(rows.numel()),
                            "input_has_nan": in_nan,
                        }
                    )
                    return

        return hook

    def pre_model(module, inputs):
        state["pass"] += 1
        state["seen"] = None

    model.register_forward_pre_hook(pre_model)
    count = 0
    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(make_hook(name))
            count += 1
    return count


def _zero_compressor_latent(worker) -> str:
    """Make DeepseekCompressor.forward allocate its latent with zeros instead of torch.empty: it writes only the
    group-boundary rows, so the rest keep whatever the caching allocator held. Patches this worker process."""
    import torch

    import vllm.models.deepseek_v41.compressor as compressor

    class _TorchWithZeroEmpty:
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def empty(*args, **kwargs):
            return torch.zeros(*args, **kwargs)

    compressor.torch = _TorchWithZeroEmpty()
    return "patched"


def _install_nan_origin_hooks(worker) -> int:
    """Forward hooks recording every module that turns clean inputs into a NaN output (skipping the compressor,
    whose non-boundary latent rows are unwritten by design). Needs eager mode."""
    import torch

    model = getattr(getattr(worker, "model_runner", worker), "model", None)
    state = {"pass": 0, "events": []}
    worker._nan_probe_state = state

    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from tensors(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                yield from tensors(item)

    def has_nan(obj):
        return any(t.is_floating_point() and t.numel() and bool(torch.isnan(t).any()) for t in tensors(obj))

    def make_hook(name):
        def hook(module, args, kwargs, output):
            if name.endswith("compressor") or not has_nan(output) or has_nan(args) or has_nan(kwargs):
                return
            out = next(t for t in tensors(output) if t.is_floating_point() and bool(torch.isnan(t).any()))
            rows = torch.isnan(out.reshape(out.shape[0], -1)).any(-1).nonzero().flatten()
            state["events"].append(
                {
                    "pass": state["pass"],
                    "module": name,
                    "type": type(module).__name__,
                    "shape": list(out.shape),
                    "nan_rows": rows[:16].tolist(),
                    "n_nan_rows": int(rows.numel()),
                }
            )

        return hook

    def pre_model(module, inputs):
        state["pass"] += 1

    model.register_forward_pre_hook(pre_model)
    count = 0
    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(make_hook(name), with_kwargs=True)
            count += 1
    return count


def _disable_aux_streams(worker) -> int:
    """Set every module's aux_stream_list to None, so maybe_execute_in_parallel runs both halves on the default
    stream (tests for a missing cross-stream dependency). Returns how many modules were changed."""
    model = getattr(getattr(worker, "model_runner", worker), "model", None)
    count = 0
    for _, module in model.named_modules():
        for attr in ("aux_stream_list", "aux_stream", "alt_stream"):
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
                count += 1
    return count


def _zero_attn_out(worker) -> int:
    """Patch every V4.1 attention class's _alloc_attn_out (torch.empty, filled in place by the attention op) to
    torch.zeros, so rows or padded heads the op leaves unwritten read as 0 instead of allocator garbage."""
    import torch

    model = getattr(getattr(worker, "model_runner", worker), "model", None)
    patched = set()
    for _, module in model.named_modules():
        cls = type(module)
        if hasattr(cls, "_alloc_attn_out") and cls not in patched:

            def _alloc_attn_out(self, num_tokens, hidden_states):
                return torch.zeros(
                    (num_tokens, self.padded_heads, self.head_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )

            cls._alloc_attn_out = _alloc_attn_out
            patched.add(cls)
    return len(patched)


NAN_FUNC_MODULES = (
    "vllm.models.deepseek_v41.attention",
    "vllm.v1.attention.backends.mla.sparse_swa",
    "vllm.v1.attention.backends.mla.sparse_indexer",
    "vllm.v1.attention.backends.mla.indexer",
)


def _install_nan_func_hooks(worker) -> int:
    """Wrap the functions and class methods defined in NAN_FUNC_MODULES (plus vLLM functions they import). After
    each call, report it if a tensor argument or output now holds NaN that none of its tensor arguments held
    before the call (catches kernels writing in place). Needs eager mode."""
    import functools
    import importlib
    import inspect

    import torch

    model = getattr(getattr(worker, "model_runner", worker), "model", None)
    state = {"pass": 0, "events": []}
    worker._nan_probe_state = state

    def tensors(obj, depth=0):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif depth < 2 and isinstance(obj, (list, tuple)):
            for item in obj:
                yield from tensors(item, depth + 1)
        elif depth < 2 and isinstance(obj, dict):
            for item in obj.values():
                yield from tensors(item, depth + 1)

    def nan_any(obj):
        return any(t.is_floating_point() and t.numel() and bool(torch.isnan(t).any()) for t in tensors(obj))

    def wrap(qualname, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            before = nan_any(args) or nan_any(kwargs)
            out = fn(*args, **kwargs)
            if not before and (nan_any(out) or nan_any(args) or nan_any(kwargs)):
                where = "output" if nan_any(out) else "in-place argument"
                state["events"].append({"pass": state["pass"], "func": qualname, "where": where})
            return out

        wrapper._nan_wrapped = True
        return wrapper

    wrapped = 0
    for mod_name in NAN_FUNC_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in list(vars(mod).items()):
            if getattr(obj, "_nan_wrapped", False):
                continue
            if inspect.isfunction(obj) and (obj.__module__ or "").startswith("vllm"):
                setattr(mod, name, wrap(f"{obj.__module__}.{name}", obj))
                wrapped += 1
            elif inspect.isclass(obj) and obj.__module__ == mod_name:
                for attr, member in list(vars(obj).items()):
                    if (
                        inspect.isfunction(member)
                        and not attr.startswith("__")
                        and not getattr(member, "_nan_wrapped", False)
                    ):
                        setattr(obj, attr, wrap(f"{mod_name}.{obj.__name__}.{attr}", member))
                        wrapped += 1

    def pre_model(module, inputs):
        state["pass"] += 1

    model.register_forward_pre_hook(pre_model)
    return wrapped


def _fill_prefill_workspace(worker, value) -> str:
    """Wrap the workspace manager the V4.1 FlashMLA prefill path gathers KV into, filling the gathered-KV buffer
    (the first tensor of get_simultaneous) with `value` (0.0 or NaN) before every use."""
    import vllm.models.deepseek_v41.nvidia.flashmla as flashmla

    original = flashmla.current_workspace_manager

    class _Filled:
        def __init__(self, manager):
            self._manager = manager

        def __getattr__(self, name):
            return getattr(self._manager, name)

        def get_simultaneous(self, *specs):
            tensors = self._manager.get_simultaneous(*specs)
            tensors[0].fill_(value)
            return tensors

    flashmla.current_workspace_manager = lambda: _Filled(original())
    return f"filled with {value}"


def _zero_prefill_workspace(worker) -> str:
    return _fill_prefill_workspace(worker, 0.0)


def _nan_prefill_workspace(worker) -> str:
    return _fill_prefill_workspace(worker, float("nan"))


def _apply_skyrl_workspace_patch(worker) -> bool:
    """Install SkyRL's real fix (patches/vllm/patch_deepseek_v41_prefill_workspace.py) in this worker."""
    from skyrl.backends.skyrl_train.patches.vllm.patch_deepseek_v41_prefill_workspace import (
        apply_deepseek_v41_prefill_workspace_patch,
    )

    return apply_deepseek_v41_prefill_workspace_patch()


def _collect_nan_events(worker):
    return getattr(worker, "_nan_probe_state", {"events": []})["events"]


def _is_bad(lp) -> bool:
    return lp is None or not math.isfinite(lp) or lp <= -9999.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--prompts", type=int, default=128)
    p.add_argument("--prompt-offset", type=int, default=0)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--poison-kv", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--attention-backend", default="FLASHMLA_SPARSE_DSV41", help="'default' leaves vLLM's choice")
    p.add_argument("--engram-gpu", action="store_true")
    p.add_argument("--repeats", type=int, default=1, help="generate the same batch this many times")
    p.add_argument("--replay", default=None, help="bad.jsonl from an earlier run: resubmit only those prompts")
    p.add_argument("--replay-alone", action="store_true", help="with --replay: one request per generate call")
    p.add_argument("--zero-compressor-latent", action="store_true", help="torch.zeros for the compressor latent")
    p.add_argument("--prefill-workspace", default=None, help="pre-fill gathered-KV workspace: zero, nan or a float")
    p.add_argument("--skyrl-workspace-patch", action="store_true", help="install SkyRL's prefill workspace fix")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dump-all", default=None, help="write every request's first-token (prompt, token, logprob) here")
    p.add_argument("--nan-func-hooks", action="store_true", help="report functions that create NaN (eager)")
    p.add_argument("--zero-attn-out", action="store_true", help="torch.zeros for the attention output buffer")
    p.add_argument("--no-aux-streams", action="store_true", help="run aux-stream work on the default stream")
    p.add_argument("--nan-origin-hooks", action="store_true", help="report modules turning clean inputs into NaN")
    p.add_argument(
        "--nan-hooks", action="store_true", help="report the first module emitting NaN (needs --enforce-eager)"
    )
    args = p.parse_args()

    df = pd.read_parquet(args.data).iloc[args.prompt_offset : args.prompt_offset + args.prompts]
    engine_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=2048 + args.max_tokens + 512,
        gpu_memory_utilization=args.gpu_memory_utilization,
        logprobs_mode="processed_logprobs",
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    if args.attention_backend != "default":
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.max_num_seqs:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.no_chunked_prefill:
        engine_kwargs["enable_chunked_prefill"] = False
    if args.engram_gpu:
        engine_kwargs["engram_config"] = {"cpu_offload": False}
    print("PROBE engine kwargs:", engine_kwargs, flush=True)
    llm = LLM(**engine_kwargs)
    if args.poison_kv:
        print("PROBE poisoned KV tensors per rank:", llm.collective_rpc(_poison_kv_cache), flush=True)
    if args.zero_compressor_latent:
        print("PROBE zero compressor latent:", llm.collective_rpc(_zero_compressor_latent), flush=True)
    if args.skyrl_workspace_patch:
        print("PROBE skyrl workspace patch:", llm.collective_rpc(_apply_skyrl_workspace_patch), flush=True)
    if args.prefill_workspace:
        value = {"zero": 0.0, "nan": float("nan")}.get(args.prefill_workspace)
        value = float(args.prefill_workspace) if value is None else value
        print("PROBE prefill workspace:", llm.collective_rpc(_fill_prefill_workspace, args=(value,)), flush=True)
    if args.nan_func_hooks:
        print("PROBE nan func hooks per rank:", llm.collective_rpc(_install_nan_func_hooks), flush=True)
    if args.zero_attn_out:
        print("PROBE zero attn_out classes per rank:", llm.collective_rpc(_zero_attn_out), flush=True)
    if args.no_aux_streams:
        print("PROBE aux streams disabled per rank:", llm.collective_rpc(_disable_aux_streams), flush=True)
    if args.nan_origin_hooks:
        print("PROBE nan origin hooks per rank:", llm.collective_rpc(_install_nan_origin_hooks), flush=True)
    if args.nan_hooks:
        print("PROBE nan hooks per rank:", llm.collective_rpc(_install_nan_hooks), flush=True)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
        for _, row in df.iterrows()
    ]
    params = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, logprobs=1, seed=args.seed
    )

    batches = [list(range(len(prompts)))]
    if args.replay:
        wanted = sorted({json.loads(line)["prompt"] - args.prompt_offset for line in open(args.replay)})
        batches = [[i] for i in wanted] if args.replay_alone else [wanted]
        print(f"PROBE replaying {len(wanted)} prompts ({'alone' if args.replay_alone else 'together'})", flush=True)

    n_tokens = n_bad = n_requests = 0
    dump = open(args.dump_all, "w") if args.dump_all else None
    with open(args.out, "w") as f:
        for repeat in range(args.repeats):
            for batch in batches:
                start = time.time()
                outputs = llm.generate([prompts[i] for i in batch], params, use_tqdm=False)
                for pi, out in zip(batch, outputs):
                    for si, comp in enumerate(out.outputs):
                        n_requests += 1
                        if dump is not None and comp.token_ids:
                            first = comp.logprobs[0].get(comp.token_ids[0]) if comp.logprobs else None
                            dump.write(
                                json.dumps(
                                    [
                                        repeat,
                                        pi + args.prompt_offset,
                                        si,
                                        comp.token_ids[0],
                                        None if first is None else first.logprob,
                                    ]
                                )
                                + "\n"
                            )
                        for pos, (tid, lp_dict) in enumerate(zip(comp.token_ids, comp.logprobs)):
                            n_tokens += 1
                            entry = lp_dict.get(tid) if lp_dict else None
                            lp = entry.logprob if entry is not None else None
                            if _is_bad(lp):
                                n_bad += 1
                                f.write(
                                    json.dumps(
                                        {
                                            "repeat": repeat,
                                            "prompt": pi + args.prompt_offset,
                                            "sample": si,
                                            "pos": pos,
                                            "prompt_len": len(out.prompt_token_ids),
                                            "token": tid,
                                            "logprob": None if lp is None else repr(lp),
                                            "len": len(comp.token_ids),
                                            "finish": comp.finish_reason,
                                        }
                                    )
                                    + "\n"
                                )
                if len(batches) == 1:
                    print(f"PROBE repeat {repeat}: {time.time() - start:.0f}s, bad so far {n_bad}", flush=True)
    if dump is not None:
        dump.close()
    if args.nan_hooks or args.nan_origin_hooks or args.nan_func_hooks:
        events = llm.collective_rpc(_collect_nan_events)
        with open(args.out + ".nan_events.json", "w") as f:
            json.dump(events, f)
        from collections import Counter

        for rank, rank_events in enumerate(events):
            first = Counter(e.get("module") or f'{e["func"]} [{e["where"]}]' for e in rank_events)
            print(f"PROBE rank {rank}: {len(rank_events)} NaN passes; first modules {first.most_common(6)}", flush=True)
    print(
        f"PROBE requests={n_requests} tokens={n_tokens} bad={n_bad} "
        f"bad_per_request={n_bad / max(1, n_requests):.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
