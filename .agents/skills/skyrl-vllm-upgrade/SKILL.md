---
name: skyrl-vllm-upgrade
description: End-to-end procedure for bumping SkyRL's pinned vLLM version (pyproject + uv.lock, torch/flashinfer coupling, runtime patches under patches/vllm, weight-sync / sleep / HTTP API breakages) and validating it with focused Anyscale GPU jobs. Use when asked to upgrade or bump vLLM, check whether a vLLM release is safe for SkyRL, or fix fallout from a vLLM bump.
---

# Bumping vLLM in SkyRL

A vLLM bump touches four layers, and each previous bump broke at least one of them in a way
that did not show up as an import error. Work through them in order; the GPU jobs at the end
are what actually validate the bump.

Read first: `.agents/docs/dependency_bumps.md`, `.agents/docs/inference.md`,
`.agents/docs/weight_sync.md`, `.agents/docs/ci.md`.

Prior bump PRs worth skimming for their failure modes (`gh pr view <n> --repo NovaSky-AI/SkyRL`):
#2114 (0.28: torch held below vLLM's pin, `packed` moved to init info), #2470 (0.31: flashinfer
per-arch JIT cache split, R3 sleep/AuxOutput), #2557 + #2582 (0.32: patch retirement, cumem
pools, `max_tokens` clamping), and the PR that added this skill (0.33).

## 1. Dependencies

1. Diff vLLM's own requirements between the old and new release (skip extras):
   ```bash
   for v in OLD NEW; do curl -s https://pypi.org/pypi/vllm/$v/json | python3 -c \
     "import json,sys; print('\n'.join(sorted(r for r in json.load(sys.stdin)['info']['requires_dist'] if 'extra ==' not in r)))" > /tmp/vllm_$v.txt; done
   diff /tmp/vllm_OLD.txt /tmp/vllm_NEW.txt
   ```
   The lines that matter: `torch`, `torchvision`, `flashinfer-python`, `transformers`,
   `nvidia-cutlass-dsl`, `quack-kernels`, `tilelang`, `apache-tvm-ffi`, `xgrammar`. Every one of
   them has an entry in `[tool.uv] override-dependencies` with a comment that names the vLLM
   version -- update the comments even when the pin does not move.
2. **torch.** SkyRL's CUDA extension wheels (flash-attn FA2+FA4 combined wheel, causal-conv1d,
   mamba-ssm, fast-hadamard-transform, transformer-engine-torch, flash-mla) are built against
   one exact torch and hosted on `NovaSky-AI/skyrl-wheels` releases. If vLLM moves torch, the
   default is to **hold torch** with the override and keep `torchvision` pinned to the release
   that matches the held torch (torchvision is not stable-ABI). Moving torch is its own PR:
   it means rebuilding all of those wheels.
   Holding torch is only safe if vLLM's compiled extensions load against the older torch.
   Check statically before spending GPU time -- see `scripts/check_vllm_torch_abi.sh`:
   every `vllm/**/*.so` must import zero `c10::`/`at::`/`torch::` C++ symbols (stable ABI),
   and every `aoti_torch_*` shim symbol it imports must be exported by the held torch's
   `libtorch_cpu.so` (CUDA-only shim symbols live in `libtorch_cuda.so`; compare them against
   the previous vLLM wheel instead). In 0.28 `third_party/deep_gemm/_C` was not stable-ABI and
   silently fell back; in 0.33 it is.
   Python-level use of new-torch APIs is not covered by this check -- only the GPU jobs are.
3. **flashinfer.** python, cubin and jit-cache must move together (flashinfer hard-errors on
   a cubin version mismatch). Since 0.7 the jit-cache is a meta package over per-arch
   `flashinfer-jit-cache-sm*` packages on the cu130 index; sm110a/sm121a are aarch64-only.
   Check the new version exists on both indexes before relocking:
   ```bash
   curl -sL https://flashinfer.ai/whl/cu130/flashinfer-jit-cache-sm90a/ | grep -o '0\.[0-9.]*post[0-9]*' | sort -u
   curl -sL https://flashinfer.ai/whl/flashinfer-cubin/ | grep -o 'flashinfer_cubin-[^"]*whl' | sort -u
   ```
4. Relock with the uv version that keeps `revision = 3` (`uv` >= 0.13 rewrites to revision 5
   and touches every marker), then check that only the intended packages changed:
   ```bash
   uvx --from uv==0.12.0 uv lock
   git diff uv.lock | grep -E '^[-+]name = |^[-+]version = ' | paste - - | sort | uniq
   ```
   Resolution with the cross-platform forks takes 10+ minutes; run it in the background.

## 2. Runtime patches (`skyrl/backends/skyrl_train/patches/vllm/`)

Unpack the old and new wheels side by side (python files only are enough) and `diff -u`
each patch target. For every patch decide: keep / update docstring / fix / remove.

- Text-match or source-anchor patches **silently stop applying** when the anchor moves.
- A backport whose upstream PR landed must be removed -- and check that its "already fixed"
  probe would actually have noticed (in 0.32 the routed-experts rebind probe looked for an
  attribute upstream never used, so it would have double-applied).
- CPU tests under `tests/backends/skyrl_train/patches/vllm/` stub the patched function's
  collaborators; new upstream calls in those functions need stubs too.
- Patches are installed from `inference_servers/new_inference_worker_wrap.py` and the
  `vllm.general_plugins` entry points in `pyproject.toml`.

Record the per-patch verdict table in the PR description (see #2557 for the format).

## 3. Integration points that broke before

| Surface | What broke | Where it is tested |
|---|---|---|
| Weight-transfer engine ABCs / init infos | 0.28 moved `packed` into `NCCLWeightTransferInitInfo` (default False): trainer packs, worker unpacks one-by-one, **NCCL hangs** instead of erroring | `inference_servers/test_weight_sync.py`, `test_policy_local_engines_e2e.py` |
| Vendored `weight_sync/sharded_rdt/*` | Imports vLLM internals; check each `from vllm...` still resolves and the upstream originals' changes | same, `sharded_rdt` params; megatron `test_megatron_policy_weight_sync[sharded_rdt_*]` |
| Sleep / wake (`skyrl_sleep_for_weight_sync`) | 0.32 added `runtime` / `cudagraph` cumem pools; offloading only `kv_cache` caused illegal memory access after wake | `test_offload_kv_weight_sync.py` |
| `/sleep` with routed-experts capture (R3) | 0.31 AuxOutput connector rejected resets unless `pause(mode=keep)` | `megatron/test_router_replay.py` (`h100` marker) |
| OpenAI API behavior | 0.32 clamps `max_tokens` to the context instead of 400 | `inference_servers/test_new_inference_generation.py -k context_length` |
| LoRA loading | `PEFTHelper.validate_legal`, `_validate_moe_lora_format`, in-memory adapter patch | `test_lora.py`, `inference_servers/test_multi_lora_serving.py`, `patches/vllm` GPU tests |
| Speculative decoding (MTP) | draft weights must be resynced and drafting must continue after a sync | `megatron/test_mtp_weight_sync.py`, `test_mtp_weight_roundtrip.py` |

## 4. Validation

Do not rely on the label-gated CI for the first pass: the full suites take hours and mix in
training-only failures. Run the focused slice on the CI cloud first.

Anyscale: default CLI credentials must be in the NovaSky org (CI cloud
`sky-anyscale-aws-us-east-1`, compute config `l4_ci` = 4xL4). Submit from the repo root of
the bump branch (the working dir is uploaded):

```bash
export HF_TOKEN=...   # or have it in .env.ray
bash .agents/skills/skyrl-vllm-upgrade/scripts/submit_focused.sh fsdp <tag>      # cpu_vllm inference weight_sync mooncake
bash .agents/skills/skyrl-vllm-upgrade/scripts/submit_focused.sh megatron <tag>  # megatron weight sync, sharded_rdt, MTP, LoRA
bash .agents/skills/skyrl-vllm-upgrade/scripts/submit_focused.sh fsdp <tag> inference   # rerun one group
```

Track jobs by the printed `prodjob_...` ID (names resolve to the most recent job). Read
results with `anyscale job logs --id <id> | grep -A20 '== SUMMARY'`. A failed job with zero
runs is a capacity failure, not a test failure -- resubmit.

Every failure must be compared against `main` before it is attributed to the bump: rerun the
same group on a `main` checkout when unsure.

After the focused jobs are green, apply the full labels (`run_gpu_ci`, `run_megatron_gpu_ci`,
and `run_h100_gpu_ci` for R3 / FP8) on the PR.

## 5. PR description

Include: the package table (from -> to, where), why torch did or did not move, the override
changes, the per-patch verdict table, every code change with the upstream change that forced
it, behavior changes users will see, and a results table per focused group (pass / fail /
skip, with each failure triaged as bump / pre-existing / infra).
