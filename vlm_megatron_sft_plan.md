# VLM SFT on the Megatron Backend — Learning Guide & Implementation Plan

**Audience:** you have a high-level grasp of SkyRL (entrypoints, FSDP worker, SFT trainer) and zero Megatron experience. You have one node with 8× H100 and want to learn-by-doing.

**Goal:** ship VLM SFT on the Megatron backend in SkyRL, starting with **Qwen3-VL-2B-Instruct** (`Qwen/Qwen3-VL-2B-Instruct`), with a CI test that proves loss parity vs. the FSDP path.

**Why Qwen3-VL-2B over Qwen3-VL-2B or Qwen3.5-VL:**
- Recent enough to be representative of current VLM architectures (3D RoPE, native dynamic-resolution vision encoder).
- Dense (not MoE) — sidesteps the MoE+VLM upstream-immaturity trap entirely for v1.
- 2B dense fits comfortably with TP=1/PP=1 on a single H100, so iteration is fast and parity tests are cheap.
- Bridge has a full `qwen3_vl_provider.py` and the provider goes through `AutoBridge.from_hf_pretrained(hf_path)`, so any Qwen3-VL HF checkpoint works even if the recipe defaults are 8B.
- Qwen3.5-VL: bridge recipes exist but reference HF paths (`Qwen/Qwen3.5-2B`, etc.) that aren't publicly released as of writing. Stretch target — if weights are out by Phase 5, swap.

**Working style:** every phase ends with a *test you can run and pass* before moving on. When a step says "step through with `breakpoint()`", do that — single-stepping a real tensor through real code is faster and stickier than reading source. Reading is reserved for the few places where the call graph is too deep to step through productively.

**Learning goal beyond shipping the feature:** by the end you should be able to write a minimal distributed VLM-SFT trainer from scratch — not by copying SkyRL/Megatron, but by understanding *why* every layer exists. To that end the plan includes **Concept sections** between the implementation phases that explain design decisions the official docs don't cover well. Read them when you hit them; don't skip them even when you're impatient to code.

**Repo orientation (referenced throughout):**
- FSDP VLM reference: `skyrl/backends/skyrl_train/workers/model_wrapper.py:90,118-385`, `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py:181`, `skyrl/backends/skyrl_train/workers/worker.py:807,1009,1238`
- Megatron worker (text-only today): `skyrl/backends/skyrl_train/workers/megatron/{megatron_worker.py, megatron_model_wrapper.py, model_bridges.py}`
- SFT trainer: `skyrl/train/sft_trainer.py` (one class, dispatches FSDP vs Megatron at line 237)
- Existing Megatron tests: `tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_worker.py`, `test_megatron_models.py`
- Bridge VLM building blocks (already installed): `.venv/lib/python3.12/site-packages/megatron/bridge/{models/qwen_vl, recipes/qwen_vl, data/vlm_datasets}/`
- Existing example shells: `examples/train/sft/run_sft_megatron.sh` (text), `examples/train/geometry3k/run_geometry3k.sh` (FSDP VLM RL)

---

## Phase 0 — Environment & Megatron-on-8×H100 setup (1–1.5 days) — DONE

You've never run Megatron, so before any code change: get the existing text-only Megatron SFT working end-to-end on your box. That alone teaches you what the "happy path" looks like.

### 0.1 Run the existing text Megatron SFT

```bash
bash examples/train/sft/run_sft_megatron.sh \
    placement.num_gpus_per_node=8 \
    micro_train_batch_size_per_gpu=1 batch_size=8
```

This should "just work" with the `megatron` extra. If it doesn't, fix env issues *now* — none of the rest matters until this runs.

### 0.2 What the parallelism knobs mean (read, then poke)

Read **only these** docs, in this order — don't go wider yet:

1. NVIDIA Megatron-Core "Parallelisms" overview (Megatron-Core docs site, "Parallelisms" page) — focus on TP, PP, DP. Skip CP, EP, virtual PP for now.
2. Megatron-Bridge README (the `megatron-bridge` GitHub repo's README). You only need to understand: *bridge wraps Megatron-Core, exposes HF-checkpoint loading, and provides "providers" that build a Megatron model from a model family.*

Then *poke* by changing the shell knobs and observing failure modes:

| Setting | Run with 8 H100s | What you should observe |
|---|---|---|
| `tensor_model_parallel_size=1 pipeline_model_parallel_size=1` | DP=8 | All ranks load full model, all do forward/backward |
| `tensor_model_parallel_size=2 pipeline_model_parallel_size=1` | TP=2, DP=4 | Each rank holds half the attention/MLP weights |
| `tensor_model_parallel_size=1 pipeline_model_parallel_size=2` | PP=2, DP=4 | Layers split across 2 ranks; only last PP stage computes loss |
| `tensor_model_parallel_size=2 pipeline_model_parallel_size=2` | TP×PP=4, DP=2 | Combo |

`placement.num_gpus_per_node=8` and the rule `world_size = TP × PP × CP × DP` is what your runs must satisfy. SkyRL launches Megatron under Ray actors — you do not run `torchrun` yourself, the trainer's `_init_workers` does it (`sft_trainer.py:237`).

### Concept A — Why these axes exist (and what you'd reinvent if you started over)

Distributed training has the parallelism axes it has because each one solves a *different* bottleneck. You can't pick one and call it done. If you were designing this from scratch for VLM SFT, you'd discover the axes in roughly this order:

1. **Data parallel (DP).** You have N GPUs and a batch. Easiest thing: replicate the model on each GPU, give each one B/N samples, average gradients via `all_reduce` before the optimizer step. Cost: each GPU stores the *full* model + optimizer state. For a 2B model in fp32 with Adam, that's ~32 GB just for optimizer state per replica. Hits a wall fast.
2. **Sharded DP (FSDP / ZeRO).** Same logical model on every rank, but you *shard* parameters and optimizer state across DP ranks and gather them just-in-time around each forward/backward. Memory wins, comm cost grows. This is what SkyRL's FSDP path does. It scales to ~30B-class dense models on a node before you hit the next wall: a single layer's *activations* don't fit.
3. **Tensor parallel (TP).** Split a single matmul across GPUs along its hidden dim. `XW` becomes `[X W_1 | X W_2]` with a final `all_reduce`. Cuts per-GPU activation memory roughly proportionally. Communication is *bandwidth-bound and intra-step*, so TP only works well within an NVLink domain (one node). 8×H100 with NVLink is the canonical TP-friendly setup.
4. **Pipeline parallel (PP).** Split the model *by layer* across ranks. Rank 0 has layers 1–N/k, rank 1 has the next chunk, and so on. Forward activations flow rank→rank, backward gradients flow back. This breaks the "model fits on one GPU" assumption entirely. Cost: you create a *bubble* — early ranks idle while waiting for later ranks during warmup/cooldown — unless you split the batch into microbatches and pipeline them (1F1B schedule). Communication is *low-bandwidth point-to-point*, so PP scales across nodes.
5. **Sequence/context parallel (SP/CP).** Even with TP, attention's `O(S²)` activation memory dominates for long contexts. Split the sequence dim across GPUs. Comes with painful constraints around RoPE and causal masking — which is exactly why VLMs (3D RoPE) currently disable it.
6. **Expert parallel (EP).** Only relevant for MoE — distribute experts across GPUs so each holds a subset.

For VLM SFT specifically: you'll use TP+PP+DP. SP/CP/EP are out of scope for v1 (and you'll add asserts blocking them). The thing the docs don't tell you: these axes *compose multiplicatively* into a "device mesh" and every collective has to know which axis it lives on. Megatron-Core's `parallel_state` module is the global registry that tracks "which rank am I on along each axis." When you read its source later, that's the mental model.

### 0.3 First `breakpoint()` exercise — meet a Megatron model

Add `breakpoint()` inside `megatron_model_wrapper.py` right after the model is constructed (find where the bridge provider returns a model object — likely `model_bridges.py` or where it's called from `megatron_worker.py`). Run the text SFT shell with `placement.num_gpus_per_node=2` and:

```python
# at the breakpoint, on rank 0:
type(self.model)              # GPTModel? or list (vpp)?
[type(m) for m in self.model] # if list
self.model[0].config          # TransformerConfig — TP/PP/etc
[(n, p.shape) for n, p in self.model[0].named_parameters()][:10]
```

You're looking for: *what type is a Megatron model object*, *what does its config look like*, *and where do params live*. **Test for Phase 0:** you can answer those three questions without re-reading the docs.

---

## Phase 1 — Read the FSDP VLM path & run a VLM RL example (2 days) - DONE

Before adding VLM to Megatron, you need to know exactly what "VLM support" means in SkyRL — what extra tensors flow, what model type, what guards are in place.

### 1.1 Trace the FSDP VLM path with `breakpoint()`

Run an FSDP VLM example end-to-end (RL is fine, the SFT data path is a subset):

```bash
bash examples/train/geometry3k/run_geometry3k.sh placement.num_gpus_per_node=2 \
    trainer.num_episodes=1 trainer.batch_size=2
```

Then place breakpoints in:

1. `skyrl/backends/skyrl_train/workers/model_wrapper.py:118` — confirm `is_vlm` becomes True; inspect `model_config.vision_config`.
2. `model_wrapper.py:314` (forward signature) — inspect shapes of `pixel_values`, `image_grid_thw`, `mm_token_type_ids`. Write the shapes down. **You'll re-create exactly these shapes on the Megatron side.**
3. `model_wrapper.py:367-385` — see how `vlm_kwargs` is assembled and passed to `model.forward`.
4. `worker.py:807,1009,1238` — see where these tensors enter the worker from a `TrainingInputBatch`.

### 1.2 Read the FSDP VLM-only constraints

In `model_wrapper.py:324-325`, two asserts: no sample packing, no SP. Read the comment block above. **Write down in `scratch/vlm_constraints.md`** why these exist (3D RoPE, multimodal token positions). You will mirror these constraints on Megatron and reference this note in the assert error messages.

### 1.3 Phase 1 test (informal but explicit)

Open `scratch/vlm_constraints.md` and answer in your own words:
- Which tensors does VLM forward need beyond `(input_ids, attention_mask, labels)`?
- What are their dtypes/shapes for a batch with B sequences and N images?
- Why is sample packing disabled for VLMs?
- Where does the vision encoder live in HuggingFace VLM models, and how is it called from `model.forward`?

If you can't answer one, re-breakpoint into the path that exercises it.

### Concept B — SkyRL's actor model and why a "worker" is a thing

The official SkyRL docs explain *what* the worker is, not *why* the architecture is shaped this way. Here's the design rationale you need:

**The core problem an RL/SFT framework has to solve:** training and rollout (and, in RL, reward computation) want different process layouts. Training wants `world_size = TP×PP×DP` GPUs in a tight collective group with NCCL handles set up once. Rollout (vLLM) wants its *own* process group with its *own* TP layout, possibly different from training. Reward models want yet another layout. If you put all of this in one Python process with one `torch.distributed` init, you'll fight `ProcessGroup` lifecycle bugs forever.

**SkyRL's answer:** Ray actors as the unit of "a process with its own torch.distributed world." Each "worker" is a Ray actor; a `PPORayActorGroup` (in `worker.py`) is a collection of N such actors that together form one logical training cluster. The group is constructed once with a parallelism config; from that point on, calling a method on the group dispatches to all actors with the right collective semantics. You see this in `WorkerDispatch` (`worker_dispatch.py`) — it's the indirection that turns "`group.forward(batch)`" into "fan out to actors with mesh-aware sharding of the batch."

**Why this matters for your project:** when you add VLM tensors (`pixel_values`, `image_grid_thw`) to the `TrainingInputBatch`, you're not just adding fields to a dict — you're declaring "these need to participate in dispatch." The replay buffer already knows about them (`TensorList` types in `replay_buffer.py:76`) precisely because dispatch needs to know how to *split* a batch across DP ranks without breaking inside an image. A `TensorList` is a list-of-tensors-of-different-shapes that survives sharding, where a normal stacked tensor wouldn't (different images have different patch counts).

**If you wrote this from scratch:** you'd discover, the hard way, that homogeneous batches (every sample same shape) are easy and heterogeneous batches (variable image counts) require a custom collation type. SkyRL's `TensorList` is one valid answer; you might also use jagged tensors, or pad-to-max with masks. The choice affects everything downstream.

---

## Phase 2 — Stand-alone Megatron-Bridge VLM SFT (2 days, no SkyRL changes) - DONE

This phase is pure learning. You will run Qwen3-VL-2B SFT *without SkyRL in the loop*, using only `megatron-bridge`. This teaches you how a VLM is constructed in Megatron land and what data the model wants — independent of SkyRL plumbing.

### 2.1 Read targeted bridge code (with breakpoints, not just eyes)

Files to read:
- `.venv/.../megatron/bridge/recipes/qwen_vl/qwen3_vl.py` — the recipe entrypoint.
- `.venv/.../megatron/bridge/models/qwen_vl/qwen3_vl_provider.py` — model construction.
- `.venv/.../megatron/bridge/data/vlm_datasets/conversation_dataset.py` and `collate.py` — data shape.
- `.venv/.../megatron/bridge/training/finetune.py` — training loop entry.

Skim first, don't memorize. The point is to know which file does what so you can navigate.

### 2.2 Run a tiny SFT job using bridge directly

Write `scratch/bridge_vlm_sft.py` that calls the bridge recipe on a tiny dataset (e.g. 16 samples from a small VLM HF dataset). Use `tensor_model_parallel_size=1 pipeline_model_parallel_size=1` first, on 2 GPUs (DP=2).

Drop a `breakpoint()` inside the bridge `forward_step` function (find it via `grep -n "def forward_step" .venv/.../megatron/bridge/training/`) and inspect:

```python
batch.keys()                       # what fields does Megatron-bridge feed in?
batch["pixel_values"].shape
batch["image_grid_thw"].shape
batch["tokens"].shape, batch["labels"].shape
# now step into the model call:
n          # advance to the model() call
s          # step into model.forward
# inspect: how is pixel_values consumed? where does the vision encoder run?
```

**Critical observation:** does `pixel_values` exist on all PP ranks or only rank 0? Run again with `pipeline_model_parallel_size=2` and put breakpoints on both ranks (`if torch.distributed.get_rank() in (0,1): breakpoint()`). The answer determines how you'll dispatch image tensors in SkyRL.

### 2.3 Phase 2 test

Pass when you can:
1. Train Qwen3-VL-2B for 5 SFT steps on bridge-only with PP=1, TP=1, DP=2 — loss decreases.
2. Re-run with PP=2 — loss decreases.
3. Diagram (in `scratch/bridge_vlm_dataflow.md`) the path from a raw `{messages, images}` HF row to the tensors fed into `GPTModel.forward`, marking which PP rank touches which tensor.

### Concept C — Megatron's "model is a list" and the `forward_step` contract

This is the single weirdest thing about Megatron coming from HF/PyTorch and the docs gloss over it.

**In a normal PyTorch model**, `model` is one `nn.Module` and `model(batch)` returns logits. Loss is computed outside, `.backward()` is called outside, optimizer steps outside.

**In Megatron-Core**, when you have PP, `model` is a *list* of `nn.Module`s — one per *virtual pipeline stage* held by this rank. Even with PP=1 the convention is `[model]` (a one-element list). And you do not call `model(batch)` directly. You hand Megatron-Core's *pipeline scheduler* a `forward_step_func` with a fixed signature:

```python
def forward_step(data_iterator, model) -> tuple[torch.Tensor, callable]:
    batch = next(data_iterator)
    output = model(batch["tokens"], ...)        # this rank's portion of the forward
    def loss_func(output_tensor):
        return loss, {"lm_loss": loss.detach()}
    return output, loss_func
```

The scheduler then orchestrates: feed microbatch i forward through the local stage, send activations to next PP rank, receive activations from prev PP rank, eventually run `loss_func` on the *last* PP stage, then run the backward pass in reverse. The 1F1B schedule interleaves forward of microbatch k+1 with backward of microbatch k to keep all stages busy.

**Implications you'll feel directly:**

- You cannot just "call the model." You have to *describe* a forward step and let Megatron drive. Look at `bridge/training/finetune.py` for an example.
- The `loss_func` returned from forward only runs on the last PP rank. Earlier ranks return `(activations, no-op)`.
- `data_iterator` is consumed *one microbatch at a time per scheduler tick*, not "give me the whole batch." If your iterator has state, it must be PP-rank-aware.
- **For VLM specifically**: only the *first* PP rank's forward sees `pixel_values`. Subsequent ranks just receive hidden states from the prior rank — they have no idea images existed. This is why your dispatch in Phase 3.4 only sends image tensors to PP rank 0.

**If you wrote this from scratch:** you'd start with the naive "send activations rank-to-rank in a for-loop" approach, see GPU utilization tank to ~50% because of the bubble, then reinvent 1F1B. You'd also notice that gradient sync across DP needs to happen *after* the full PP backward completes — and that overlapping it with the optimizer's first parameter updates buys real wallclock. Megatron-Core's `DistributedDataParallel` and "overlap_grad_reduce" flags exist for exactly this; reading those flags' docstrings now will make sense after you've felt the pain.

---

## Phase 3 — Plumb VLM through the SkyRL Megatron worker (4–5 days) - DONE 

Now we ship code. Each sub-phase has a runnable test.

### 3.1 Detect VLM models in Megatron worker

**Read first:** `model_wrapper.py:90,118-128` (FSDP detection logic — copy it).

**Implement:** add `self.is_vlm` to `megatron_model_wrapper.py` using the same `hasattr(model_config, "vision_config")` check. Branch in the model construction path: if VLM, call the bridge VLM provider (`qwen3_vl_provider`); else the existing text path.

For now, hard-code Qwen3-VL-2B — `if "Qwen3-VL-2B" in cfg.model.path: use_vlm_provider = True`. Generalize later.

**Test 3.1** (write this *first*, then implement):
Add `tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_vlm_init.py`. Mirror `test_megatron_worker.py:get_test_actor_config` but with `model.path="Qwen/Qwen3-VL-2B-3B-Instruct"`. The test instantiates the worker and asserts `worker.model_wrapper.is_vlm == True`, `len(list(model.parameters())) > N_text_only`. **No forward step yet.** This proves the model loads on the Megatron side.

Step through with `breakpoint()` inside the new VLM branch the first time you run it; compare the constructed model against what you saw in Phase 2.

### 3.2 Pass image tensors through `TrainingInputBatch` & dispatch

**Read first:** `skyrl/train/dataset/replay_buffer.py:76,103-104` (image fields already exist) and `skyrl/backends/skyrl_train/workers/worker_utils.py:126-127`.

**Implement:**
- In `sft_trainer.py::tokenize_chat_example` and `collate_sft_batch`, extend to extract `pixel_values` / `image_grid_thw` from VLM dataset rows and put them on the `TrainingInputBatch`. Use the HF processor for Qwen3-VL-2B exactly as the FSDP VLM path does (re-breakpoint there if unsure).
- In `megatron_worker.py`'s training step entry, plumb these through to the model wrapper's forward (mirror `worker.py:1009-1020`).

**Test 3.2:** unit test that constructs a 2-sample VLM `TrainingInputBatch` (1 image each), calls the worker's *forward only* (no backward), and asserts that loss is finite. Use TP=1, PP=1, DP=2. Mirror the structure of `test_megatron_worker.py::test_forward` (find it via `grep -n "def test" test_megatron_worker.py`).

`breakpoint()` inside the forward to verify: shapes match what you saw in Phase 1.1 step 2.

### 3.3 Backward + optimizer step

Add the loss + backward for the VLM path. The Megatron-bridge `forward_step` already returns a loss function — your job is to make sure the wrapper's forward returns the same kind of loss tuple as the text path does.

**Test 3.3:** 5-step SFT smoke test (mirror `test_megatron_worker.py::test_train_step` if present; otherwise create one based on the text SFT example loop). Assert loss at step 5 < loss at step 0.

`breakpoint()` once at step 1's optimizer step and confirm vision tower params have nonzero `.grad`. If they don't, you wired the loss to ignore vision params.

### 3.4 Pipeline-parallel correctness (the landmine)

**Read first:** Megatron-Core's "Pipeline parallelism" doc page — only the section on "interleaved schedules and the role of `_micro_batch_size`". Skim, don't memorize.

#### Concept D — PP scheduling intuition you need before debugging

The naive PP schedule ("GPipe") is: forward microbatch 1 through all stages, forward 2, ..., forward M, *then* backward M, ..., backward 1. The first stage is idle for `(P-1) * t_fwd + (P-1) * t_bwd` time — the "bubble". For P=2, M=8, that's a 25% efficiency loss. For P=8, M=8, it's catastrophic.

**1F1B** ("one forward, one backward"): once stage P-1 finishes forward of microbatch 1, it immediately starts backward 1. Meanwhile stage 0 starts forward of microbatch (P+1). Each stage alternates F and B in steady state. Bubble is the same in absolute terms but you can amortize it across more microbatches without quadratically growing memory.

**Why this matters when debugging:** if your VLM loss diverges between PP=1 and PP=2, you might suspect a numerics bug. But the more common cause is *activation recomputation* interacting with the PP schedule — Megatron defaults to recomputing activations during backward, which means your vision encoder's forward runs *twice* on PP rank 0 unless you opt it out. Before chasing numerics, check `recompute_granularity` and `recompute_method` in the `TransformerConfig`.

The other landmine: PP point-to-point sends use `torch.distributed.P2POp` and require both peers to post matched send/recv in the same order. If your VLM forward conditionally returns different shapes depending on whether images are present, you'll deadlock the second microbatch. Solution: make the forward shape-stable regardless of whether the current microbatch has images (pad/no-op the vision tower for image-less microbatches, or batch homogeneously).

The vision encoder lives only on the first PP stage. Two failure modes to guard against:
1. `pixel_values` materialized on all PP ranks (wastes memory, may even error on non-stage-0 ranks).
2. Loss on last PP stage doesn't propagate gradients back to the vision encoder on stage 0.

**Implement:** in the worker's per-microbatch dispatch, only attach `pixel_values` / `image_grid_thw` to the batch sent to PP rank 0. The bridge VLM provider should already handle the gradient flow — verify, don't recreate.

**Test 3.4:** re-run the test from 3.3 with `pipeline_model_parallel_size=2`. Loss should decrease at the same rate (within noise) as PP=1.

`breakpoint()` on both PP ranks (use `if mpu.get_pipeline_model_parallel_rank() == X: breakpoint()`); confirm rank 0 sees `pixel_values`, rank 1 does not.

### 3.5 Mirror the FSDP guards

Add asserts in the Megatron VLM forward path matching `model_wrapper.py:324-325`:
- `assert not use_sample_packing, "..."`
- `assert sequence_parallel_size == 1, "..."`
- `assert context_parallel_size == 1, "..."` (extra Megatron-only one)

Reference `scratch/vlm_constraints.md` in the message text.

**Test 3.5:** parametrized test that constructs a VLM config with each of those flags enabled and asserts the worker raises `AssertionError` with the expected substring.

---

## Phase 4 — Loss parity vs. FSDP (1.5 days) - DONE

This is the test that proves correctness, not just "it runs."

### 4.1 Build the parity harness

**Read first:** any existing parity tests — `grep -rn "fsdp.*megatron\|parity" tests/` to find prior art. If none exist for SFT, model the structure on `test_megatron_extractor_consistency.py`.

**Implement** `tests/.../megatron/test_vlm_sft_parity.py`:
- Same model, same fixed batch (1 sample, 1 image), same seed, fp32.
- Run 5 forward+backward steps on FSDP (`strategy=fsdp2`) and Megatron (`strategy=megatron`, TP=1, PP=1, DP=1).
- Assert per-step loss matches within `rtol=1e-3, atol=1e-4`.

Use `Qwen/Qwen3-VL-2B-3B-Instruct` if it fits, otherwise the smallest VL HF model that exercises both LM and vision tower. Document the choice in the test's docstring.

### 4.2 If parity fails

Single most useful debugger move: `breakpoint()` at the *first* forward, dump `logits.sum()`, `logits[0,0,:5]`, `loss.item()` from both paths. Diverges at logits → model construction bug. Diverges only at loss → loss masking / ignore-index mismatch. Diverges only after step 1 → optimizer config mismatch (Megatron uses Adam impl from Apex, double-check betas/eps).

### 4.3 Phase 4 test

The parity test passes in CI on 1 GPU. That is the deliverable.

---

## Phase 5 — Parallelism sweep, checkpointing, example, docs (2–3 days)

### 5.1 Parallelism sweep

Add a parametrized integration test that runs 5 steps under each combo on 8 GPUs:

| TP | PP | DP | Notes |
|----|----|----|-------|
| 1  | 1  | 8  | pure DP baseline |
| 2  | 1  | 4  | TP only |
| 1  | 2  | 4  | PP only — exercises 3.4 |
| 2  | 2  | 2  | combo |

Loss-decrease assertion is enough; full parity is in 4.

### 5.2 Checkpoint round-trip

**Read first:** how the existing Megatron text path saves/loads — search for `save_checkpoint` in `megatron_worker.py` and follow into bridge.

**Test:** save after step 3, reload into a fresh worker, run step 4, assert resumed loss matches uninterrupted run within tolerance. Crucially, **inspect** the checkpoint dir to confirm vision-tower weights are present. `breakpoint()` after save and `os.listdir` the dir.

### 5.3 Example shell + docs

- `examples/train/sft/run_sft_megatron_vlm.sh` — copy `run_sft_megatron.sh`, swap to a small VLM HF dataset (e.g. `HuggingFaceM4/the_cauldron` subset), set `model.path=Qwen/Qwen3-VL-2B-3B-Instruct`, and document the parallelism settings.
- Update `examples/train/sft/README.md` with a VLM section + the constraints from `scratch/vlm_constraints.md`.
- Update `docs/content/docs/tutorials/vision_language_rl.mdx` and `docs/content/docs/examples/megatron.mdx` to remove "VLM is FSDP-only" caveats and link to the new shell.

### 5.4 Final acceptance test

```bash
bash examples/train/sft/run_sft_megatron_vlm.sh placement.num_gpus_per_node=8 \
    megatron_config.tensor_model_parallel_size=2 \
    megatron_config.pipeline_model_parallel_size=2
```

Loss decreases over 50 steps, no OOM, checkpoint saves and reloads.

---

## Concept E — Gradient sync, optimizer state, and what FSDP and Megatron actually share

The single most useful realization for understanding both frameworks: under the hood they're solving the same three problems, just with different tradeoffs.

**Problem 1: parameters don't fit per GPU.** FSDP shards parameters across DP ranks and gathers them just-before-forward, releases them just-after-backward (per layer or per group). Megatron shards parameters along the TP/PP axes and *keeps them sharded* — there's no gather, the math itself is rewritten to operate on shards. FSDP optimizes for "I just want my single-GPU model to scale"; Megatron optimizes for "I want the absolute minimum memory and comm cost for a fixed model."

**Problem 2: gradients have to be synchronized.** With pure DP, every rank has a full gradient and you `all_reduce` once per step. With FSDP, gradients are sharded the same way params are, so it's a `reduce_scatter`. With Megatron's TP, gradients along TP-sharded dims need a TP-group `all_reduce`; gradients along PP-sharded dims don't sync (different ranks own different params). Megatron's DP-group gradient sync is its own pass *on top* of TP/PP — DP groups are "ranks at the same TP/PP coords across replicas."

**Problem 3: optimizer state.** Adam needs 2× param-size of state (m, v), often in fp32 even for bf16 models. FSDP shards optimizer state along DP automatically. Megatron has its own "distributed optimizer" that shards optimizer state across DP, independent of the param/grad sharding. You'll see this as a flag — turn it on for >2B models or you'll OOM.

**The thing that ties this back to your VLM project:** the vision tower is small (~600M params for Qwen3-VL-2B). It will live entirely on PP rank 0. Its gradients sync only across the *DP group at PP=0*, not across all DP ranks globally — because PP rank 1+ doesn't have these params. If you ever see "DP rank N has a different param count than rank 0," that's expected for VLM+PP and the optimizer needs to handle it. Bridge handles it; you just need to know not to be surprised.

**If you wrote this from scratch:** start with all-reduce-after-backward DP. Realize you OOM on optimizer state. Add ZeRO-1 (shard optimizer). Realize forward activations OOM. Add ZeRO-3 (shard params, that's FSDP). Realize a single attention block's activations don't fit. Add TP. Realize you can't fit even one layer of a 70B model on one node's worth of TP. Add PP. Each step is a response to a specific bottleneck — the design is *forced*, not arbitrary.

---

## Concept F — vLLM weight sync (preview for the RL extra)

You're not doing this for SFT, but it's the next thing you'll touch if you do the VLM-RL extra, and it surfaces a design pattern worth knowing.

In RL, the policy that *generates* rollouts (vLLM, optimized for throughput, flat weight layout) is a different process than the policy that *trains* (Megatron, sharded weight layout). After every training step you need to copy updated weights from the trainer into the inference engine. Naively this means: gather all sharded params on rank 0, materialize the full state dict, ship it to vLLM. That's an OOM on big models.

The trick: vLLM exposes a `collective_rpc` to update specific weight tensors in place, and the Megatron side ships *one tensor at a time*, gathering shards just for that tensor before sending. The "metadata" you'll see referenced in recent commits (`0cad388e`, `125588ea`) is the mapping between Megatron's sharded names and vLLM's flat names — including the vision tower's params, which you'll need to verify exist when you add VLM-RL.

For your VLM SFT scope this is just useful background; come back to it if you do the RL extra.

---

## If you had to write this from scratch — minimal blueprint

By Phase 5 you should be able to sketch this in <300 lines of pseudocode without looking. As a self-test for the end of the project:

```
class MiniVLMTrainer:
    # 1. Process group setup
    init_torch_distributed()
    build_device_mesh(tp=2, pp=2, dp=2)   # Concept A
    parallel_state.set_groups(...)

    # 2. Model construction (Concept C)
    if is_first_pp_rank: build vision_tower + lm_layers[0:N//PP]
    elif is_last_pp_rank: build lm_layers[(PP-1)*N//PP:] + lm_head
    else:                  build lm_layers[i*N//PP:(i+1)*N//PP]
    shard_params_along_TP(model)

    # 3. Optimizer (Concept E)
    optimizer = DistributedAdam(local_params, dp_group=mesh.dp)

    # 4. Data (Concept B)
    for batch in dataloader:
        microbatches = split(batch, num_microbatches)
        if not is_first_pp_rank:
            for mb in microbatches: mb.pop("pixel_values")  # Concept D

        # 5. PP-scheduled fwd/bwd (Concept C, D)
        loss = pipeline_1f1b(forward_step_func, microbatches, model)

        # 6. Gradient sync (Concept E)
        all_reduce(grads, group=mesh.tp)        # tp-replicated grads
        reduce_scatter(grads, group=mesh.dp)    # dp-sharded
        optimizer.step()
```

If you can fill in each line of that pseudocode with the actual primitive (which `torch.distributed` collective, which Megatron-Core helper, which bridge utility), you've internalized enough to call this project a success.

---

## Open the PR

PR title: `[train] VLM SFT support on Megatron backend (Qwen3-VL-2B)`
PR body must list: (a) which constraints carry over from FSDP VLM, (b) parity-test result, (c) parallelism combos validated, (d) what's *not* supported (CP, SP, sample packing, MoE+VLM).

---

## Extras / Follow-ups (in priority order)

1. **Second VLM family.** Add Gemma3-VL (`recipes/gemma3_vl`) or GLM-VL via the same path. Forces the Qwen-specific code into a clean abstraction. ~3–5 days.
2. **VLM RL on Megatron.** With SFT done, RL is mostly: extend `worker.py::actor_forward_step` and `compute_log_probs_step` Megatron branches with the same `pixel_values`/`image_grid_thw` plumbing you just added; confirm vLLM weight sync covers the vision tower (recent commits `0cad388e`, `125588ea` touched metadata maps — verify vision params are included). Re-run `geometry3k` / `visgym` examples on Megatron. ~1.5–2 weeks.
3. **LoRA on the VLM.** SkyRL has Megatron LoRA for text (`run_megatron_lora_qwen3-0.6b.sh`). Extend to freezing-vision-tower / projector-only LoRA. ~3–4 days.
4. **Sequence/context parallelism for VLM.** Hard. 3D RoPE means you can't naively split sequence dim. Real research problem and likely needs upstream Megatron-Core changes. Punt unless someone asks.
5. **Mixed text+image microbatches.** Today the path effectively requires homogeneous batches. Heterogeneous batching improves throughput on real datasets. Medium-hard.
6. **MoE+VLM.** Avoid until both mature upstream.

---

## Cheatsheet — when to `breakpoint()` vs. read

**Use `breakpoint()`:** any time you need shapes, dtypes, device placement, what's-on-which-rank, or "does this branch fire". This is 80% of the work.

**Read source:** (a) when the call graph crosses 5+ files and stepping is too slow, (b) when reading a config dataclass is faster than constructing one, (c) when the file is small (<200 lines) and central to your task.

**Read docs:** narrowly and on-demand — Parallelisms overview before Phase 0.2, PP scheduling before Phase 3.4, that's it. Don't pre-read the whole Megatron-Core docs site; you'll forget 90% before you need it.
