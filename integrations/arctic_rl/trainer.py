"""ArcticPPOTrainer — subclass of RayPPOTrainer for the Arctic RL backend.

Overrides ``build_models()`` to route training operations (forward,
backward, optimizer step, weight sync) to the Arctic RL server via the
``arctic_platform.rl`` client.

The server owns the full GRPO computation:
  - per-token log-probs (old + new)
  - clipped PPO surrogate loss + backward
  - optimizer step + grad-norm + lr-schedule

The client (this file) is responsible for:
  - rollout generation (via ArcticGenerator → server vLLM)
  - reward scoring (via skyrl-gym)
  - group-relative advantage estimation (delegated to SkyRL's
    ``compute_advantages_and_returns`` — outcome-only, no value model)
  - **wire-protocol shaping**: translating SkyRL's batch layout to the
    server's expected `(input_ids, attention_mask, prompts, responses,
    response_mask, position_ids)` + `meta` dict. This mirrors the
    validated verl adapter at
    ``verl/workers/remote_client/arctic_rl.py``; the server is
    framework-agnostic and consumes the same payload from either.

Dependencies:
    arctic_platform  — on-prem-and-dss-platform Arctic RL client
"""

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    TrainingOutputBatch,
)
from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.trainer import RayPPOTrainer


def _run(coro):
    """Run an awaitable from sync context.

    SkyRL's WorkerDispatch protocol is synchronous but the new
    ``arctic_platform.rl`` client methods are async, so the dispatch
    blocks on the coroutine here. SkyRL's ``train()`` runs inside an
    asyncio loop (``asyncio.run(trainer.train())``), so ``asyncio.run``
    from inside it raises ``RuntimeError`` — we then fall back to a
    fresh loop in a worker thread (the ``arctic_platform.rl`` ray client
    only does ``ray.get`` under the hood, which works across loops).
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda c=coro: asyncio.new_event_loop().run_until_complete(c)).result()


class ArcticPPOTrainer(RayPPOTrainer):
    """PPO Trainer backed by Arctic RL server (DeepSpeed).

    Drop-in replacement for ``RayPPOTrainer``.  Requires
    ``colocate_all: false`` since all GPU work is on the Arctic RL server.
    """

    def __init__(self, *args, arctic_client, **kwargs):
        self._arctic_client = arctic_client
        self._stashed_rewards = None
        super().__init__(*args, **kwargs)

    def build_models(self, PolicyWorker=None, CriticWorker=None, RefWorker=None):
        """Replace GPU actor creation with a lightweight dispatch to Arctic RL."""
        self.dispatch = _ArcticDispatch(self.cfg, self._arctic_client)
        self.policy_model = _ArcticPolicyStub()
        self.ref_model = None
        self.critic_model = None
        logger.info("ArcticPPOTrainer: build_models → training routed to Arctic RL server")

    def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch):
        """No-op — old log-probs are computed server-side inside
        :meth:`_ArcticDispatch.forward_backward` (one ``fwd_no_grad``
        call before each ``fwd_bwd``), mirroring verl's actor worker
        which calls ``compute_log_prob`` before ``update_actor``."""
        return training_input

    def compute_advantages_and_returns(self, training_input: TrainingInputBatch):
        """Compute GRPO advantages client-side, then send them to the server.

        We stash the raw rewards so train_critic_and_policy can restore them
        (parent's train() pops rewards before calling train_critic_and_policy),
        then delegate to the parent to compute group-relative advantages normally.
        """
        self._stashed_rewards = training_input["rewards"].clone()
        training_input["values"] = None
        return super().compute_advantages_and_returns(training_input)

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """Send the full batch to the Arctic RL server for training.

        The server handles gradient accumulation internally via
        ``_forward_maybe_backward`` (splitting into micro-batches based on
        DeepSpeed's ``gradient_accumulation_steps``).  Each epoch sends a
        single fwd_bwd + step pair; ``set_gradient_accumulation_boundary``
        ensures the step always triggers a real optimizer update.

        Restricted to ``update_epochs_per_batch == 1``: the per-epoch loop
        in ``_ArcticDispatch.forward_backward`` calls ``_compute_old_log_probs``
        on every iteration, so >1 epoch would refresh ``old_log_probs`` and
        collapse PPO ``ratio`` to 1, defeating clipping. To support multi-epoch
        later, hoist the old-log-prob call into ``fwd_logprobs_values_reward``
        (matches verl's pre-update_actor ``compute_log_prob`` placement).
        """
        ep = int(self.cfg.trainer.update_epochs_per_batch)
        assert ep == 1, f"ArcticPPOTrainer requires update_epochs_per_batch=1, got {ep}"

        if self._stashed_rewards is not None:
            data["rewards"] = self._stashed_rewards
            self._stashed_rewards = None

        data.metadata["global_step"] = self.global_step
        n_samples = self.cfg.generator.n_samples_per_prompt

        all_metrics: Dict[str, List[float]] = defaultdict(list)

        status = self.dispatch.forward_backward(
            "policy",
            data,
            loss_fn="verl_grpo",
            loss_fn_config={"n_samples": n_samples},
        )
        for k, v in status.items():
            all_metrics[k].append(v)

        grad_norm = self.dispatch.optim_step("policy")
        if grad_norm is not None:
            all_metrics["grad_norm"].append(grad_norm)

        all_metrics.pop("loss_fn_outputs", None)
        all_metrics.pop("post_process_outputs", None)
        reduced = reduce_metrics(dict(all_metrics))

        for k, v in reduced.items():
            self.all_metrics[f"policy/{k}"] = v

        return reduced


# ---------------------------------------------------------------------------
# Dispatch — duck-types the WorkerDispatch interface used by RayPPOTrainer
# ---------------------------------------------------------------------------


class _ArcticDispatch:
    """Routes ``WorkerDispatch`` calls to the Arctic RL server.

    Owns the SkyRL→verl wire-protocol translation: takes SkyRL's
    ``TrainingInputBatch`` (left-padded ``[PAD…|prompt|response]``
    layout, with response-only tensors right-aligned to ``max_response``)
    and produces the dense padded payload that the
    ``arctic_platform.rl`` server expects (verl-style left-padded
    prompt + right-padded response, with ``meta`` carrying
    ``pad_token_id``, ``temperature``, ``actor_config``,
    ``policy_loss_config``, … — see :meth:`_build_meta` for the full
    list).
    """

    def __init__(self, cfg: SkyRLTrainConfig, client):
        self.cfg = cfg
        self.client = client
        # Policy flags read from `cfg.trainer.arctic_rl`, not `client.config`:
        # Arctic-Platform's `ArcticRLRayClient.reconnect_config()` strips the
        # schema to {backend, model_name, *_job_id, comm_protocol} when
        # serializing the client to Ray actors, dropping `colocate` etc.
        arl = cfg.trainer.arctic_rl
        self._colocate = bool(getattr(arl, "colocate", False)) if arl is not None else False
        self._cuda_ipc = bool(getattr(arl, "cuda_ipc_weight_sync", False)) if arl is not None else False
        self._low_memory = bool(getattr(arl, "low_memory_weight_sync", False)) if arl is not None else False

        # Load tokenizer once for pad_token_id (needed in every wire
        # payload). Same path the server itself uses — no risk of
        # version skew.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
        self._pad_token_id: int = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        # DP size = number of training GPUs (Arctic-RL training is DP-only;
        # no TP/PP on the training side). The verl_grpo loss uses this to
        # compute the correct global-mean normalization.
        self._dp_size: int = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes

        # Sampling temperature is part of the loss math (apply_temperature
        # post-processor divides logits by it). Recipe knob.
        self._temperature: float = float(getattr(cfg.generator.sampling_params, "temperature", 1.0) or 1.0)

        # ZoRRO toggle — mirror SkyRL's `arl.use_zorro` onto the server-side
        # `meta.zorro_train_enable` so the two stay in sync. Defaults to False.
        # When True the server expects the model to have been built with the
        # zorro-aware ds_worker_config (`response_len`, `max_token_len`,
        # `rollout_n`, `use_unpad=True`) — `build_rl_config` in `config.py`
        # already wires those when `arl.use_zorro=True`. Wire payloads stay
        # ZoRRO-compatible regardless (response-only tensors are left-padded
        # to seq_len in `forward_backward`); flipping this only changes the
        # server-side compute path, not our outgoing payload shape.
        self._zorro_train_enable: bool = bool(getattr(arl, "use_zorro", False)) if arl is not None else False

        # Per-GPU token budget for tiled compute. SkyRL has no direct
        # equivalent of verl's `actor.ppo_max_token_len_per_gpu`; derive a
        # ZoRRO-aware safe lower bound:
        #
        # * Non-ZoRRO (per-sequence packing): max_prompt + max_response —
        #   the worst-case single-sequence length.
        # * ZoRRO: max_prompt + n_samples * max_response — one deduplicated
        #   prompt followed by n_samples concatenated responses. Required
        #   because Arctic's `create_prompt_groups`
        #   (`arctic_platform/rl/zorro_train/seqlen_balancing.py:462`)
        #   packs an entire prompt-group into a single micro-batch and
        #   asserts ``max_token_len >= max_prompt +
        #   max_group_length_threshold * max_response``. With
        #   ``max_group_length_threshold == n_samples`` (the default),
        #   undersizing this budget produces ``ValueError: max_token_len=X
        #   is smaller than Y`` on the first ``fwd_no_grad`` of step 1.
        n_samples = cfg.generator.n_samples_per_prompt
        max_prompt = cfg.trainer.max_prompt_length
        max_resp = cfg.generator.sampling_params.max_generate_length
        if self._zorro_train_enable:
            self._max_token_len_per_gpu: int = int(max_prompt + n_samples * max_resp)
        else:
            self._max_token_len_per_gpu: int = int(max_prompt + max_resp)

        # Fixed response_len baked into Qwen3ModelOncePatcher at engine build
        # (arctic_rl/config.py:412 -> deepspeed_worker.py:328). Used by
        # `_repack_to_verl_shape` to keep the patcher's prompt/response split
        # aligned with the per-call attention mask.
        self._patcher_response_len: int = int(max_resp)

        if self._zorro_train_enable:
            logger.warning(
                "Arctic RL ZoRRO path enabled — make sure ds_worker_config "
                "carries response_len / max_token_len / rollout_n / use_unpad. "
                "Wire payload from this bridge is already ZoRRO-compatible "
                "(response-only tensors left-padded to seq_len); only the "
                "server-side compute path differs. Bridge derived "
                f"max_token_len_per_gpu={self._max_token_len_per_gpu} "
                f"(= {max_prompt} + {n_samples} * {max_resp})."
            )

    # ------------------------------------------------------------------ #
    # Wire-shape conversion: SkyRL TrainingInputBatch → verl batch dict
    # ------------------------------------------------------------------ #
    # SkyRL layout produced by `convert_prompts_responses_to_batch_tensors`
    # (skyrl/train/dataset/preprocess.py:124):
    #     sequences      : [B, S]   [PAD * pad_i | prompt_i | response_i]
    #     attention_mask : [B, S]   [0   * pad_i | 1        | 1        ]
    #     response_mask  : [B, A]   right-aligned [0 * (A - r_i) | 1 * r_i]
    #     loss_mask      : [B, A]   same shape as response_mask
    #     advantages     : [B, A]   same shape (after compute_advantages)
    # where S = max_i (p_i + r_i), A = max_i r_i (independent of S).
    #
    # verl/Arctic-RL server layout (per
    # `verl/workers/remote_client/arctic_rl.py`):
    #     input_ids      : [B, P+R]   left-pad prompt to P, right-pad
    #                                  response to R, total = P + R
    #     attention_mask : [B, P+R]   [0*(P-p_i) | 1*p_i | 1*r_i | 0*(R-r_i)]
    #     prompts        : [B, P]     (server reads .shape[1] only — see
    #                                  processors/pipeline.py:236)
    #     responses      : [B, R]
    #     response_mask  : [B, R]     right-padded with zeros
    #     position_ids   : [B, P+R]   cumsum(attention_mask)-1, pad→0
    #
    # The repack is per-sample: PAD region moves from the LEFT (SkyRL)
    # to BETWEEN prompt and response (verl) — i.e., prompts get
    # left-padded individually, responses get right-padded.

    def _repack_to_verl_shape(self, data: TrainingInputBatch) -> dict:
        """Per-sample repack of SkyRL's ``[PAD|prompt|response]`` into
        verl's ``[PAD|prompt|response|PAD]`` layout.

        Returns a dict with keys ``input_ids``, ``attention_mask``,
        ``prompts``, ``responses``, ``response_mask``, ``position_ids``,
        ``advantages``, ``loss_mask`` (and any other 2D response-shape
        tensors found in ``data``), all left/right-padded to a uniform
        ``[B, P + R]`` shape where ``P = max(prompt_len)`` and
        ``R = max(response_len)``.
        """
        sequences = data["sequences"]  # [B, S]
        attention_mask = data["attention_mask"]  # [B, S]
        response_mask = data["response_mask"]  # [B, R_sky]  R_sky = max_r
        B, S = sequences.shape
        device = sequences.device
        pad_id = self._pad_token_id

        # Real per-sample lengths.
        total_lens = attention_mask.sum(dim=1)  # [B] = p_i + r_i
        response_lens = response_mask.sum(dim=1)  # [B] = r_i
        prompt_lens = total_lens - response_lens  # [B] = p_i

        max_p = int(prompt_lens.max().item())
        max_r = int(response_lens.max().item())
        # The ZoRRO patcher derives prompt_len = seq_len - patcher_response_len
        # per forward; the server-side unpack uses meta["max_prompt_len"] = max_p.
        # For the two to agree, the response region must be exactly
        # patcher_response_len tokens wide. Padded positions get attention_mask=0
        # so the model and loss treat them as masked.
        if self._zorro_train_enable and max_r < self._patcher_response_len:
            max_r = self._patcher_response_len
        new_S = max_p + max_r

        # Pre-allocate output tensors.
        new_input_ids = torch.full((B, new_S), pad_id, dtype=sequences.dtype, device=device)
        new_attn = torch.zeros((B, new_S), dtype=attention_mask.dtype, device=device)
        new_resp_mask = torch.zeros((B, max_r), dtype=response_mask.dtype, device=device)

        # Optional response-shape tensors that need the same right-pad treatment.
        # We re-pad them inside the same per-row loop to keep alignment trivial.
        optional_response_keys: List[str] = []
        for k in ("advantages", "loss_mask", "rollout_logprobs", "returns"):
            if k in data.keys() and torch.is_tensor(data[k]) and data[k].ndim == 2:
                optional_response_keys.append(k)
        new_opts: Dict[str, torch.Tensor] = {}
        for k in optional_response_keys:
            t = data[k]
            new_opts[k] = torch.zeros((B, max_r), dtype=t.dtype, device=t.device)

        for i in range(B):
            p_i = int(prompt_lens[i].item())
            r_i = int(response_lens[i].item())
            pad_i = S - p_i - r_i  # left pad in SkyRL layout

            # SkyRL slice locations:
            prompt_slice = sequences[i, pad_i : pad_i + p_i]  # [p_i]
            response_slice = sequences[i, pad_i + p_i :]  # [r_i]

            # verl placement: prompt sits in positions [max_p - p_i : max_p],
            # response sits in positions [max_p : max_p + r_i].
            new_input_ids[i, max_p - p_i : max_p] = prompt_slice
            new_input_ids[i, max_p : max_p + r_i] = response_slice
            new_attn[i, max_p - p_i : max_p + r_i] = 1
            new_resp_mask[i, :r_i] = 1

            # Response-shape tensors in SkyRL are RIGHT-ALIGNED to max_r_sky
            # (i.e., the real values occupy the LAST r_i positions). After
            # repack we put them LEFT-ALIGNED in [B, max_r] (positions
            # [0 : r_i]) so they line up with the new response region of
            # input_ids. This matches verl's response-only tensor convention.
            R_sky = response_mask.shape[1]
            for k in optional_response_keys:
                src = data[k][i, R_sky - r_i :]  # the real per-token values
                new_opts[k][i, :r_i] = src

        # position_ids derived from attention_mask: cumsum-1, pad→0 (drops
        # negative -1 to 0 so the embedding lookup is safe).
        pos = new_attn.long().cumsum(-1) - 1
        pos.masked_fill_(new_attn == 0, 0)

        batch: Dict[str, Any] = {
            "input_ids": new_input_ids,
            "attention_mask": new_attn,
            "prompts": new_input_ids[:, :max_p],
            "responses": new_input_ids[:, max_p:],
            "response_mask": new_resp_mask,
            "position_ids": pos,
        }
        # Tag the repacked response-shape tensors so the loss / post-procs
        # find them in the same wire location as the verl adapter ships.
        for k, v in new_opts.items():
            batch[k] = v

        batch["_max_prompt_len"] = max_p
        batch["_max_response_len"] = max_r
        return batch

    @staticmethod
    def _left_pad_to_seq(t: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Left-pad a ``[B, R]`` response-only tensor to ``[B, P+R]`` with
        zeros so the loss can index it by full-sequence positions.

        Mirrors ``_send_update_actor._left_pad`` in
        ``verl/workers/remote_client/arctic_rl.py``: the verl_grpo loss
        gathers per-token positions via the response_mask, which is
        itself left-padded to seq_len — so every tensor it reads must
        have the same shape.
        """
        pad_len = seq_len - t.shape[-1]
        if pad_len <= 0:
            return t
        pad = torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)
        return torch.cat([pad, t], dim=-1)

    def _build_meta(
        self,
        *,
        max_prompt_len: int,
        max_response_len: int,
        batch_num_tokens: int,
        global_batch_size: int,
        calculate_entropy: bool,
        actor_config: Optional[Dict[str, Any]] = None,
        policy_loss_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the ``meta`` dict the server's processing pipeline
        and verl_grpo loss read.

        Mirrors the verl adapter's meta exactly (see
        ``verl/workers/remote_client/arctic_rl.py:198-279``). Defaults
        for ``actor_config`` / ``policy_loss_config`` come from the
        SkyRL recipe's algorithm block — GRPO without KL, no entropy
        bonus, clip_ratio=0.2 (matches verl PR #6 BIRD hyperparams).
        """
        cfg = self.cfg
        algo = cfg.trainer.algorithm
        n_samples = cfg.generator.n_samples_per_prompt

        # actor_config — fields read by verl_grpo.VerlPolicyConfig.
        # SkyRL stores most of these on `cfg.trainer.algorithm`.
        ac: Dict[str, Any] = {
            "loss_agg_mode": getattr(algo, "loss_agg_mode", "token-mean"),
            "kl_loss_coef": float(getattr(algo, "kl_loss_coef", 0.001) or 0.0),
            "kl_loss_type": getattr(algo, "kl_loss_type", "low_var_kl"),
            "clip_ratio": float(getattr(algo, "clip_ratio", 0.2)),
            "clip_ratio_low": float(getattr(algo, "clip_ratio_low", getattr(algo, "clip_ratio", 0.2))),
            "clip_ratio_high": float(getattr(algo, "clip_ratio_high", getattr(algo, "clip_ratio", 0.2))),
            "clip_ratio_c": float(getattr(algo, "clip_ratio_c", 3.0)),
            "entropy_coeff": float(getattr(algo, "entropy_coeff", 0.0)),
            "use_kl_loss": bool(getattr(algo, "use_kl_loss", False)),
            "calculate_entropy": calculate_entropy,
            "rollout_n": n_samples,
        }
        if actor_config:
            ac.update(actor_config)

        plc: Dict[str, Any] = {"loss_mode": "vanilla"}
        if policy_loss_config:
            plc.update(policy_loss_config)

        meta: Dict[str, Any] = {
            # Server-side processors & loss require these literally —
            # see `arctic_platform/rl/processors/pipeline.py:401`
            # (`pad_token = meta["pad_token_id"]`) and
            # `processors/verl_grpo.py:395-408`.
            "pad_token_id": self._pad_token_id,
            "rollout_n": n_samples,
            "max_prompt_len": max_prompt_len,
            "max_response_len": max_response_len,
            "max_token_len_per_gpu": self._max_token_len_per_gpu,
            "temperature": self._temperature,
            "calculate_entropy": calculate_entropy,
            # Per-call meta values. These must match what the public-branch
            # bridge sent (hardcoded "none" / False / 4 / False) -- empirically
            # validated through 7 converged steps of run gqpa0syk.
            #
            # IMPORTANT: do NOT source these from arl.* config. Pushing
            # ``logits_optimization="memory"`` here triggered a server-side
            # ZoRRO unpack mismatch at step 5 of run 1ikq295l ("value tensor
            # of shape [8026] cannot be broadcast to indexing result of
            # shape [7042]"), even though the same value is valid when set
            # on ``ds_worker_config`` at engine-build time. The two layers
            # take different code paths: ds_worker_config drives the model
            # patcher (Qwen3ModelOncePatcher), per-call meta drives the
            # non-ZoRRO processor pipeline (processors/pipeline.py:815-827),
            # and pushing "memory" into the non-ZoRRO path corrupts the
            # response-length packing contract.
            #
            # Likewise ``drop_position_ids=True`` while still sending
            # position_ids in the batch caused position_id-vs-attention_mask
            # divergence on certain batch compositions. Hardcoding False
            # restores the matched shapes.
            "drop_position_ids": False,
            "logits_optimization": "none",
            "logits_optimization_peak_mem_size_in_gib": 4,
            "logits_compute_in_fp32": False,
            # verl_grpo global-normalization knobs.
            "dp_size": self._dp_size,
            "batch_num_tokens": int(batch_num_tokens),
            "global_batch_size": int(global_batch_size),
            "rollout_is_weights": None,
            # ZoRRO is opt-in via SkyRL's `arl.use_zorro`; we mirror it onto
            # the server's per-call meta so the dispatch and ds_worker_config
            # stay aligned. `zorro_train_max_rollouts == n_samples_per_prompt`
            # is the verl-recommended value (best perf, no group-balancing
            # needed).
            "zorro_train_enable": self._zorro_train_enable,
            "zorro_train_max_rollouts": n_samples,
            "zorro_train_load_balancer": True,
            # Serialized loss config — see verl_grpo.VerlPolicyConfig.
            "actor_config": ac,
            "policy_loss_config": plc,
        }
        return meta

    # ------------------------------------------------------------------ #
    # WorkerDispatch interface
    # ------------------------------------------------------------------ #

    def forward(self, model: str, data: TrainingInputBatch) -> TrainingOutputBatch:
        """Compute log-probs only (no grad). Currently unused: the
        Arctic trainer overrides ``fwd_logprobs_values_reward`` as a
        no-op and computes old log-probs inside ``forward_backward``."""
        batch = self._repack_to_verl_shape(data)
        max_p = batch.pop("_max_prompt_len")
        max_r = batch.pop("_max_response_len")
        meta = self._build_meta(
            max_prompt_len=max_p,
            max_response_len=max_r,
            batch_num_tokens=int(batch["response_mask"].sum().item()),
            global_batch_size=int(data["sequences"].shape[0]),
            calculate_entropy=True,
        )
        payload = dict(
            batch={k: v for k, v in batch.items()},
            meta=meta,
            processing={"post": ["compute_entropy_and_logprobs"], "loss_fn": None},
        )
        result = _run(self.client.fwd_no_grad(payload, reference_model=False))
        out = TrainingOutputBatch()
        for k, v in result.get("batch", {}).items():
            out[k] = torch.tensor(v) if isinstance(v, list) else v
        out.metadata = {"model": model}
        return out

    def _compute_old_log_probs(self, repacked: Dict[str, Any], meta: Dict[str, Any]) -> torch.Tensor:
        """Run ``fwd_no_grad`` with the entropy/logprobs post-processor
        to obtain on-policy old log-probs for the PPO ratio.

        Mirrors verl's ``compute_log_prob`` call before
        ``update_actor`` — at step 1 (no policy update yet) this gives
        ``old_log_probs == new_log_probs`` so ``ppo_kl == 0`` and
        ``clipfrac == 0``, matching the verl PR #6 step-1 invariants.

        Payload mirrors ``_prepare_padded_arctic_batch_dict`` (verl
        adapter): only the keys the server needs to derive
        ``response_lens`` (``attention_mask`` + ``prompts.shape[1]``)
        and run the forward (``input_ids`` + ``position_ids``). No
        response-region tensors — those would be skipped by
        ``pack_with_unpad`` anyway (shape mismatch with attention_mask)
        but sending them needlessly bloats the wire payload and risks
        confusing the model's ``forward(**kwargs)`` signature.
        """
        fwd_batch = {
            "input_ids": repacked["input_ids"],
            "attention_mask": repacked["attention_mask"],
            "prompts": repacked["prompts"],
            "position_ids": repacked["position_ids"],
        }
        # `compute_entropy_and_logprobs` doesn't apply temperature here —
        # matches verl's compute_log_prob path (verl_grpo loss only
        # applies temperature inside update_actor / fwd_bwd). With
        # temperature=1.0 (BIRD recipe) this is a no-op either way.
        sub_meta = dict(meta)
        sub_meta["calculate_entropy"] = False  # cheaper; we only need log_probs
        # actor_config / policy_loss_config are update_actor-only; drop
        # them so the pipeline's `calculate_entropy` gating
        # (`if "entropy_coeff" in actor_config: meta["calculate_entropy"]
        # = (entropy_coeff != 0)`) doesn't accidentally flip on/off.
        sub_meta.pop("actor_config", None)
        sub_meta.pop("policy_loss_config", None)
        payload = dict(
            batch=fwd_batch,
            meta=sub_meta,
            processing={"post": ["compute_entropy_and_logprobs"], "loss_fn": None},
        )
        result = _run(self.client.fwd_no_grad(payload, reference_model=False))
        # Server returns the post-processor outputs under "batch", unpacked
        # back to [B, S] (pad regions filled with the pad_token_id by
        # `unpadded_tensor_1d_to_padded_tensor_2d`); the
        # `compute_entropy_and_logprobs` post writes "logprobs". Verl
        # renames to "log_probs"; accept either.
        lp = result["batch"].get("logprobs")
        if lp is None:
            lp = result["batch"].get("log_probs")
        if lp is None:
            raise RuntimeError(f"fwd_no_grad returned no logprobs; got keys={list(result.get('batch', {}).keys())}")
        if isinstance(lp, list):
            lp = torch.tensor(lp)
        return lp

    def forward_backward(
        self,
        model: str,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """One PPO update epoch: compute old log-probs, build the
        verl-shape payload, run server-side fwd+bwd with verl_grpo.

        Wire contract (mirrors
        ``verl/workers/remote_client/arctic_rl.py:_send_update_actor``):
          processing.post   = ["apply_temperature", "compute_entropy_and_logprobs"]
          processing.loss_fn = "verl_grpo"
          batch.input_ids        [B, P+R]
          batch.attention_mask   [B, P+R]
          batch.prompts          [B, P]     (server reads only .shape[1])
          batch.responses        [B, R]
          batch.position_ids     [B, P+R]
          batch.response_mask    [B, P+R]   (left-padded from [B, R])
          batch.loss_mask        [B, P+R]   (== response_mask after pad)
          batch.advantages       [B, P+R]   (left-padded with zeros)
          batch.old_log_probs    [B, P+R]   (left-padded with zeros)
        """
        repacked = self._repack_to_verl_shape(data)
        max_p = repacked.pop("_max_prompt_len")
        max_r = repacked.pop("_max_response_len")
        n_samples = self.cfg.generator.n_samples_per_prompt
        global_batch_size = self.cfg.trainer.policy_mini_batch_size * n_samples

        meta = self._build_meta(
            max_prompt_len=max_p,
            max_response_len=max_r,
            batch_num_tokens=int(repacked["response_mask"].sum().item()),
            global_batch_size=global_batch_size,
            calculate_entropy=True,
        )

        # 1) Old log-probs via fwd_no_grad (matches verl's two-call pattern).
        old_log_probs_resp = self._compute_old_log_probs(repacked, meta)  # [B, R]

        # 2) Left-pad response-only tensors to full seq_len. The verl_grpo
        #    loss + apply_temperature post-processor index by full-seq
        #    positions, so every response-shape tensor must match.
        seq_len = repacked["input_ids"].shape[-1]
        batch: Dict[str, Any] = {
            "input_ids": repacked["input_ids"],
            "attention_mask": repacked["attention_mask"],
            "prompts": repacked["prompts"],
            "responses": repacked["responses"],
            "position_ids": repacked["position_ids"],
        }
        batch["response_mask"] = self._left_pad_to_seq(repacked["response_mask"], seq_len)
        batch["advantages"] = self._left_pad_to_seq(repacked["advantages"], seq_len)
        batch["old_log_probs"] = self._left_pad_to_seq(old_log_probs_resp.to(repacked["input_ids"].device), seq_len)
        # verl: `loss_mask = response_mask` after the same left-pad.
        batch["loss_mask"] = batch["response_mask"]

        payload = dict(
            batch=batch,
            meta=meta,
            processing={
                "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                "loss_fn": "verl_grpo",
            },
        )

        result = _run(self.client.fwd_bwd(payload))
        result.pop("job_id", None)
        return result.get("metrics", result)

    def optim_step(self, model: str) -> Optional[float]:
        resp = _run(self.client.step())
        metrics = resp.get("metrics", resp)
        # The server reports per-DP-rank grad_norm as a flat list (length =
        # num training workers); SkyRL's `reduce_metrics` expects a scalar
        # per call and warns "Metrics for key grad_norm are not all numbers"
        # if given a nested list. With ZeRO-3 + DDP every rank sees the same
        # globally-reduced grad_norm, so picking the first is equivalent to
        # averaging — and avoids the warning churn in the trainer logs.
        grad_norm = metrics.get("grad_norm")
        if isinstance(grad_norm, (list, tuple)):
            flat = []
            for v in grad_norm:
                if isinstance(v, (list, tuple)):
                    flat.extend(v)
                else:
                    flat.append(v)
            grad_norm = float(flat[0]) if flat else None
        return grad_norm

    def get_lcm_dp_size(self) -> int:
        return 1

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        _run(self.client.save_checkpoint())

    def load_checkpoint(self, model: str, ckpt_dir: str, **kwargs) -> None:
        logger.info(f"Arctic RL: load_checkpoint for {model} — delegated to server")

    def save_hf_model(self, model: str, export_dir: str, tokenizer) -> None:
        logger.info(f"Arctic RL: save_hf_model for {model} — delegated to server")

    def set_lr(self, model: str, learning_rate: float) -> None:
        pass

    def init_weight_sync_state(self, inference_engine_client) -> None:
        pass

    async def save_weights_for_sampler(self) -> None:
        if self._colocate:
            await self.client.empty_training_cache()
            await self.client.wake_training()
            await self.client.wake_inference()
            await self.client.sync_weights(cuda_ipc=self._cuda_ipc, low_memory=self._low_memory)
        else:
            await self.client.sync_weights(cuda_ipc=False, low_memory=self._low_memory)

    def mark_all_offloaded(self) -> None:
        pass

    def empty_cache(self, model=None) -> None:
        pass

    def get_node_ids(self) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# Stub — satisfies self.policy_model usage in RayPPOTrainer / FullyAsync
# ---------------------------------------------------------------------------


class _ArcticPolicyStub:
    """No-op stub for ``self.policy_model``."""

    async def async_run_method(self, *args, **kwargs):
        pass

    def async_run_ray_method(self, *args, **kwargs):
        return []


class _ArcticInferenceEngineStub:
    """Stub for ``self.inference_engine_client``.

    Routes sleep/wake to the Arctic RL server for colocated mode.
    pause/resume are no-ops (server manages its own engine).
    """

    def __init__(self, client=None, colocate: bool = False):
        self._client = client
        # `colocate` from cfg.trainer.arctic_rl, set at entrypoint._setup_trainer.
        self._colocate = colocate

    async def sleep(self, **kwargs):
        if self._client and self._colocate:
            # level=2 releases bf16 weight pages in addition to KV cache.
            # Required at 32B colocation — level=1 keeps ~64 GiB of weights
            # resident and the colocated DeepSpeed worker OOMs on the first
            # MLP allocation of step 2. Mirrors arctic-verl's VLLM_SLEEP_LEVEL=2.
            level = kwargs.get("level", 2)
            await self._client.sleep_inference(level=level)

    async def wake_up(self, **kwargs):
        if self._client and self._colocate:
            tags = kwargs.get("tags")
            await self._client.wake_inference(tags=tags)

    async def pause_generation(self, **kwargs):
        pass

    async def resume_generation(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Fully-async variant
# ---------------------------------------------------------------------------


def _make_arctic_fully_async_trainer_class():
    """Build the fully-async Arctic trainer class with a deferred import.

    The import of ``FullyAsyncRayPPOTrainer`` is deferred so that
    ``arctic_trainer.py`` can be imported without pulling in the
    fully-async module (and its extra dependencies) when only the
    sync trainer is needed.
    """
    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

    class _ArcticFullyAsyncPPOTrainer(FullyAsyncRayPPOTrainer):
        """Fully-async PPO Trainer backed by Arctic RL server (DeepSpeed).

        Drop-in replacement for ``FullyAsyncRayPPOTrainer``.  Applies the
        same Arctic RL overrides as ``ArcticPPOTrainer`` (no-op fwd_logprobs,
        server-side GRPO loss, DeepSpeed gradient accumulation).
        """

        def __init__(self, *args, arctic_client, **kwargs):
            self._arctic_client = arctic_client
            self._stashed_rewards = None
            super().__init__(*args, **kwargs)

        def build_models(self, PolicyWorker=None, CriticWorker=None, RefWorker=None):
            self.dispatch = _ArcticDispatch(self.cfg, self._arctic_client)
            self.policy_model = _ArcticPolicyStub()
            self.ref_model = None
            self.critic_model = None
            logger.info("ArcticFullyAsyncPPOTrainer: build_models → training routed to Arctic RL server")

        def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch):
            return training_input

        def compute_advantages_and_returns(self, training_input: TrainingInputBatch):
            self._stashed_rewards = training_input["rewards"].clone()
            training_input["values"] = None
            return super().compute_advantages_and_returns(training_input)

        def train_critic_and_policy(self, data: TrainingInputBatch):
            # See ArcticPPOTrainer.train_critic_and_policy docstring — the
            # bridge's old-log-prob call lives inside forward_backward, so
            # multi-epoch would collapse `ratio == 1` every epoch.
            ep = int(self.cfg.trainer.update_epochs_per_batch)
            assert ep == 1, f"ArcticFullyAsyncPPOTrainer: update_epochs_per_batch must " f"be 1; got {ep}."

            if self._stashed_rewards is not None:
                data["rewards"] = self._stashed_rewards
                self._stashed_rewards = None

            data.metadata["global_step"] = self.global_step
            n_samples = self.cfg.generator.n_samples_per_prompt

            all_metrics: Dict[str, List[float]] = defaultdict(list)

            status = self.dispatch.forward_backward(
                "policy",
                data,
                loss_fn="verl_grpo",
                loss_fn_config={"n_samples": n_samples},
            )
            for k, v in status.items():
                all_metrics[k].append(v)

            grad_norm = self.dispatch.optim_step("policy")
            if grad_norm is not None:
                all_metrics["grad_norm"].append(grad_norm)

            all_metrics.pop("loss_fn_outputs", None)
            all_metrics.pop("post_process_outputs", None)
            reduced = reduce_metrics(dict(all_metrics))

            for k, v in reduced.items():
                self.all_metrics[f"policy/{k}"] = v

            return reduced

        async def async_sync_policy_weights_to_inference_engines(self):
            """Sync weights via Arctic RL server instead of policy_model."""
            await self._arctic_client.sync_weights()

    return _ArcticFullyAsyncPPOTrainer


def ArcticFullyAsyncPPOTrainer(*args, **kwargs):
    """Factory for the fully-async Arctic RL trainer.

    Defers the import of ``FullyAsyncRayPPOTrainer`` until first use.
    """
    cls = _make_arctic_fully_async_trainer_class()
    return cls(*args, **kwargs)
