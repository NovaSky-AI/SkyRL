"""Arctic RL configuration types.

Defines:
- ``ArcticRLTrainerConfig``: backend-specific knobs (colocate, zero_stage, ...)
- ``ArcticTrainerConfig``: extends core ``TrainerConfig`` with ``arctic_rl`` field
- ``ArcticSkyRLConfig``: top-level config used by the integration's entrypoint
- ``build_rl_config(cfg)``: translates ``SkyRLTrainConfig`` → ``ArcticRLClientConfig``

These live in the integration to keep core SkyRL integration-agnostic — core only
knows about a generic ``trainer.override_entrypoint: Optional[str]`` field that
lazily dispatches here. All shared knobs (GPU counts, vLLM settings, colocation)
are derived from existing SkyRL config fields by ``build_rl_config``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from arctic_platform.rl import ArcticRLClientConfig
from omegaconf import OmegaConf

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import BaseConfig, TrainerConfig, make_config

# ---------------------------------------------------------------------------
# Arctic RL backend configuration
# ---------------------------------------------------------------------------


@dataclass
class ArcticRLTrainerConfig(BaseConfig):
    """Arctic RL (DeepSpeed) backend settings.

    Mirrors the surface area of verl's
    ``arctic-verl/verl/workers/remote_client/arctic_rl.py``
    (``ArcticRLClientWrapper``). Each field corresponds 1:1 to a knob that
    the verl xid2pl9f BIRD-1.7B converged-reference run sets in its
    ``launch_1.7b_newshape.sh``; defaults pick safe non-Arctic values so
    flipping ``trainer.arctic_rl={}`` alone is sufficient for a smoke
    test.
    """

    colocate: bool = False
    """Share GPUs between training and inference on the ARL server.

    Distinct from ``trainer.placement.colocate_all`` which controls Ray placement
    groups.  ARL colocation is server-side GPU sharing — ``colocate_all`` must
    stay ``false`` when using the ARL backend.
    """
    use_zorro: bool = False
    """Enable ZoRRO (prompt deduplication) on the training server."""
    zero_stage: int = 0
    """DeepSpeed ZeRO stage (0, 2, or 3)."""
    log_prob_gpus: int = 0
    """Number of GPUs for log-prob computation (0 = skip separate log-prob)."""
    offload_optimizer: bool = False
    """Offload optimizer state to CPU when ``zero_stage >= 2``."""

    # -- Model / kernel knobs (verl: actor_rollout_ref.model.*) -------------
    use_liger: bool = False
    """Enable Liger fused MLP/RMSNorm kernels on the training engine.

    Server reads from ``ds_worker_config.use_liger``
    (``arctic_platform/rl/deepspeed_worker.py:155``). Requires
    ``liger_kernel`` to be installed.
    """
    attn_implementation: str = "flash_attention_2"
    """HF attention implementation passed to ``Qwen3ForCausalLM.from_pretrained``.

    Server reads from ``ds_worker_config.attn_implementation``
    (``arctic_platform/rl/deepspeed_worker.py:147``). verl's converged
    reference uses ``flash_attention_3`` (requires the ``flash_attn_3``
    wheel) for O(N) attention memory on long sequences.
    """
    enable_gradient_checkpointing: bool = True
    """Activation checkpointing on the training engine.

    Server reads from ``ds_worker_config.enable_gradient_checkpointing``
    (``arctic_platform/rl/deepspeed_worker.py:202``); defaults to True so
    long sequences fit in colocated mode.
    """
    ulysses_sequence_parallel_size: int = 1
    """Ulysses sequence-parallel degree (DeepSpeed ``sequence_parallel_size``).

    Splits a single sequence's compute across this many GPUs, cutting
    per-GPU activation memory at the same factor. verl's converged
    reference uses 2.
    """

    # -- Logits / loss memory knobs (verl: arctic_rl.train.logits.*) --------
    logits_optimization: Optional[str] = None
    """``"memory"`` enables chunked logits compute on the server (ZoRRO
    only). None = full materialization. verl converged reference: "memory".
    Maps to ``ppo_trainer.yaml: arctic_rl.train.logits.optimization``.
    """
    logits_optimization_peak_mem_size_in_gib: float = 4.0
    """Per-chunk peak memory budget for the chunked-logits path. Matches
    verl ``ppo_trainer.yaml: arctic_rl.train.logits.optimization_peak_mem_size_in_gib``
    default (4). Smaller -> more, smaller chunks -> lower peak memory but
    more compute overhead.
    """
    logits_compute_from_fp32_inputs: bool = False
    """Cast hidden states to fp32 before the LM head (ZoRRO logits path).
    Maps to ``arctic_rl.train.logits.compute_from_fp32_inputs``.
    """
    logits_compute_in_fp32: bool = False
    """Run the LM-head matmul itself in fp32 (ZoRRO logits path).
    Maps to ``arctic_rl.train.logits.compute_in_fp32``.
    """

    # -- Determinism (verl: arctic_rl.train.determinism.*) -----------------
    determinism_full: bool = False
    """Enable ``transformers.enable_full_determinism`` on the server.
    Maps to ``arctic_rl.train.determinism.full``.
    """
    determinism_seed: int = 42
    """Seed used when ``determinism_full=True``. Maps to
    ``arctic_rl.train.determinism.seed``.
    """

    # -- Weight sync (verl: arctic_rl.cuda_ipc_weight_sync) ----------------
    cuda_ipc_weight_sync: bool = False
    """Use CUDA-IPC zero-copy weight sync (colocate-only). Avoids the
    extra GPU buffer that the NCCL/CPU paths need; verl converged
    reference enables it.
    """
    low_memory_weight_sync: bool = False
    """Stream one param at a time during CUDA-IPC sync (more round-trips,
    bounded peak extra GPU memory). Matches verl
    ``arctic_rl.low_memory_weight_sync`` default (False).
    """

    # -- DeepSpeed config knobs not exposed elsewhere (verl: arctic_rl.train.deepspeed.*)
    offload_param: bool = False
    """Offload ZeRO-3 parameter shards to CPU when ``zero_stage >= 2``.
    verl ``arctic_rl.train.deepspeed.zero_optimization.offload_param.device``
    default is ``"none"`` (off); xid2pl9f also leaves it off.
    """
    torch_autocast_enabled: bool = False
    """Enable DeepSpeed ``torch_autocast`` on the training engine. verl
    default and xid2pl9f: False (relies on the bf16 path).
    """
    torch_autocast_dtype: Optional[str] = None
    """``"bfloat16"`` etc; only consumed when ``torch_autocast_enabled=True``."""

    # -- Comms (verl: arctic_rl.comms.*) -----------------------------------
    drop_position_ids: bool = True
    """Drop position_ids from the wire payload and rebuild them server-side
    from attention_mask. verl default: True; only set False for models that
    need non-arange position_ids (mrope / 3D rope).
    """

    # -- LR scheduler (verl: actor.optim.*) ---------------------------------
    lr_warmup_ratio: float = 0.0
    """Linear warmup fraction of the total training horizon. verl uses 0.05."""
    lr_scheduler_type: str = "constant"
    """Server-side LR scheduler. "constant" matches verl's default."""
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    """AdamW betas. verl converged reference uses (0.9, 0.95)."""

    # -- vLLM knobs not exposed in SkyRL's generator config -----------------
    vllm_max_num_seqs: int = 256
    """vLLM ``max_num_seqs``. verl converged reference: 256."""
    vllm_enforce_eager: bool = True
    """Disable vLLM CUDA graph capture. verl converged reference: False
    (graphs on; uses more memory but is faster). Default True is safer in
    colocated mode.
    """
    vllm_enable_chunked_prefill: bool = True
    """Chunked prefill (lower KV memory peak)."""
    vllm_enable_prefix_caching: bool = True
    """vLLM ``enable_prefix_caching``. Verl converged reference: True.
    Big win for long-context bird prompts since the schema prefix is
    shared across all 16 rollouts of every prompt.
    """
    vllm_max_model_len: Optional[int] = None
    """vLLM ``max_model_len``; if None, derived from
    ``max_prompt_length + max_generate_length``.
    """
    vllm_max_num_batched_tokens: Optional[int] = None
    """vLLM ``max_num_batched_tokens`` (prefill batch capacity).
    Verl converged reference: 40960 (= ``ROLLOUT_MAX_BATCHED``).
    Higher = more parallel prefill at the cost of GPU memory.
    None = leave vLLM's default (typically 8192, which throttles
    long-prompt prefill on bird).
    """

    # -- ArcticInference (FCA / fast-continuous-attention; verl: rollout.name=arctic)
    use_arctic_inference: bool = False
    """Enable Arctic Inference (FCA + custom kernels) for the vLLM
    sampling engine. Passes ``arctic_inference_config={}`` to
    ``ArcticRLClientConfig`` which sets ``ARCTIC_INFERENCE_ENABLED=1``
    on the sampling Ray workers. Matches verl's ``rollout.name=arctic``
    converged path; typical 2-3x rollout speedup on long-context prompts.
    Requires ``arctic_inference`` to be installed.
    """
    speculative_model: Optional[str] = None
    """HF id or local path of an Arctic draft-head checkpoint. When set
    (and ``use_arctic_inference=True``), the integration auto-injects::

        speculative_config: {
            method: arctic,
            model: <speculative_model>,
            num_speculative_tokens: <num_speculative_tokens>,
        }

    into the rollout vLLM engine. Leave ``None`` to disable speculative
    decoding. End users normally set just this field — no raw
    ``vllm_config`` block needed.
    """
    num_speculative_tokens: int = 3
    """Number of draft tokens proposed per target-model step. Only used
    when ``speculative_model`` is set.
    """
    arctic_inference_config: Optional[dict] = None
    """Optional ArcticInference high-level schema (e.g. ``zorro_inference:
    {enable: true}``, ``speculative_decoding: {model: ...}``). Forwarded
    as-is to ``ArcticRLClientConfig.arctic_inference_config``, where
    arctic-platform's ``parse_arctic_inference_rollout`` translates the
    recognised keys (``use_fca``, ``spec_model``) into ``ModelConfig``
    fields. Most users should leave this ``None`` and use the typed
    ``speculative_model`` / ``use_arctic_inference`` knobs above — the
    integration injects the matching raw vLLM kwargs automatically.
    """
    vllm_config: Optional[dict] = None
    """Escape hatch for raw ``vllm.AsyncEngineArgs`` keys that aren't yet
    exposed as typed ``trainer.arctic_rl.*`` fields.

    Merged on top of the integration's defaults (TP, ``max_num_seqs``,
    auto-injected FCA / speculative_config / ``fuse_allreduce_rms``
    workaround); **user keys win on conflict**. Mirrors arctic-platform's
    own ``ArcticRLClientConfig.vllm_config`` field name and semantics:
    every key is forwarded to vLLM (those matching ``ModelConfig`` fields
    become typed fields, the rest land in ``extra_engine_kwargs``).

    The simple case never needs this — ``use_arctic_inference=true`` and
    ``speculative_model=<id>`` cover FCA + Arctic spec-dec automatically.
    Reach for ``vllm_config`` only when you need an inference-engine knob
    the typed fields don't cover, e.g.::

        trainer.arctic_rl.vllm_config={
            compilation_config: {cudagraph_mode: FULL},
        }
    """

    host: str = "localhost"
    """Server host for HTTP comm protocol; ignored for Ray."""
    port: int = 7000
    """Server port for HTTP comm protocol; ignored for Ray."""
    startup_timeout: float = 300.0
    """Seconds to wait for server jobs to come up."""
    server_logs: bool = False
    """Forward server logs to stdout for debugging."""


@dataclass
class ArcticTrainerConfig(TrainerConfig):
    """``TrainerConfig`` extended with the Arctic RL field. Used when
    ``trainer.override_entrypoint=integrations.arctic_rl.entrypoint`` is set."""

    arctic_rl: Optional[ArcticRLTrainerConfig] = None
    """Arctic RL backend settings. ``None`` falls back to defaults."""


# Top-level config for arctic_rl recipes; parsed by the integration's entrypoint
# after core dispatch (``trainer.override_entrypoint=integrations.arctic_rl.entrypoint``).
ArcticSkyRLConfig = make_config(trainer_cls=ArcticTrainerConfig)


# ---------------------------------------------------------------------------
# Translation: SkyRLTrainConfig → ArcticRLClientConfig
# ---------------------------------------------------------------------------


def build_rl_config(cfg: SkyRLTrainConfig) -> ArcticRLClientConfig:
    """Build ``ArcticRLClientConfig`` from ``SkyRLTrainConfig``.

    Mirrors ``arctic-verl/verl/workers/remote_client/arctic_rl.py``
    (``ArcticRLClientWrapper._initialize_client``,
    ``_create_ds_config``, ``_create_ds_worker_config``) so the SkyRL
    bridge sends the Arctic server the same payload shape as the
    verl xid2pl9f BIRD-1.7B converged-reference run.

    Raises ``ValueError`` if ``cfg.trainer.arctic_rl`` is not set.
    """
    arl = cfg.trainer.arctic_rl
    if arl is None:
        raise ValueError(
            "trainer.arctic_rl must be set when using the Arctic RL entrypoint. "
            "Add 'trainer.arctic_rl={}' to your config overrides to enable it "
            "with defaults, or set individual fields like "
            "'trainer.arctic_rl.zero_stage=2'."
        )

    # -- Derived from existing SkyRL configs ---------------------------------
    training_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
    # Arctic-Platform's `sampling_gpus` is the TOTAL sampling-side GPU count,
    # not the replica count (replicas = sampling_gpus // tp_size); SkyRL's
    # `num_engines` is the replica count, so multiply by `tp_size`.
    tp_size = cfg.generator.inference_engine.tensor_parallel_size
    sampling_gpus = cfg.generator.inference_engine.num_engines * tp_size
    colocate = arl.colocate
    vllm_gpu_mem = cfg.generator.inference_engine.gpu_memory_utilization

    # -- From ARL-specific config --------------------------------------------
    opt = cfg.trainer.policy.optimizer_config
    lr = opt.lr
    weight_decay = float(getattr(opt, "weight_decay", 0.0) or 0.0)
    eps = float(getattr(opt, "eps", 1e-8))
    grad_clip = float(opt.max_grad_norm) if opt.max_grad_norm else 0.0
    betas_from_cfg = getattr(opt, "betas", None)
    betas = tuple(betas_from_cfg) if betas_from_cfg else tuple(arl.optimizer_betas)

    n_samples = cfg.generator.n_samples_per_prompt
    mini_batch_size = cfg.trainer.policy_mini_batch_size * n_samples

    max_prompt = int(cfg.trainer.max_prompt_length)
    max_resp = int(cfg.generator.sampling_params.max_generate_length)
    max_length = max_prompt + max_resp

    # DeepSpeed micro-batch sizing — mirrors verl's
    # `actor.ppo_micro_batch_size_per_gpu` convention
    # (`verl/workers/remote_client/arctic_rl.py:_create_ds_config`). Each
    # fwd / fwd+bwd RPC ships one per-GPU mini-batch shard; the server then
    # splits it into `gradient_accumulation_steps` micro-batches inside
    # DeepSpeed.
    #
    # Default depends on ZoRRO:
    #   * ZoRRO off: 1 sample / micro-batch (verl's long-seq GRPO default;
    #     safe under ZeRO-3 + vLLM colocation).
    #   * ZoRRO on:  n_samples_per_prompt — each micro-batch must contain
    #     a full prompt-group (1 prompt + n_samples responses) so the
    #     dedup + bin-pack in `arctic_platform/rl/zorro_train/seqlen_balancing.py
    #     ::create_prompt_groups` can run. With micro-batch < n_samples,
    #     only one rollout's worth of prompt is present; the server-side
    #     `unpadded_tensor_1d_response_to_padded_tensor_2d_full` then
    #     emits a tensor of the wrong shape and the run dies with a
    #     `RuntimeError: shape mismatch: value tensor of shape [T_resp]
    #     cannot be broadcast to indexing result of shape [B*R]`.
    train_micro_batch_size_per_gpu = n_samples if arl.use_zorro else 1
    # DeepSpeed `_batch_assertion` requires:
    #   train_batch_size == micro_batch * grad_accum * world_size
    # When ``sequence_parallel_size > 1`` DeepSpeed sets
    #   world_size = dist.get_world_size() / sequence_parallel_size
    # (see ``deepspeed/runtime/config.py``: "if 'sequence_parallel_size' in
    # config: self.world_size = dist.get_world_size() / config[...]"),
    # so the effective DP world is ``training_gpus // ulysses_sp``.
    ulysses_sp = max(1, int(arl.ulysses_sequence_parallel_size))
    assert training_gpus % ulysses_sp == 0, (
        f"training_gpus ({training_gpus}) must be divisible by " f"ulysses_sequence_parallel_size ({ulysses_sp})"
    )
    dp_world = max(1, training_gpus // ulysses_sp)
    mini_per_dp = max(1, mini_batch_size // dp_world)
    assert mini_per_dp % train_micro_batch_size_per_gpu == 0, (
        f"per-DP-rank mini-batch ({mini_per_dp}) must be divisible by "
        f"train_micro_batch_size_per_gpu ({train_micro_batch_size_per_gpu}); "
        f"got mini_batch_size={mini_batch_size}, training_gpus={training_gpus}, "
        f"ulysses_sp={ulysses_sp}, dp_world={dp_world}, "
        f"n_samples_per_prompt={n_samples}, use_zorro={arl.use_zorro}"
    )
    grad_accum_steps = max(1, mini_per_dp // train_micro_batch_size_per_gpu)
    assert mini_batch_size == train_micro_batch_size_per_gpu * grad_accum_steps * dp_world, (
        f"DeepSpeed batch assertion would fail: train_batch_size={mini_batch_size} "
        f"!= micro={train_micro_batch_size_per_gpu} * grad_accum={grad_accum_steps} "
        f"* dp_world={dp_world} (= {train_micro_batch_size_per_gpu * grad_accum_steps * dp_world})"
    )

    # -- vLLM config ---------------------------------------------------------
    # Always populate (not just colocated) so server gets the same shape
    # verl sends to Arctic. Mirrors ``arctic_rl.py:vllm_config`` block.
    # ``max_num_batched_tokens`` and ``enable_prefix_caching`` are not
    # native ModelConfig fields -- they're passed through to vLLM via
    # ``extra_engine_kwargs`` (see arctic_platform/rl/ray_server.py:121).
    vllm_cfg: dict = {
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": vllm_gpu_mem,
        "max_model_len": int(arl.vllm_max_model_len) if arl.vllm_max_model_len else max_length,
        "max_num_seqs": int(arl.vllm_max_num_seqs),
        "enforce_eager": bool(arl.vllm_enforce_eager),
        "enable_chunked_prefill": bool(arl.vllm_enable_chunked_prefill),
        "enable_prefix_caching": bool(arl.vllm_enable_prefix_caching),
    }
    if arl.vllm_max_num_batched_tokens is not None:
        vllm_cfg["max_num_batched_tokens"] = int(arl.vllm_max_num_batched_tokens)

    # ------------------------------------------------------------------
    # Auto-inject Arctic Inference vLLM kwargs from typed top-level knobs.
    # The intent: keep the user's recipe free of raw ``vllm_config`` blocks
    # for the common Arctic flows. User-supplied ``arl.vllm_config`` is
    # still merged on top below and wins on conflict.
    # ------------------------------------------------------------------
    num_engines = int(cfg.generator.inference_engine.num_engines)

    if arl.use_arctic_inference:
        # FCA is the headline Arctic Inference optimization; enable it
        # whenever the user opts into Arctic Inference. ``"{}"`` =
        # arctic-inference defaults.
        vllm_cfg.setdefault("forest_cascade_attn_configs", "{}")

        # Multi-replica FlashInfer workaround. arctic-inference's
        # ``use_fca=True`` default also enables ``fuse_allreduce_rms``,
        # which collides with per-process FlashInfer IPC port allocation
        # when ``world_size > tp_size`` (multiple replicas per node).
        # Symptom: ``AssertionError: Flashinfer workspace must be
        # initialized when using flashinfer``. Auto-disable for the
        # multi-replica topology; single-replica keeps the fused path.
        # Long-term fix belongs inside arctic-inference; this is the
        # interim workaround so end users never have to know.
        if num_engines > 1:
            comp_cfg = vllm_cfg.setdefault("compilation_config", {})
            pass_cfg = comp_cfg.setdefault("pass_config", {})
            pass_cfg.setdefault("fuse_allreduce_rms", False)

    # Arctic speculative decoding: opt-in via a single typed flag.
    if arl.speculative_model:
        vllm_cfg.setdefault(
            "speculative_config",
            {
                "method": "arctic",
                "model": str(arl.speculative_model),
                "num_speculative_tokens": int(arl.num_speculative_tokens),
            },
        )

    # Layer user-supplied raw vLLM kwargs on top of the integration's
    # defaults + auto-injected Arctic kwargs. arctic-platform forwards
    # this dict to vLLM verbatim (any key not on ``ModelConfig`` lands in
    # ``extra_engine_kwargs``); user keys win on conflict.
    if arl.vllm_config is not None:
        user_vllm = OmegaConf.to_container(
            OmegaConf.create(arl.vllm_config),
            resolve=True,
        )
        if not isinstance(user_vllm, dict):
            raise TypeError(f"trainer.arctic_rl.vllm_config must be a dict, got " f"{type(user_vllm).__name__}")
        vllm_cfg.update(user_vllm)

    # -- DeepSpeed engine config (training) ---------------------------------
    # Mirrors ``_create_ds_config`` in verl's arctic_rl.py. ``sequence_parallel_size``
    # = ulysses degree (per-sequence compute split across this many GPUs);
    # 1 = disabled.
    # ``train_batch_size`` here is the per-optimizer-step global batch
    # (DeepSpeed's view: micro * grad_accum * dp_world). With
    # ``update_epochs_per_batch=1`` and ``policy_mini_batch == train_batch``
    # in the launcher this equals the global training batch.
    ds_config: dict = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "train_batch_size": mini_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "sequence_parallel_size": ulysses_sp,
        "bf16": {"enabled": True},
        "gradient_clipping": grad_clip,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": list(betas),
                "eps": eps,
                "weight_decay": weight_decay,
            },
        },
    }

    zero_cfg: dict = {"stage": int(arl.zero_stage)}
    if arl.zero_stage >= 2 and arl.offload_optimizer:
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    else:
        # Explicit "none" mirrors verl xid2pl9f's
        # arctic_rl.train.deepspeed.zero_optimization.offload_optimizer.device.
        zero_cfg["offload_optimizer"] = {"device": "none"}
    if arl.zero_stage >= 3 and arl.offload_param:
        zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}
    else:
        zero_cfg["offload_param"] = {"device": "none"}
    ds_config["zero_optimization"] = zero_cfg

    # torch_autocast: verl xid2pl9f leaves disabled, but expose the knob.
    if arl.torch_autocast_enabled:
        autocast_cfg: dict = {"enabled": True}
        if arl.torch_autocast_dtype:
            autocast_cfg["dtype"] = str(arl.torch_autocast_dtype)
        ds_config["torch_autocast"] = autocast_cfg
    # Determinism handled at the ArcticRLClientConfig top level below
    # (NOT in ds_config — Arctic's server reads full_determinism/seed
    # from job_config, not the DeepSpeed config).

    # -- DeepSpeed worker config (per-process; model + kernel + ZoRRO) ------
    # Mirrors ``_create_ds_worker_config`` in verl's arctic_rl.py. These
    # are read by ``arctic_platform/rl/deepspeed_worker.py`` at engine
    # build time -- NOT per-call. Without ``attn_implementation`` and
    # ``use_liger`` here the Qwen3 forward defaults to eager attention
    # (O(N^2) memory) and unfused MLP, which OOMs in colocated mode.
    ds_worker_config: dict = {
        "use_liger": bool(arl.use_liger),
        "enable_gradient_checkpointing": bool(arl.enable_gradient_checkpointing),
        "attn_implementation": str(arl.attn_implementation),
    }
    if arl.use_zorro:
        # ZoRRO budget: ``max_token_len`` must be >= max_prompt +
        # n_samples * max_response (see _ArcticDispatch._max_token_len_per_gpu
        # for the full reasoning).
        ds_worker_config.update(
            zorro_train_enable=True,
            response_len=max_resp,
            max_token_len=int(max_prompt + n_samples * max_resp),
            rollout_n=int(n_samples),
            temperature=float(getattr(cfg.generator.sampling_params, "temperature", 1.0) or 1.0),
            use_unpad=True,
            logits_optimization=arl.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=float(arl.logits_optimization_peak_mem_size_in_gib),
            logits_compute_from_fp32_inputs=bool(arl.logits_compute_from_fp32_inputs),
            logits_compute_in_fp32=bool(arl.logits_compute_in_fp32),
        )

    # -- Top-level training_config (LR schedule, model meta, etc.) ----------
    # Mirrors ``training_config`` arg in verl's _initialize_client.
    # ``training_horizon`` is the number of optimizer steps over which the
    # LR scheduler interpolates (warmup + decay); for SkyRL we approximate
    # it as ``epochs * (rows / train_batch_size)`` -- the same
    # ``trainer.epochs * data_size / train_batch_size`` quantity SkyRL uses
    # internally for its scheduler. Falls back to a generous default so the
    # constant-LR path (verl's reference) is robust.
    training_horizon = int(getattr(cfg.trainer, "total_training_steps", 0) or 0)
    if training_horizon <= 0:
        # ``constant`` scheduler ignores horizon; non-constant schedulers
        # will get a default 1k horizon (caller can override via
        # cfg.trainer.total_training_steps).
        training_horizon = 1000

    optimizer_config: dict = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": list(betas),
    }
    if grad_clip > 0:
        optimizer_config["gradient_clipping"] = grad_clip

    lr_scheduler_config: dict = {
        "type": str(arl.lr_scheduler_type),
        "warmup_ratio": float(arl.lr_warmup_ratio),
    }

    training_config: dict = {
        "training_horizon": training_horizon,
        "optimizer": optimizer_config,
        "lr_scheduler": lr_scheduler_config,
        "max_length": max_length,
        "model_config": None,
        "attn_implementation": str(arl.attn_implementation),
    }

    return ArcticRLClientConfig(
        model_name=cfg.trainer.policy.model.path,
        backend="local",
        # arctic_platform.rl defaults comm_protocol to "http"; preserve the
        # public-branch behavior (in-process Ray actor) by setting it explicitly.
        # Host/port are auto-derived from comm_protocol (None for ray).
        comm_protocol="ray",
        checkpoint_path=cfg.trainer.ckpt_path,
        training_gpus=training_gpus,
        sampling_gpus=sampling_gpus,
        log_prob_gpus=arl.log_prob_gpus,
        log_prob_engine="deepspeed" if arl.log_prob_gpus > 0 else "vllm",
        colocate=colocate,
        vllm_config=vllm_cfg,
        ds_config=ds_config,
        ds_worker_config=ds_worker_config,
        training_config=training_config,
        # Deep-convert OmegaConf containers to plain Python so that vLLM's
        # ``AsyncEngineArgs.__post_init__`` recognizes nested overrides
        # (``compilation_config``, ``speculative_config``, ...). The
        # ``OmegaConf.create(x)`` round-trip normalizes both plain ``dict``
        # and ``DictConfig`` inputs, matching the ``OmegaConf.to_container``
        # idiom used in ``arctic-verl/verl/workers/remote_client/arctic_rl.py``.
        arctic_inference_config=(
            OmegaConf.to_container(
                OmegaConf.create(arl.arctic_inference_config),
                resolve=True,
            )
            if arl.use_arctic_inference and arl.arctic_inference_config is not None
            else None
        ),
        full_determinism=bool(arl.determinism_full),
        seed=int(arl.determinism_seed),
        startup_timeout=arl.startup_timeout,
        server_logs=arl.server_logs,
    )
