"""Local configuration; no fields are added to SkyRL's core configuration."""

import math
from dataclasses import dataclass, field

from skyrl.train.config import (
    AlgorithmConfig,
    PlacementConfig,
    TrainerConfig,
    make_config,
)


@dataclass
class TeacherConfig:
    kind: str = "ema"  # self: student net; ema: hosted policy copy + lerp; frozen: ema with update_rate 0
    path: str | None = None  # optional ema/frozen checkpoint; defaults to the policy path
    update_rate: float = 0.05  # kind=frozen sets it to 0
    storage: str = "cuda"  # "cpu" trades GPU memory for per-microbatch transfers

    def __post_init__(self):
        if self.kind == "frozen":
            self.update_rate = 0.0


@dataclass
class DistillationConfig:
    trajectory_distillation: bool = False  # allow feedback-only multi-turn; masks still come from the generator
    use_sample_routing: bool = False  # route rows between SDPO and GRPO; requires G>1
    srpo_route_positive_rewards: bool = False  # reward>0 GRPO; reward==0 with a teacher SDPO; otherwise skip
    srpo_route_success_only: bool = False  # reward>=threshold GRPO; lower rewards SDPO with a teacher, else skip
    vision_language: bool = False  # required for image inputs, visual feedback, and VLM policies
    limit_response_logits: bool = False  # FSDP passes logits_to_keep and scores response positions
    megatron_teacher_cache_gib: float = 128.0
    """CPU response-logit budget per final-stage TP rank; minimum one microbatch."""
    teacher_first: bool = False
    """Score a CPU-stored hosted teacher before student forward and retain its logits on CPU."""
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    loss: str = "distribution"  # or "sampled_reverse_kl"
    alpha: float = 0.5  # 0: forward KL; 1: reverse KL; interior: generalized JSD
    topk: int | None = 100
    add_tail: bool = True
    loss_chunk_size: int = 128
    reduction: str = "global_token_mean"
    rollout_is_clip: float | None = None
    advantage_is_clip: float | None = None  # clip at this multiple of the running mean |log p - log q|
    skip_correct_singletons: bool = True  # skip correct responses whose prompt group has size 1
    success_reward_threshold: float = 1.0
    sibling_selection: str = "first_success"  # or group-wide best_reward
    exclude_self_demonstration: bool = True
    remove_thinking: bool = True
    train_on_reasoning_tokens: bool = True
    """Include student <think> spans in the loss; independent of teacher hint stripping."""
    include_student_response: bool = False
    """Include the current attempt in teacher context using remove_thinking."""
    student_response_template: str = "\n\nBaseline response from your first attempt:\n\n{student_response}\n"
    include_environment_feedback: bool = True
    feedback_only_without_solution: bool = True
    feedback_key: str = "feedback"  # optional per-row env_metrics entry
    demonstration_source: str = "sibling"
    expert_trace_key: str = "expert_trace"  # dataset column, read from env_extras
    expert_trace_template: str = "\n\nExpert reference solution:\n\n{expert_trace}\n"
    expert_trace_include_thinking: bool = False
    max_teacher_prompt_length: int = 4096
    max_teacher_sequence_length: int = 8192
    overflow: str = "skip"  # or "error" or "truncate_right"
    chat_template_kwargs: dict = field(default_factory=dict)
    filter_ineligible_demonstrations: bool = True
    reprompt_template: str = (
        "{prompt}{student_response}{solution}{feedback}\n\nCorrectly solve the original question.\n"
    )
    solution_template: str = "\n\nCorrect solution:\n\n{successful_previous_attempt}\n"
    feedback_template: str = "\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n"
    use_entropy_weighting: bool = False  # independent of sample routing
    entropy_beta: float = 1.0  # weight is exp(-beta * teacher entropy); 0 is uniform

    def __post_init__(self):
        if not math.isfinite(self.megatron_teacher_cache_gib) or self.megatron_teacher_cache_gib <= 0:
            raise ValueError("megatron_teacher_cache_gib must be finite and positive")
        if self.srpo_route_positive_rewards and self.srpo_route_success_only:
            raise ValueError("Choose only one strict SRPO routing mode")
        if self.sibling_selection not in {"first_success", "best_reward"}:
            raise ValueError("Unknown sibling_selection")
        if self.demonstration_source not in {
            "sibling",
            "feedback_only",
            "expert_only",
            "expert_then_sibling",
            "sibling_then_expert",
        }:
            raise ValueError("Unknown demonstration_source")
        if not isinstance(self.expert_trace_key, str) or not self.expert_trace_key.strip():
            raise ValueError("expert_trace_key must be a nonempty dataset column name")
        if self.expert_trace_key in {
            "reward_model",
            "extra_info",
            "data_source",
            "ability",
        }:
            raise ValueError("expert_trace_key must not replace standard environment metadata")
        if "{expert_trace}" not in self.expert_trace_template:
            raise ValueError("expert_trace_template must contain {expert_trace}")
        if self.reduction not in {"global_token_mean", "upstream_microbatch_mean"}:
            raise ValueError("Unknown SDPO reduction")
        if self.teacher.kind not in {"self", "ema", "frozen"}:
            raise ValueError("teacher.kind must be self, ema or frozen")
        if self.teacher.kind == "self" and self.teacher.path:
            raise ValueError("teacher.path requires kind ema or frozen")
        if self.teacher.kind == "ema" and not 0 <= self.teacher.update_rate <= 1:
            raise ValueError("teacher.update_rate must be in [0, 1]")
        if self.teacher.storage not in {"cpu", "cuda"}:
            raise ValueError("teacher.storage must be cpu or cuda")
        if self.teacher_first and (self.teacher.kind == "self" or self.teacher.storage != "cpu"):
            raise ValueError("teacher_first requires an ema or frozen teacher with CPU storage")
        if self.loss not in {"distribution", "sampled_reverse_kl"}:
            raise ValueError("loss must be distribution or sampled_reverse_kl")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.loss == "sampled_reverse_kl" and self.alpha != 1:
            raise ValueError("sampled_reverse_kl requires alpha=1")
        if self.topk is not None and (isinstance(self.topk, bool) or self.topk < 1):
            raise ValueError("topk must be positive or null")
        for name in (
            "loss_chunk_size",
            "max_teacher_prompt_length",
            "max_teacher_sequence_length",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rollout_is_clip is not None and (not math.isfinite(self.rollout_is_clip) or self.rollout_is_clip <= 0):
            raise ValueError("rollout_is_clip must be finite and positive or null")
        if self.advantage_is_clip is not None and (
            not math.isfinite(self.advantage_is_clip) or self.advantage_is_clip <= 0
        ):
            raise ValueError("advantage_is_clip must be finite and positive or null")
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")
        if self.overflow not in {"skip", "error", "truncate_right"}:
            raise ValueError("overflow must be skip, error, or truncate_right")
        if not math.isfinite(self.entropy_beta) or self.entropy_beta < 0:
            raise ValueError("entropy_beta must be finite and >= 0")


@dataclass
class SDPOAlgorithmConfig(AlgorithmConfig):
    use_kl_loss: bool = False
    # The local worker computes the loss directly. This valid core registry name
    # is only used by the base worker constructor, never for SDPO optimization.
    policy_loss_type: str = "regular"


@dataclass
class SDPOTrainerConfig(TrainerConfig):
    placement: PlacementConfig = field(
        default_factory=lambda: PlacementConfig(colocate_all=False, colocate_policy_ref=False)
    )
    algorithm: SDPOAlgorithmConfig = field(default_factory=SDPOAlgorithmConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    train_batch_size: int = 32
    policy_mini_batch_size: int = 32
    remove_microbatch_padding: bool = False


SDPOConfig = make_config(trainer_cls=SDPOTrainerConfig)


def validate_sdpo_config(cfg: SDPOConfig) -> None:
    t, g, a = cfg.trainer, cfg.generator, cfg.trainer.algorithm
    if g.n_samples_per_prompt == 1:
        d = t.distillation
        if d.use_sample_routing:
            raise ValueError("G=1 requires distillation.use_sample_routing=false (no SRPO)")
        if d.demonstration_source == "sibling" and not d.include_environment_feedback:
            raise ValueError("G=1 requires expert traces or environment feedback; no siblings are available")
        if d.demonstration_source != "expert_only" and not d.exclude_self_demonstration:
            raise ValueError("G=1 requires exclude_self_demonstration=true to prevent self-conditioning")
    requirements = [
        (
            not g.vision_language_generator or t.distillation.vision_language,
            "distillation.vision_language=true for image inputs",
        ),
        (
            not t.distillation.vision_language or t.distillation.overflow != "truncate_right",
            "skip/error overflow for image inputs (never truncate image tokens)",
        ),
        (t.strategy in {"fsdp", "megatron"}, "trainer.strategy=fsdp or megatron"),
        (
            not t.fully_async.enabled and not t.fully_async.simulate_training,
            "synchronous training",
        ),
        (
            not g.step_wise_trajectories
            and (
                g.max_turns == 1
                or (
                    t.distillation.trajectory_distillation
                    and not g.batched
                    and g.use_conversation_multi_turn
                    and g.chat_template == type(g.chat_template)()
                    and not t.distillation.use_sample_routing
                    and t.distillation.demonstration_source == "feedback_only"
                )
            ),
            "single-turn generation or explicitly enabled feedback-only token-preserving trajectories (no SRPO)",
        ),
        (t.update_epochs_per_batch == 1, "trainer.update_epochs_per_batch=1"),
        (
            t.policy_mini_batch_size == t.train_batch_size,
            "policy_mini_batch_size=train_batch_size",
        ),
        (not t.remove_microbatch_padding, "trainer.remove_microbatch_padding=false"),
        (t.max_tokens_per_microbatch <= 0, "sample-based microbatching"),
        (t.policy.sequence_parallel_size == 1, "policy.sequence_parallel_size=1"),
        (not t.policy.inference_only_init, "a trainable policy"),
        (
            t.policy.model.lora.dropout == 0,
            "zero LoRA dropout for deterministic scoring",
        ),
        (t.critic.model.path is None, "no critic"),
        (
            not a.use_kl_loss and not a.use_kl_in_reward and not a.use_entropy_loss,
            "no auxiliary KL/entropy loss",
        ),
        (
            a.dynamic_sampling.type is None and not a.zero_variance_filter,
            "no reward-variance filtering",
        ),
        (
            not a.advantage_batch_normalize and a.loss_reduction == "token_mean",
            "token_mean without advantage normalization",
        ),
        (not t.update_ref_every_epoch, "trainer.update_ref_every_epoch=false"),
        (
            not g.inference_engine.enable_return_routed_experts
            and not t.policy.megatron_config.moe_enable_routing_replay,
            "no router replay",
        ),
        (not t.mtp.enabled, "no MTP"),
        (
            a.off_policy_correction == type(a.off_policy_correction)(),
            "default algorithm.off_policy_correction; use distillation.rollout_is_clip instead",
        ),
        (
            g.sampling_params.temperature > 0 and a.temperature == g.sampling_params.temperature,
            "matching positive sampling and loss temperatures",
        ),
        (
            t.distillation.max_teacher_sequence_length >= t.max_prompt_length + g.sampling_params.max_generate_length,
            "max_teacher_sequence_length >= max_prompt_length + max_generate_length (including fallback rows)",
        ),
        (
            not t.distillation.use_sample_routing or a.advantage_estimator == "grpo",
            "advantage_estimator=grpo when distillation.use_sample_routing=true",
        ),
        (
            t.distillation.advantage_is_clip is None or t.distillation.alpha == 1,
            "alpha=1 (reverse KL) for distillation.advantage_is_clip",
        ),
    ]
    for valid, description in requirements:
        if not valid:
            raise ValueError(f"This SDPO example requires {description}")
    if t.strategy == "megatron":
        validate_megatron_sdpo(t)


def validate_megatron_sdpo(t):
    """Explicit initial support boundary, also checked inside each GPU worker."""
    d, mg = t.distillation, t.policy.megatron_config
    requirements = [
        (
            mg.pipeline_model_parallel_size in {1, 2, 3},
            "pipeline_model_parallel_size=1, 2 or 3",
        ),
        (mg.context_parallel_size == 1, "context_parallel_size=1 (including VLMs)"),
        (mg.expert_model_parallel_size == 1, "expert_model_parallel_size=1"),
        (d.loss == "distribution", "distribution loss"),
        (d.reduction == "global_token_mean", "global_token_mean reduction"),
        (
            not d.use_sample_routing,
            "no SRPO",
        ),
        (
            d.advantage_is_clip is None or d.alpha == 1,
            "alpha=1 for advantage clipping",
        ),
        (
            not d.limit_response_logits,
            "limit_response_logits=false (Megatron has no HF logits_to_keep)",
        ),
        (not t.remove_microbatch_padding, "remove_microbatch_padding=false"),
        (not t.fused_lm_head_logprob, "fused_lm_head_logprob=false"),
        (mg.lora_config.lora_type == "lora", "standard LoRA"),
        (not mg.transformer_config_kwargs.get("fp8"), "BF16/FP32 training (no FP8)"),
        (
            not mg.ddp_config.overlap_grad_reduce,
            "overlap_grad_reduce=false for deferred gradient synchronization",
        ),
    ]
    for valid, description in requirements:
        if not valid:
            raise ValueError(f"Megatron SDPO currently requires {description}")
