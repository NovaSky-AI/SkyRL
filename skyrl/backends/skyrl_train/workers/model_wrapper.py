# This code is adapted from OpenRLHF and OpenReasonerZero
# https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/models.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from flash_attn.bert_padding import pad_input, unpad_input
from loguru import logger
from packaging.version import Version
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    canonical_token_metadata_layout,
)
from skyrl.backends.skyrl_train.distributed.ulysses.utils import (
    gather_outputs_and_unpad,
    ulysses_pad_and_slice_inputs,
)
from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_ENTROPY_MASK_KEY,
    SAMPLE_SUPPORT_NO_ROW,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    missing_sample_support_message,
    sample_support_row_ids_in_batch_positions,
    score_aligned_sample_support,
)
from skyrl.backends.skyrl_train.utils.torch_utils import (
    chunked_entropy_from_logits,
    logprobs_from_logits,
)

_NO_TRAJECTORY = -1


@dataclass(frozen=True)
class _SampleSupportChannels:
    """Token-aligned metadata transformed alongside model inputs."""

    row_ids: torch.Tensor
    loss_mask: torch.Tensor
    trajectory_ids: torch.Tensor

    @classmethod
    def build(
        cls,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_support: PackedTensor,
        loss_mask: torch.Tensor,
    ) -> "_SampleSupportChannels":
        """Place channels in canonical ``[batch, seq_len]`` positions."""
        layout = canonical_token_metadata_layout(attention_mask)
        target_loss_mask = torch.zeros_like(sequences, dtype=torch.bool)
        target_loss_mask[:, sequences.shape[1] - loss_mask.shape[1] :] = loss_mask.to(torch.bool)
        trajectory_ids = torch.arange(sequences.shape[0], device=sequences.device).unsqueeze(1)
        return cls(
            row_ids=sample_support_row_ids_in_batch_positions(sample_support, layout),
            loss_mask=target_loss_mask,
            trajectory_ids=trajectory_ids.expand_as(sequences),
        )

    def unpad(self, nnz_indices: torch.Tensor) -> "_SampleSupportChannels":
        """Drop padding positions exactly as ``unpad_input`` does for the tokens."""

        def gather(channel: torch.Tensor) -> torch.Tensor:
            return channel.reshape(-1).index_select(0, nnz_indices).unsqueeze(0)

        return _SampleSupportChannels(
            row_ids=gather(self.row_ids),
            loss_mask=gather(self.loss_mask),
            trajectory_ids=gather(self.trajectory_ids),
        )

    def roll_to_logit_positions(self) -> "_SampleSupportChannels":
        """Shift the loss mask onto the logit that predicts each token, like ``sequences_rolled``."""
        return _SampleSupportChannels(
            row_ids=self.row_ids,
            loss_mask=torch.roll(self.loss_mask, shifts=-1, dims=1),
            trajectory_ids=self.trajectory_ids,
        )

    def slice_for_sequence_parallel(self, sp_size: int) -> "_SampleSupportChannels":
        """Take this rank's Ulysses shard, padding with each channel's own sentinel."""
        return _SampleSupportChannels(
            row_ids=ulysses_pad_and_slice_inputs(
                self.row_ids, sp_size=sp_size, input_padding_value=SAMPLE_SUPPORT_NO_ROW
            )[0],
            loss_mask=ulysses_pad_and_slice_inputs(self.loss_mask, sp_size=sp_size)[0],
            trajectory_ids=ulysses_pad_and_slice_inputs(
                self.trajectory_ids, sp_size=sp_size, input_padding_value=_NO_TRAJECTORY
            )[0],
        )


class HFModelWrapper(nn.Module):
    """
    Base class for wrapped HF models in reinforcement learning.

    This class serves as a foundation for implementing various model roles.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        lora_init_method (str, optional): Initialization method for LoRA layers. Defaults to "kaiming".
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        exclude_modules (list, optional): List of modules to exclude from applying LoRA. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        # TODO(shu): combine all LoRA specific configs into one place?
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        lora_init_method="kaiming",
        target_modules=None,
        exclude_modules=None,
        device_map=None,
        temperature=1.0,
        use_liger_kernel=False,
        sequence_parallel_size=1,
        remove_microbatch_padding: bool = False,
        use_torch_compile: bool = False,
        model_config_kwargs: dict = {},
        meta_init: bool = False,
        language_model_only: bool = False,
        logprobs_chunk_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.sequence_parallel_size = sequence_parallel_size
        self.attn_implementation = "flash_attention_2" if use_flash_attention_2 else "sdpa"
        self.remove_microbatch_padding = remove_microbatch_padding
        self.is_vlm = False
        if remove_microbatch_padding:
            assert (
                self.attn_implementation == "flash_attention_2"
            ), "Flash attention 2 should be used for `remove_microbatch_padding`"

        if isinstance(pretrain_or_model, str):
            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            model_config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True, **model_config_kwargs)

            if language_model_only:
                logger.info("[VLM] language_model_only=True, skipping vision encoder initialization")
            else:
                self.is_vlm = (
                    hasattr(model_config, "vision_config") and getattr(model_config, "vision_config") is not None
                )
                if self.is_vlm:
                    logger.info(
                        f"[VLM] Config {type(model_config).__name__} has a vision config, "
                        "using AutoModelForImageTextToText"
                    )
                    # NOTE: In future transformers releases (> 5.0.0), all multimodal models can use AutoModelForMultimodalLM.
                    model_class = AutoModelForImageTextToText

            model_config._attn_implementation = self.attn_implementation

            if meta_init:
                with torch.device("meta"):
                    self.model = model_class.from_config(model_config, trust_remote_code=True)
                target_dtype = torch.bfloat16 if bf16 else torch.float32
                # Cast just params + persistent buffers to the target dtype;
                # leave non-persistent buffers at their init dtype. For example, transformers
                # builds `Qwen3RotaryEmbedding.inv_freq` via a RoPE init that hardcodes
                # `dtype=torch.float`, so it is fp32 no matter the model dtype, which rank-0's
                # `from_pretrained` preserves but a blanket `.to` would clobber.
                # SEE: https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/modeling_rope_utils.py#L177-L178
                # Otherwise, e.g., non-rank-0's `inv_freq` is bf16 while rank-0's stays fp32,
                # so the init-time rank-0→all broadcast that seeds these buffers copies
                # rank-0's fp32 bytes into that half-width bf16 buffer as huge garbage values
                non_persistent_names = {n for n, _ in self.model.named_non_persistent_buffers()}
                for p in self.model.parameters():
                    p.data = p.data.to(target_dtype)
                for name, buf in self.model.named_buffers():
                    if name not in non_persistent_names:
                        buf.data = buf.data.to(target_dtype)
            else:
                self.model = model_class.from_pretrained(
                    pretrain_or_model,
                    config=model_config,
                    trust_remote_code=True,
                    attn_implementation=self.attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                    device_map=device_map,
                )

            # gpt oss
            if Version(transformers.__version__) >= Version("4.56.2"):
                from transformers import GptOssConfig

                if isinstance(self.model.config, GptOssConfig):
                    # patch attention with Unsloth's flex attn
                    from transformers import AttentionInterface, AttentionMaskInterface

                    from skyrl.backends.skyrl_train.patches.gptoss.patch_transformers import (
                        custom_attention,
                        custom_attention_mask,
                        patch_GptOssAttention,
                    )

                    AttentionInterface.register("custom_flex", custom_attention)
                    AttentionMaskInterface.register("custom_flex", custom_attention_mask)
                    # set attention implementation to be `custom_flex`
                    self.model.set_attn_implementation("custom_flex")
                    self.attn_implementation = "custom_flex"
                    # NOTE: Even though we set a custom attn implementation, we
                    # also patch the full attention function for GPT OSS
                    patch_GptOssAttention()

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    exclude_modules=exclude_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    init_lora_weights=True if lora_init_method == "kaiming" else lora_init_method,
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                # Skip for granitemoehybrid: its decoder layers don't return router
                # logits, so enabling this flag causes an IndexError in
                # load_balancing_loss_func when it tries to access empty gate_logits.
                if model_config.get("model_type") == "granitemoehybrid":
                    logger.info(
                        "[MoE] granitemoehybrid detected, skipping output_router_logits (decoder layers don't return router logits)"
                    )
                else:
                    logger.info("[MoE] set output_router_logits as True")
                    self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

        self.logprobs_chunk_size = logprobs_chunk_size

        # TODO (sumanthrh): do the same for `logprobs_from_logits` and test.
        # Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/#efficient-solution
        self.chunked_entropy_from_logits_fn = (
            torch.compile(chunked_entropy_from_logits, dynamic=True)
            if use_torch_compile
            else chunked_entropy_from_logits
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, list[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_output=False,
        compute_entropy=False,
        entropy_requires_grad=True,
        pixel_values: Optional[TensorList] = None,
        image_grid_thw: Optional[TensorList] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        sample_support: Optional[PackedTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        enable_sample_support_replay: bool = False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        has_image_inputs = pixel_values is not None or image_grid_thw is not None
        support_channels = None
        if enable_sample_support_replay:
            if sample_support is None:
                raise ValueError(missing_sample_support_message("FSDP"))
            if loss_mask is None:
                raise ValueError("sample-support replay is enabled but the microbatch has no loss mask")
            support_channels = _SampleSupportChannels.build(sequences, attention_mask, sample_support, loss_mask)
        if self.is_vlm:
            # VLMs use model specific 3D positional IDs, meaning sequence packing can not be supported.
            # Sequence packing requires computing position IDs, but position IDs for VLMs are 3D and require
            # model specific logic to compute.
            assert (
                not self.remove_microbatch_padding
            ), "remove_microbatch_padding is not supported with VLM vision inputs"
            assert self.sequence_parallel_size == 1, "Sequence parallelism is not supported with VLM vision inputs"

            if has_image_inputs:
                # Convert TensorList -> concatenated tensors for the HF model
                if isinstance(pixel_values, TensorList):
                    pixel_values = torch.cat(pixel_values.tensors, dim=0)
                if isinstance(image_grid_thw, TensorList):
                    image_grid_thw = torch.cat(image_grid_thw.tensors, dim=0)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        sequences_fwd = sequences
        position_ids_fwd = position_ids
        attention_mask_fwd = attention_mask
        if self.remove_microbatch_padding:
            with torch.no_grad():
                # Removes padding to get a packed tensor. `unpad_input` expects 3 dimensional tensor so we unsqueeze first
                sequences_fwd, nnz_indices, _, _, _ = unpad_input(
                    sequences.unsqueeze(-1), attention_mask=attention_mask
                )
                # (nnz, 1) -> (1, nnz)
                sequences_fwd = sequences_fwd.transpose(0, 1)
                position_ids_fwd, _, _, _, _ = unpad_input(position_ids.unsqueeze(-1), attention_mask)
                # (nnz, 1) -> (1, nnz)
                position_ids_fwd = position_ids_fwd.transpose(0, 1)
                if support_channels is not None:
                    support_channels = support_channels.unpad(nnz_indices)
                attention_mask_fwd = None  # no attention mask with FA 2

        sequences_rolled = torch.roll(sequences_fwd, shifts=-1, dims=1)
        if support_channels is not None:
            support_channels = support_channels.roll_to_logit_positions()
        if self.sequence_parallel_size > 1:
            # NOTE: don't pass any attn mask with sample packing
            attention_mask_fwd = None if self.remove_microbatch_padding else attention_mask_fwd

            # slice for sequence parallelism
            # (bsz, seqlen) -> (bsz, seqlen//sp_size)
            sequences_fwd, position_ids_fwd, attention_mask_fwd, pad_size = ulysses_pad_and_slice_inputs(
                sequences_fwd, position_ids_fwd, attention_mask_fwd, self.sequence_parallel_size
            )
            sequences_rolled, _, _, _ = ulysses_pad_and_slice_inputs(
                sequences_rolled, None, None, self.sequence_parallel_size
            )
            if support_channels is not None:
                support_channels = support_channels.slice_for_sequence_parallel(self.sequence_parallel_size)

        if self.is_vlm:
            # NOTE: transformers v5 introduced `mm_token_type_ids` to distinguish text
            # vs. multimodal tokens, and expects it to be populated at tokenization.
            # However, vLLM doesn't support transformers v5 yet so no mm_token_type_ids are
            # returned. For now we populate it here for images (0 = text, 1 = image token).
            if image_grid_thw is not None and mm_token_type_ids is None:
                mm_token_type_ids = (sequences_fwd == self.model.config.image_token_id).long()

            vlm_kwargs = dict(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            if mm_token_type_ids is not None:
                vlm_kwargs["mm_token_type_ids"] = mm_token_type_ids
            output = self.model(
                sequences_fwd,
                attention_mask=attention_mask_fwd,
                position_ids=None,
                **vlm_kwargs,
            )
        # NOTE (sumanthrh): Once we have position_ids, we don't need attention mask with flash attention.
        elif self.remove_microbatch_padding and self.attn_implementation == "flash_attention_2":
            # NOTE (sumanthrh): Don't use attention mask. position_ids is enough.
            # Not using attention mask leads to higher perf since flash attention varlen func is enabled
            output = self.model(sequences_fwd, attention_mask=None, position_ids=position_ids_fwd)
        else:
            output = self.model(sequences_fwd, attention_mask=attention_mask_fwd, position_ids=position_ids_fwd)

        logits_BSV = output["logits"]
        logits_BSV.div_(temperature)

        support_entropy = None
        support_entropy_mask = None
        if support_channels is not None:
            # FSDP supplies unsharded, temperature-scaled logits.
            support_scores = score_aligned_sample_support(
                logits_BSV,
                sequences_rolled,
                support_channels.row_ids,
                support_channels.loss_mask,
                sample_support,
                vocab_start_index=0,
                vocab_end_index=logits_BSV.shape[-1],
                tp_group=None,
                inference_only=not torch.is_grad_enabled(),
                compute_entropy=compute_entropy,
                entropy_requires_grad=compute_entropy and entropy_requires_grad,
                trajectory_ids=support_channels.trajectory_ids,
                num_trajectories=sequences.shape[0],
            )
            log_probs = support_scores.logprobs
            support_entropy = support_scores.entropy
            support_entropy_mask = support_scores.valid_mask
        else:
            # NOTE: this is slightly inaccurate with sample packing because last token from nth seq -> first token of n+1th seq loss is added.
            log_probs = logprobs_from_logits(
                logits_BSV,
                sequences_rolled,
                inplace_backward=True,
            )

        batch_size, seqlen = attention_mask.shape

        def to_canonical_batch_positions(values: torch.Tensor) -> torch.Tensor:
            """Undo the Ulysses slice and then the microbatch unpadding, as ``log_probs`` takes.

            Result is (1, nnz) with packing and no SP, else (B, S).
            """
            if self.sequence_parallel_size > 1:
                dim = values.ndim - 1
                values = gather_outputs_and_unpad(values, gather_dim=dim, unpad_dim=dim, padding_size=pad_size)
            if self.remove_microbatch_padding:
                # Pad token ID used by flash attention is 0.
                values = pad_input(
                    values.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
                ).squeeze(-1)
            return values

        log_probs = to_canonical_batch_positions(log_probs)

        if compute_entropy:
            if support_entropy is not None:
                assert support_entropy_mask is not None
                output["entropy"] = to_canonical_batch_positions(support_entropy)
                # The gather and the unpad both carry a float payload, so the mask rides as one.
                # Padding positions score no support row, which is what a zero fill means here.
                output[SAMPLE_SUPPORT_ENTROPY_MASK_KEY] = (
                    to_canonical_batch_positions(support_entropy_mask.to(support_entropy.dtype)) > 0
                )
            else:
                # For sample packing: entropy is calculated on unpacked data, so no attention mask needed
                # For non-sample packing: pass the attention mask to exclude padding tokens.
                # attention_mask_fwd may be sliced (if sequence_parallel_size > 1) or full.
                padding_mask = None if self.remove_microbatch_padding else attention_mask_fwd
                output["entropy"] = to_canonical_batch_positions(
                    self.chunked_entropy_from_logits_fn(
                        logits_BSV,
                        requires_grad=entropy_requires_grad,
                        attention_mask=padding_mask,
                        chunk_size=self.logprobs_chunk_size,
                    )
                )

        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                num_actions = np.array(num_actions)
        action_log_probs = log_probs[:, -num_actions - 1 : -1]

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()


def _get_critic_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="value_head",
    sequence_parallel_size=1,
    remove_microbatch_padding: bool = False,
):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.sequence_parallel_size = sequence_parallel_size
            self.remove_microbatch_padding = remove_microbatch_padding
            if remove_microbatch_padding:
                assert (
                    config._attn_implementation == "flash_attention_2"
                ), "Flash attention must be used with remove_microbatch_padding"

            if self.sequence_parallel_size > 1:
                logger.info("Critic model using sequence parallelism with size: ", self.sequence_parallel_size)

            self.post_init()

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            input_ids_fwd = input_ids
            position_ids_fwd = position_ids
            attention_mask_fwd = attention_mask

            if self.remove_microbatch_padding:
                with torch.no_grad():
                    # remove padding. `unpad_input` expects 3 dimensional tensor
                    input_ids_fwd, nnz_indices, _, _, _ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    input_ids_fwd = input_ids_fwd.transpose(0, 1)
                    position_ids_fwd, _, _, _, _ = unpad_input(
                        position_ids.unsqueeze(-1), attention_mask=attention_mask
                    )
                    # (nnz, 1) -> (1, nnz)
                    position_ids_fwd = position_ids_fwd.transpose(0, 1)
                    # don't use attention mask with FA2
                    attention_mask_fwd = None

            if self.sequence_parallel_size > 1:
                assert self.remove_microbatch_padding, "remove_microbatch_padding must be true for sequence parallelism"
                # don't pass any attention mask for flash attention 2. this will save an all gather.
                attention_mask_fwd = None if self.config._attn_implementation == "flash_attention_2" else attention_mask
                # slice for sequence parallelism
                # (bsz, seqlen) -> (bsz, seqlen//sp_size)
                input_ids_fwd, position_ids_fwd, attention_mask_fwd, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_fwd, position_ids_fwd, attention_mask_fwd, self.sequence_parallel_size
                )

            if self.sequence_parallel_size > 1 and self.config._attn_implementation == "flash_attention_2":
                outputs = getattr(self, self.base_model_prefix)(input_ids_fwd, position_ids=position_ids_fwd)
            else:
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids_fwd, attention_mask=attention_mask_fwd, position_ids=position_ids_fwd
                )
            last_hidden_states_BSH = outputs["last_hidden_state"]

            if self.sequence_parallel_size > 1:
                last_hidden_states_SH = last_hidden_states_BSH.squeeze(0)
                # (seqlen*bsz//sp_size, 1) -> (seqlen*bsz, 1)
                last_hidden_states_SH = gather_outputs_and_unpad(
                    last_hidden_states_SH, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
                last_hidden_states_BSH = last_hidden_states_SH.unsqueeze(0)

            values_BSH = getattr(self, self.value_head_prefix)(last_hidden_states_BSH)

            if self.remove_microbatch_padding:
                # add padding back - postprocess logits to be compatible with original tensors
                batch_size, seqlen = attention_mask.shape
                # (1, nnz, 1) -> (nnz, 1) -> (batch_size, seqlen, 1)
                values_BSH = pad_input(values_BSH.squeeze(0), indices=nnz_indices, batch=batch_size, seqlen=seqlen)

            values = values_BSH.squeeze(-1)[:, :-1]

            if num_actions is None:
                assert return_output
                return outputs

            action_values = values[:, -num_actions:]

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    exclude_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    init_value_head: bool = False,
    value_head_prefix="value_head",
    device_map=None,
    sequence_parallel_size=1,
    remove_microbatch_padding: bool = False,
    model_config_kwargs: dict = {},
    meta_init: bool = False,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Type of sequence classification model. Only `critic` is supported.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert model_type == "critic", f"Only model_type critic is supported, got: {model_type}."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, **model_config_kwargs)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "sdpa"

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_critic_model(
        base_pretrained_class,
        base_class,
        value_head_prefix,
        sequence_parallel_size=sequence_parallel_size,
        remove_microbatch_padding=remove_microbatch_padding,
    )

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    if meta_init:
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(), torch.device("meta"):
            model = cls_class(config)
            model.to(dtype=torch.bfloat16 if bf16 else torch.float32)
    else:
        model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            quantization_config=nf4_config,
            device_map=device_map,
            **kwargs,
        )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        if model_config.get("model_type") == "granitemoehybrid":
            logger.info(
                "[MoE] granitemoehybrid detected, skipping output_router_logits (decoder layers don't return router logits)"
            )
        else:
            logger.info("[MoE] set output_router_logits as True")
            model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model
