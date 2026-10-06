"""GLM-5.3-Flash vision-language model: the HF vision tower in front of the Megatron language model.

Port of the ``GLM53FlashModel`` wrapper in NVIDIA-NeMo/Megatron-Bridge#6044 (open), built on
SkyRL's ``GPTModel`` + ``glm5_next`` layer spec rather than that PR's ``HybridModel``.

The HF ``Glm5NextVisionModel`` (patch embed, ViT blocks, downsample, merger) is replicated on
every tensor-parallel rank of the first pipeline stage; its gradients are averaged across TP by
megatron-core's ``finalize_model_grads`` (``average_gradients_across_tp_domain``). Image features
replace the ``<|image|>`` placeholder embeddings before the sequence-parallel scatter, so the
language model is built with ``scatter_embedding_sequence_parallel=False``.

GLM-5.3-Flash's language model has no positional encoding (NoPE MLA/DSA, KDA), so a packed THD
stream needs no per-sample position rebuild: the placeholder mask is a token-id match over the
packed ``[1, T]`` stream, and the images' features are consumed in the same sample order. That
is what ``model_owns_packing`` declares to SkyRL's Megatron wrapper.
"""

import types
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


class Glm5NextVLModel(MegatronModule):
    """HF ``Glm5NextVisionModel`` + Megatron ``GPTModel`` for ``Glm5NextForConditionalGeneration``.

    Parameter names: ``visual.*`` (HF names under ``model.visual.*``) and ``language_model.*``
    (the text-only ``GPTModel`` names, prefixed).
    """

    # SkyRL's Megatron wrapper (megatron_utils.model_owns_vlm_packing) sends this model the
    # packed THD stream and leaves image placement to it.
    model_owns_packing = True

    def __init__(
        self,
        config,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)
        # TODO(xgui): CP > 1 (needs KDA CP, NovaSky-AI/SkyRL#2418). With model_owns_packing the
        # model receives the full packed stream on every CP rank, so it must inject vision
        # features first and then apply SkyRL's zigzag CP split itself.
        if config.context_parallel_size > 1:
            raise NotImplementedError(
                "GLM-5.3-Flash vision-language training supports context_parallel_size=1 only; "
                f"got {config.context_parallel_size}."
            )

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextModel,
            Glm5NextVisionModel,
        )

        if pre_process:
            self.visual = Glm5NextVisionModel._from_config(
                config.vision_config, attn_implementation=config.vision_attn_implementation
            )
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # finalize_model_grads reads these off the outer module.
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # HF feature extraction and placeholder masking, bound to this module: they read
        # ``self.visual`` and ``self.config.{image_token_id,video_start_token_id,...}``.
        self.get_image_features = types.MethodType(Glm5NextModel.get_image_features, self)
        self.get_placeholder_mask = types.MethodType(Glm5NextModel.get_placeholder_mask, self)

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def _keep_vision_rope_fp32(self) -> None:
        """Recompute the ViT rotary ``inv_freq`` in fp32 if a module-wide cast lowered it.

        It is a non-persistent buffer, so Megatron's ``Float16Module`` (``module.bfloat16()``)
        casts it with the parameters, while HF and vLLM keep it in fp32.
        """
        rope = self.visual.rotary_pos_emb
        if rope.inv_freq.dtype != torch.float32:
            device = rope.inv_freq.device
            rope.inv_freq = 1.0 / (
                rope.theta ** (torch.arange(0, rope.dim, 2, dtype=torch.float32, device=device) / rope.dim)
            )

    def _vision_trainable(self) -> bool:
        return any(p.requires_grad for p in self.visual.parameters())

    def _dummy_vision_term(self, like: Tensor) -> Tensor:
        """A zero-valued term that touches every trainable vision parameter.

        A microbatch without images would otherwise leave the vision tower out of the autograd
        graph, and DDP with ``overlap_grad_reduce`` asserts that every bucketed parameter's
        gradient hook fired. One 2x2-patch image (``spatial_merge_size**2`` patches, one token).
        """
        vc = self.config.vision_config
        merge = vc.spatial_merge_size
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        pixel_values = torch.zeros(merge * merge, patch_dim, device=like.device, dtype=like.dtype)
        grid_thw = torch.tensor([[1, merge, merge]], device=like.device, dtype=torch.long)
        features = self.get_image_features(pixel_values, grid_thw).pooler_output
        return torch.cat(features, dim=0).sum() * 0.0

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        pixel_values_videos: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        loss_mask: Optional[Tensor] = None,
        output_processor: Optional[Callable[..., Any]] = None,
        output_processor_context: Optional[Any] = None,
        **language_model_kwargs,
    ) -> Tensor:
        """Same positional layout as ``GPTModel.forward`` (input_ids, position_ids, attention_mask).

        ``input_ids`` is ``[B, S]`` (unpacked) or the packed ``[1, T]`` THD stream.
        ``position_ids`` are passed through unchanged; the language model is NoPE.
        ``language_model_kwargs`` (e.g. ``padding_mask`` from router replay) go to ``GPTModel``.
        """
        # TODO(xgui): video (pixel_values_videos / video_grid_thw; image and video share the
        # placeholder id and are told apart by the <|begin_of_video|> spans).
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError("GLM-5.3-Flash video inputs are not supported on the Megatron backend yet.")

        if self.pre_process and decoder_input is None:
            self._keep_vision_rope_fp32()
            # [S, B, H], not scattered across TP (scatter_embedding_sequence_parallel=False).
            embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)
            embeds = embeds.transpose(0, 1).contiguous()  # [B, S, H] for the HF placeholder mask

            # An image-free microbatch carries an empty [0, D] pixel_values tensor, not None.
            if pixel_values is not None and pixel_values.numel() > 0:
                if image_grid_thw is None:
                    raise ValueError("pixel_values were given without image_grid_thw.")
                image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
                image_embeds = torch.cat(image_embeds, dim=0).to(embeds.device, embeds.dtype)
                # Raises when the <|image|> count differs from the number of image features.
                image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=embeds, image_features=image_embeds)
                embeds = embeds.masked_scatter(image_mask, image_embeds)
            elif self.training and self._vision_trainable():
                embeds = embeds + self._dummy_vision_term(embeds)

            decoder_input = embeds.transpose(0, 1).contiguous()  # back to [S, B, H]
            if self.config.sequence_parallel:
                pg_collection = getattr(self.config, "_pg_collection", None)
                decoder_input = scatter_to_sequence_parallel_region(
                    decoder_input, group=pg_collection.tp if pg_collection is not None else None
                )

        if output_processor is not None:
            language_model_kwargs["output_processor"] = output_processor
            language_model_kwargs["output_processor_context"] = output_processor_context

        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,
            **language_model_kwargs,
        )

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Set ``requires_grad=False`` on the selected parts (same signature as Megatron-Bridge's VLMs).

        Vision model: patch embed, ViT blocks and post-norm. Vision projection: the 2x2
        downsample conv and the merger MLP, which map into the language model's hidden size.
        """
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if self.pre_process and freeze_vision_model:
            modules += [self.visual.patch_embed, self.visual.blocks, self.visual.post_layernorm]
        if self.pre_process and freeze_vision_projection:
            modules += [self.visual.downsample, self.visual.merger]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
