"""Utilities for processing image data for VLM training."""

import io
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoImageProcessor


@dataclass
class ProcessedImage:
    """Result of processing an ImageChunk through the HF image processor."""

    pixel_values: torch.Tensor  # (num_patches, patch_embed_dim)
    image_grid_thw: torch.Tensor  # (1, 3)
    num_tokens: int  # number of LM placeholder tokens for this image


class VisionProcessor:
    """Converts ImageChunks to model-ready tensors using an HF image processor."""

    def __init__(self, processor: AutoImageProcessor):
        self._processor = processor

    @staticmethod
    def from_pretrained(model_path: str) -> "VisionProcessor":
        processor = AutoImageProcessor.from_pretrained(model_path)
        return VisionProcessor(processor)

    @property
    def image_token(self) -> str:
        """The placeholder token string for images, e.g. '<|image_pad|>'."""
        return self._processor.image_token

    def process_image(self, image_bytes: bytes, format: str) -> ProcessedImage:
        """Decode image bytes, run HF processor, return tensors + token count."""
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        output = self._processor.preprocess(pil_image, return_tensors="pt")
        pixel_values = output["pixel_values"]  # (num_patches, embed_dim)
        image_grid_thw = output["image_grid_thw"]  # (1, 3)
        grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
        num_tokens = int(grid_t * grid_h * grid_w // self._processor.merge_size**2)
        return ProcessedImage(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_tokens=num_tokens,
        )
