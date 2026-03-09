"""Multi-modal data processing utilities for SkyRL training."""

from typing import Optional

import torch

from skyrl.backends.skyrl_train.training_batch import TensorList


def extract_vision_data(
    all_pixel_values: list[torch.Tensor],
    all_image_grid_thw: list[torch.Tensor],
) -> tuple[Optional[TensorList], Optional[TensorList]]:
    """Extract and validate vision data for training batches.

    Args:
        all_pixel_values: Per-example pixel values, each of shape [num_patches, dim].
        all_image_grid_thw: Per-example image grid info, each of shape [num_images, 3].

    Returns:
        (pixel_values_tl, image_grid_thw_tl): TensorList wrappers, or (None, None) if
        all inputs are empty/None.
    """
    if not all_pixel_values or all(pv is None for pv in all_pixel_values):
        return None, None

    pixel_values_tl = TensorList([pv if pv is not None else torch.empty(0) for pv in all_pixel_values])
    image_grid_thw_tl = TensorList([thw if thw is not None else torch.empty(0, 3) for thw in all_image_grid_thw])
    return pixel_values_tl, image_grid_thw_tl
