"""Flex attention utilities from Unsloth Zoo"""

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "HAS_FLEX_ATTENTION",
    "FLEX_ATTENTION_BLOCK_SIZE",
    "_flex_attention",
    "flex_attention",
    "create_block_mask_cached",
    "create_block_mask",
    "compiled_create_block_mask",
    "FlexAttentionCache",
    "causal_mask",
    "generate_causal_mask_with_padding",
    "generate_decoding_causal_mask_with_padding",
    "generate_sliding_window_mask",
    "generate_sliding_window_mask_with_padding",
    "generate_decoding_sliding_window_mask_with_padding",
]

import torch
import functools

FLEX_ATTENTION_KV_INCREMENT = 512

try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )

    HAS_FLEX_ATTENTION = True
    from torch.nn.attention.flex_attention import _score_mod_signature, _mask_mod_signature

    try:
        # Determine kernel_options since low memory GPUs will go out of memory
        # InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 65536 Hardware limit:65536 Reducing block sizes or `num_stages` may help.
        # See https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459
        # https://github.com/pytorch/pytorch/issues/133254#issuecomment-2539969593
        vram_of_gpu = min(
            torch.cuda.memory.mem_get_info(i)[-1] / 1024 / 1024 / 1024 for i in range(torch.cuda.device_count())
        )
        kernel_options = None
        if vram_of_gpu <= 16:
            kernel_options = {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
            }
            _flex_attention = functools.partial(_flex_attention, kernel_options=kernel_options)
        elif vram_of_gpu <= 24:
            kernel_options = {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            }
            _flex_attention = functools.partial(_flex_attention, kernel_options=kernel_options)
    except Exception:
        pass
    flex_attention = torch.compile(_flex_attention)

    @functools.lru_cache
    def create_block_mask_cached(mask_mod, M, N, device="cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return _create_block_mask(mask_mod, None, None, M, N, device=device)

    @functools.lru_cache
    def create_block_mask(mask_mod, bsz, head, M, N, device="cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return _create_block_mask(mask_mod, bsz, head, M, N, device=device)

    def compiled_create_block_mask_cached(mask_mod, M, N, device="cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        # See https://github.com/meta-pytorch/attention-gym/issues/15#issuecomment-2284148665
        # _compile MUST be on to reduce VRAM otherwise O(N^2) usage
        return _create_block_mask(mask_mod, None, None, M, N, device=device, _compile=True)

    def compiled_create_block_mask(mask_mod, bsz, head, M, N, device="cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        # _compile MUST be on to reduce VRAM otherwise O(N^2) usage
        return _create_block_mask(mask_mod, bsz, head, M, N, device=device, _compile=True)

    def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
        """Causal mask for Flex Attention"""
        return q_idx >= kv_idx

    def generate_causal_mask_with_padding(padding_start_idx=None):
        """
        Causal mask for Flex Attention with left padding support.
        Normal causal mask:
            k0 k1 k2 k3 k4
        q0   X
        q1   X  X
        q2   X  X  X
        q3   X  X  X  X
        q4   X  X  X  X  X
        If we add 2 tokens as padded tokens, we get:
            #0 #1 k2 k3 k4
        #0
        #1
        q2         X
        q3         X  X
        q4         X  X  X
        Assume padding_start_idx == [2]
        """
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
            """Causal mask for Flex Attention"""
            q_start = q_idx >= padding_start_idx[batch_idx]
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return q_start & k_start & (q_idx >= kv_idx)

        causal_mask.__name__ = causal_mask.__doc__ = "causal_mask_with_left_padding"
        return causal_mask

    def generate_decoding_causal_mask_with_padding(padding_start_idx=None):
        """
        For decoding purposes only. We remove q_padded since decoding attends to 1 q
        Assume padded tokens = 5
            #0 #1 #2 #3 #4 k5 k6
        #0   #
        #1   #  #
        #2   #  #  #
        #3   #  #  #  #
        #4   #  #  #  #  #
        q5   #  #  #  #  #  X
        q6   #  #  #  #  #  X  X
        """
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
            """Causal mask for Flex Attention"""
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return k_start & (q_idx >= kv_idx)

        causal_mask.__name__ = causal_mask.__doc__ = "decoding_causal_mask_with_left_padding"
        return causal_mask

    @functools.lru_cache
    def generate_sliding_window_mask(window_size: int):
        """Sliding window mask for Flex Attention"""

        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            """
            [NOTE] Official PyTorch uses <= window_size which is (window_size+1) tokens
            Ie if window_size = 3, HuggingFace does the below:
            X
            X X
            X X X
              X X X
                X X X
            Whilst official PyTorch does (ie attends to the last 3 and itself)
            X
            X X
            X X X
            X X X X
              X X X X
            """
            windowed_mask = q_idx - kv_idx < window_size
            # Official PyTorch attends to 1 extra token
            # windowed_mask = q_idx - kv_idx <= window_size
            return causal_mask & windowed_mask

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}"
        return sliding_window

    def generate_sliding_window_mask_with_padding(window_size: int, padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            windowed_mask = q_idx - kv_idx < window_size
            q_padded = q_idx >= padding_start_idx[batch_idx]
            k_padded = kv_idx >= padding_start_idx[batch_idx]
            return q_padded & k_padded & causal_mask & windowed_mask

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_with_left_padding_{window_size}"
        return sliding_window

    def generate_decoding_sliding_window_mask_with_padding(window_size: int, padding_start_idx=None):
        """
        We cannot use padding_start_idx[batch_idx] for SWA decoding since
        assume padding_start_idx=[3406, 4000, 0] and SW=128 then it'll always
        be masked since the KV size=128.

        Since we set padded tokens = 0 always, we simply return the generic SWA.
        """
        return generate_sliding_window_mask(window_size)

    # For inference see https://pytorch.org/blog/flexattention-for-inference
    def get_score_mod_w_offset(score_mod: _score_mod_signature, _offset: torch.tensor):
        def _score_mod(score, b, h, q, kv):
            return score_mod(score, b, h, q + _offset, kv)

        return _score_mod

    def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: torch.tensor):
        def _mask_mod(b, h, q, kv):
            return mask_mod(b, h, q + _offset, kv)

        return _mask_mod

    # Used for every attention layer
    class FlexAttentionCache:
        __slots__ = (
            "offset",
            "offset_tensor",
            "mask_mod_with_offset",
            "block_mask",
            "mask_mod",
            "max_length",
            "block_size",
            "sliding_window",
            "block_mask_slice",
        )

        def __init__(self, key, mask_mod, sliding_window):
            bsz, heads_KV, qlen_KV, dim = key.shape
            if sliding_window is None:
                """
                Normal causal mask:
                    k0 k1 k2 k3 k4
                q0   X
                q1   X  X
                q2   X  X  X
                q3   X  X  X  X
                q4   X  X  X  X  X
                During decoding:
                    k0 k1 k2 k3 k4
                q0
                q1
                q2
                q3
                q4   X  X  X  X  X
                But q_index = 0, so we need an offset = 4
                If no offset, we get:
                   k0 k1 k2 k3 k4
                q0
                q1
                q2
                q3
                q4  X
                Which is wrong. We need q_idx=0 + offset=4 = 0+4 = 4
                Note it's offset==index since it's indexed from 0 as seen in
                https://pytorch.org/blog/flexattention-for-inference/
                https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L91
                """
                # Get next multiple of FLEX_ATTENTION_KV_INCREMENT
                div, mod = divmod(qlen_KV, FLEX_ATTENTION_KV_INCREMENT)
                n = FLEX_ATTENTION_KV_INCREMENT * div + (FLEX_ATTENTION_KV_INCREMENT if mod != 0 else 0)
                self.offset = qlen_KV - 2  # Minus two since we need the block mask to use the saved offset_tensor
                # Minus 2 and not -1 since we do pre-incrementing and not post-incrementing
                # See self.offset += 1
                if self.offset <= -2:
                    # Minimum is -1
                    self.offset = -1
                    # During decoding we do self.offset += 1, so self.offset = 0
                self.sliding_window = None
            else:
                """
                Sliding window of 2 + causal mask:
                    k0 k1 k2 k3 k4
                q0   X
                q1   X  X
                q2      X  X
                q3         X  X
                q4            X  X
                During decoding:
                    k0 k1 k2 k3 k4
                q0
                q1
                q2
                q3
                q4            X  X
                If we set cache_implementation = "static" which we assume, we don't use an offset
                since the K is a rolling matrix of the past window size.
                Ie if sliding_window = 2, K is shape (2, dim). So in actuality, we get:
                    -- -- -- k3 k4
                q0
                q1
                q2
                q3
                q4            X  X
                But since we use a rolling matrix, offset = sliding_window-1 always ie 2-1 = 1
                    -- -- -- k0 k1
                q0
                q1
                q2
                q3(0)
                q4(1)         X  X
                Note it's offset==index since it's indexed from 0 as seen in
                https://pytorch.org/blog/flexattention-for-inference/
                https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L91
                For sliding window, it's always sliding_window - 1
                since 128 means index 127
                """
                n = sliding_window
                self.offset = min(sliding_window, qlen_KV) - 2  # Minus 2 since block mask is indexing
                if self.offset <= -2:
                    # Minimum is -1
                    self.offset = -1
                    # During decoding we do self.offset += 1, so self.offset = 0
                self.sliding_window = sliding_window - 1  # Minus 1 since token 128 means index 127
            self.offset_tensor = torch.tensor(self.offset, device=key.device, dtype=torch.int32)
            self.block_mask = compiled_create_block_mask(mask_mod, bsz, heads_KV, n, n, device=key.device)
            self.mask_mod = mask_mod
            self.max_length = n
            self.block_size = self.block_mask.BLOCK_SIZE[0]
            self.mask_mod_with_offset = get_mask_mod_w_offset(self.mask_mod, self.offset_tensor)
            self.block_mask_slice = None

        def __call__(self, key):
            bsz, heads_KV, qlen_KV, dim = key.shape
            # We increment beforehand to get the correct index since offset_tensor is used
            #                                    Assume sliding_window=128-1 = 127
            #                                    offset=126, so offset+1 = 127
            if (self.sliding_window is None) or (self.offset < self.sliding_window):
                self.offset += 1
                self.offset_tensor.add_(1)
            elif self.sliding_window is not None:
                # Quick return since sliding window mask has the same block mask always
                # Can only enter here if (self.offset < self.sliding_window) fails
                # ie the maximum sliding window has been reached already
                return self.block_mask_slice
            if self.offset >= self.max_length:
                # Must be >= since offset=127, max_length=128 means size=127+1=128
                # since we do zero indexing
                self.max_length += FLEX_ATTENTION_KV_INCREMENT
                self.block_mask = compiled_create_block_mask(
                    self.mask_mod, bsz, heads_KV, self.max_length, self.max_length, device=key.device
                )
                self.block_size = self.block_mask.BLOCK_SIZE[0]
            block_offset = self.offset // self.block_size
            block_mask_slice = self.block_mask[:, :, block_offset]
            block_mask_slice.mask_mod = self.mask_mod_with_offset
            # Must set seq_lengths as seen in
            # https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L80
            block_mask_slice.seq_lengths = (1, qlen_KV)
            self.block_mask_slice = block_mask_slice
            return block_mask_slice

    pass

except Exception:
    HAS_FLEX_ATTENTION = False
    FLEX_ATTENTION_BLOCK_SIZE = None
    flex_attention = None
    create_block_mask_cached = None
    causal_mask = None
    generate_sliding_window_mask = None
    FlexAttentionCache = None
pass
