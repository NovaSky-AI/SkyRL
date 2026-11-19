"""Inkling2 (HF ``inkling2_mm_model``) for the Megatron backend.

Shaped for an upstream home in Megatron-Bridge:

- ``quant``: dequantization of the packed NVFP4 / MXFP8 routed-expert weights of the released
  checkpoint to BF16 (used by the offline converter and the Bridge import path).
"""
