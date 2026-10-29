"""Mistral-format ML4 checkpoint as a Megatron-Bridge state source.

The bridge maps Megatron parameters to the ML4 trainer's module names, which are vLLM's
names for the Pixtral-wrapped checkpoint (``language_model.model.layers.{i}.self_attn.q_proj``
...). Exported LoRA adapters therefore carry vLLM's keys, as the FSDP trainer's do.

The checkpoint itself uses Mistral names (``layers.{i}.attention.wq``) and ModelOpt
quantization. This source presents it under the trainer names, dequantized:

* FP8 block weights (FP8_PB_WO and the FP8 format): ``fp8 * scale`` in FP32, exact
  (``loading.CheckpointReader.rows_dequant``);
* NVFP4 routed experts: ``E2M1(code) * E4M3(sf)`` (``qat.decode_nvfp4``, exact in BF16)
  times ``weight_scale_2`` in FP32, as ``<expert>.<proj>.weight``. The packed bytes and
  scales are exposed as well (``.weight_packed``, ``.weight_scale``, ``.weight_scale_2``,
  ``.input_scale``) for modules that keep the NVFP4 base;
* stacked params (indexer ``wk_weights_proj`` = [wk; weights_proj]) concatenated by rows.

Everything else is returned as stored. Tensor names and the plan come from
``ml4_train.loading.build_plan``, the FSDP loader's own name map.
"""

from typing import Dict, List

import torch
from megatron.bridge.models.hf_pretrained.state import StateSource

from skyrl.backends.skyrl_train.patches.megatron.ml4.ml4_train import loading, qat

_NVFP4_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


class ML4CheckpointStateSource(StateSource):
    """Read-only view of a consolidated ML4 checkpoint under trainer (vLLM) names."""

    def __init__(self, ckpt_dir: str) -> None:
        self.ckpt_dir = ckpt_dir
        self.reader = loading.CheckpointReader(ckpt_dir)
        plan, _ = loading.build_plan(self.reader)
        self._plan = plan
        self._keys: Dict[str, tuple] = {}
        for pname, srcs in plan.items():
            dts = {self.reader.dtype(s.ckpt_name) for s in srcs}
            if pname.endswith(".weight") and dts == {"U8"}:
                base = pname[: -len(".weight")]
                self._keys[pname] = ("nvfp4_dequant", base)
                self._keys[base + ".weight_packed"] = ("raw", pname)
            elif "U8" in dts:
                raise ValueError(f"{pname}: unexpected U8 tensor in a non-NVFP4 weight")
            elif (
                pname.endswith(_NVFP4_SUFFIXES)
                and pname.rsplit(".", 1)[0] + ".weight" in plan
                and {self.reader.dtype(s.ckpt_name) for s in plan[pname.rsplit(".", 1)[0] + ".weight"]} == {"U8"}
            ):
                self._keys[pname] = ("raw", pname)
            else:
                self._keys[pname] = ("dequant", pname)

    def __deepcopy__(self, memo):
        # Read-only view holding open safetensors handles (not copyable). Megatron deep-copies its config,
        # which reaches this source through the bridge's weight-loading pre-wrap hook; share it instead.
        return self

    def get_all_keys(self) -> List[str]:
        return sorted(self._keys)

    def _rows(self, pname: str, raw: bool) -> torch.Tensor:
        # stacked params: sources in row order (the plan lists them by checkpoint name)
        srcs = sorted(self._plan[pname], key=lambda s: s.row_offset)
        parts = []
        for s in srcs:
            shape = self.reader.shape(s.ckpt_name)
            if not shape:
                return self.reader.full(s.ckpt_name)
            if raw:
                parts.append(self.reader.rows(s.ckpt_name, 0, shape[0]))
            else:
                parts.append(self.reader.rows_dequant(s.ckpt_name, 0, shape[0]))
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    def _load_one(self, key: str) -> torch.Tensor:
        kind, ref = self._keys[key]
        if kind == "raw":
            t = self._rows(ref, raw=True)
            return t if t.dim() == 0 else t.view(torch.uint8)
        if kind == "nvfp4_dequant":
            packed = self._rows(ref + ".weight", raw=True).view(torch.uint8)
            scale = self._rows(ref + ".weight_scale", raw=True).view(torch.uint8)
            ws2 = self._rows(ref + ".weight_scale_2", raw=True).float()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            w = qat.decode_nvfp4(packed.to(dev), scale.to(dev), torch.float32)
            return (w * ws2.to(dev)).cpu()
        return self._rows(key, raw=False)

    def load_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        return {k: self._load_one(k) for k in keys}

    def has_glob(self, pattern: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatch(k, pattern) for k in self._keys)
