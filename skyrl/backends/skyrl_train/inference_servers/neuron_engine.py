"""In-process, transformers-native inference engine for TorchNeuron (Trainium).

Unlike the vLLM-backed :class:`RemoteInferenceClient`, this engine drives the model
directly with ``transformers`` ``model.generate`` on the ``neuron`` (PrivateUse1)
device.  It exists because vLLM cannot serve on Trainium: vLLM requires CUDA torch and
ships no weight-synchronization path for RL, while ``torch-neuronx`` exposes a plain
``torch.nn.Module`` that ``model.generate`` already supports (this is the same path the
TRL GRPO leg uses).

This first cut implements the *text* RL path end-to-end, plus the multimodal
(VLM) path over a vLLM-free codec.  Two follow-ups remain:

1. **Processed-logprob exactness** -- ``_sampled_token_logprobs`` re-implements
   transformers' sampling filters (temperature/top-k/min-p/top-p) to reproduce vLLM's
   "processed logprob of the sampled token".  It must be validated bit-for-bit against
   the trainer's ``processed_logprobs`` forward on a neuron instance.
2. **Full-weight sync routing** -- SkyRL syncs full weights by bypassing
   ``InferenceEngineInterface`` (vLLM's own trainer engine).  Routing that through
   ``update_named_weights`` (and placing the engine relative to the FSDP workers) is a
   separate change; LoRA sync already flows through this interface.

The VLM path uses ``skyrl.backends.mm_codec`` (safetensors + a ``skyrl:torch:`` prefix)
for ``kwargs_data`` instead of vLLM's ``mm_serde``, so it does not require vLLM.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from skyrl.backends.mm_codec import decode_mm_kwargs_item, encode_mm_kwargs_item
from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)
from skyrl.backends.skyrl_train.inference_servers.engine_utils import get_neuron_sampling_params
from skyrl.backends.skyrl_train.weight_sync.base import LoraLoadRequest
from skyrl.utils.tok import check_is_vlm, get_processor, get_tokenizer


class NeuronInferenceEngine(InferenceEngineInterface):
    """Runs the policy model in-process on the ``neuron`` device via ``model.generate``."""

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer=None,
        processor=None,
        device: str = "neuron",
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._torch_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16

        self._tokenizer = tokenizer if tokenizer is not None else get_tokenizer(model_path)
        self._processor = processor

        self._is_vlm = check_is_vlm(model_path)
        self._base_model = self._load_base_model(model_path, attn_implementation)
        self._base_model.to(self._device)
        self._base_model.eval()

        # ``self._model`` is what ``generate`` drives: the raw base model until a LoRA
        # adapter is loaded, after which it is the corresponding ``PeftModel``.
        self._model = self._base_model
        self._adapters: Dict[str, Any] = {}
        self._active_lora_name: Optional[str] = None

    # -- construction ---------------------------------------------------------

    def _load_base_model(self, model_path: str, attn_implementation: Optional[str]):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": self._torch_dtype}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        if self._is_vlm:
            self._processor = self._processor if self._processor is not None else get_processor(model_path)
            return AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # -- InferenceEngineInterface ---------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_path

    async def generate(self, input_batch: InferenceEngineInput, model: Optional[str] = None) -> InferenceEngineOutput:
        mm_features = input_batch.get("mm_features")

        prompt_token_ids = input_batch.get("prompt_token_ids")
        if not prompt_token_ids:
            raise ValueError("NeuronInferenceEngine.generate requires prompt_token_ids")

        sampling_params = input_batch.get("sampling_params") or {}
        generate_kwargs = get_neuron_sampling_params(sampling_params)

        if model is not None and model != self._model_path:
            if model not in self._adapters:
                raise ValueError(f"Unknown LoRA adapter requested for sampling: {model!r}")
            if model != self._active_lora_name:
                self._model = self._adapters[model]
                self._active_lora_name = model

        # ``model.generate`` is a blocking call; keep the driver's event loop free.
        return await asyncio.to_thread(self._generate_sync, prompt_token_ids, generate_kwargs, mm_features)

    def get_endpoint_url(self) -> str:
        raise NotImplementedError("The native engine has no HTTP endpoint; it runs in-process.")

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("The native engine does not serve the OpenAI HTTP API.")

    async def render_chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = (request_payload.get("json") or {}).get("messages")
        if messages is None:
            raise ValueError("render_chat_completion requires request_payload['json']['messages']")
        if self._is_vlm:
            return self._render_vlm(messages)
        token_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return {"token_ids": token_ids, "features": None}

    def _render_vlm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render a VLM conversation to token ids + encoded vision tensors.

        The processor is the single source of truth for both token expansion (image
        placeholder counts depend on the per-image patch grid) and vision-tensor layout:
        ``pixel_values`` / ``image_grid_thw`` are produced here and replayed verbatim in
        :meth:`_vlm_inputs`, so no layout is re-derived by hand. The tensors travel as one
        ``mm_codec`` blob in ``kwargs_data["image"]`` (the trainer's ``decode_mm_kwargs``
        concatenates items along dim 0, so a single item holding the full concatenation
        round-trips unchanged).
        """
        messages = [
            {**message, "content": self._normalize_content(message.get("content"))}
            for message in messages
        ]
        images = self._extract_images(messages)
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if images:
            inputs = self._processor(text=[text], images=images, return_tensors="pt", padding=True)
        else:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)

        input_ids = inputs["input_ids"][0].tolist()
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Some processors return one pixel_values tensor per image; collapse to the full
        # [num_patches, dim] tensor the trainer's decoder expects.
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(list(pixel_values), dim=0) if pixel_values else None

        features: Dict[str, Any] = {"mm_hashes": {}, "mm_placeholders": {}}
        if pixel_values is not None and image_grid_thw is not None:
            features["kwargs_data"] = {
                "image": [encode_mm_kwargs_item({"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})]
            }
        else:
            features["kwargs_data"] = None
        return {"token_ids": input_ids, "features": features}

    @staticmethod
    def _normalize_content(content: Any) -> Any:
        """Normalize one message ``content`` field into a clean OpenAI content list.

        Two parquet/pyarrow round-trip artifacts must be undone before the chat
        template sees the content:

        * a ``list<struct>`` content column arrives as a ``numpy.ndarray`` of dicts
          (or a list of dicts) rather than a plain ``list``; and
        * the struct schema is unified across heterogeneous part types, so a ``text``
          part also carries an ``image_url: None`` key (and an ``image_url`` part a
          ``text: None`` key).  The Qwen3.5-VL chat template treats any part whose
          dict contains an ``image_url`` *key* (even ``None``) as an image, so those
          spurious keys must be stripped -- otherwise the placeholder count diverges
          from the number of images actually extracted and the processor raises
          ``StopIteration``.

        Plain-string content (text-only messages) is returned unchanged.
        """
        if content is None:
            return []
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            content = [content]
        elif hasattr(content, "tolist"):  # numpy array from a pyarrow round-trip
            content = list(content.tolist())
        else:
            content = list(content)

        normalized: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url is not None:
                    normalized.append({"type": "image_url", "image_url": {"url": url}})
            elif part_type == "image":
                image = part.get("image")
                if image is not None:
                    normalized.append({"type": "image", "image": image})
            elif part_type == "text":
                text = part.get("text")
                if text is not None:
                    normalized.append({"type": "text", "text": text})
            # Unknown or empty-typed parts are dropped.
        return normalized

    @staticmethod
    def _extract_images(messages: List[Dict[str, Any]]) -> List[Any]:
        """Collect image references from an OpenAI-style message list, in conversation order.

        Handles the content-part spellings the env/renderer emit: ``image_url``
        (``{"image_url": {"url": ...}}``) and bare ``image`` (path, data URI, or PIL image).
        """
        images: List[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        images.append(NeuronInferenceEngine._load_image(url))
                elif part_type == "image" and part.get("image") is not None:
                    images.append(NeuronInferenceEngine._load_image(part["image"]))
        return images

    @staticmethod
    def _load_image(source: Any) -> Any:
        """Resolve an image reference (data URI, file URI/path, or PIL image) to a PIL image.

        Remote ``http(s)`` URLs are not fetched here; the env is expected to emit data URIs
        or local paths (as the tinker renderer does).
        """
        from PIL import Image

        if isinstance(source, Image.Image):
            return source
        if not isinstance(source, str):
            raise TypeError(f"Unsupported image reference type: {type(source)!r}")

        if source.startswith("data:"):
            import base64
            import io

            _, b64 = source.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

        path = source[7:] if source.startswith("file://") else source
        return Image.open(path).convert("RGB")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("The native engine does not serve the OpenAI HTTP API.")

    async def wake_up(self, *args: Any, **kwargs: Any) -> None:
        # In-process engine is always awake; no-op for interface compatibility.
        return None

    async def sleep(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def update_named_weights(self, request: "LoraLoadRequest | Dict[str, Any]") -> None:
        if isinstance(request, LoraLoadRequest):
            await self.load_lora_adapter(request.lora_name, request.lora_path)
            return

        # Full-weight update: ``request`` maps HF state-dict names to tensors.
        state_dict = {}
        for name, value in request.items():
            if torch.is_tensor(value):
                state_dict[name] = value.to(self._device)
            else:
                state_dict[name] = torch.tensor(value, device=self._device)
        self._base_model.load_state_dict(state_dict, strict=False)
        logger.info("NeuronInferenceEngine updated %d named weights in-place.", len(state_dict))

    async def load_lora_adapter(self, lora_name: str, lora_path: str) -> Dict[str, Any]:
        if lora_name in self._adapters:
            self._model = self._adapters[lora_name]
            self._active_lora_name = lora_name
            return {"status": "already_loaded", "lora_name": lora_name}

        from peft import PeftModel

        adapter = PeftModel.from_pretrained(
            self._base_model, lora_path, adapter_name=lora_name, is_trainable=False
        )
        adapter.to(self._device)
        adapter.eval()
        self._adapters[lora_name] = adapter
        self._model = adapter
        self._active_lora_name = lora_name
        return {"status": "ok", "lora_name": lora_name}

    async def teardown(self) -> None:
        del self._model
        del self._base_model
        self._adapters.clear()
        if self._device == "neuron":
            import torch_neuronx  # noqa: F401

            torch_neuronx.reset()  # type: ignore[attr-defined]

    async def reset_prefix_cache(self, reset_running_requests: bool = False) -> None:
        # transformers has no persistent prefix cache in this serving mode.
        return None

    async def pause_generation(self) -> None:
        return None

    async def resume_generation(self) -> None:
        return None

    async def finish_session(self, session_id: str) -> None:
        # No session-held replica capacity to release in-process.
        return None

    async def get_world_size(self) -> Tuple[int, int]:
        return (1, 1)

    # -- generation core -------------------------------------------------------

    def _generate_sync(
        self,
        prompt_token_ids: List[List[int]],
        generate_kwargs: Dict[str, Any],
        mm_features: Optional[List[Dict[str, Any]]] = None,
    ) -> InferenceEngineOutput:
        if mm_features:
            inputs, prompt_lens = self._vlm_inputs(prompt_token_ids, mm_features)
        else:
            inputs, prompt_lens = self._text_inputs(prompt_token_ids)
        batch = len(prompt_lens)

        outputs = self._model.generate(
            **inputs,
            output_logits=True,
            return_dict_in_generate=True,
            **generate_kwargs,
        )
        sequences = outputs.sequences  # [batch, padded_prompt + generated], right-padded

        max_new_tokens = generate_kwargs.get("max_new_tokens")
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id

        response_ids: List[List[int]] = []
        responses: List[str] = []
        stop_reasons: List[str] = []
        for b in range(batch):
            ids = sequences[b, prompt_lens[b] :].tolist()
            ids = [t for t in ids if t != pad_id]
            if eos_id is not None and eos_id in ids:
                ids = ids[: ids.index(eos_id) + 1]
            response_ids.append(ids)
            responses.append(self._tokenizer.decode(ids, skip_special_tokens=True))
            stop_reasons.append("length" if max_new_tokens is not None and len(ids) >= max_new_tokens else "stop")

        response_logprobs = self._response_logprobs(outputs.logits, response_ids, generate_kwargs)

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=response_logprobs,
            prompt_logprobs=None,
            rollout_expert_indices=None,
            rollout_sample_support=None,
        )

    def _text_inputs(self, prompt_token_ids: List[List[int]]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """Pad a text batch into the ``{input_ids, attention_mask}`` ``generate`` inputs."""
        prompt_lens = [len(ids) for ids in prompt_token_ids]
        max_prompt_len = max(prompt_lens)

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        input_ids = torch.tensor(
            [ids + [pad_id] * (max_prompt_len - len(ids)) for ids in prompt_token_ids],
            device=self._device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1] * pl + [0] * (max_prompt_len - pl) for pl in prompt_lens],
            device=self._device,
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, prompt_lens

    def _vlm_inputs(
        self,
        prompt_token_ids: List[List[int]],
        mm_features: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """Rebuild a single-sample VLM ``generate`` input from token ids + encoded vision tensors.

        The VLM generator runs unbatched (``batched=False``), so there is exactly one
        sample and no padding: the input is the sample's token ids plus the vision tensors
        the render step stashed in ``kwargs_data``. Those tensors were produced by the
        processor and are replayed verbatim, so their layout is whatever the model expects.
        """
        sample_ids = prompt_token_ids[0]
        input_ids = torch.tensor([sample_ids], device=self._device, dtype=torch.long)
        inputs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

        kwargs_data = (mm_features[0].get("kwargs_data") or {}) if mm_features else None
        image_blobs = (kwargs_data or {}).get("image") or []
        if image_blobs and image_blobs[0] is not None:
            data = decode_mm_kwargs_item(image_blobs[0])
            pixel_values = data.get("pixel_values")
            image_grid_thw = data.get("image_grid_thw")
            if pixel_values is not None:
                inputs["pixel_values"] = pixel_values.to(self._device)
            if image_grid_thw is not None:
                inputs["image_grid_thw"] = image_grid_thw.to(self._device)

        return inputs, [len(sample_ids)]

    def _response_logprobs(
        self,
        step_logits: Tuple[torch.Tensor, ...],
        response_ids: List[List[int]],
        generate_kwargs: Dict[str, Any],
    ) -> List[List[float]]:
        """Per-token logprob of the sampled token, aligned 1:1 with ``response_ids``.

        Follow-up 2 in the module docstring: validate this against the trainer's
        ``processed_logprobs`` computation on a neuron instance.

        The per-step log-softmax/top-k/gather math runs on CPU: neuronx-cc cannot lower
        some of these ops under an NKI context (``No backend set`` / ``No active NKI
        context``), and the per-token results are a handful of floats anyway, so the
        ``[batch, vocab]`` round-trip is cheaper than a compiled-graph fallback.
        """
        temperature = generate_kwargs.get("temperature")
        top_k = generate_kwargs.get("top_k")
        top_p = generate_kwargs.get("top_p")
        min_p = generate_kwargs.get("min_p")

        batch = len(response_ids)
        result: List[List[float]] = [[] for _ in range(batch)]
        for t, logits_t in enumerate(step_logits):
            active = [b for b in range(batch) if t < len(response_ids[b])]
            if not active:
                continue
            logits_cpu = logits_t.detach().to("cpu").float()
            sampled = torch.tensor([response_ids[b][t] for b in active], dtype=torch.long)
            lp = self._sampled_token_logprobs(
                logits_cpu[active], sampled, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p
            )
            for b, value in zip(active, lp.tolist()):
                result[b].append(value)
        return result

    @staticmethod
    def _sampled_token_logprobs(
        logits: torch.Tensor,
        sampled_ids: torch.Tensor,
        *,
        temperature: Optional[float],
        top_k: Optional[float],
        top_p: Optional[float],
        min_p: Optional[float],
    ) -> torch.Tensor:
        """Log-prob of each sampled token under the (filtered) sampling distribution.

        Mirrors transformers' sampling filter order: temperature, top-k, min-p, top-p,
        then softmax.  Returns one log-prob per row of ``logits``.
        """
        logits = logits.float()
        if temperature is not None and temperature != 1.0:
            logits = logits / temperature

        vocab = logits.size(-1)
        if top_k is not None and top_k > 0:
            top_k = min(int(top_k), vocab)
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if min_p is not None and min_p > 0:
            probs = logits.softmax(dim=-1)
            threshold = probs.max(dim=-1, keepdim=True).values * min_p
            logits = logits.masked_fill(probs < threshold, float("-inf"))

        if top_p is not None and top_p < 1.0:
            probs = logits.softmax(dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            # ``cumsum - sorted_probs`` is the cumulative mass of tokens strictly above
            # this one in sorted order; when it already exceeds ``top_p``, this token and
            # everything below it are cut. This mirrors transformers' ``_sample`` filter.
            remove = cumsum - sorted_probs > top_p
            idx_to_remove = remove.scatter(dim=-1, index=sorted_idx, src=remove)
            logits = logits.masked_fill(idx_to_remove, float("-inf"))

        log_probs = logits.log_softmax(dim=-1)
        return log_probs.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)
