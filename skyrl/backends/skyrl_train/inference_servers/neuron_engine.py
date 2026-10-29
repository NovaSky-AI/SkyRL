"""In-process, transformers-native inference engine for TorchNeuron (Trainium).

Unlike the vLLM-backed :class:`RemoteInferenceClient`, this engine drives the model
directly with ``transformers`` ``model.generate`` on the ``neuron`` (PrivateUse1)
device.  It exists because vLLM cannot serve on Trainium: vLLM requires CUDA torch and
ships no weight-synchronization path for RL, while ``torch-neuronx`` exposes a plain
``torch.nn.Module`` that ``model.generate`` already supports (this is the same path the
TRL GRPO leg uses).

This first cut implements the *text* RL path end-to-end, plus the multimodal
(VLM) path over a vLLM-free codec.

**Processed logprobs.** ``model.generate(output_scores=True)`` returns the
fully-processed per-step logits the sampler softmaxes before drawing a token, so the
response logprob of the sampled token is ``log_softmax(scores)[sampled]`` -- bit-for-bit
the distribution the token was actually drawn from, with no filter re-implementation.

**Weight sync.** LoRA sync flows through ``update_named_weights`` (the driver reloads the
exported adapter). Full-weight (non-LoRA) sync is not wired: the FSDP worker has no
engine handle under the neuron backend, and ``WorkerDispatch._neuron_save_weights_for_sampler``
raises ``NotImplementedError`` for ``lora.rank <= 0``. The engine's ``update_named_weights``
full-weight branch is present but unreached until that routing is added.

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
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_neuron_sampling_params,
)
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
        # ``model.generate`` is not thread-safe and the caller fans out every prompt in
        # a batch concurrently, so serialize generation (and the adapter swap it races
        # with) on the driver. On a single NeuronCore this is also the right memory
        # shape: no concurrent forward passes.
        self._generate_lock = asyncio.Lock()

    # -- construction ---------------------------------------------------------

    def _load_base_model(self, model_path: str, attn_implementation: Optional[str]):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": self._torch_dtype}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        if self._is_vlm:
            # Qwen3.5-VL's vision tower has device-unsafe RoPE / repeat_interleave paths
            # that crash on TorchNeuron. Apply the shared device-safety patch before the
            # model is built -- but only for the model family it targets. The patch module
            # is absent on transformers releases predating Qwen3.5 (the base floor is
            # >=5.6.1), so importing it unconditionally would crash every VLM load there.
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if config.model_type == "qwen3_5":
                try:
                    from skyrl.backends.skyrl_train.patches.neuron.qwen3_5_vl import (
                        apply_qwen3_5_vl_device_patch,
                    )

                    apply_qwen3_5_vl_device_patch()
                except ImportError:
                    logger.warning(
                        "Qwen3.5-VL detected but the neuron device-safety patch is unavailable "
                        "on this transformers release; running unpatched."
                    )
            load_kwargs["config"] = config
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

        # ``model.generate`` is not thread-safe and mutates ``self._model`` when an
        # adapter is switched, so serialize the whole swap+generate critical section.
        # The caller fans out every prompt in a batch, so this also stops the default
        # executor's threads from issuing concurrent forwards on one NeuronCore.
        async with self._generate_lock:
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
        token_ids = self._as_token_id_list(token_ids)
        return {"token_ids": token_ids, "features": None}

    @staticmethod
    def _as_token_id_list(token_ids: Any) -> List[int]:
        """Normalize ``apply_chat_template`` output to a flat ``List[int]``.

        transformers 4.x returned the tokenized prompt directly as ``List[int]``;
        transformers >= 5.0 returns a ``BatchEncoding`` (a ``UserDict``) whose
        ``input_ids`` is a flat list for a single conversation and a list of lists for a
        batch.  The caller wraps the result into a single-sample batch, so a flat
        ``List[int]`` must come out.
        """
        if not isinstance(token_ids, (list, tuple, torch.Tensor)):
            # transformers >= 5.0 ``BatchEncoding``.
            token_ids = token_ids["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            # A batched template call returns a list of lists; unwrap the batch dim.
            token_ids = token_ids[0]
        return [int(t) for t in token_ids]

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
        messages = [{**message, "content": self._normalize_content(message.get("content"))} for message in messages]
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

        # Full-weight (non-LoRA) sync is not wired: the FSDP worker has no engine handle
        # under the neuron backend and `WorkerDispatch._neuron_save_weights_for_sampler`
        # already raises `NotImplementedError` for `lora.rank <= 0`. Refuse loudly here
        # rather than half-applying a state dict into the raw base model (which would
        # bypass the active PeftModel and silently drop key/shape mismatches).
        raise NotImplementedError(
            "NeuronInferenceEngine only supports LoRA weight sync; full-weight sync is not wired."
        )

    async def load_lora_adapter(self, lora_name: str, lora_path: str) -> Dict[str, Any]:
        # A single-tenant run re-exports the adapter under the same name after every
        # train step. A cached PeftModel keeps the step-0 tensors, so on a repeat name
        # reload the freshly exported safetensors into the existing adapter instead of
        # short-circuiting.
        if lora_name in self._adapters:
            adapter = self._adapters[lora_name]
            adapter.load_adapter(lora_path, adapter_name=lora_name, is_trainable=False)
            self._model = adapter
            self._active_lora_name = lora_name
            logger.info("Reloaded LoRA adapter %r from %s", lora_name, lora_path)
            return {"status": "reloaded", "lora_name": lora_name}

        from peft import PeftModel

        adapter = PeftModel.from_pretrained(self._base_model, lora_path, adapter_name=lora_name, is_trainable=False)
        adapter.to(self._device)
        adapter.eval()
        self._adapters[lora_name] = adapter
        self._model = adapter
        self._active_lora_name = lora_name
        logger.info("Loaded LoRA adapter %r from %s", lora_name, lora_path)
        return {"status": "ok", "lora_name": lora_name}

    async def teardown(self) -> None:
        del self._model
        del self._base_model
        self._adapters.clear()
        if self._device == "neuron":
            import torch_neuronx  # noqa: F401

            # torch-neuronx has no ``reset()``; ``empty_cache()`` releases the
            # device memory allocator's cached blocks (the analog of
            # ``torch.cuda.empty_cache()``).
            torch_neuronx.empty_cache()

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
            output_scores=True,
            return_dict_in_generate=True,
            tokenizer=self._tokenizer,
            **generate_kwargs,
        )
        sequences = outputs.sequences  # [batch, padded_prompt + generated], right-padded

        max_new_tokens = generate_kwargs.get("max_new_tokens")
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            # A tokenizer with no pad token (common in the Qwen family) still needs a
            # value to identify the right-padding ``generate`` appends; transformers
            # itself falls back to EOS there. Keep the two distinct below so a real EOS
            # stop token is not stripped.
            pad_id = eos_id

        # ``sequences`` is ``[batch, input_width + max_new_tokens]``: the padded prompt
        # (real tokens left-aligned, ``pad_id`` right-filling to ``max(prompt_lens)``)
        # followed by the generated tokens (right-padded with ``pad_id``). Slice from the
        # padded input width so only generated tokens remain; do not slice from
        # ``prompt_lens[b]`` (that leaves prompt-side pads at the head of shorter prompts).
        input_width = max(prompt_lens)

        response_ids: List[List[int]] = []
        responses: List[str] = []
        stop_reasons: List[str] = []
        for b in range(batch):
            ids = sequences[b, input_width:].tolist()
            # Strip only the trailing pad tokens ``generate`` appends after the response.
            # Do not strip EOS: the vLLM path sets ``include_stop_str_in_output=True``, so
            # the terminating EOS must survive for stop-reason and loss-mask parity.
            if pad_id is not None and pad_id != eos_id:
                while ids and ids[-1] == pad_id:
                    ids.pop()
            if eos_id is not None and eos_id in ids:
                ids = ids[: ids.index(eos_id) + 1]
            response_ids.append(ids)
            responses.append(self._tokenizer.decode(ids, skip_special_tokens=True))
            stop_reasons.append("length" if max_new_tokens is not None and len(ids) >= max_new_tokens else "stop")

        response_logprobs = self._response_logprobs(outputs.scores, response_ids)

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

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
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

        The VLM generator runs unbatched (``batched=False``), so exactly one sample and
        one ``mm_features`` entry are expected; the invariant is asserted rather than
        silently truncating. The input is the sample's token ids plus the vision tensors
        the render step stashed in ``kwargs_data``. Those tensors were produced by the
        processor and are replayed verbatim, so their layout is whatever the model expects.
        """
        assert len(prompt_token_ids) == 1, "VLM generation is single-sample (batched=False)"
        assert len(mm_features) == 1, "VLM generation expects exactly one mm_features entry"
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
        step_scores: Tuple[torch.Tensor, ...],
        response_ids: List[List[int]],
    ) -> List[List[float]]:
        """Per-token logprob of the sampled token, aligned 1:1 with ``response_ids``.

        ``model.generate(output_scores=True)`` returns the fully-processed per-step
        logits -- temperature / top-k / top-p / min-p / penalties already applied -- that
        the sampler softmaxes before drawing a token, so ``log_softmax(scores)[sampled]``
        is bit-for-bit the distribution the token was actually drawn from.

        The ``[batch, vocab]`` log-softmax runs on CPU: neuronx-cc cannot lower it under
        an NKI context, and per-token results are a handful of floats, so the round-trip
        is cheaper than a compiled-graph fallback.
        """
        batch = len(response_ids)
        result: List[List[float]] = [[] for _ in range(batch)]
        for t, scores_t in enumerate(step_scores):
            active = [b for b in range(batch) if t < len(response_ids[b])]
            if not active:
                continue
            scores_cpu = scores_t.detach().to("cpu").float()
            sampled = torch.tensor([response_ids[b][t] for b in active], dtype=torch.long)
            lp = scores_cpu[active].log_softmax(dim=-1).gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
            for b, value in zip(active, lp.tolist()):
                result[b].append(value)
        return result
