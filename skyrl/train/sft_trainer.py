"""
SFT (Supervised Fine-Tuning) trainer for SkyRL.

Supports both FSDP and Megatron backends via a single ``SFTTrainer`` class.
The backend is selected dynamically based on ``SFTConfig.strategy``.

Usage::

    from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
    from skyrl.train.sft_trainer import SFTTrainer

    cfg = SFTConfig(strategy="megatron")
    trainer = SFTTrainer(cfg)
    trainer.setup()
    trainer.train()
    trainer.shutdown()

Or as a CLI entrypoint::

    python -m skyrl.train.main_sft strategy=megatron model.path=Qwen/Qwen3-0.6B
"""

import functools
import json
import multiprocessing as mp
import os
import tempfile
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import ray
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from ray.util.placement_group import placement_group
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.sft_config import (
    SFTConfig,
    TrainOnWhat,
    _normalize_dataset_cfg,
    build_skyrl_config_for_sft,
)
from skyrl.train.dataset.pretokenized import load_from_pretokenized
from skyrl.train.generators.utils import (
    get_response_ids_and_loss_mask_from_messages,
)
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.callbacks import (
    CallbackHandler,
    CallbackInput,
    TrainingCallback,
    TrainingControl,
)
from skyrl.train.utils.ray_gpu_monitor import RayGpuMonitor
from skyrl.train.utils.tracking import Tracking
from skyrl.train.utils.trainer_utils import (
    GLOBAL_STEP_PREFIX,
    cleanup_old_checkpoints,
    extract_step_from_path,
    validate_consistency_for_latest_checkpoint,
)
from skyrl.train.utils.utils import ResolvedPlacementGroup, Timer
from skyrl.utils.tok import (
    check_is_vlm,
    get_processor,
    get_tokenizer,
)

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _tokenize_chat_slice_worker(args):
    """Worker function for parallel chat-format tokenization with slice-based loading.

    Each worker loads the full dataset (HF caches it locally after the parent's
    first call) and tokenizes only its assigned index range.

    Must be top-level for pickling with spawn.
    """
    (
        dataset_name,
        dataset_split,
        start_idx,
        end_idx,
        tokenizer_path,
        max_length,
        messages_key,
        train_on_what_str,
        tools_key,
        system_key,
    ) = args

    # Worker loads tokenizer from cached path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )

    train_on_what = TrainOnWhat(train_on_what_str)

    # Reload the dataset using the original split string and slice by index.
    # The parent has already loaded once so this hits the HF cache.
    dataset = load_dataset(dataset_name, split=dataset_split)
    dataset_slice = dataset.select(range(start_idx, end_idx))

    # Tokenize and filter inline
    results = []
    for example in dataset_slice:
        tokenized = tokenize_chat_example(
            example,
            tokenizer,
            max_length=max_length,
            messages_key=messages_key,
            train_on_what=train_on_what,
            tools_key=tools_key,
            system_key=system_key,
        )
        if tokenized is not None:
            results.append(tokenized)

    return results


def _tokenize_alpaca_slice_worker(args):
    """Worker function for parallel Alpaca-format tokenization with slice-based loading.

    Each worker loads the full dataset (HF caches it locally after the parent's
    first call) and tokenizes only its assigned index range.

    Must be top-level for pickling with spawn.
    """
    dataset_name, dataset_split, start_idx, end_idx, tokenizer_path, max_length = args

    # Worker loads tokenizer from cached path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )

    # Reload the dataset using the original split string and slice by index.
    dataset = load_dataset(dataset_name, split=dataset_split)
    dataset_slice = dataset.select(range(start_idx, end_idx))

    # Tokenize and filter inline
    results = []
    for example in dataset_slice:
        tokenized = tokenize_sft_example(example, tokenizer, max_length)
        if tokenized is not None:
            results.append(tokenized)

    return results


def _compute_cache_key(
    dataset_name: str,
    dataset_split: str,
    model_path: str,
    max_length: Optional[int],
    messages_key: str,
    train_on_what: str,
    tools_key: Optional[str],
    system_key: Optional[str],
) -> str:
    """Compute a cache key (hash) for a tokenized dataset.

    The hash uniquely identifies the dataset and tokenization parameters so
    that cached results can be safely reused when parameters match.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split string (e.g., "train[:100000]")
        model_path: Model name/path (tokenizer identity)
        max_length: Maximum sequence length for truncation
        messages_key: Column name for messages
        train_on_what: Training target (last_assistant_message or all_assistant_messages)
        tools_key: Column name for tools (if applicable)
        system_key: Column name for system prompt (if applicable)

    Returns:
        A hex string hash (e.g., "a3f2c1...")
    """
    import hashlib

    # Build a deterministic string from all relevant parameters
    cache_params = json.dumps(
        {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "model_path": model_path,
            "max_length": max_length,
            "messages_key": messages_key,
            "train_on_what": train_on_what,
            "tools_key": tools_key,
            "system_key": system_key,
        },
        sort_keys=True,
    )
    return hashlib.sha256(cache_params.encode()).hexdigest()[:16]


def _get_cache_path(cache_dir: str, cache_key: str) -> str:
    """Get the full path to a cached tokenized dataset.

    The cache is stored as an arrow-backed HuggingFace dataset on disk
    (``Dataset.save_to_disk``), so this path is a directory, not a file.

    Args:
        cache_dir: Base cache directory
        cache_key: Cache key (hash) for this dataset

    Returns:
        Path to the cache directory (e.g., /path/to/cache/a3f2c1).
    """
    return os.path.join(cache_dir, cache_key)


def _load_from_cache(cache_path: str) -> Optional[list]:
    """Load tokenized dataset from cache.

    Reads an arrow-backed HF ``Dataset`` directory written by
    :func:`_save_to_cache` and materializes it back to the ``list[dict]``
    representation expected by the trainer (which slices, shuffles, and
    concatenates the result during the training loop).

    Args:
        cache_path: Path to cached dataset directory.

    Returns:
        List of tokenized examples, or ``None`` if the cache directory
        does not exist or fails to load.
    """
    if not os.path.isdir(cache_path):
        return None

    try:
        logger.info(f"Loading tokenized dataset from cache: {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
        tokenized = dataset.to_list()
        logger.info(f"Loaded {len(tokenized)} examples from cache")
        return tokenized
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def _save_to_cache(cache_path: str, tokenized: list) -> None:
    """Save tokenized dataset to cache.

    Materializes the in-memory ``list[dict]`` as a HuggingFace ``Dataset``
    and writes it via ``save_to_disk``. At 1M-row scale, the arrow-backed,
    memory-mapped format reads and writes dramatically faster than pickle
    while also being portable across Python versions. The write goes to a
    sibling ``<cache_path>.tmp`` directory which is then atomically renamed
    onto ``cache_path`` for NFS safety.

    Args:
        cache_path: Path to the cache directory to create.
        tokenized: List of tokenized examples.
    """
    try:
        import shutil

        parent_dir = os.path.dirname(cache_path)
        os.makedirs(parent_dir, exist_ok=True)

        logger.info(f"Saving {len(tokenized)} examples to cache: {cache_path}")
        # Build the HF Dataset from rows and write to a sibling temp dir.
        # An atomic rename onto cache_path makes concurrent readers see only
        # a fully-written cache (NFS-safe; matches the previous pickle path).
        dataset = Dataset.from_list(tokenized)
        temp_path = cache_path + ".tmp"
        # Clean up any stale temp dir from an interrupted prior run.
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
        dataset.save_to_disk(temp_path)
        # If a previous cache exists at the final path, drop it before
        # rename so the swap is the only visible state change.
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)
        os.rename(temp_path, cache_path)
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


@functools.lru_cache(maxsize=512)
def _parse_tools_str(tools: str) -> Optional[tuple]:
    """Parse a JSON-encoded tools string. Cached because tool-calling datasets
    typically share one schema across thousands of rows (e.g. APIGen's airline
    domain), and `apply_chat_template` re-tokenizes the schema on every row."""
    tools = tools.strip()
    if not tools:
        return None
    parsed = json.loads(tools)
    if not parsed:
        return None
    # Return a tuple so the cache stores an immutable value; caller re-lists it.
    return tuple(parsed)


def _coerce_tools(tools: Any) -> Optional[list]:
    """Coerce a dataset's ``tools`` field into a list[dict] for ``apply_chat_template``.

    Tool-calling datasets ship the schema list as ``list[dict]`` (parquet-typed),
    JSON-encoded ``str``, or absent. HF chat templates expect ``list[dict]``;
    returns ``None`` when there are no tools so the caller can omit the kwarg.
    """
    if tools is None:
        return None
    if isinstance(tools, str):
        cached = _parse_tools_str(tools)
        return list(cached) if cached else None
    if isinstance(tools, list):
        return tools or None
    raise TypeError(f"Unsupported `tools` type: {type(tools).__name__}")


def _normalize_tool_call_payload(tc: Any) -> Optional[list]:
    """Normalize an assistant message's ``tool_calls`` into the OpenAI-style list.

    Datasets in the wild use:

    * ``[]`` / ``""`` / ``None`` — no tool call;
    * a single JSON-encoded ``{"name": ..., "arguments": ...}`` dict (APIGen-MT);
    * a JSON-encoded list of such dicts (xLAM, ToolACE);
    * an already-parsed list of OpenAI-style ``{"type": "function", "function": {...}}``.

    HF chat templates expect ``[{"type": "function", "function": {"name", "arguments"}}]``,
    so we coerce to that shape. Returns ``None`` when the message has no tool call.
    """
    if tc is None:
        return None
    if isinstance(tc, str):
        tc = tc.strip()
        if not tc or tc == "[]":
            return None
        tc = json.loads(tc)
    if isinstance(tc, dict):
        tc = [tc]
    if not isinstance(tc, list) or not tc:
        return None

    out = []
    for call in tc:
        if not isinstance(call, dict):
            raise TypeError(f"tool call entry must be a dict, got {type(call).__name__}")
        fn = call["function"] if isinstance(call.get("function"), dict) else call
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        out.append({"type": "function", "function": {"name": fn.get("name"), "arguments": arguments}})
    return out


# Fields we explicitly normalize; everything else on the message is forwarded
# so model-specific templates can read e.g. ``reasoning_content`` (Qwen3,
# Nemotron-3 thinking models) or ``tool_call_id``.
_NORMALIZED_KEYS = frozenset({"role", "content", "tool_calls"})


def _normalize_chat_messages(messages: list[dict]) -> list[dict]:
    """Return messages in a shape that HF chat templates accept.

    Normalizes ``content`` (``None`` → ``""``), promotes string-encoded
    ``tool_calls`` on assistant turns into the OpenAI list-of-dicts form,
    drops empty/placeholder ``tool_calls`` on every role, and preserves any
    other fields on the message verbatim.
    """
    out = []
    for msg in messages:
        role = msg["role"]
        new_msg = {k: v for k, v in msg.items() if k not in _NORMALIZED_KEYS}
        new_msg["role"] = role
        new_msg["content"] = msg.get("content", "") or ""
        if role == "assistant":
            tool_calls = _normalize_tool_call_payload(msg.get("tool_calls"))
            if tool_calls:
                new_msg["tool_calls"] = tool_calls
        out.append(new_msg)
    return out


def tokenize_sft_example(example: dict, tokenizer, max_length: int = 512, **tokenizer_kwargs) -> dict | None:
    """Tokenize an Alpaca-format SFT example via ``apply_chat_template``.

    Converts the instruction/input/output fields into a two-message chat
    (user + assistant) and delegates to :func:`tokenize_chat_example`.
    This ensures tokenization matches the HF / TRL convention (proper
    special tokens, chat template formatting).

    Returns dict with input_ids, attention_mask, num_actions (response length),
    or None if the example was fully truncated.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Build user content: instruction + optional input
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    user_content = user_content.strip()

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    return tokenize_chat_example(
        {"messages": messages},
        tokenizer,
        max_length=max_length,
        messages_key="messages",
        **tokenizer_kwargs,
    )


def tokenize_chat_example(
    example: dict,
    tokenizer,
    max_length: Optional[int] = None,
    messages_key: str = "messages",
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    tools_key: Optional[str] = "tools",
    system_key: Optional[str] = "system",
    processor=None,
    **tokenizer_kwargs,
) -> dict | None:
    """Tokenize a chat-format example with configurable loss targets.

    Uses ``apply_chat_template`` to tokenize the conversation and determine
    which tokens to train on based on ``train_on_what``.

    For tool-calling datasets (e.g. APIGen-MT, xLAM, ToolACE), if the example
    has a ``tools_key`` column, its parsed value is forwarded as ``tools=`` to
    ``apply_chat_template`` so the model sees the function schemas. Likewise,
    a ``system_key`` column is prepended as a leading system message when no
    system message is already present.

    Args:
        example: Dict containing a ``messages_key`` column with chat messages.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        max_length: Maximum sequence length (truncation boundary).
        messages_key: Key in *example* that holds the messages list.
        train_on_what: Which tokens to compute loss on.
        tools_key: Key in *example* whose value is the per-row tool schema list
            (or JSON-encoded string thereof). ``None`` disables the lookup.
        system_key: Key in *example* whose value is a system prompt string.
            ``None`` disables the lookup.
        **tokenizer_kwargs: Extra kwargs forwarded to ``apply_chat_template``
            (e.g. ``enable_thinking``).

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``num_actions``, and
        optionally ``loss_mask`` (a per-token list of 0/1 within the action
        window).  Returns ``None`` when the example should be skipped.
    """
    # Validate supported modes
    _SUPPORTED = {TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.ALL_ASSISTANT_MESSAGES}
    if train_on_what not in _SUPPORTED:
        raise NotImplementedError(
            f"train_on_what={train_on_what!r} is not yet supported. "
            f"Supported values: {sorted(v.value for v in _SUPPORTED)}"
        )
    messages = list(example[messages_key])

    # Trim trailing tool observations with no follow-up assistant response
    # (common in APIGen-MT). Trailing user turns still drop the row.
    while messages and messages[-1]["role"] == "tool":
        messages.pop()

    if not messages or messages[-1]["role"] != "assistant":
        return None

    messages = _normalize_chat_messages(messages)

    system_prompt = example.get(system_key) if system_key else None
    if system_prompt and messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    # Per-row schemas yield to an explicit caller-provided ``tools`` kwarg.
    tools = _coerce_tools(example.get(tools_key)) if tools_key else None
    if tools is not None:
        tokenizer_kwargs = {"tools": tools, **tokenizer_kwargs}

    # Detect image content. VLM tokenization runs through the HF processor and
    # only supports last-assistant training (image token positions are tied to
    # a single forward over the full conversation).
    has_images = any(
        isinstance(m.get("content"), list)
        and any(isinstance(d, dict) and d.get("type") == "image" for d in m["content"])
        for m in messages
    )
    if has_images and train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
        raise NotImplementedError("Training on all assistant messages with vision inputs is not yet supported")

    if train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE:
        return _tokenize_chat_last_assistant(
            messages, tokenizer, max_length, processor if has_images else None, **tokenizer_kwargs
        )
    else:
        # ALL_ASSISTANT_MESSAGES
        return _tokenize_chat_all_assistants(messages, tokenizer, max_length, **tokenizer_kwargs)


def _unbatch(proc_tok_output):
    """Strip the outer batch list from a processor output.

    ``processor.apply_chat_template`` returns batched ids (``list[list[int]]``)
    while a plain tokenizer returns a flat ``list[int]``. Normalize to flat.
    """
    if proc_tok_output and isinstance(proc_tok_output[0], list):
        return proc_tok_output[0]
    return proc_tok_output


def _tokenize_chat_last_assistant(
    messages: list[dict],
    tokenizer,
    max_length: Optional[int] = None,
    processor=None,
    **tokenizer_kwargs,
) -> dict | None:
    """Tokenize a conversation and compute loss only on the last assistant message.

    Args:
        messages: Full conversation (must end with an assistant message).
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        max_length: Optional sequence length cap; truncates both prompt and full
            conversation to this limit.
        processor: Optional HF processor used for VLM examples. When provided,
            image tensors (``pixel_values``, ``image_grid_thw``) are produced and
            returned alongside the token ids.
        **tokenizer_kwargs: Extra kwargs forwarded to ``apply_chat_template``.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``num_actions`` (number of
        last-assistant tokens), and, for VLM examples, ``pixel_values`` and
        ``image_grid_thw``. Returns ``None`` if truncation left no response
        tokens or dropped an image's placeholder tokens.
    """
    # The processor (when present) tokenizes text and images together; otherwise
    # the plain tokenizer handles text. ``return_dict=True`` so we can read both
    # token ids and any image tensors back out.
    processing_class = processor or tokenizer

    length_kwargs = {}
    if max_length is not None and processor is None:
        length_kwargs = dict(truncation=True, max_length=max_length)

    # Tokenize prompt (everything except last assistant message)
    prompt_ids = processing_class.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        **length_kwargs,
        **tokenizer_kwargs,
    )

    # Tokenize full conversation
    full_ids = processing_class.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        **length_kwargs,
        **tokenizer_kwargs,
    )

    full_input_ids = _unbatch(full_ids["input_ids"])
    full_prompt_ids = _unbatch(prompt_ids["input_ids"])

    # VLM samples can't be safely truncated (it would drop image placeholder tokens
    # and break image/text alignment), so drop anything that exceeds the limit.
    if processor is not None and max_length is not None and len(full_input_ids) > max_length:
        logger.warning(
            f"Dropping VLM sample longer than max_length={max_length}, consider increasing max_length if you see this warning too much"
        )
        return None

    vlm_kwargs = {}  # We only support Qwen-style image kwargs at the moment
    if "pixel_values" in full_ids and "image_grid_thw" in full_ids:
        vlm_kwargs = dict(
            pixel_values=full_ids["pixel_values"],
            image_grid_thw=full_ids["image_grid_thw"],
        )

    num_actions = len(full_input_ids) - len(full_prompt_ids)
    if num_actions <= 0:
        return None

    return {
        "input_ids": full_input_ids,
        "attention_mask": [1] * len(full_input_ids),
        "num_actions": num_actions,
        "loss_mask": [1] * num_actions,
        **vlm_kwargs,
    }


def _tokenize_chat_all_assistants(
    messages: list[dict],
    tokenizer,
    max_length: Optional[int] = None,
    **tokenizer_kwargs,
) -> dict | None:
    """Tokenize a conversation and compute loss on all assistant messages.

    Builds a per-token loss mask covering every assistant turn. ``num_actions``
    spans from the first assistant token to the end of the conversation, with
    interior 0s masking out user/system tokens between assistant turns.

    Args:
        messages: Full conversation. May start with system/user messages;
            must contain at least one assistant message.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        max_length: Optional sequence length cap; truncates to this limit.
        **tokenizer_kwargs: Extra kwargs forwarded to ``apply_chat_template``.

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``num_actions``, and
        ``loss_mask`` (per-token 0/1 list within the action window), or
        ``None`` if no assistant tokens survived after truncation.
    """

    # Find the index of the first assistant message.
    i = 0
    while i < len(messages) and messages[i]["role"] != "assistant":
        i += 1

    # Encode leading non-assistant messages separately because
    # `get_response_ids_and_loss_mask_from_messages` does not accept system messages.

    initial_token_ids = tokenizer.apply_chat_template(
        messages[:i],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **tokenizer_kwargs,
    )
    # no assistant message
    if i >= len(messages):
        return None

    later_token_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
        messages[i:], tokenizer, tokenizer_kwargs=tokenizer_kwargs
    )
    input_ids = initial_token_ids + later_token_ids

    # truncate
    if max_length is not None:
        input_ids = input_ids[:max_length]
        max_assistant_length = max(max_length - len(initial_token_ids), 0)
        loss_mask = loss_mask[:max_assistant_length]

    if sum(loss_mask) == 0:
        return None  # No assistant tokens survived truncation

    num_actions = len(loss_mask)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "num_actions": num_actions,
        "loss_mask": loss_mask,
    }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_sft_batch(examples: list, tokenizer) -> TrainingInputBatch:
    """Collate tokenized examples into a TrainingInputBatch.

    Creates the batch format expected by forward_backward with cross_entropy loss:
    - sequences: [batch_size, seq_len] - token IDs (left-padded)
    - attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
    - loss_mask: [batch_size, num_actions] - 1 for tokens to compute loss on

    All examples are expected to carry a ``loss_mask`` key (guaranteed by both
    ``_tokenize_chat_last_assistant`` and ``_tokenize_chat_all_assistants``).
    """
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_num_actions = max(ex["num_actions"] for ex in examples)
    num_examples = len(examples)

    # Fill NumPy buffers by slice, then convert once.
    sequences_np = np.full((num_examples, max_len), tokenizer.pad_token_id, dtype=np.int64)
    attention_mask_np = np.zeros((num_examples, max_len), dtype=np.int64)
    loss_mask_np = np.zeros((num_examples, max_num_actions), dtype=np.int64)

    # VLM image tensors travel as a TensorList (one variable-shape tensor per
    # sample). Mixed text+image batches are not supported; every sample in a VLM
    # batch must carry images. Check homogeneity up front so a mixed batch fails
    # here with a clear message rather than a KeyError deep in the pad loop.
    num_with_images = sum("pixel_values" in ex for ex in examples)
    if num_with_images not in (0, len(examples)):
        raise ValueError(
            f"Mixed text+image batches are not supported: {num_with_images}/{len(examples)} "
            "samples carry 'pixel_values'. Every sample in a VLM batch must carry images."
        )
    batch_has_images = num_with_images > 0
    pixel_values = []
    image_grid_thw = []

    for i, ex in enumerate(examples):
        # Left-pad sequences; right-align response loss masks.
        pad_len = max_len - len(ex["input_ids"])
        sequences_np[i, pad_len:] = ex["input_ids"]
        attention_mask_np[i, pad_len:] = ex["attention_mask"]

        action_pad = max_num_actions - ex["num_actions"]
        loss_mask_np[i, action_pad:] = ex["loss_mask"]

        if batch_has_images:
            pixel_values.append(torch.as_tensor(ex["pixel_values"]))
            image_grid_thw.append(torch.as_tensor(ex["image_grid_thw"]))

    batch = TrainingInputBatch(
        {
            "sequences": torch.from_numpy(sequences_np),
            "attention_mask": torch.from_numpy(attention_mask_np),
            "loss_mask": torch.from_numpy(loss_mask_np),
            "pixel_values": TensorList(pixel_values) if batch_has_images else None,
            "image_grid_thw": TensorList(image_grid_thw) if batch_has_images else None,
        }
    )
    batch.metadata = {"response_length": max_num_actions}
    return batch


def collate_sft_examples(
    examples: list, collator, batch_size: int, pad_to_batch_size: bool = False
) -> TrainingInputBatch:
    """Top-level collate function for the SFT ``StatefulDataLoader``.

    Defined at module scope (not a lambda/closure) so it is picklable when the
    dataloader uses worker processes with the ``spawn`` start method. Delegates
    to the trainer's configured ``collator`` (``DefaultCollator`` or
    ``PackedDataCollator``).

    When ``pad_to_batch_size`` is set (the train path, non-packed), a final
    short batch is padded up to ``batch_size`` rows via
    :func:`pad_training_input_batch`, which zeros ``loss_mask`` on the padded
    rows so they contribute no gradient. This lets every example in an epoch be
    trained on (instead of dropping the tail) while still dispatching a full,
    evenly-shardable ``batch_size`` batch. Packed batches are never row-padded
    (their rows are FFD bins, already rounded to a multiple of ``dp_size``).
    """
    batch = collator(examples, batch_size=batch_size)
    if pad_to_batch_size:
        pad_rows = batch_size - len(examples)
        if pad_rows > 0:
            batch = pad_training_input_batch(batch, pad_rows)
    return batch


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------


def _format_eval_metrics(eval_metrics: dict) -> str:
    """Render per-dataset eval metrics (``{name}/loss``) for stdout logging."""
    return ", ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())


class SFTTrainer:
    """SFT trainer supporting FSDP and Megatron backends.

    Unlike RayPPOTrainer, this does NOT subclass it. SFT's concerns are
    fundamentally different: no generation, no critic, no advantages, no
    KL penalty. Sharing a base class would create confusing dead code paths.

    Usage::

        trainer = SFTTrainer(SFTConfig(strategy="megatron"))
        trainer.setup()
        trainer.train()
        trainer.shutdown()
    """

    def __init__(
        self,
        cfg: SFTConfig,
        skyrl_cfg: SkyRLTrainConfig | None = None,
        callbacks: Optional[list[TrainingCallback]] = None,
    ):
        self.sft_cfg = cfg
        _normalize_dataset_cfg(cfg)
        # Accept a pre-built bridge config to avoid redundant rebuilds.
        # When not provided (e.g. standalone usage), build it here.
        self.cfg = skyrl_cfg if skyrl_cfg is not None else build_skyrl_config_for_sft(cfg)
        self.tokenizer = None
        self.processor = None  # set in setup() for VLM models
        self.is_vlm = False
        self.dispatch: WorkerDispatch | None = None
        self.tracker: Tracking | None = None
        # Stateful dataloaders, built in train() once data is tokenized.
        self.train_dataloader: StatefulDataLoader | None = None
        # One ``(name, dataloader)`` pair per configured eval dataset; ``None``
        # when eval is disabled. Names are unique (enforced in config validation)
        # and namespace the eval metrics as ``eval/{name}/...``.
        self.eval_dataloaders: list[tuple[str, StatefulDataLoader]] | None = None
        self.global_step = 0
        # running count of total non-padding tokens trained on
        self._total_tokens_processed = 0
        self.collator = None  # built in setup() once the tokenizer is available

        self._num_training_gpus: int = cfg.placement.num_nodes * cfg.placement.num_gpus_per_node
        self._ray_gpu_monitor = RayGpuMonitor() if cfg.enable_ray_gpu_monitor else None

        self._callback_handler = CallbackHandler(callbacks)
        self._training_control = TrainingControl()
        # Loop metadata used to build CallbackInput. Populated in train().
        self._total_steps: int = 0
        self._steps_per_epoch: int = 0
        self._current_epoch: int = 0

    @property
    def _torch_profiler_enabled(self) -> bool:
        """Whether to dispatch policy profiler RPCs."""
        return self.cfg.trainer.policy.torch_profiler_config.enable

    def _build_collator(self, tokenizer):
        """Select the batch collator from the configured packing mode.

        ``PackedDataCollator`` performs controller-level FFD bin-packing
        (Megatron-only, ``use_sequence_packing=True``); ``DefaultCollator``
        left-pads each example. The choice is fixed by static config; the
        ``tokenizer`` is passed in by :meth:`setup` once it is available. The
        packed config is validated here.
        """
        # Imported lazily to avoid a circular import: ``collators`` imports
        # ``collate_sft_batch`` from this module.
        from skyrl.train.dataset.collators import DefaultCollator, PackedDataCollator

        if self.sft_cfg.use_sequence_packing:
            from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
                is_fp8_enabled,
            )

            self._validate_packing_cfg()
            transformer_config_kwargs = self.sft_cfg.megatron_config.transformer_config_kwargs or {}
            return PackedDataCollator(
                tokenizer=tokenizer,
                max_tokens_per_microbatch=self.sft_cfg.resolved_bin_capacity(),
                tp_size=self.sft_cfg.megatron_config.tensor_model_parallel_size,
                pp_size=self.sft_cfg.megatron_config.pipeline_model_parallel_size,
                cp_size=self.sft_cfg.megatron_config.context_parallel_size,
                dp_size=self._dp_size(),
                batch_size=self.sft_cfg.batch_size,
                micro_train_batch_size_per_gpu=self.sft_cfg.micro_train_batch_size_per_gpu,
                fp8_enabled=is_fp8_enabled(transformer_config_kwargs.get("fp8")),
            )
        return DefaultCollator(
            tokenizer=tokenizer,
            micro_train_batch_size_per_gpu=self.sft_cfg.micro_train_batch_size_per_gpu,
        )

    def _dp_size(self) -> int:
        """Number of DP ranks under the configured Megatron parallelism."""
        total_gpus = self.sft_cfg.placement.num_nodes * self.sft_cfg.placement.num_gpus_per_node
        tp = self.sft_cfg.megatron_config.tensor_model_parallel_size
        pp = self.sft_cfg.megatron_config.pipeline_model_parallel_size
        cp = self.sft_cfg.megatron_config.context_parallel_size
        return total_gpus // (tp * pp * cp)

    def _validate_packing_cfg(self):
        """Validate the config when ``use_sequence_packing=True``."""
        if self.sft_cfg.strategy != "megatron":
            raise ValueError(
                f"use_sequence_packing=True only supports strategy='megatron'; got "
                f"{self.sft_cfg.strategy!r}. Use the FSDP packing path instead."
            )
        # Sequence packing needs the THD layout, so it implies
        # remove_microbatch_padding=True. Auto-enable it (warning if the user
        # explicitly set it False) instead of erroring on the contradiction.
        if not self.sft_cfg.remove_microbatch_padding:
            logger.warning(
                "use_sequence_packing=True requires the THD layout; "
                "setting remove_microbatch_padding=True (was False)."
            )
            self.sft_cfg.remove_microbatch_padding = True

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup(self):
        """Initialize tokenizer, workers, dispatch, and tracker.

        Ray must already be initialized before calling this (either via
        ``initialize_ray`` on the head node or inside a Ray task).
        """
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": not self.cfg.trainer.disable_fast_tokenizer,
            "padding_side": "left",
        }

        self.is_vlm = check_is_vlm(self.cfg.trainer.policy.model.path)
        if self.is_vlm:
            self.processor = get_processor(self.cfg.trainer.policy.model.path, **tokenizer_kwargs)
            # Sequence packing / microbatch padding removal are unsupported for
            # VLMs (3D RoPE + image token positions). ``remove_microbatch_padding``
            # defaults to True, so disable both unconditionally and mirror the
            # change onto the already-built trainer config the workers receive.
            if self.sft_cfg.use_sequence_packing or self.sft_cfg.remove_microbatch_padding:
                logger.warning("VLM detected: disabling sequence packing / microbatch padding removal.")
            self.sft_cfg.use_sequence_packing = False
            self.sft_cfg.remove_microbatch_padding = False
            self.cfg.trainer.remove_microbatch_padding = False

        self.tokenizer = get_tokenizer(self.cfg.trainer.policy.model.path, **tokenizer_kwargs)
        self.collator = self._build_collator(self.tokenizer)
        self._init_tracker()
        self._init_workers()

    def _init_workers(self):
        """Create PPORayActorGroup and WorkerDispatch.

        Selects the correct PolicyWorker based on strategy.
        """
        if self.sft_cfg.strategy == "megatron":
            from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
                PolicyWorker,
            )
        else:
            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker

        num_gpus = self.sft_cfg.placement.num_gpus_per_node
        raw_pg = placement_group(
            [{"GPU": num_gpus, "CPU": num_gpus}] * self.sft_cfg.placement.num_nodes,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(raw_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        pg = ResolvedPlacementGroup(raw_pg)

        actor_group = PPORayActorGroup(
            self.cfg.trainer,
            num_nodes=self.sft_cfg.placement.num_nodes,
            num_gpus_per_node=num_gpus,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
            record_memory=self.cfg.trainer.policy.record_memory,
        )
        num_training_steps = (
            self.sft_cfg.dummy_run_max_steps if self.sft_cfg.dummy_run_full_ctx else self.sft_cfg.num_steps
        )
        if self.sft_cfg.max_training_steps is not None:
            num_training_steps = (
                self.sft_cfg.max_training_steps
                if num_training_steps is None
                else min(num_training_steps, self.sft_cfg.max_training_steps)
            )
        # num_steps may be None when num_epochs is used; without an explicit cap,
        # the worker will use its default large value for the LR scheduler.
        ray.get(
            actor_group.async_init_model(
                self.sft_cfg.model.path,
                num_training_steps=num_training_steps,
            )
        )
        ray.get(actor_group.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id))

        self.dispatch = WorkerDispatch(self.cfg, policy_actor_group=actor_group)

    def _init_tracker(self):
        self.tracker = Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backend=self.cfg.trainer.logger,
            config=self.sft_cfg,
            tags=self.cfg.trainer.tags,
        )

    def add_callback(self, callback: TrainingCallback) -> None:
        """Register a callback. Can be called anytime; events fired after this
        call will reach the new callback."""
        self._callback_handler.add(callback)

    def _build_callback_input(self, **fields) -> CallbackInput:
        """Snapshot loop counters + per-event fields into a CallbackInput."""
        return CallbackInput(
            global_step=self.global_step,
            epoch=self._current_epoch,
            total_steps=self._total_steps,
            steps_per_epoch=self._steps_per_epoch,
            **fields,
        )

    def _fire(self, event_name: str, **fields) -> None:
        """Build a CallbackInput and dispatch the given event to all callbacks."""
        cb_input = self._build_callback_input(**fields)
        getattr(self._callback_handler, event_name)(self, cb_input, self._training_control)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #

    def _load_and_tokenize(self, dataset_name: str, dataset_split: str) -> list:
        """Load and tokenize a dataset with caching support.

        Auto-detects the dataset format based on column names:
        - If a ``messages_key`` column exists, uses chat-format tokenization.
        - If ``instruction`` and ``output`` columns exist, uses Alpaca-format
          tokenization.

        Uses manual multiprocessing for parallel tokenization when num_workers > 0.
        With parallel mode, uses slice-based loading where each worker loads its
        own data slice directly from HuggingFace to eliminate pickle overhead.

        Caching:
        - Tokenized datasets are cached to disk as a HuggingFace ``Dataset``
          (arrow-backed, memory-mapped) for reuse across runs.
        - Cache key is a hash of dataset name, split, model, and tokenization params.
        - Set ``force_recache=True`` to ignore cache and re-tokenize.
        - Set ``disable_cache=True`` to disable caching entirely.

        Args:
            dataset_name: HuggingFace dataset name (e.g. ``"yahma/alpaca-cleaned"``).
            dataset_split: Dataset split (e.g. ``"train[:100]"`` or ``"test"``).

        Returns a list of tokenized examples (dicts with ``input_ids``,
        ``attention_mask``, ``num_actions``).
        """
        # Check cache first (unless disabled or force_recache)
        if not self.sft_cfg.disable_cache:
            cache_dir = self.sft_cfg.cache_dir

            # Compute cache key
            tools_key = self.sft_cfg.tools_key if self.sft_cfg.tools_key else None
            system_key = self.sft_cfg.system_key if self.sft_cfg.system_key else None
            cache_key = _compute_cache_key(
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                model_path=self.sft_cfg.model.path,
                max_length=self.sft_cfg.max_length,
                messages_key=self.sft_cfg.messages_key,
                train_on_what=self.sft_cfg.train_on_what.value,
                tools_key=tools_key,
                system_key=system_key,
            )
            cache_path = _get_cache_path(cache_dir, cache_key)

            # Try to load from cache (unless force_recache)
            if not self.sft_cfg.force_recache:
                cached = _load_from_cache(cache_path)
                if cached is not None:
                    return cached

            logger.info("Cache miss or force_recache=True, tokenizing dataset...")
            logger.info(f"Cache key: {cache_key}")

        logger.info(f"Loading dataset '{dataset_name}' split='{dataset_split}'...")
        dataset = load_dataset(dataset_name, split=dataset_split)

        columns = dataset.column_names
        num_workers = self.sft_cfg.num_workers

        # The HF processor needed for VLM tokenization does not round-trip
        # cleanly through the spawn-based worker pool, so VLM tokenization runs
        # sequentially.
        if self.is_vlm and num_workers != 0:
            logger.warning("VLM detected: forcing sequential tokenization (num_workers=0).")
            num_workers = 0

        # Sequential tokenization path
        if num_workers == 0:
            logger.info("Tokenizing dataset (sequential)...")
            if self.sft_cfg.messages_key in columns:
                tools_key = self.sft_cfg.tools_key if self.sft_cfg.tools_key in columns else None
                system_key = self.sft_cfg.system_key if self.sft_cfg.system_key in columns else None
                tokenized = [
                    tokenize_chat_example(
                        ex,
                        self.tokenizer,
                        self.sft_cfg.max_length,
                        self.sft_cfg.messages_key,
                        train_on_what=self.sft_cfg.train_on_what,
                        tools_key=tools_key,
                        system_key=system_key,
                        processor=self.processor,
                    )
                    for ex in dataset
                ]
            elif "instruction" in columns and "output" in columns:
                tokenized = [tokenize_sft_example(ex, self.tokenizer, self.sft_cfg.max_length) for ex in dataset]
            else:
                raise ValueError(
                    f"Unrecognized dataset format. Expected '{self.sft_cfg.messages_key}' column "
                    f"(chat format) or 'instruction'+'output' columns (Alpaca format). "
                    f"Found columns: {columns}"
                )
            tokenized = [ex for ex in tokenized if ex is not None]
            logger.info(f"Tokenized {len(tokenized)} examples (filtered from {len(dataset)})")

            # Save to cache if enabled
            if not self.sft_cfg.disable_cache:
                # TODO (sumanthrh): Currently we use a simple list instead of dataset + stateful dataloader
                # for simplicity but for caching we use HF Dataset since file sizes can get large
                # We should migrate to using HF datasets + a dataloader so that we don't materialize
                # the full dataset in memory
                _save_to_cache(cache_path, tokenized)

            return tokenized

        # Parallel tokenization path with slice-based loading
        logger.info(f"Tokenizing dataset with {num_workers} workers (slice-based loading)...")

        # Cache tokenizer to temp dir for fast worker loading
        tokenizer_cache_dir = tempfile.mkdtemp(prefix="skyrl_tokenizer_")
        try:
            self.tokenizer.save_pretrained(tokenizer_cache_dir)

            # Slice the already-loaded dataset; the original split string is
            # forwarded to workers verbatim so HF parses it (no local regex).
            dataset_size = len(dataset)
            chunk_size = max(1, dataset_size // num_workers)

            # Generate worker slice boundaries
            worker_args = []
            for worker_idx in range(num_workers):
                worker_start = worker_idx * chunk_size
                # Last worker takes any remainder
                if worker_idx == num_workers - 1:
                    worker_end = dataset_size
                else:
                    worker_end = min((worker_idx + 1) * chunk_size, dataset_size)

                # Skip empty slices
                if worker_start >= worker_end:
                    continue

                # Prepare worker arguments based on format
                if self.sft_cfg.messages_key in columns:
                    tools_key = self.sft_cfg.tools_key if self.sft_cfg.tools_key in columns else None
                    system_key = self.sft_cfg.system_key if self.sft_cfg.system_key in columns else None
                    worker_args.append(
                        (
                            dataset_name,
                            dataset_split,
                            worker_start,
                            worker_end,
                            tokenizer_cache_dir,
                            self.sft_cfg.max_length,
                            self.sft_cfg.messages_key,
                            self.sft_cfg.train_on_what.value,
                            tools_key,
                            system_key,
                        )
                    )
                elif "instruction" in columns and "output" in columns:
                    worker_args.append(
                        (
                            dataset_name,
                            dataset_split,
                            worker_start,
                            worker_end,
                            tokenizer_cache_dir,
                            self.sft_cfg.max_length,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unrecognized dataset format. Expected '{self.sft_cfg.messages_key}' column "
                        f"(chat format) or 'instruction'+'output' columns (Alpaca format). "
                        f"Found columns: {columns}"
                    )

            # Select worker function based on format
            if self.sft_cfg.messages_key in columns:
                worker_fn = _tokenize_chat_slice_worker
            else:
                worker_fn = _tokenize_alpaca_slice_worker

            logger.info(f"Dividing {dataset_size} examples among {len(worker_args)} workers")

            # Use spawn to avoid Ray fork issues
            ctx = mp.get_context("spawn")

            # Process in parallel
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(worker_fn, worker_args)

            # Flatten results
            tokenized = []
            for chunk_results in results:
                tokenized.extend(chunk_results)

            logger.info(f"Tokenized {len(tokenized)} examples (filtered from {dataset_size})")

            # Save to cache if enabled
            if not self.sft_cfg.disable_cache:
                _save_to_cache(cache_path, tokenized)

            return tokenized

        finally:
            # Cleanup temp tokenizer cache
            import shutil

            shutil.rmtree(tokenizer_cache_dir, ignore_errors=True)

    def load_dataset(self) -> tuple[list, list[int]]:
        """Load the training dataset(s): pretokenized stores or tokenize-on-load.

        When ``pretokenized_dataset_paths`` is set, each store is loaded through
        :func:`~skyrl.train.dataset.pretokenized.load_from_pretokenized` (same
        ``list[dict]`` shape as :meth:`_load_and_tokenize`, no online
        tokenization) and concatenated in config order. Otherwise each
        ``(name, split)`` pair from ``train_datasets``/``train_dataset_splits``
        is tokenized independently through :meth:`_load_and_tokenize`
        (preserving per-dataset cache keys), then concatenated in config order.
        Either way, multiple sources are mixed per ``train_dataset_weights``.

        Returns:
            ``(tokenized, dataset_lengths)`` where ``dataset_lengths`` holds the
            tokenized size of each source, used to configure weighted mixing in
            :meth:`build_train_sampler`.
        """
        tokenized: list = []
        dataset_lengths: list[int] = []
        if self.sft_cfg.pretokenized_dataset_paths:
            for path in self.sft_cfg.pretokenized_dataset_paths:
                # The loader raises on 0 usable rows, so no empty-source check.
                source = load_from_pretokenized(path, max_length=self.sft_cfg.max_length)
                tokenized.extend(source)
                dataset_lengths.append(len(source))
            if len(dataset_lengths) > 1:
                per_dataset = ", ".join(
                    f"{path}={length}" for path, length in zip(self.sft_cfg.pretokenized_dataset_paths, dataset_lengths)
                )
                logger.info(f"Concatenated {len(dataset_lengths)} pretokenized datasets: {per_dataset}")
            return tokenized, dataset_lengths
        for name, split in zip(self.sft_cfg.train_datasets, self.sft_cfg.train_dataset_splits):
            source = self._load_and_tokenize(name, split)
            if len(source) == 0:
                raise ValueError(f"Training dataset '{name}' (split '{split}') tokenized to 0 examples.")
            tokenized.extend(source)
            dataset_lengths.append(len(source))
        if len(dataset_lengths) > 1:
            per_dataset = ", ".join(
                f"{name}={length}" for name, length in zip(self.sft_cfg.train_datasets, dataset_lengths)
            )
            logger.info(f"Concatenated {len(dataset_lengths)} training datasets: {per_dataset}")
        return tokenized, dataset_lengths

    def load_eval_datasets(self) -> Optional[list[tuple[str, list]]]:
        """Load and tokenize the eval dataset(s), or return ``None`` if not configured.

        When ``eval_pretokenized_dataset_paths`` is set, each store is loaded
        through :func:`~skyrl.train.dataset.pretokenized.load_from_pretokenized`
        and named by the corresponding entry of ``eval_dataset_names`` (filled
        from the path basenames by config normalization when not set
        explicitly).

        Returns:
            One ``(name, tokenized)`` pair per eval source, where ``name``
            namespaces the eval metrics (``eval/{name}/...``).
        """
        if self.sft_cfg.eval_pretokenized_dataset_paths:
            return [
                (name, load_from_pretokenized(path, max_length=self.sft_cfg.max_length))
                for name, path in zip(self.sft_cfg.eval_dataset_names, self.sft_cfg.eval_pretokenized_dataset_paths)
            ]
        if not self.sft_cfg.eval_datasets:
            return None
        eval_sets: list[tuple[str, list]] = []
        for name, dataset, split in zip(
            self.sft_cfg.eval_dataset_names, self.sft_cfg.eval_datasets, self.sft_cfg.eval_dataset_splits
        ):
            eval_tokenized = self._load_and_tokenize(dataset, split)
            if len(eval_tokenized) == 0:
                raise ValueError(
                    f"Eval dataset '{dataset}' (split '{split}') tokenized to 0 examples. "
                    f"Provide a non-empty eval split or remove it from eval_datasets."
                )
            eval_sets.append((name, eval_tokenized))
        return eval_sets

    def _log_dataset_stats(self, tokenized: list) -> None:
        """Log tokenized sequence length statistics over the training set.

        Reports count, mean, median (q50), q25, q75, min, max of the tokenized
        ``input_ids`` lengths. Logs once via ``logger.info``.
        """
        if not tokenized:
            logger.warning("No tokenized examples to compute stats over")
            return

        lengths = [len(ex["input_ids"]) for ex in tokenized]
        n = len(lengths)
        sorted_lengths = sorted(lengths)

        def pct(p: float) -> int:
            # Simple nearest-rank percentile over ints; adequate for dataset stats.
            idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return sorted_lengths[idx]

        mean_len = sum(lengths) / n
        q25 = pct(25)
        q50 = pct(50)
        q75 = pct(75)
        min_len = sorted_lengths[0]
        max_len = sorted_lengths[-1]

        logger.info(
            f"Dataset stats (tokenized lengths over {n} examples):\n"
            f"total={sum(lengths)}, mean={mean_len:.1f}, median={q50}, q25={q25}, q75={q75}, min={min_len}, max={max_len}"
        )

    def collate_batch(self, examples: list, batch_size: int) -> TrainingInputBatch:
        """Collate examples into a TrainingInputBatch via the configured collator.

        Delegates to ``self.collator`` (``DefaultCollator`` or, when sequence
        packing is enabled, ``PackedDataCollator``).

        Args:
            examples: Tokenized examples to collate.
            batch_size: Global batch dimension. The train path passes
                ``sft_cfg.batch_size`` and the eval path passes its
                per-dispatch chunk size.
        """
        return self.collator(examples, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Dataloaders & samplers
    # ------------------------------------------------------------------ #

    def build_train_sampler(
        self, tokenized: list, dataset_lengths: Optional[list[int]] = None
    ) -> Optional[torch.utils.data.Sampler]:
        """Build the training sampler from ``sft_cfg.sampler``.

        Returns ``None`` for the default ``"random"`` strategy over a single
        dataset, signalling :meth:`build_train_dataloader` to use the
        dataloader's built-in ``shuffle=True`` path (which is statefully
        checkpointed by ``StatefulDataLoader``). With multiple training
        datasets, ``"random"`` instead returns a :class:`DataMixingSampler`
        configured with the per-dataset lengths and ``train_dataset_weights``.
        For ``"sequential"`` and ``"custom"`` it returns an explicit stateful
        sampler.

        Custom samplers are imported from ``sft_cfg.sampler_class_path`` and
        instantiated as ``ClassName(tokenized, **sft_cfg.sampler_kwargs)``. With
        multiple datasets, the per-dataset ``lengths`` are injected into the
        kwargs (unless the user already supplied ``lengths``), so the sampler
        constructor must accept them.

        Args:
            tokenized: The (concatenated) tokenized training dataset.
            dataset_lengths: Tokenized size of each source dataset, in order.
                ``None`` is treated as a single source spanning ``tokenized``.
        """
        from skyrl.train.dataset.samplers import (
            DataMixingSampler,
            StatefulSequentialSampler,
            import_sampler_class,
        )

        multi_dataset = dataset_lengths is not None and len(dataset_lengths) > 1
        sampler_type = self.sft_cfg.sampler
        if sampler_type == "random":
            if not multi_dataset:
                return None
            # Config normalization (validate_sft_cfg) fills equal weights for
            # the random sampler on every construction path.
            return DataMixingSampler(
                tokenized,
                lengths=dataset_lengths,
                weights=self.sft_cfg.train_dataset_weights,
                seed=self.sft_cfg.seed,
            )
        if sampler_type == "sequential":
            return StatefulSequentialSampler(tokenized)
        if sampler_type == "custom":
            if not self.sft_cfg.sampler_class_path:
                raise ValueError("sampler='custom' requires sampler_class_path to be set.")
            sampler_cls = import_sampler_class(self.sft_cfg.sampler_class_path)
            sampler_kwargs = self.sft_cfg.sampler_kwargs
            if multi_dataset:
                # User-provided kwargs win over the injected lengths.
                sampler_kwargs = {"lengths": dataset_lengths, **sampler_kwargs}
            return sampler_cls(tokenized, **sampler_kwargs)
        raise ValueError(f"Unknown sampler '{sampler_type}'. Must be one of 'random', 'sequential', 'custom'.")

    def build_train_dataloader(
        self, tokenized: list, dataset_lengths: Optional[list[int]] = None
    ) -> StatefulDataLoader:
        """Build the training ``StatefulDataLoader``.

        Sampling order is seeded for reproducibility and captured in the dataloader's
        ``state_dict`` for checkpoint/resume. Uses ``drop_last=False`` so every
        example in an epoch is trained on: a final short batch is padded up to
        ``batch_size`` in :func:`collate_sft_examples` (padded rows are masked
        out of the loss), instead of being dropped. (Packed batches are not
        row-padded; the FFD packer already handles a short example list.)

        Resume note: ``StatefulDataLoader`` restores the *in-progress* epoch
        bit-exactly (the common case). For the built-in ``"random"`` sampler,
        epochs after the resumed one are re-shuffled into a valid but not
        byte-identical order (the generator advances differently after a
        partially-replayed epoch) -- this matches the RL trainer's dataloader.
        Custom samplers that span the whole run in a single pass (e.g. the
        ``CurriculumLearningSampler`` example under ``examples/train/sft/`` with
        ``num_samples=num_steps*batch_size``) resume bit-exactly across the
        entire schedule, since the iterator is never re-created.
        """
        collate_fn = functools.partial(
            collate_sft_examples,
            collator=self.collator,
            batch_size=self.sft_cfg.batch_size,
            # Packed batches dispatch FFD-bin rows (already a multiple of
            # dp_size), so only the un-packed path pads to batch_size.
            pad_to_batch_size=not self.sft_cfg.use_sequence_packing,
        )

        seeded_generator = torch.Generator()
        seeded_generator.manual_seed(self.sft_cfg.seed)

        sampler = self.build_train_sampler(tokenized, dataset_lengths)
        num_workers = self.sft_cfg.dataloader_num_workers

        return StatefulDataLoader(
            tokenized,
            batch_size=self.sft_cfg.batch_size,
            sampler=sampler,
            # ``shuffle`` and an explicit ``sampler`` are mutually exclusive;
            # only enable the built-in random sampler when none was provided.
            shuffle=sampler is None,
            collate_fn=collate_fn,
            # Keep the trailing partial batch (padded in collate) so no example
            # is dropped within an epoch.
            drop_last=False,
            generator=seeded_generator,
            num_workers=num_workers,
            persistent_workers=self.sft_cfg.dataloader_persistent_workers and num_workers > 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    def build_eval_dataloader(self, eval_tokenized: list) -> StatefulDataLoader:
        """Build the eval ``StatefulDataLoader``.

        Order is sequential and the final short chunk is kept (``drop_last=False``);
        :meth:`run_eval` pads it.
        """
        # One micro-batch per DP rank per dispatch call — keeps memory usage bounded
        # and removes the need for a separate `eval_batch_size` knob.
        dp_size = self.dispatch.dp_size("policy")
        eval_chunk_size = self.sft_cfg.micro_train_batch_size_per_gpu * dp_size
        collate_fn = functools.partial(
            collate_sft_examples,
            collator=self.collator,
            batch_size=eval_chunk_size,
        )
        num_workers = self.sft_cfg.dataloader_num_workers
        return StatefulDataLoader(
            eval_tokenized,
            batch_size=eval_chunk_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=self.sft_cfg.dataloader_persistent_workers and num_workers > 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint resume
    # ------------------------------------------------------------------ #

    def load_checkpoint(self) -> int:
        """Load a checkpoint and return the step number to resume from.

        Behaviour depends on ``sft_cfg.resume_from``:
        - ``""`` (empty): no resume, return 0.
        - ``"latest"``: read ``latest_ckpt_global_step.txt`` from ``ckpt_path``.
        - otherwise: treat as a direct path to a ``global_step_N`` directory.

        Returns:
            The global step to resume from (0 if no checkpoint loaded).
        """
        resume_from = self.sft_cfg.resume_from
        if not resume_from:
            return 0

        if resume_from == "latest":
            if not self.sft_cfg.ckpt_path:
                logger.info("resume_from='latest' but ckpt_path is empty, starting from scratch")
                return 0
            latest_file = os.path.join(self.sft_cfg.ckpt_path, "latest_ckpt_global_step.txt")
            if not io.exists(latest_file):
                logger.info("No latest checkpoint marker found, starting from scratch")
                return 0
            with io.open_file(latest_file, "r") as f:
                ckpt_step = int(f.read().strip())
            checkpoint_path = os.path.join(self.sft_cfg.ckpt_path, f"{GLOBAL_STEP_PREFIX}{ckpt_step}")
            # Validate consistency: ensure no stale checkpoint folders from prior runs
            validate_consistency_for_latest_checkpoint(
                self.sft_cfg.ckpt_path,
                ckpt_step,
                checkpoint_path,
                latest_file,
                self.sft_cfg.ckpt_interval,
            )
        else:
            checkpoint_path = resume_from

        if not io.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        global_step = extract_step_from_path(checkpoint_path)
        if global_step == -1:
            raise ValueError(
                f"Cannot extract step number from checkpoint path: {checkpoint_path}. "
                f"Expected a directory named '{GLOBAL_STEP_PREFIX}<N>'."
            )

        # Load and validate trainer state if available
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if io.exists(trainer_state_path):
            with io.open_file(trainer_state_path, "rb") as f:
                trainer_state = torch.load(f, map_location="cpu", weights_only=False)
            saved_global_step = trainer_state.get("global_step", global_step)
            logger.info("Successfully loaded trainer state")
            if saved_global_step != global_step:
                logger.warning(
                    f"Global step mismatch: path={global_step}, saved={saved_global_step}. Using path value."
                )
        else:
            logger.warning(
                f"No trainer_state.pt found at {trainer_state_path}. "
                "This checkpoint was likely saved by an older version."
            )

        policy_ckpt_dir = os.path.join(checkpoint_path, "policy")
        logger.info(f"Loading checkpoint from {checkpoint_path} (step {global_step})")
        self.dispatch.load_checkpoint(
            "policy",
            policy_ckpt_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )

        # Restore train dataloader / sampler position so sampling resumes from
        # the exact next example (mirrors the RL trainer's data.pt handling).
        dataloader_state_path = os.path.join(checkpoint_path, "data.pt")
        if io.exists(dataloader_state_path):
            try:
                with io.open_file(dataloader_state_path, "rb") as f:
                    dataloader_state = torch.load(f, map_location="cpu", weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state)
                logger.info("Restored train dataloader state")
            except Exception as e:
                logger.warning(f"Failed to restore dataloader state: {e}")
        else:
            logger.warning(
                f"No data.pt found at {dataloader_state_path}; dataloader will start from the "
                "beginning of its sampling order (older checkpoint or RNG-only resume)."
            )

        logger.info(f"Successfully resumed from global_step_{global_step}")
        return global_step

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def run_eval(self) -> tuple[dict, int]:
        """Compute eval loss over every configured eval dataset.

        Runs :meth:`_run_eval_one` per ``(name, dataloader)`` pair in
        :attr:`eval_dataloaders`, namespacing each dataset's metrics by its
        name. The keys are later prefixed with ``eval/`` at the logging sites,
        yielding ``eval/{name}/loss`` — nested even with a single eval dataset,
        so runs with and without dataset mixing chart the same metric keys.

        Returns:
            ``(metrics, num_eval_batches)`` where ``metrics`` maps
            ``{name}/loss`` to that dataset's token-weighted mean loss and
            ``num_eval_batches`` is the total batch count across datasets
            (stdout bookkeeping, not a wandb metric).
        """
        if not self.eval_dataloaders:
            raise ValueError(
                "run_eval called without eval dataloaders. Provide non-empty eval splits or "
                "disable eval by setting eval_datasets=None."
            )
        metrics: dict[str, float] = {}
        total_eval_batches = 0
        for name, eval_dataloader in self.eval_dataloaders:
            eval_loss, num_eval_batches = self._run_eval_one(eval_dataloader)
            metrics[f"{name}/loss"] = eval_loss
            total_eval_batches += num_eval_batches
            logger.info(f"Eval dataset '{name}': loss={eval_loss:.4f} over {num_eval_batches} batches")
        return metrics, total_eval_batches

    def _run_eval_one(self, eval_dataloader: StatefulDataLoader) -> tuple[float, int]:
        """Compute eval loss over one eval dataset.

        Iterates the dataloader (chunks of ``micro_train_batch_size_per_gpu * dp_size``,
        i.e. exactly one micro-batch per DP rank per dispatch call), calls
        :meth:`WorkerDispatch.forward` with ``loss_fn="cross_entropy"`` (which
        runs the model in ``eval()`` mode under ``no_grad``), and aggregates the
        per-batch losses into a token-weighted mean.

        The aggregated loss is a token-weighted mean of the per-batch losses,
        which are themselves per-non-pad-token means within each batch. This
        yields the true per-non-pad-token mean across the eval dataset.

        Returns:
            ``(eval_loss, num_eval_batches)``.
        """
        # The dataloader yields one chunk per DP rank's micro-batch; the final
        # (possibly short) chunk is padded below up to the full chunk size.
        eval_chunk_size = eval_dataloader.batch_size

        # Pad a trailing partial batch up to ``eval_chunk_size`` via
        # ``pad_training_input_batch`` (which zeros ``loss_mask`` on padded rows).
        # Padded rows contribute 0 to the cross-entropy numerator, and the
        # pre-padding ``total_nonpad`` scaling in ``collate_batch`` excludes
        # them from the denominator, so the reported ``eval_loss`` is the
        # per-real-token mean over the full (non-padded) eval set.
        total_loss_weighted = 0.0
        total_tokens = 0
        num_eval_batches = 0
        for batch in eval_dataloader:
            num_eval_batches += 1
            # Pad the last (possibly-short) chunk so every dispatch sees exactly
            # ``eval_chunk_size`` rows. ``pad_training_input_batch`` zeros the
            # ``loss_mask`` for padding rows; with ``pad_size=0`` it is a no-op.
            num_rows = batch["sequences"].shape[0]
            pad_rows = eval_chunk_size - num_rows
            if pad_rows > 0:
                logger.info(
                    f"Padding final eval batch by {pad_rows} rows "
                    f"({num_rows} real -> {eval_chunk_size} total); "
                    f"padded rows are masked out of the loss."
                )
                batch = pad_training_input_batch(batch, pad_rows)
            # Count non-pad response tokens (from the unscaled mask, recovered from the batch)
            # We use the attention_mask response window via collate_sft_batch's loss_mask which
            # was 0/1 before scaling. Recover the count from the batch by counting positive entries.
            # Padded rows have loss_mask=0 so they are excluded here.
            nonpad_tokens = int((batch["loss_mask"] > 0).sum().item())
            output = self.dispatch.forward(
                "policy",
                batch,
                loss_fn="cross_entropy",
                loss_fn_config=None,
            )
            batch_loss = float(output.metrics.get("loss", float("nan")))
            total_loss_weighted += batch_loss * nonpad_tokens
            total_tokens += nonpad_tokens

        eval_loss = total_loss_weighted / max(total_tokens, 1)
        return eval_loss, num_eval_batches

    def train_step(self, batch: TrainingInputBatch, step: int) -> dict:
        """Execute a single training step: forward_backward + optim_step.

        Args:
            batch: The collated training batch.
            step: Current global step (reserved for future use, e.g. scheduling).

        Returns:
            Dict with ``loss``, ``grad_norm``, and ``timings``.
        """
        timings: dict[str, float] = {}
        with Timer("forward_backward", timings):
            output = self.dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
        with Timer("optim_step", timings):
            grad_norm = self.dispatch.optim_step("policy")

        metrics = output.metrics

        # One profiler step per SFT global step.
        if self._torch_profiler_enabled:
            self.dispatch.profile_step("policy")

        loss_val = metrics.get("final_loss", metrics.get("loss", float("nan")))
        return {
            "loss": loss_val,
            "grad_norm": grad_norm,
            "timings": timings,
        }

    def _validate_batch_parallelism(self):
        """Validate that batch_size is compatible with data-parallel and micro-batch sizes."""
        batch_size = self.sft_cfg.batch_size
        total_gpus = self.sft_cfg.placement.num_nodes * self.sft_cfg.placement.num_gpus_per_node
        if self.sft_cfg.use_sequence_packing:
            # With packing, batch_size is the *example* count (not bins) and the
            # per-DP-rank bin count == bins_per_shard. The worker micro batch
            # size refers to bin rows per micro-batch, derived from the
            # ``max_tokens_per_microbatch`` token budget. We only require
            # batch_size >= dp_size (every DP rank needs >= 1 bin) and do NOT
            # require batch_size % micro_train_batch_size_per_gpu == 0, because
            # micro_train_batch_size_per_gpu refers to bins-per-MB, not
            # examples-per-MB; FFD rounds the bin count up to a multiple of
            # dp_size, and bins/MB is a separate knob.
            dp_size = self._dp_size()
            if batch_size < dp_size:
                raise ValueError(
                    f"batch_size ({batch_size}) must be >= dp_size ({dp_size}) when "
                    f"use_sequence_packing=True (each DP rank needs at least one bin)."
                )
            return
        if self.sft_cfg.strategy == "megatron":
            tp = self.sft_cfg.megatron_config.tensor_model_parallel_size
            pp = self.sft_cfg.megatron_config.pipeline_model_parallel_size
            dp_size = total_gpus // (tp * pp)
        else:
            # FSDP: all GPUs are data-parallel
            dp_size = total_gpus
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by data-parallel size ({dp_size})")
        per_dp_batch = batch_size // dp_size
        micro_batch = self.sft_cfg.micro_train_batch_size_per_gpu
        if per_dp_batch % micro_batch != 0:
            raise ValueError(
                f"batch_size ({self.sft_cfg.batch_size}) / dp_size ({dp_size}) must be divisible by "
                f"micro_train_batch_size_per_gpu ({micro_batch})"
            )

    def _build_dummy_batch(self) -> TrainingInputBatch:
        """Build a dummy batch of random full-context sequences for benchmarking."""
        batch_size = self.sft_cfg.batch_size
        max_length = self.sft_cfg.max_length
        vocab_size = self.tokenizer.vocab_size

        # num_actions is max_length - 1 because the autoregressive model
        # produces log-probs for positions 1..T (predicting next token),
        # so the first token has no corresponding log-prob.
        num_actions = max_length - 1

        sequences = torch.randint(0, vocab_size, (batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
        # All tokens are non-pad in the dummy batch, so total_nonpad = batch_size * num_actions.
        # Scaling = 1 / total_nonpad.
        total_nonpad = batch_size * num_actions
        loss_mask = torch.ones(batch_size, num_actions, dtype=torch.float) / total_nonpad

        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
            }
        )
        batch.metadata = {"response_length": num_actions}
        return batch

    def _train_dummy(self):
        """Dummy training loop for benchmarking. Skips real data, checkpoints, and resume."""
        self._validate_batch_parallelism()
        batch = self._build_dummy_batch()
        num_steps = self.sft_cfg.dummy_run_max_steps

        logger.info(
            f"Starting dummy SFT training for {num_steps} steps "
            f"(batch_size={self.sft_cfg.batch_size}, max_length={self.sft_cfg.max_length})..."
        )

        if self._ray_gpu_monitor is not None:
            self._ray_gpu_monitor.start()
        if self._torch_profiler_enabled:
            self.dispatch.start_profile("policy")
        try:
            for step in range(num_steps):
                all_timings: dict[str, float] = {}

                with Timer("step", all_timings):
                    step_result = self.train_step(batch, step)
                    all_timings.update(step_result["timings"])

                actual_num_tokens = batch["attention_mask"].sum().item()
                self._total_tokens_processed += actual_num_tokens
                tokens_per_second = actual_num_tokens / all_timings["step"]

                log_dict = {
                    "train/loss": step_result["loss"],
                    "train/grad_norm": step_result["grad_norm"],
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens_per_second_per_gpu": tokens_per_second / self._num_training_gpus,
                    "train/actual_num_tokens": actual_num_tokens,
                    "train/total_tokens_processed": self._total_tokens_processed,
                }
                log_dict.update({f"timing/{k}": v for k, v in all_timings.items()})
                if self._ray_gpu_monitor is not None:
                    log_dict.update(self._ray_gpu_monitor.flush())

                self.tracker.log(log_dict, step=step, commit=True)
                logger.info(
                    f"Step {step}: loss={step_result['loss']:.4f}, "
                    f"grad_norm={step_result['grad_norm']}, "
                    f"tokens_per_second={tokens_per_second:.0f}"
                )
        finally:
            if self._torch_profiler_enabled:
                self.dispatch.stop_profile("policy")

        logger.info("Dummy SFT training complete!")

    @staticmethod
    def _resolve_num_steps(
        *,
        num_steps: Optional[int],
        num_epochs: int,
        steps_per_epoch: int,
        max_training_steps: Optional[int],
    ) -> int:
        """Resolve the total number of training steps.

        Explicit ``num_steps`` takes precedence; otherwise it is derived from
        ``num_epochs``. When ``max_training_steps`` is set it caps the result.
        """
        resolved = num_steps if num_steps is not None else num_epochs * steps_per_epoch
        if max_training_steps is not None:
            resolved = min(resolved, max_training_steps)
        return resolved

    def train(self):
        """Full training loop: load data, iterate, log, checkpoint."""
        if self.sft_cfg.dummy_run_full_ctx:
            if self.sft_cfg.resume_from:
                logger.warning("resume_from is ignored in dummy run mode")
            return self._train_dummy()

        tokenized, dataset_lengths = self.load_dataset()

        # Log tokenized sequence length statistics (once, before training loop)
        self._log_dataset_stats(tokenized)

        # Load eval datasets (if configured). We load once up-front so the
        # tokenization cost is amortized across all eval invocations.
        eval_datasets = self.load_eval_datasets()
        if eval_datasets is not None:
            for eval_name, eval_tokenized in eval_datasets:
                logger.info(f"Eval dataset '{eval_name}' loaded: {len(eval_tokenized)} examples")

        batch_size = self.sft_cfg.batch_size

        self._validate_batch_parallelism()

        # Build stateful dataloaders (replaces manual list shuffling/slicing).
        # The training sampler is selected by ``sft_cfg.sampler`` and its
        # position is captured in the checkpoint for resume.
        self.train_dataloader = self.build_train_dataloader(tokenized, dataset_lengths)
        if eval_datasets is not None:
            self.eval_dataloaders = [
                (eval_name, self.build_eval_dataloader(eval_tokenized)) for eval_name, eval_tokenized in eval_datasets
            ]

        # Validate the invariant the training loop relies on: the dataloader must
        # yield at least one batch. With drop_last=False (the final short batch is
        # padded, not dropped) this only happens when the sampler yields nothing
        # at all -- an empty dataset or a custom sampler with num_samples=0.
        # Catching it here turns an otherwise opaque StopIteration in the training
        # loop into a clear error.
        if len(self.train_dataloader) == 0:
            raise ValueError(
                f"Train dataloader is empty (0 batches): the sampler yields no indices "
                f"(dataset has {len(tokenized)} examples). "
                f"Provide a non-empty dataset, or set the custom sampler's num_samples > 0."
            )

        # steps_per_epoch is derived from the dataloader. With drop_last=False it
        # is ceil(len(sampler) / batch_size) -- the trailing partial batch counts
        # as a step. Callbacks rely on it; guaranteed >= 1 by the check above.
        steps_per_epoch = len(self.train_dataloader)

        if self.sft_cfg.num_steps is None:
            logger.info(
                f"num_steps not set; deriving from num_epochs={self.sft_cfg.num_epochs}: "
                f"{len(self.train_dataloader)} steps/epoch * {self.sft_cfg.num_epochs} = "
                f"{self.sft_cfg.num_epochs * steps_per_epoch} steps"
            )

        num_steps = self._resolve_num_steps(
            num_steps=self.sft_cfg.num_steps,
            num_epochs=self.sft_cfg.num_epochs,
            steps_per_epoch=steps_per_epoch,
            max_training_steps=self.sft_cfg.max_training_steps,
        )

        if self.sft_cfg.max_training_steps is not None:
            logger.info(f"Capping training at max_training_steps={self.sft_cfg.max_training_steps}")

        # Resume from checkpoint if configured. This also restores the train
        # dataloader's sampling position (when a data.pt is present), so the
        # first pass over ``self.train_dataloader`` below continues mid-epoch.
        # start_step is the last *completed* step (checkpoint is saved AFTER the
        # optimizer update), so we begin at start_step + 1 to avoid replaying it.
        start_step = self.load_checkpoint()

        start_epoch = start_step // steps_per_epoch
        current_epoch = start_epoch

        # Initialize `global_step`
        self.global_step = start_step

        # Publish loop metadata so CallbackInput can be built consistently.
        self._total_steps = num_steps
        self._steps_per_epoch = steps_per_epoch
        self._current_epoch = current_epoch
        self._training_control.reset()

        logger.info(f"Starting SFT training for {num_steps} steps (batch_size={batch_size})...")
        if start_step > 0:
            logger.info(f"Resuming from step {start_step}")

        if self._ray_gpu_monitor is not None:
            self._ray_gpu_monitor.start()

        # Tracks whether the most recent in-loop iteration saved a checkpoint
        # (either via the ckpt_interval or via a callback-driven ``should_save``).
        did_save_last_step = False

        self._fire("on_train_start")

        # Baseline eval before training begins (logged at step 0).
        # Wandb's step counter starts at 0; the training loop's first commit
        # advances it to >=1, so step=0 here does not conflict with later steps.
        if self.sft_cfg.eval_before_train and self.eval_dataloaders is not None:
            self._fire("on_eval_start")
            eval_metrics, num_eval_batches = self.run_eval()
            self._fire("on_eval_end", metrics=eval_metrics)
            baseline_log = {f"eval/{k}": v for k, v in eval_metrics.items()}
            self._fire("on_log", logs=baseline_log)
            self.tracker.log(baseline_log, step=self.global_step, commit=True)
            logger.info(
                f"Baseline eval before training: {_format_eval_metrics(eval_metrics)} "
                f"over {num_eval_batches} batches"
            )

        # SkyRL starts counting at step 1
        self.global_step = start_step + 1 if start_step > 0 else 1
        self._fire("on_epoch_start")

        # Iterate once on global_step rather than looping epoch-by-epoch: a
        # single iterator is advanced across steps, and only re-created at an
        # epoch boundary (StopIteration). Custom samplers that span the whole
        # run in one pass therefore never re-create the iterator, preserving
        # their state across the (conceptual) epoch boundaries.
        data_iter = iter(self.train_dataloader)

        if self._torch_profiler_enabled:
            self.dispatch.start_profile("policy")
        try:
            while self.global_step <= num_steps:
                all_timings: dict[str, float] = {}

                with Timer("step", all_timings):

                    # Fetch the next batch; on epoch exhaustion, close the epoch and
                    # restart the iterator (reshuffles the random/sequential samplers).
                    with Timer("data_loading", all_timings):
                        batch = next(data_iter, None)
                    if batch is None:
                        self._fire("on_epoch_end")
                        current_epoch += 1
                        self._current_epoch = current_epoch
                        self._fire("on_epoch_start")
                        data_iter = iter(self.train_dataloader)
                        with Timer("data_loading", all_timings):
                            batch = next(data_iter)

                    self._fire("on_step_start", batch=batch)

                    # Training step
                    step_result = self.train_step(batch, self.global_step)
                    all_timings.update(step_result["timings"])

                # Compute throughput using actual (non-padding) tokens. A padded
                # tail batch appends ``pad_size`` rows (copies of row 0) that are
                # masked out of the loss; exclude them from the token count so the
                # throughput metric reflects only real tokens.
                batch_padded_seq_len = batch["sequences"].shape[1]
                pad_size = batch.metadata.get("pad_size", 0) if batch.metadata else 0
                real_rows = batch["attention_mask"].shape[0] - pad_size
                actual_num_tokens = batch["attention_mask"][:real_rows].sum().item()
                self._total_tokens_processed += actual_num_tokens
                tokens_per_second = actual_num_tokens / all_timings["step"]

                # Build log dict
                log_dict = {
                    "train/loss": step_result["loss"],
                    "train/grad_norm": step_result["grad_norm"],
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens_per_second_per_gpu": tokens_per_second / self._num_training_gpus,
                    "train/actual_num_tokens": actual_num_tokens,
                    "train/batch_padded_seq_len": batch_padded_seq_len,
                    "train/total_tokens_processed": self._total_tokens_processed,
                }
                log_dict.update({f"timing/{k}": v for k, v in all_timings.items()})
                if self._ray_gpu_monitor is not None:
                    log_dict.update(self._ray_gpu_monitor.flush())

                self._fire("on_step_end", batch=batch, metrics=step_result)

                # Capture callback-driven triggers, then reset so they only fire once.
                force_save = self._training_control.should_save
                force_eval = self._training_control.should_evaluate
                self._training_control.should_save = False
                self._training_control.should_evaluate = False

                # Checkpoint: interval-driven or callback-requested.
                interval_save = (
                    self.sft_cfg.ckpt_interval > 0
                    and self.global_step > 0
                    and self.global_step % self.sft_cfg.ckpt_interval == 0
                )
                did_save_last_step = force_save or interval_save
                if did_save_last_step:
                    with Timer("save_checkpoint", all_timings):
                        ckpt_path = self.save_checkpoint()
                    log_dict["timing/save_checkpoint"] = all_timings["save_checkpoint"]
                    self._fire("on_save", ckpt_path=ckpt_path)

                # HF export at regular intervals
                if self.sft_cfg.hf_save_interval > 0 and self.global_step % self.sft_cfg.hf_save_interval == 0:
                    with Timer("save_hf_model", all_timings):
                        self.save_hf_model()
                    log_dict["timing/save_hf_model"] = all_timings["save_hf_model"]

                eval_metrics = None
                num_eval_batches: int | None = None
                # Eval fires at step N where N % eval_interval == 0 and N > 0, OR
                # whenever a callback set ``control.should_evaluate``.
                interval_eval = self.sft_cfg.eval_interval > 0 and self.global_step % self.sft_cfg.eval_interval == 0
                if self.eval_dataloaders is not None and (force_eval or interval_eval):
                    self._fire("on_eval_start")
                    with Timer("eval", all_timings):
                        eval_metrics, num_eval_batches = self.run_eval()
                    self._fire("on_eval_end", metrics=eval_metrics)
                    if eval_metrics:
                        log_dict.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                        log_dict["timing/eval"] = all_timings["eval"]

                log_dict.update({"train/epoch": current_epoch, "train/global_step": self.global_step})
                # Callbacks may mutate log_dict in place via on_log.
                self._fire("on_log", logs=log_dict)
                self.tracker.log(log_dict, step=self.global_step, commit=True)

                if self.global_step % 5 == 0:
                    logger.info(
                        f"Step {self.global_step}: loss={step_result['loss']:.4f}, "
                        f"grad_norm={step_result['grad_norm']}"
                    )

                if eval_metrics:
                    logger.info(
                        f"Step {self.global_step}: {_format_eval_metrics(eval_metrics)} "
                        f"over {num_eval_batches} batches"
                    )

                # Epoch boundaries are detected at the top of the loop when the
                # dataloader iterator is exhausted (StopIteration), not here.

                self.global_step += 1
        finally:
            if self._torch_profiler_enabled:
                self.dispatch.stop_profile("policy")
        self.global_step = min(self.global_step, num_steps)

        # Close the final epoch. The loop always exits with exactly one epoch
        # open (boundaries are detected lazily at the top of the loop and
        # immediately re-opened), so this is the single matching on_epoch_end
        # for the last on_epoch_start.
        self._fire("on_epoch_end")

        # Save final checkpoint (if checkpointing is enabled). Skip if the last
        # in-loop iteration already saved (either via ckpt_interval or via a
        # callback-driven force-save) so we don't double-save.
        if self.sft_cfg.ckpt_path and not did_save_last_step:
            final_step = num_steps
            logger.info(f"Saving final checkpoint at step {final_step}")
            ckpt_path = self.save_checkpoint()
            self._fire("on_save", ckpt_path=ckpt_path)

        # Save final HF model if enabled (only if not already saved at last step)
        if self.sft_cfg.hf_save_interval > 0:
            final_step = num_steps
            already_saved = final_step % self.sft_cfg.hf_save_interval == 0
            if not already_saved:
                self.global_step = final_step
                logger.info(f"Saving final HF model at step {final_step}")
                self.save_hf_model()

        # Final eval pass (skip if the last step already ran eval).
        # NOTE: The last in-loop tracker.log(..., commit=True) at step=num_steps
        # advanced wandb's internal step counter to num_steps+1. Logging the
        # final eval at step=num_steps would be rejected by wandb with
        # "step N < current step N+1". We log the final eval at num_steps+1
        # (one past the last committed train step) in a single combined
        # tracker.log() call, preserving wandb step ordering. We use a local
        # ``final_eval_step`` rather than mutating ``self.global_step``: the
        # bump is purely a wandb-step accounting concern, not real trainer
        # state.
        if self.eval_dataloaders is not None:
            already_ran = self.sft_cfg.eval_interval > 0 and num_steps % self.sft_cfg.eval_interval == 0
            if not already_ran:
                final_eval_step = num_steps + 1
                eval_timings: dict[str, float] = {}
                self._fire("on_eval_start")
                with Timer("eval", eval_timings):
                    eval_metrics, num_eval_batches = self.run_eval()
                self._fire("on_eval_end", metrics=eval_metrics)
                if eval_metrics:
                    eval_log = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    eval_log["timing/eval"] = eval_timings["eval"]
                    self._fire("on_log", logs=eval_log)
                    self.tracker.log(eval_log, step=final_eval_step, commit=True)
                    logger.info(
                        f"Final eval at step {final_eval_step}: {_format_eval_metrics(eval_metrics)} "
                        f"over {num_eval_batches} batches"
                    )

        self._fire("on_train_end")
        logger.info("SFT training complete!")

    def save_checkpoint(self) -> str:
        """Save a checkpoint at the given step. Returns the checkpoint folder path."""
        step = self.global_step
        global_step_folder = os.path.join(self.sft_cfg.ckpt_path, f"{GLOBAL_STEP_PREFIX}{step}")
        policy_save_dir = os.path.join(global_step_folder, "policy")
        io.makedirs(global_step_folder, exist_ok=True)
        logger.info(f"Saving checkpoint at step {step} to {global_step_folder}")
        self.dispatch.save_checkpoint("policy", policy_save_dir, self.tokenizer)

        # Save train dataloader state (sampler position) for resume.
        if self.train_dataloader is not None:
            dataloader_save_path = os.path.join(global_step_folder, "data.pt")
            try:
                with io.open_file(dataloader_save_path, "wb") as f:
                    torch.save(self.train_dataloader.state_dict(), f)
                logger.info(f"Saved dataloader state to {dataloader_save_path}")
            except Exception as e:
                logger.warning(f"Failed to save dataloader state: {e}")

        # Save trainer state for cross-validation on resume (mirrors PPO's trainer_state.pt)
        trainer_state = {
            "global_step": step,
            "config": asdict(self.sft_cfg),
        }
        trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
        with io.open_file(trainer_state_path, "wb") as f:
            torch.save(trainer_state, f)
        logger.info(f"Saved trainer state to {trainer_state_path}")

        # Atomic tracking -- write this last after all saves succeed
        latest_file = os.path.join(self.sft_cfg.ckpt_path, "latest_ckpt_global_step.txt")
        with io.open_file(latest_file, "w") as f:
            f.write(str(step))
        logger.info(f"Checkpoint saved for global_step_{step}")

        # Clean up old checkpoints after successful save
        cleanup_old_checkpoints(self.sft_cfg.ckpt_path, self.sft_cfg.max_ckpts_to_keep)
        return global_step_folder

    def save_hf_model(self):
        """Save policy weights in HuggingFace format.

        Export path: cfg.trainer.export_path/global_step_{step}/policy
        Mirrors the pattern used by the RL trainer's save_models().
        """
        step = self.global_step
        policy_export_dir = os.path.join(
            self.cfg.trainer.export_path,
            f"{GLOBAL_STEP_PREFIX}{step}",
            "policy",
        )
        self.dispatch.save_hf_model("policy", policy_export_dir, self.tokenizer)
        logger.info(f"Saved HF model weights at step {step} to {policy_export_dir}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def shutdown(self):
        """Finish tracking.

        Does NOT call ``ray.shutdown()`` -- when running inside a Ray task
        (the normal path via ``sft_entrypoint``), shutting down Ray from
        within the task would be incorrect.  The head-node process owns
        the Ray lifecycle.
        """
        if self._ray_gpu_monitor is not None:
            self._ray_gpu_monitor.stop()
        if self.tracker is not None:
            self.tracker.finish()
