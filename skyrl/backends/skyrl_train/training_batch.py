"""Defines interfaces for training data."""

import io
import pickle
from typing import Any, Dict, Generic, List, Optional, TypedDict, TypeVar

import numpy as np
import torch
from jaxtyping import Float, Integer

DictType = TypeVar("DictType")


def _serialize_tensor(value: torch.Tensor) -> dict:
    """Serialize a single tensor for pickle protocol."""
    try:
        # Fast path: direct memory copy via numpy (works for most dtypes)
        arr = value.numpy()
        return {
            "format": "numpy",
            "data": arr.tobytes(),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }
    except TypeError:
        # Fallback for dtypes not supported by numpy (e.g., bfloat16)
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return {
            "format": "torch",
            "data": buffer.getvalue(),
        }


def _deserialize_tensor(value: dict) -> torch.Tensor:
    """Deserialize a single tensor from pickle format."""
    if value.get("format") == "torch":
        # Fallback path: torch.load for unsupported dtypes
        buffer = io.BytesIO(value["data"])
        return torch.load(buffer, weights_only=True)
    else:
        # Fast path: reconstruct from numpy bytes
        # Also handles legacy format without "format" key
        arr = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        arr = arr.reshape(value["shape"])
        return torch.from_numpy(arr.copy())


class TensorList:
    """A list of tensors with variable shapes, indexed by batch position.

    Each element can have a different shape (e.g., pixel_values[i] has shape
    [num_patches_i, dim]). Supports the same batch operations as tensors:
    slicing, chunking, concatenation, device transfer, and serialization.
    """

    def __init__(self, tensors: list[torch.Tensor]):
        if len(tensors) == 0:
            raise ValueError("Cannot create a TensorList with no tensors.")
        self.tensors = tensors
        expected_device = tensors[0].device
        for tensor in tensors:
            if tensor.device != expected_device:
                raise ValueError(
                    f"All tensors must be on the same device. Expected {expected_device}, got {tensor.device}"
                )

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TensorList(self.tensors[index])
        return self.tensors[index]

    def to(self, device=None, dtype=None, non_blocking=False):
        return TensorList([t.to(device=device, dtype=dtype, non_blocking=non_blocking) for t in self.tensors])

    def contiguous(self):
        return TensorList([t.contiguous() for t in self.tensors])

    @property
    def device(self):
        return self.tensors[0].device if self.tensors else None

    def repeat(self, repeats: int):
        return TensorList(self.tensors * repeats)

    def repeat_interleave(self, repeats: int):
        return TensorList([t for t in self.tensors for _ in range(repeats)])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorList):
            return False
        if len(self) != len(other):
            return False
        return all(torch.equal(a, b) for a, b in zip(self.tensors, other.tensors))

    @staticmethod
    def cat(lists: list["TensorList"]) -> "TensorList":
        return TensorList([t for tl in lists for t in tl.tensors])


def _rebuild_tensor_batch(cls, state: Dict[str, Any]):
    """Module-level helper for unpickling TensorBatch (must be importable by name)."""
    obj = dict.__new__(cls)
    obj.__setstate__(state)
    return obj


# NOTE (sumanthrh): This is inspired by `TensorDict` but is much simpler.
class TensorBatch(dict, Generic[DictType]):
    """Base class for training batches

    This defines a generic container for a batch of training data (inputs or outputs).
    Consists of a dictionary of tensors along with some metadata.
    """

    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = None
        self._device = None
        self._check_consistency()

    def select(self, keys: List[str], metadata_keys: Optional[List[str]] = None) -> "TensorBatch[DictType]":
        """Select a subset of the data batch.

        Args:
            keys: The keys to select
            metadata_keys: The metadata keys to select

        Returns:
            A new `TensorBatch` object with the selected keys and metadata
        """
        selected_batch_data = {}
        for key in keys:
            selected_batch_data[key] = self[key]
        selected_metadata = {}
        if metadata_keys is None:
            selected_metadata = self.metadata
        else:
            selected_metadata = {}
            for key in metadata_keys:
                selected_metadata[key] = self.metadata[key]
        new_batch = self.__class__(selected_batch_data)
        new_batch.metadata = selected_metadata
        return new_batch

    def _check_consistency(self):
        """Check consistency of all present fields"""
        keys = list(self.keys())
        if len(keys) == 0:
            return

        batch_size = len(self[keys[0]])
        self._batch_size = batch_size
        for key in keys:
            value = self[key]
            if value is None:
                continue
            if not isinstance(value, (torch.Tensor, TensorList)):
                raise ValueError(f"Field {key} must be a tensor or TensorList, got {type(value)}")
            self._device = value.device if self._device is None else self._device
            if len(value) != batch_size:
                raise ValueError(f"Batch size mismatch in {key}")
            if value.device != self._device:
                raise ValueError(f"Device mismatch in {key}. Expected {self._device}, got {value.device}")

    def __getitem__(self, index) -> "TensorBatch[DictType]":
        if isinstance(index, slice):
            return self.slice(index.start, index.stop, index.step)
        elif isinstance(index, int):
            return self.slice(index, index + 1)
        else:
            return super().__getitem__(index)

    def __setitem__(self, key: str, value: Optional[torch.Tensor | TensorList]) -> None:
        if value is None:
            super().__setitem__(key, value)
            return

        if not isinstance(value, (torch.Tensor, TensorList)):
            raise ValueError(f"Field {key} must be a tensor or TensorList, got {type(value)}")

        if hasattr(self, "_batch_size") and self._batch_size is not None and len(value) != self._batch_size:
            raise ValueError(f"Batch size mismatch in {key}. Expected size {self._batch_size}, got {len(value)}.")

        super().__setitem__(key, value)

        if hasattr(self, "_batch_size") and self._batch_size is None:
            self._batch_size = len(value)

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None, *, non_blocking: bool = False
    ) -> "TensorBatch":
        """Move tensors to device and/or cast to dtype.

        Args:
            device: The device to move the tensors to
            dtype: The dtype to cast the tensors to
            non_blocking: Whether the operation should be non-blocking
        """
        for key, value in self.items():
            if value is None:
                continue
            assert isinstance(
                value, (torch.Tensor, TensorList)
            ), f"Field {key} must be a tensor or TensorList, got {type(value)}"
            self[key] = value.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def contiguous(self) -> "TensorBatch":
        """Make the tensors contiguous"""
        for key, value in self.items():
            if value is None:
                continue
            assert isinstance(
                value, (torch.Tensor, TensorList)
            ), f"Field {key} must be a tensor or TensorList, got {type(value)}"
            self[key] = value.contiguous()
        return self

    @property
    def batch_size(self) -> int:
        """Batch size for the tensors"""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """Get the device for the tensors"""
        return self._device

    def __reduce__(self):
        """Override pickle reduce to avoid separately pickling raw dict items.

        The default dict-subclass pickle calls iter(self.items()) which pickles
        each tensor through torch's storage-level pickle. That path fails for
        dtypes backed by UntypedStorage (e.g. uint16). By returning only
        (__getstate__,) we force all tensor serialization through our custom
        numpy/torch.save paths which handle every dtype.
        """
        return (_rebuild_tensor_batch, (type(self), self.__getstate__()))

    def __getstate__(self):
        """Serialize the `TensorBatch` object for pickle protocol.

        Uses fast numpy-based serialization when possible, with fallback to torch.save
        for dtypes not supported by numpy (e.g., bfloat16).
        """
        self.contiguous()
        if self._device is not None:
            assert self._device == torch.device("cpu"), "Tensors must be on CPU before serialization"
        batch_dict = {}
        for key, value in self.items():
            if value is None:
                batch_dict[key] = None
            elif isinstance(value, TensorList):
                batch_dict[key] = {
                    "format": "tensor_list",
                    "items": [_serialize_tensor(t) for t in value.tensors],
                }
            else:
                batch_dict[key] = _serialize_tensor(value)

        return {
            "batch_dict": batch_dict,
            "batch_size": self._batch_size,
            "device": self._device,
            "metadata": self.metadata,
        }

    def __setstate__(self, state):
        """Deserialize the `TensorBatch` object and load it into memory.

        Handles both numpy-based format (fast path) and torch format (fallback for bfloat16 etc).
        """
        for key, value in state["batch_dict"].items():
            if value is None:
                self[key] = None
            elif value.get("format") == "tensor_list":
                self[key] = TensorList([_deserialize_tensor(item) for item in value["items"]])
            else:
                self[key] = _deserialize_tensor(value)

        self._batch_size = state["batch_size"]
        self._device = state["device"]
        self.metadata = state["metadata"]
        self._check_consistency()
        return self

    def repeat(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat` (and `numpy.tile`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            elif isinstance(value, TensorList):
                new_batch[key] = value.repeat(repeats)
            else:
                assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat(repeats)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def repeat_interleave(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat_interleave` (and `numpy.repeat`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            elif isinstance(value, TensorList):
                new_batch[key] = value.repeat_interleave(repeats)
            else:
                assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat_interleave(repeats)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def chunk(self, chunk_size: int) -> List["TensorBatch[DictType]"]:
        """Split into smaller chunks"""
        chunks = []
        for i in range(0, self.batch_size, chunk_size):
            chunk_data = {}
            for key, value in self.items():
                if value is not None:
                    if isinstance(value, (torch.Tensor, TensorList)):
                        chunk_data[key] = value[i : i + chunk_size]
                    else:
                        raise ValueError(f"Unsupported type {type(value)} for key {key}")
                else:
                    # `None` values are not chunked
                    chunk_data[key] = value
            chunk = self.__class__(chunk_data)
            chunk.metadata = self.metadata
            chunks.append(chunk)
        return chunks

    def slice(self, start: int, end: int, step: int = 1) -> "TensorBatch[DictType]":
        """Slice the data batch.

        Args:
            start: The start index
            end: The end index
            step: The step size

        Returns:
            A new `TensorBatch` object with the view of the specified slice.
        """
        slice_obj = slice(start, end, step)
        sliced_data = {}
        for key, value in self.items():
            if value is not None:
                if isinstance(value, (torch.Tensor, TensorList)):
                    sliced_data[key] = value[slice_obj]
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                # `None` values are not sliced
                sliced_data[key] = value
        sliced_batch = self.__class__(sliced_data)
        sliced_batch.metadata = self.metadata
        return sliced_batch

    def save(self, path: str):
        """Save the data to a pickle file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the data from a pickle file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def cat(cls, shards: List["TensorBatch[DictType]"]) -> "TensorBatch[DictType]":
        """Concatenate shards.

        Args:
            shards: The list of `TensorBatch` objects to cat

        Returns:
            A new `TensorBatch` object with the concatenated data
        """
        cat_data = {}
        assert len(shards) > 0, "Cannot cat an empty list of shards"
        for key, value in shards[0].items():
            if value is not None:
                if isinstance(value, TensorList):
                    cat_data[key] = TensorList.cat([shard[key] for shard in shards])
                elif isinstance(value, torch.Tensor):
                    cat_data[key] = torch.cat([shard[key] for shard in shards])
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                # `None` values are not cat'd
                cat_data[key] = value
        metadata = shards[0].metadata
        cat_batch = cls(cat_data)
        cat_batch.metadata = metadata
        return cat_batch

    def __len__(self) -> int:
        """Length of the batch.

        Note that this is the same as the batch size rather than the number of keys in the batch.
        """
        return self._batch_size

    def __eq__(self, other: Any) -> bool:
        """Check if two `TensorBatch` objects are equal"""
        if not isinstance(other, TensorBatch):
            return False
        if self.metadata != other.metadata:
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for k, v in self.items():
            if isinstance(v, torch.Tensor) and isinstance(other[k], torch.Tensor):
                if not torch.equal(v, other[k]):
                    return False
            elif v != other[k]:
                return False
        return True

    def __str__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return f"TensorBatch(batch_size={self.batch_size}, device={self.device}, metadata={self.metadata}), items={self.items()}"

    def __repr__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return self.__str__()


class TrainingInput(TypedDict, total=False):
    """Schema for training input batch"""

    sequences: Integer[torch.Tensor, "batch_size seq_len"]
    attention_mask: Integer[torch.Tensor, "batch_size seq_len"]
    loss_mask: Integer[torch.Tensor, "batch_size seq_len"]
    response_mask: Integer[torch.Tensor, "batch_size seq_len"]
    action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    base_action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    values: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    returns: Float[torch.Tensor, "batch_size seq_len"]
    advantages: Float[torch.Tensor, "batch_size seq_len"]
    kl: Float[torch.Tensor, "batch_size seq_len"]
    rewards: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    rollout_expert_indices: Optional[Integer[torch.Tensor, "batch_size seq_len layer_num topk"]]
    pixel_values: Optional[TensorList]  # list of [num_patches_i, dim]
    image_grid_thw: Optional[TensorList]  # list of [num_images_i, 3]


class TrainingInputBatch(TensorBatch[TrainingInput]):
    """Training input data"""

    pass


class TrainingOutputBatch(TensorBatch[Dict[str, torch.Tensor]]):
    """Training output data"""

    pass
