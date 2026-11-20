"""Write a ``torch_dist`` checkpoint from the live tensors in a background thread.

With ``async_dist_ckpt_save``, the NVRx writer process gets a host copy of every
checkpoint tensor before it writes. The strategy here plans the same save, but its
async request reads the live model and optimizer tensors while it writes. PyTorch's
``FileSystemWriter`` copies each tensor to host memory when it writes it (one tensor
plus a 10 MB copy-ahead window), so the host never holds a copy of the whole
checkpoint. ``StreamingCheckpointWrite`` runs the request in a thread of the
training process. The caller must call ``finish`` before it changes a tensor in
the checkpoint.
"""

import threading
from typing import Optional

import torch
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.api import _wrap_exception


class _LiveTensorFileSystemWriter(FileSystemWriter):
    """``FileSystemWriter`` with the plan, write and finalize steps of the NVRx saver.

    ``save_state_dict_async_plan`` gives the plan to ``prepare_write_data``, and
    ``save_state_dict_async_finalize`` reads ``retrieve_write_results``. The write
    runs in between, in ``write_prepared_data``, and reads each tensor only when
    it writes it.
    """

    def __init__(self, checkpoint_dir):
        super().__init__(checkpoint_dir, thread_count=1)
        self._plan = None
        self._planner = None
        self._write_results = None

    def prepare_write_data(self, plan, planner) -> None:
        self._plan = plan
        self._planner = planner

    def write_prepared_data(self, device: Optional[int], ready: Optional[torch.cuda.Event]) -> None:
        """Write the planned items. A failure becomes the write result of this rank.

        ``save_state_dict_async_finalize`` gathers the results of all ranks and raises
        ``CheckpointException`` on every rank if one of them is an exception.
        """
        try:
            if device is not None:
                # A new thread uses device 0. The copies to host must run on the
                # streams of the device that holds the tensors.
                torch.cuda.set_device(device)
                ready.synchronize()
            self._write_results = self.write_data(self._plan, self._planner).wait()
        except BaseException as exc:
            self._write_results = _wrap_exception(exc)

    def retrieve_write_results(self):
        return self._write_results


class StreamingTorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """``torch_dist`` save strategy whose async request writes the live tensors.

    It uses the same planner and file writer as the synchronous ``torch_dist`` save,
    so the format is the same. The returned request has no ``preload_fn``, and its
    ``async_fn`` must run in the training process (see ``StreamingCheckpointWrite``).
    """

    def async_save(self, sharded_state_dict, checkpoint_dir) -> AsyncRequest:
        if self.thread_count != 1:
            # PyTorch's FileSystemWriter does not report errors of its extra writer
            # threads: https://github.com/pytorch/pytorch/issues/191391
            raise ValueError("StreamingTorchDistSaveShardedStrategy supports thread_count=1 only")
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        writer = _LiveTensorFileSystemWriter(checkpoint_dir)
        save_state_dict_ret = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            0,
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
        )

        # The write thread waits for the GPU work that is queued now, which
        # produced the values to save.
        device, ready = None, None
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            ready = torch.cuda.Event()
            ready.record()

        def finalize_fn():
            save_state_dict_async_finalize(*save_state_dict_ret)

        return AsyncRequest(writer.write_prepared_data, (device, ready), [finalize_fn], async_fn_kwargs={})


class StreamingCheckpointWrite:
    """Run the write of a streaming async request in a thread of this process.

    ``finish`` waits for the thread and then runs the finalize functions of the
    request. They are collective, so every rank must call ``finish`` at the same
    point.
    """

    def __init__(self, request: AsyncRequest):
        if request.preload_fn is not None:
            raise ValueError("A streaming checkpoint write must not stage tensors to host memory")
        self._request = request
        self._thread = threading.Thread(
            target=request.async_fn,
            args=request.async_fn_args,
            kwargs=request.async_fn_kwargs or {},
            name="skyrl-streaming-ckpt-write",
            daemon=True,
        )
        self._thread.start()

    def finish(self) -> None:
        self._thread.join()
        for finalize_fn in self._request.finalize_fns:
            finalize_fn()
