"""
GPU integration tests for weight transfer strategies without involving the actual training workers and inference engines.

Tests the full execution flow with multiple training ranks:
    1. create_init_info: Extract config-derived args (master addr/port, dtype, etc.)
    2. create_sender/create_receiver: Initialize transfer components on both sides (multiple ranks)
    3. send_chunks/receive_weights: Transfer weight tensors between actors

This test uses 2 sender actors (simulating 2 training worker ranks) and 2 receiver actors
(simulating 2 inference engines, each with 1 worker) to test rank-specific logic.

GPU Requirements:
    - CUDA IPC test: 2 GPUs (each sender-receiver pair colocated on 1 GPU)
    - Broadcast test: 4 GPUs (each sender and receiver uses 1 GPU).

Run with:
    uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_transfer_strategy.py -v
"""

import asyncio
import ray
import torch
import torch.distributed as dist

from skyrl_train.config import SkyRLConfig
from skyrl_train.weight_sync import (
    WeightChunk,
    CudaIpcTransferStrategy,
    BroadcastTransferStrategy,
    WeightSyncInitInfo,
)
from skyrl_train.utils.utils import get_free_port, str_to_torch_dtype
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group


def make_cfg(
    weight_sync_backend: str,
    model_dtype: str,
    num_inference_engines: int,
    colocate_all: bool,
):
    """Create a config object.

    Assumes no intra-engine parallelism (tp=pp=dp=1).
    """
    cfg = SkyRLConfig()
    cfg.generator.weight_sync_backend = weight_sync_backend
    cfg.generator.model_dtype = model_dtype
    cfg.generator.num_inference_engines = num_inference_engines
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.trainer.placement.colocate_all = colocate_all
    return cfg


@ray.remote
class SenderActor:
    """Generic sender actor for transfer strategies."""

    def __init__(self, rank: int, world_size: int):
        """Initialize sender with its rank in the training worker group.

        Args:
            rank: Rank in the training worker group.
            world_size: Total number of training workers.
        """
        self.rank = rank
        self.world_size = world_size

    def get_master_addr_and_port(self):
        """Get the node IP address and a free port (only valid on rank 0)."""
        return ray._private.services.get_node_ip_address(), get_free_port()

    def init_process_group(self, master_addr: str, master_port: int):
        """Initialize the training worker process group.

        Args:
            master_addr: Master address for the process group.
            master_port: Port for the training worker process group.
        """
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=self.world_size,
                rank=self.rank,
            )

    def create_init_info(self, strategy_cls, cfg):
        """Create init info using the strategy (only rank 0)."""
        if self.rank == 0:
            return strategy_cls.create_init_info(cfg)
        return None

    def create_sender(self, strategy_cls, init_info, receiver_handles):
        """Create sender (must be called concurrently with receiver init for broadcast strategy)."""

        class MockInferenceClient:
            def __init__(self, receiver_handles):
                self.receiver_handles = receiver_handles

            async def update_named_weights(self, request):
                # Start receive_weights on all receivers and await completion
                # Ray ObjectRefs are awaitable, and the broadcast collective coordinates the communication
                await asyncio.gather(*[r.receive_weights.remote(request) for r in self.receiver_handles])

        mock_client = MockInferenceClient(receiver_handles)
        self._sender = strategy_cls.create_sender(init_info, mock_client)

    async def send_weights(
        self,
        init_info,
        names: list,
        shapes: list,
        send_individually: bool = False,
    ):
        """Send weights using the pre-created sender."""
        assert hasattr(self, "_sender"), "Sender not created. Call create_sender() first."
        dtype_str = init_info.model_dtype_str
        dtype = str_to_torch_dtype(dtype_str)

        # All ranks generate tensors because:
        # - Broadcast strategy: All ranks iterate through chunks (simulating FSDP collective ops
        #   during weight extraction), but only rank 0's tensor values are actually broadcast
        # - CUDA IPC strategy: All ranks create IPC handles from their tensors
        # Use same seed on all ranks so they generate identical tensors for easier testing
        torch.manual_seed(42)
        tensors = [torch.randn(shape, device="cuda", dtype=dtype) for shape in shapes]

        if send_individually:
            for name, tensor, shape in zip(names, tensors, shapes):
                chunk = WeightChunk(names=[name], dtypes=[dtype_str], shapes=[shape], tensors=[tensor])
                await self._sender.send_chunks([chunk])
        else:
            chunk = WeightChunk(names=names, dtypes=[dtype_str] * len(names), shapes=shapes, tensors=tensors)
            await self._sender.send_chunks([chunk])

        # Only rank 0 returns the tensors for comparison
        if self.rank == 0:
            return [t.cpu() for t in tensors]
        return None

    def teardown_sender(self):
        """Teardown the sender (tests cleanup logic)."""
        if hasattr(self, "_sender"):
            self._sender.teardown()


@ray.remote
class ReceiverActor:
    """Generic receiver actor for transfer strategies.

    Each actor represents a complete inference engine with 1 worker (no intra-engine parallelism).
    """

    def __init__(self, strategy_cls, init_info):
        """Initialize receiver for a single-worker inference engine.

        Args:
            strategy_cls: Transfer strategy class.
            init_info: Init info for creating the receiver.
        """
        self.strategy_cls = strategy_cls
        self.init_info = init_info
        self.receiver = None
        self.received_weights = []

    def init_process_group(self):
        """Initialize the engine's internal default process group.

        Gets a free port on this node and initializes the process group for a single-worker engine.
        """
        # Get a free port on this receiver's node
        master_port = get_free_port()

        # Initialize the engine's internal default process group (simulates TP/PP/DP group)
        # Even with 1 worker, this is needed for torch.distributed.get_rank() calls
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://localhost:{master_port}",
                world_size=1,
                rank=0,
            )

    def init_receiver(self):
        """Initialize the receiver (must be called before receive_weights).

        For broadcast strategy, this must be called concurrently with sender creation
        to avoid deadlock when joining the model_update_group (a named group).

        The model_update_group is separate from the engine's internal default group.
        """
        self.receiver = self.strategy_cls.create_receiver(self.init_info)

    def receive_weights(self, request):
        """Receive weights using the pre-created receiver."""
        assert self.receiver is not None, "Receiver not initialized. Call init_receiver() first."
        received = list(self.receiver.receive_weights(request))
        self.received_weights.extend([(name, tensor.cpu()) for name, tensor in received])

    def get_received_weights(self):
        return self.received_weights

    def teardown_receiver(self):
        """Teardown the receiver (tests cleanup logic)."""
        if self.receiver is not None:
            self.receiver.teardown()


def _run_weight_sync_e2e(
    strategy_cls,
    cfg,
    num_training_ranks: int,
    num_inference_engines: int,
    send_individually: bool,
    colocate: bool,
):
    """Run end-to-end weight sync test for a given strategy.

    Args:
        strategy_cls: Transfer strategy class to test.
        cfg: Mock config object.
        num_training_ranks: Number of training worker ranks to create.
        num_inference_engines: Number of inference engines to create (each with 1 worker).
        send_individually: Whether to send weights one at a time or batched.
        colocate: Whether to colocate sender and receiver on the same GPU (required for CUDA IPC).
    """

    # Simplifying assumption: each receiver represents one complete engine with 1 rank
    # (not testing intra-engine parallelism like TP/PP/DP)
    assert num_inference_engines == cfg.generator.num_inference_engines, "Test assumes 1 rank per engine"

    # For CUDA IPC, each sender-receiver pair must share the same GPU (required for IPC handles)
    pg = None
    if colocate:
        # Create placement group: one bundle per sender-receiver pair
        # Bundle i contains: training worker rank i (0.5 GPU) + inference engine i (0.5 GPU) = 1 GPU total
        pg = placement_group([{"GPU": 1}] * num_training_ranks, strategy="PACK")
        ray.get(pg.ready())

    # Create sender actors
    senders = []
    for i in range(num_training_ranks):
        if colocate:
            # CUDA IPC: sender and receiver share GPU (0.5 each)
            sender_options = {
                "num_cpus": 0,
                "num_gpus": 0.5,
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=i
                ),
            }
        else:
            # Broadcast: each actor on separate GPU for NCCL
            sender_options = {"num_gpus": 1}

        sender = SenderActor.options(**sender_options).remote(rank=i, world_size=num_training_ranks)
        senders.append(sender)

    # Get master_addr and master_port from rank 0 (its node IP and a free port on its node)
    master_addr, training_master_port = ray.get(senders[0].get_master_addr_and_port.remote())

    # Initialize process groups on all senders
    ray.get([sender.init_process_group.remote(master_addr, training_master_port) for sender in senders])

    # Only rank 0 creates init_info
    init_info: WeightSyncInitInfo = ray.get(senders[0].create_init_info.remote(strategy_cls, cfg))

    # Create receiver actors
    receivers = []
    for i in range(num_inference_engines):
        if colocate:
            # CUDA IPC: sender and receiver share GPU (0.5 each)
            receiver_options = {
                "num_cpus": 0,
                "num_gpus": 0.5,
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=i
                ),
            }
        else:
            # Broadcast: each actor on separate GPU for NCCL
            receiver_options = {"num_gpus": 1}

        # Use for_engine() to get per-engine init_info
        # Test assumes tp_size=pp_size=1 (single worker per engine)
        receiver_init_info = init_info.for_engine(engine_index=i, tp_size=1, pp_size=1)

        receiver = ReceiverActor.options(**receiver_options).remote(strategy_cls, receiver_init_info)
        receivers.append(receiver)

    # Initialize process groups on all receivers
    ray.get([receiver.init_process_group.remote() for receiver in receivers])

    # Create senders and receivers concurrently (required for broadcast strategy to avoid deadlock)
    # Both need to join the model_update_group at the same time
    init_tasks = [receiver.init_receiver.remote() for receiver in receivers]
    create_sender_tasks = [sender.create_sender.remote(strategy_cls, init_info, receivers) for sender in senders]
    ray.get(init_tasks + create_sender_tasks)

    names = ["layer1.weight", "layer1.bias", "layer2.weight"]
    shapes = [[32, 64], [64], [16, 16]]

    # Now send weights (senders and receivers are fully initialized)
    results = ray.get(
        [
            sender.send_weights.remote(init_info, names, shapes, send_individually=send_individually)
            for sender in senders
        ]
    )

    # Only rank 0 returns tensors
    src_tensors = results[0]
    assert src_tensors is not None
    assert results[1] is None  # Non-rank-0 training worker returns None

    # All receivers should have received the weights
    for receiver in receivers:
        received = ray.get(receiver.get_received_weights.remote())

        assert len(received) == len(names)
        for i, (name, tensor) in enumerate(received):
            assert name == names[i]
            assert tensor.shape == tuple(shapes[i])
            assert torch.allclose(tensor, src_tensors[i])

    # Test teardown on all actors (should not raise exceptions)
    teardown_tasks = []
    for sender in senders:
        teardown_tasks.append(sender.teardown_sender.remote())
    for receiver in receivers:
        teardown_tasks.append(receiver.teardown_receiver.remote())
    ray.get(teardown_tasks)  # Ensure all teardowns complete successfully


class TestCudaIpcTransferStrategy:
    """Integration tests for CUDA IPC transfer strategy.

    Tests weight synchronization using CUDA IPC handles with colocated sender-receiver pairs.
    Requires 2 GPUs (one GPU per sender-receiver pair).
    """

    def test_weight_sync_e2e(self, ray_init_fixture):
        """Test CUDA IPC strategy end-to-end with 2 training ranks and 2 inference engines."""
        cfg = make_cfg(
            weight_sync_backend="nccl",
            model_dtype="bfloat16",
            num_inference_engines=2,
            colocate_all=True,
        )
        _run_weight_sync_e2e(
            CudaIpcTransferStrategy,
            cfg,
            num_training_ranks=2,
            num_inference_engines=2,
            send_individually=False,
            colocate=True,
        )


class TestBroadcastTransferStrategy:
    """Integration tests for Broadcast transfer strategy.

    Tests weight synchronization using torch.distributed.broadcast with NCCL backend.
    Requires 4 GPUs.
    """

    def test_weight_sync_e2e(self, ray_init_fixture):
        """Test Broadcast strategy end-to-end with 2 training ranks and 2 inference engines."""
        cfg = make_cfg(
            weight_sync_backend="nccl",
            model_dtype="bfloat16",
            num_inference_engines=2,
            colocate_all=False,
        )
        _run_weight_sync_e2e(
            BroadcastTransferStrategy,
            cfg,
            num_training_ranks=2,
            num_inference_engines=2,
            send_individually=True,
            colocate=False,
        )
