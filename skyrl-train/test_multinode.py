#!/usr/bin/env python
"""Script to test multi-node NCCL communication with Ray

This script is useful to debug if multi-node communication works and if the right network interfaces (eg: RDMA) is being used.
"""
import os
import sys
import ray
import torch
import torch.distributed as dist
from skyrl_train.utils.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S
from ray.util.placement_group import placement_group
from omegaconf import OmegaConf
from loguru import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-nodes", type=int, default=4)
parser.add_argument("--master-port", type=int, default=12355)
args = parser.parse_args()


def log_versions(rank):
    logger.warning(
        f"{rank} Python version: {sys.version} | "
        f"PyTorch version: {torch.__version__} | "
        f"CUDA available: {torch.cuda.is_available()} | "
        f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'} | "
        f"Ray version: {ray.__version__}"
    )


@ray.remote(num_gpus=1)
class PyTorchDistActor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.node_ip = self.get_node_ip()
        self.world_size = world_size

        logger.warning(f"Rank {self.rank} initialized with: node_ip={self.node_ip}, world_size={self.world_size}")
        log_versions(rank)

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def set_master_node_addr(self, master_addr, master_port):
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        logger.warning(f"Rank {self.rank} set MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

    def run(self):
        import time
        
        logger.warning(f"Rank {self.rank} STARTING run() method")
        logger.warning(f"Rank {self.rank} MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        
        # Initialize the process group
        logger.warning(f"Rank {self.rank} BEFORE init_process_group")
        try:
            dist.init_process_group(backend="nccl", init_method="env://", world_size=self.world_size, rank=self.rank)
            logger.warning(f"Rank {self.rank} AFTER init_process_group - SUCCESS")
        except Exception as e:
            logger.error(f"Rank {self.rank} FAILED init_process_group: {e}")
            raise

        # Create a 1GB tensor
        # 1 GB = 1,073,741,824 bytes
        # float32 = 4 bytes, so we need 268,435,456 elements
        tensor_size = 2000  # ~1 GB
        
        logger.warning(f"Rank {self.rank} BEFORE tensor creation")
        if self.rank == 0:
            # Rank 0 creates and initializes the tensor
            tensor = torch.randn(tensor_size, dtype=torch.float32, device='cuda')
            logger.warning(f"Rank {self.rank} AFTER tensor creation - shape {tensor.shape}, size {tensor.element_size() * tensor.numel() / 1e9:.2f} GB")
            # Store the checksum for verification
            logger.warning(f"Rank {self.rank} BEFORE checksum calculation")
            checksum = tensor.sum().item()
            logger.warning(f"Rank {self.rank} AFTER checksum calculation - checksum: {checksum}")
        else:
            # Other ranks create an empty tensor to receive data
            tensor = torch.zeros(tensor_size, dtype=torch.float32, device='cuda')
            logger.warning(f"Rank {self.rank} AFTER empty tensor creation - shape {tensor.shape}")

        # Broadcast the tensor
        logger.warning(f"Rank {self.rank} BEFORE broadcast")
        start_time = time.time()
        try:
            dist.broadcast(tensor, src=0)
            end_time = time.time()
            logger.warning(f"Rank {self.rank} AFTER broadcast - SUCCESS")
        except Exception as e:
            logger.error(f"Rank {self.rank} FAILED broadcast: {e}")
            raise
        
        broadcast_time = end_time - start_time
        tensor_size_gb = tensor.element_size() * tensor.numel() / 1e9
        bandwidth_gbps = tensor_size_gb / broadcast_time
        
        if self.rank != 0:
            logger.warning(f"Rank {self.rank} BEFORE received checksum calculation")
            received_checksum = tensor.sum().item()
            logger.warning(f"Rank {self.rank} AFTER received checksum - checksum: {received_checksum}")
            logger.warning(f"Rank {self.rank} broadcast took {broadcast_time:.3f}s, bandwidth: {bandwidth_gbps:.2f} GB/s")
        else:
            logger.warning(f"Rank {self.rank} broadcast completed in {broadcast_time:.3f}s, bandwidth: {bandwidth_gbps:.2f} GB/s")

        logger.warning(f"Rank {self.rank} BEFORE barrier")
        dist.barrier()
        logger.warning(f"Rank {self.rank} AFTER barrier")
        
        # Clean up
        logger.warning(f"Rank {self.rank} BEFORE destroy_process_group")
        dist.destroy_process_group()
        logger.warning(f"Rank {self.rank} AFTER destroy_process_group")
        
        logger.warning(f"Rank {self.rank} COMPLETING run() method")
        return {
            "rank": self.rank,
            "broadcast_time_s": broadcast_time,
            "bandwidth_gbps": bandwidth_gbps,
            "tensor_size_gb": tensor_size_gb
        }


if __name__ == "__main__":
    logger.warning("=== MAIN: Starting test_multinode script ===")
    
    # Initialize Ray
    logger.warning("MAIN: Creating config")
    cfg = OmegaConf.create()
    cfg.generator = OmegaConf.create()
    cfg.generator.backend = "vllm"
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer = OmegaConf.create()
    cfg.trainer.strategy = "fsdp2"
    
    logger.warning("MAIN: Initializing Ray")
    initialize_ray(cfg)
    logger.warning("MAIN: Ray initialized successfully")

    total_ranks = args.num_nodes
    actors = []
    logger.warning(f"MAIN: Will create {total_ranks} actors")

    # Create placement group for distributed training
    logger.warning("MAIN: Creating placement group")
    pg = placement_group(bundles=[{"GPU": 1, "CPU": 1}] * total_ranks, strategy="STRICT_SPREAD")
    logger.warning("MAIN: Waiting for placement group to be ready")
    get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
    logger.warning("MAIN: Placement group ready")

    # Create actors
    logger.warning("MAIN: Creating actors")
    for rank in range(total_ranks):
        logger.warning(f"MAIN: Creating actor for rank {rank}")
        actor = PyTorchDistActor.options(placement_group=pg).remote(rank, total_ranks)
        actors.append(actor)
    logger.warning(f"MAIN: All {total_ranks} actors created")

    # set master node addr
    logger.warning("MAIN: Getting master node address")
    master_addr = ray.get(actors[0].get_node_ip.remote())
    master_port = args.master_port
    logger.warning(f"MAIN: Master address is {master_addr}:{master_port}")
    
    logger.warning("MAIN: Setting master address on all actors")
    ray.get([actor.set_master_node_addr.remote(master_addr, master_port) for actor in actors])
    logger.warning("MAIN: Master address set on all actors")

    # Run the distributed operation
    iteration = 0
    while True:
        import time
        iteration += 1
        logger.warning(f"MAIN: ========== Starting iteration {iteration} ==========")
        logger.warning(f"MAIN: Launching run() on all {len(actors)} actors")
        
        try:
            results = ray.get([actor.run.remote() for actor in actors])
            logger.warning(f"MAIN: All actors completed iteration {iteration}")
            
            print("\n" + "="*80)
            print(f"Broadcast Results Summary (Iteration {iteration}):")
            for result in results:
                print(f"  Rank {result['rank']}: {result['broadcast_time_s']:.3f}s, {result['bandwidth_gbps']:.2f} GB/s")
            avg_time = sum(r['broadcast_time_s'] for r in results) / len(results)
            avg_bandwidth = sum(r['bandwidth_gbps'] for r in results) / len(results)
            print(f"  Average: {avg_time:.3f}s, {avg_bandwidth:.2f} GB/s")
            print("="*80 + "\n")
        except Exception as e:
            logger.error(f"MAIN: Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        logger.warning(f"MAIN: Sleeping for 10 seconds before next iteration")
        time.sleep(10)