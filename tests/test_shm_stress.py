#!/usr/bin/env python3
"""
Atropos-SkyRL SHM Integration Stress Test
==========================================
Multi-process test that exercises the zero-copy shared memory buffer
between an Atropos producer and a SkyRL consumer (AtroposSHMGenerator).

No vLLM, flash_attn, or model weights required.

Usage:
    python test_shm_stress.py
"""

import asyncio
import multiprocessing as mp
import os
import sys
import time
import logging
import statistics

# Ensure both repos are importable
sys.path.insert(0, "/root/SkyRL")
sys.path.insert(0, "/root/atropos")

from skyrl.train.integrations.atropos.utils import ZeroCopySHMBuffer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration

SHM_NAME = "atropos_shm_stress_test"
BUFFER_SLOTS = 1000
ENTRY_SIZE = 512          # Tokens per trajectory (smaller = faster for throughput test)
NUM_TRAJECTORIES = 50_000  # Total trajectories to push through
BATCH_SIZE = 64            # Consumer batch size


# Producer (simulates Atropos reasoning server)

def producer_process(shm_name: str, num_trajectories: int, ready_event, start_event):
    """Writes trajectories to SHM as fast as possible."""
    shm = ZeroCopySHMBuffer(
        name=shm_name, size=BUFFER_SLOTS, entry_size=ENTRY_SIZE, create=True
    )
    logger.info(f"[Producer] SHM buffer created: {shm_name} ({BUFFER_SLOTS} slots, {ENTRY_SIZE} tokens/slot)")

    # Signal consumer that SHM is ready
    ready_event.set()
    # Wait for synchronized start
    start_event.wait()

    # Pre-generate a dummy token sequence (avoids allocation in hot loop)
    dummy_tokens = list(range(1, ENTRY_SIZE + 1))

    written = 0
    overflow_count = 0
    t_start = time.perf_counter()

    for i in range(num_trajectories):
        instance_id = f"task_{i // 4}"  # 4 reps per task
        rep_id = i % 4
        score = float(i % 10) / 10.0

        # Spin-wait on overflow (consumer will drain)
        while not shm.write_trajectory(
            tokens=dummy_tokens,
            score=score,
            instance_id=instance_id,
            rep_id=rep_id,
            metadata={"step": i},
        ):
            overflow_count += 1
            time.sleep(0.0001)  # 100µs backoff

        written += 1

        if written % 10000 == 0:
            elapsed = time.perf_counter() - t_start
            rate = written / elapsed
            logger.info(f"[Producer] {written}/{num_trajectories} written ({rate:.0f} traj/s, {overflow_count} overflows)")

    elapsed = time.perf_counter() - t_start
    rate = written / elapsed
    logger.info(f"[Producer] DONE: {written} trajectories in {elapsed:.2f}s = {rate:.0f} traj/s")
    logger.info(f"[Producer] Overflows: {overflow_count}")

    # Don't unlink — consumer needs it
    shm.close(unlink=False)


# Consumer (simulates SkyRL trainer reading via direct SHM)

def consumer_process(shm_name: str, num_trajectories: int, ready_event, start_event, throughput_out):
    """Reads trajectories from SHM as fast as possible."""
    # Wait for producer to create the SHM
    ready_event.wait()

    shm = ZeroCopySHMBuffer(
        name=shm_name, size=BUFFER_SLOTS, entry_size=ENTRY_SIZE, create=False
    )
    logger.info(f"[Consumer] Attached to SHM: {shm_name}")

    # Signal synchronized start
    start_event.set()

    read_count = 0
    empty_polls = 0
    t_start = time.perf_counter()

    while read_count < num_trajectories:
        item = shm.read_next()
        if item is None:
            empty_polls += 1
            if empty_polls > 1_000_000:
                elapsed = time.perf_counter() - t_start
                if elapsed > 120:
                    logger.info(f"[Consumer] TIMEOUT after {elapsed:.0f}s with {read_count} reads")
                    break
            continue

        empty_polls = 0
        read_count += 1

        # Validate data integrity
        if read_count <= 5 or read_count % 10000 == 0:
            tokens = item["tokens"]
            assert len(tokens) == ENTRY_SIZE, f"Token length mismatch: {len(tokens)} != {ENTRY_SIZE}"
            assert tokens[0] == 1, f"First token mismatch: {tokens[0]}"
            assert tokens[-1] == ENTRY_SIZE, f"Last token mismatch: {tokens[-1]}"

        if read_count % 10000 == 0:
            elapsed = time.perf_counter() - t_start
            rate = read_count / elapsed
            logger.info(f"[Consumer] {read_count}/{num_trajectories} read ({rate:.0f} traj/s)")

    elapsed = time.perf_counter() - t_start
    rate = read_count / elapsed if elapsed > 0 else 0

    logger.info(f"[Consumer] DONE: {read_count} trajectories in {elapsed:.2f}s = {rate:.0f} traj/s")

    # Write throughput to shared value for parent
    throughput_out.value = rate

    # Cleanup
    shm.close(unlink=True)


# Main

def main():
    logger.info("Starting Atropos-SkyRL SHM Stress Test")
    logger.info(f"Buffer: {BUFFER_SLOTS} slots | Tokens/slot: {ENTRY_SIZE}")
    logger.info(f"Trajectories: {NUM_TRAJECTORIES:,} | Batch size: {BATCH_SIZE}")

    # Synchronization primitives
    ready_event = mp.Event()   # Producer signals SHM is created
    start_event = mp.Event()   # Consumer signals both can start
    throughput = mp.Value('d', 0.0)  # Shared value for consumer throughput

    # Launch processes
    prod = mp.Process(
        target=producer_process,
        args=(SHM_NAME, NUM_TRAJECTORIES, ready_event, start_event),
    )
    cons = mp.Process(
        target=consumer_process,
        args=(SHM_NAME, NUM_TRAJECTORIES, ready_event, start_event, throughput),
    )

    cons.start()
    prod.start()

    prod.join()
    cons.join()

    rate = throughput.value

    logger.info(f"STEADY-STATE THROUGHPUT: {rate:,.0f} trajectories/sec")
    logger.info(f"TARGET: 16,500 trajectories/sec")
    status = "PASS" if rate >= 16500 else "BELOW TARGET"
    logger.info(f"STATUS: {status}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
