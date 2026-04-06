import sys
import os
from unittest.mock import MagicMock

# Mock skyrl_gym to avoid ModuleNotFoundError
sys.modules["skyrl_gym"] = MagicMock()
# Mock other generators that might be imported by skyrl.train.generators.__init__
sys.modules["skyrl.train.generators.skyrl_gym_generator"] = MagicMock()
sys.modules["skyrl.train.generators.verl_generator"] = MagicMock()

import asyncio
import logging
import uuid
from typing import List

# Now we can safely import from skyrl
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from atropos_shm_generator import AtroposSHMGenerator
from atropos_shm_utils import ZeroCopySHMBuffer
from skyrl.train.generators.base import GeneratorInput, TrajectoryID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_skyrl_generator_logic():
    """
    Verifies that the SkyRL AtroposSHMGenerator correctly consumes SHM data:
    1. Mock Atropos writing out-of-order data.
    2. Request a batch in a specific order.
    3. Verify the generator re-orders and stashes correctly.
    """
    shm_name = f"test_skyrl_{uuid.uuid4().hex[:8]}"
    
    # 1. Setup Provider (Atropos side)
    provider_shm = ZeroCopySHMBuffer(name=shm_name, size=10, create=True)
    
    # 2. Setup Consumer (SkyRL side)
    generator = AtroposSHMGenerator(shm_name=shm_name, shm_size=10)

    # 3. Write data OUT-OF-ORDER to SHM
    # We want Task B first, then Task A
    provider_shm.write_trajectory(tokens=[20, 21], score=0.9, instance_id="task_B", repetition_id=0)
    provider_shm.write_trajectory(tokens=[10, 11], score=0.8, instance_id="task_A", repetition_id=0)

    # 4. Request Batch: [Task A, Task B]
    input_batch: GeneratorInput = {
        "trajectory_ids": [
            TrajectoryID(instance_id="task_A", repetition_id=0),
            TrajectoryID(instance_id="task_B", repetition_id=0)
        ]
    }

    logger.info("Polling generator for batch [Task A, Task B]...")
    output = await generator.generate(input_batch)

    # 5. Verification
    # Task A should be at index 0, Task B at index 1
    assert output["rewards"][0] == 0.8
    assert output["response_ids"][0] == [10, 11]
    assert output["rewards"][1] == 0.9
    assert output["response_ids"][1] == [20, 21]
    
    logger.info("✅ SUCCESS: Generator correctly re-ordered trajectories using stash.")

    # 6. Verify Stash is empty
    assert len(generator.stash) == 0
    
    # 7. Cleanup
    provider_shm.close(unlink=True)

if __name__ == "__main__":
    asyncio.run(test_skyrl_generator_logic())
