import asyncio
import pytest
from skyrl.train.generators.atropos_shm_generator import AtroposSHMGenerator
from skyrl.train.generators.atropos_shm_utils import ZeroCopySHMBuffer

@pytest.mark.asyncasync
async def test_generator_stashing():
    """
    Verifies that the AtroposSHMGenerator re-orders trajectories arriving from 
    different reasoners at different times.
    """
    shm_name = "test_gen_shm"
    batch_size = 2
    
    # Producer (Atropos)
    shm = ZeroCopySHMBuffer(name=shm_name, size=10, create=True)
    
    # Consumer (SkyRL)
    gen = AtroposSHMGenerator(shm_name=shm_name, batch_size=batch_size)
    
    # 1. Write OUT OF ORDER
    shm.write_trajectory(tokens=[1, 2], score=1.0, instance_id="i1", rep_id=1) # Rep 1 first
    shm.write_trajectory(tokens=[3, 4], score=0.5, instance_id="i1", rep_id=0) # Rep 0 second
    
    # 2. Consume
    batch = await gen.generate({})
    
    # 3. Verify Correct Ordering [0, 1]
    assert batch["response_ids"][0] == [3, 4]
    assert batch["response_ids"][1] == [1, 2]
    assert batch["rewards"][1] == 1.0
    
    shm.close(unlink=True)
    gen.close()

if __name__ == "__main__":
    asyncio.run(test_generator_stashing())
