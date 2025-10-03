import os
from loguru import logger
import torch
import torch.distributed


class Profiler:
    """
    A PyTorch profiler wrapper class for collecting performance metrics.
    """

    def __init__(self, config):
        """
        config contains:
        - enable: bool
        - ranks: list[int]
        - save_path: str
        """
        self.enable = config.enable
        if not config.enable:
            return
        self.config = config
        self.save_path = config.save_path
        self.ranks = config.ranks
        self.saved = False
        self.prof = None
        self.rank = torch.distributed.get_rank()
        if self.rank in self.ranks:
            logger.info(f"[Profiler] Profiler init for rank {self.rank}")

            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=1,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    def check(self):
        return self.prof is not None and self.enable

    def start(self):
        if self.check():
            logger.info(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            logger.info(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None and not self.saved:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            save_file_name = f"/prof_rank_{self.rank}.json"
            logger.info(f"[Profiler] Saving trace to {self.save_path + save_file_name}")
            self.prof.export_chrome_trace(self.save_path + save_file_name)
            self.enable = False
            self.saved = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        if self.check():
            logger.info(f"[Profiler] Trace stopped for rank {self.rank}")
            self.enable = False
