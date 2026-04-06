import asyncio
import os
import signal
import subprocess
import time
import sys
import argparse
from typing import Optional

from skyrl.train.entrypoints.main_base import BasePPOExp, main as base_main
from skyrl.train.integrations.atropos import AtroposSHMGenerator

class AtroposExp(BasePPOExp):
    """
    Atropos-specific experiment launcher that overrides the generator 
    to use the Zero-Copy SHM transport.
    """
    
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Overrides the base generator to return AtroposSHMGenerator.
        """
        # SHM parameters are pulled from the config or defaults
        shm_name = getattr(cfg.generator, "shm_name", f"atropos_shm_{self.cfg.trainer.tracking.group}")
        
        return AtroposSHMGenerator(
            shm_name=shm_name,
            batch_size=cfg.trainer.batch_size,
            poll_interval=0.001,
            timeout=300.0
        )

def run_joint_training(args):
    """
    Orchestrates joint Atropos (Reasoning) + SkyRL (PPO) training via SHM.
    """
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = f"{os.getcwd()}:{env_vars.get('PYTHONPATH', '')}"
    
    # 1. Start Atropos Server
    print("🚀 Starting Atropos Reasoning Server...")
    server_cmd = [
        "python3", "environments/skyrl_server.py",
        "--batch_size", str(args.batch_size),
        "--port", "8080"
    ]
    server_proc = subprocess.Popen(server_cmd, env=env_vars)
    
    # Wait for SHM segment to be initialized
    time.sleep(5)
    
    # 2. Start SkyRL Trainer
    # We now use the AtroposExp class directly instead of a sub-process
    print("🧠 Starting SkyRL FullyAsync Trainer (Native SHM Integration)...")
    
    # Mocking CLI args for SkyRLTrainConfig if needed, or passing them through
    # For now, we'll stick to the subprocess for the trainer to avoid Ray init conflicts 
    # in the same process as the server-driver, BUT we use the new generator.
    
    # Actually, the repo pattern is to use the launcher for the trainer.
    # To use AtroposExp, we need to call it via the standard skyrl entrypoint logic.
    
    trainer_cmd = [
        "python3", "-m", "skyrl.train.entrypoints.main_atropos",
        "--generator", "atropos_shm", # This flag is used by our specific entrypoint
        "--shm_name", f"atropos_shm_{args.group}",
        "--batch_size", str(args.batch_size),
    ]
    
    try:
        subprocess.run(trainer_cmd, env=env_vars, check=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    # If called without --generator, we are in "Launcher Mode" (starts server + child trainer)
    # If called with --generator, we are in "Trainer Mode" (the child process)
    
    if "--generator" in sys.argv:
        # Trainer Mode: Run the actual SkyRL experiment using our custom class
        from skyrl.train.config import SkyRLTrainConfig
        from skyrl.train.utils import validate_cfg
        from skyrl.train.utils.utils import initialize_ray
        import ray
        
        @ray.remote(num_cpus=1)
        def skyrl_atrop_entrypoint(cfg: SkyRLTrainConfig):
            exp = AtroposExp(cfg)
            exp.run()

        cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
        validate_cfg(cfg)
        initialize_ray(cfg)
        ray.get(skyrl_atrop_entrypoint.remote(cfg))
    else:
        # Launcher Mode
        parser = argparse.ArgumentParser(description="Atropos-SkyRL SHM Launcher")
        parser.add_argument("--group", type=str, default="default", help="WandB group name")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--micro_batch_size", type=int, default=4)
        run_joint_training(parser.parse_args())
