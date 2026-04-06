import os
import signal
import subprocess
import time
import argparse
from typing import List

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
    print("🧠 Starting SkyRL FullyAsync Trainer (SHM Mode)...")
    trainer_cmd = [
        "python3", "-m", "skyrl.train.fully_async_trainer",
        "--generator", "atropos_shm",
        "--shm_name", f"atropos_shm_{args.group}",
        "--batch_size", str(args.batch_size),
        "--micro_batch_size", str(args.micro_batch_size),
    ]
    
    try:
        subprocess.run(trainer_cmd, env=env_vars, check=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atropos-SkyRL SHM Launcher")
    parser.add_argument("--group", type=str, default="default", help="WandB group name")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    
    run_joint_training(parser.parse_args())
