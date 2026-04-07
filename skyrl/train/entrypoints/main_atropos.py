import asyncio
import os
import signal
import subprocess
import time
import sys
import argparse
import requests
import logging
from typing import Optional



from skyrl.train.entrypoints.main_base import BasePPOExp, main as base_main
from skyrl.train.integrations.atropos import AtroposSHMGenerator

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SkyRL-Atropos-Launcher: %(message)s'
)
logger = logging.getLogger(__name__)

class AtroposExp(BasePPOExp):
    """
    Atropos-specific experiment launcher that overrides the generator 
    to use the Zero-Copy SHM transport.
    """
    
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Overrides the base generator to return AtroposSHMGenerator.
        """
        shm_name = getattr(self, "shm_name", f"atropos_shm_{cfg.trainer.run_name}")
        return AtroposSHMGenerator(
            shm_name=shm_name,
            batch_size=cfg.trainer.train_batch_size,
            entry_size=512,
            poll_interval=0.001,
            timeout=300.0
        )

def run_joint_training(args, unknown_args):
    """
    Orchestrates joint Atropos (Reasoning) + SkyRL (PPO) training via SHM.
    """
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = f"{os.getcwd()}:{env_vars.get('PYTHONPATH', '')}"
    env_vars["VLLM_USE_V1"] = "0"
    
    shm_name = f"atropos_shm_{args.group}"

    # 1. Start vLLM API Server (Inference Backend on GPU 0)
    vllm_env = env_vars.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    # Use SkyRL Bridge instead of broken vLLM
    vllm_cmd = [
        sys.executable, "/root/atropos/example_trainer/skyrl_bridge_server.py", "--port", "9002"
    ]
    logger.info(f"Starting SkyRL-Native Bridge | GPU 0 | Logs: /tmp/skyrl_bridge_{args.group}.log")
    vllm_log = open(f"/tmp/skyrl_bridge_{args.group}.log", "w")
    vllm_proc = subprocess.Popen(vllm_cmd, env=vllm_env, stdout=vllm_log, stderr=subprocess.STDOUT)

    # Wait for vLLM Health
    logger.info("Waiting for vLLM Engine health check...")
    max_wait = 300
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get("http://localhost:9002/health", timeout=2)
            if resp.status_code == 200:
                logger.info("vLLM Engine is ready.")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        logger.error("vLLM Engine failed to start in time. Aborting.")
        vllm_proc.terminate()
        return

    # 2. Start Atropos Server (Reasoning Environment)
    env_vars_server = env_vars.copy()
    env_vars_server["CUDA_VISIBLE_DEVICES"] = "" # No GPU needed for coordination
    server_cmd = [
        sys.executable, "/root/atropos/environments/skyrl_server.py",
        "serve",
        "--env.batch_size", str(args.batch_size),
        "--env.shm_size", str(args.shm_size),
        "--env.shm_name", shm_name,
        "--env.transport", "SHM",
        "--env.use_wandb", "False",
        "--openai.model_name", "Qwen/Qwen2.5-1.5B-Instruct",
        "--openai.tokenizer_name", "Qwen/Qwen2.5-1.5B-Instruct",
        "--openai.server_type", "vllm",
        "--openai.base_url", "http://localhost:9002/v1",
    ]
    logger.info(f"Starting Atropos Environment | Logs: /tmp/atropos_server_{args.group}.log")
    server_log = open(f"/tmp/atropos_server_{args.group}.log", "w")
    server_proc = subprocess.Popen(server_cmd, env=env_vars_server, stdout=server_log, stderr=subprocess.STDOUT)
    
    # Wait for SHM segment to be initialized
    time.sleep(5)
    
    # 3. Start SkyRL Trainer (Training on GPU 1)
    logger.info("Starting SkyRL FullyAsync Trainer | GPU 1")
    trainer_env = env_vars.copy()
    trainer_env["CUDA_VISIBLE_DEVICES"] = "1"
    
    trainer_cmd = [
        sys.executable, "-m", "skyrl.train.entrypoints.main_atropos",
        "--generator",
        "--shm_name", shm_name,
        f"trainer.train_batch_size={args.batch_size}",
        f"trainer.policy_mini_batch_size={args.batch_size}",
        f"trainer.critic_mini_batch_size={args.batch_size}",
        "generator.inference_engine.backend=none",
        "generator.external_generation=True",
    ] + unknown_args
    
    try:
        subprocess.run(trainer_cmd, env=trainer_env, check=True)
    except KeyboardInterrupt:
        logger.info("Stopping processes...")
    finally:
        server_proc.terminate()
        vllm_proc.terminate()
        server_proc.wait()
        vllm_proc.wait()

if __name__ == "__main__":
    if "--generator" in sys.argv:
        # Trainer Mode: Run the actual SkyRL experiment using our custom class
        from skyrl.train.config import SkyRLTrainConfig
        from skyrl.train.utils import validate_cfg
        from skyrl.train.utils.utils import initialize_ray
        import ray
        
        # Capture our specific flags, let the rest be trainer config
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--generator", action="store_true")
        parser.add_argument("--shm_name", type=str)
        t_args, cfg_args = parser.parse_known_args()

        @ray.remote(num_cpus=1)
        def skyrl_atrop_entrypoint(cfg: SkyRLTrainConfig, shm_name: str):
            exp = AtroposExp(cfg)
            exp.shm_name = shm_name
            exp.run()

        cfg = SkyRLTrainConfig.from_cli_overrides(cfg_args)
        validate_cfg(cfg)
        initialize_ray(cfg)
        ray.get(skyrl_atrop_entrypoint.remote(cfg, t_args.shm_name))
    else:
        # Launcher Mode
        parser = argparse.ArgumentParser(description="Atropos-SkyRL SHM Launcher")
        parser.add_argument("--group", type=str, default="default")
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--shm_size", type=int, default=1000)
        args, unknown = parser.parse_known_args()
        
        # Filter unknown for trainer (OmegaConf style key=value only)
        trainer_overrides = [a for a in unknown if "=" in a]
        
        run_joint_training(args, trainer_overrides)
