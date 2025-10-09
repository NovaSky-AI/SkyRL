import modal
from pathlib import Path

app = modal.App("skyrl-app")

# Get the SkyRL repo root path
repo_path = Path(__file__).parent.parent.parent  # Goes up to SkyRL root

# This syncs your local code to /root/SkyRL in the container
image = (
    modal.Image.from_registry("novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8")
    .add_local_dir(
        local_path=str(repo_path),
        remote_path="/root/SkyRL",
        ignore=[".venv", "*.pyc", "__pycache__", ".git", "*.egg-info", ".pytest_cache"]
    )
)

# Create external volume for datasets

data_volume = modal.Volume.from_name("skyrl-data", create_if_missing=True)


@app.function(
    image=image, 
    gpu="L4:1",
    volumes={"/root/data": data_volume}
)
def run_script(command: str):
    """
    Run any command from the SkyRL repo.
    Example: run_script.remote("uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k")
    """
    import subprocess
    import os
    
    # Change to the repo directory
    os.chdir("/root/SkyRL/skyrl-train")
    
    print(f"Running command: {command}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run the command
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        raise Exception(f"Command failed with exit code {result.returncode}")
    
    if result.returncode != 0:
        print("=== ERROR DETAILS ===")
        print(f"Return code: {result.returncode}")
        print(f"Command: {command}")
        print(f"Working directory: {os.getcwd()}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise Exception(f"Command failed with exit code {result.returncode}")
    


@app.local_entrypoint()
def main(command: str = "nvidia-smi"):
    """
    Local entrypoint - runs on your Mac and calls the remote function.
    Usage: modal run main/run.py --command "uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
    """
    print(f"Submitting command to Modal: {command}")
    result = run_script.remote(command)
    print("\n=== Command completed successfully ===")
    return result
