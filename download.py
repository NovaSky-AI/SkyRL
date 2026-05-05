from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FleetAI/fleet-cu-trajectories",
    repo_type="dataset",
    local_dir="./fleet-cu-trajectories"
)