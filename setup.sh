# Set global git config 
git config --global user.name "Daniel Kim" && git config --global user.email "sox8502@gmail.com"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh 
# source ~/.bashrc

# Install dependencies
uv sync --extra fsdp --python 3.12

# Activate virtual environment
source .venv/bin/activate

# Install libnuma-dev & tmux
apt update && apt-get install build-essential libnuma-dev && apt install -y tmux

# Ideally I can do configuration here with env variables, not there yet

# Prepare dataset 
python examples/train/rlm/datasets/rlm_dataset_qasper.py
python examples/train/rlm/datasets/multi_paper_dataset.py
python examples/train/rlm/datasets/rlm_dataset_synthetic_multi.py

