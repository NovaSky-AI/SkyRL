# Initialize upstream git repo
git init
git remote add origin https://github.com/alphaXiv/SkyRL.git
git fetch origin daniel/sub-rlm
git checkout -f -B daniel/sub-rlm origin/daniel/sub-rlm

# Set global git config 
git config --global user.name "Daniel Kim" && git config --global user.email "sox8502@gmail.com"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh 

# Install dependencies
uv sync --extra fsdp

# Activate virtual environment
source .venv/bin/activate

# Install libnuma-dev & tmux
apt update && apt-get install build-essential libnuma-dev && apt install -y tmux

# Save current directory
CWD=$(pwd)

# Move to home directory, install AWS CLI, then return to original directory
cd ~
apt update
apt install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip
./aws/install
aws --version
cd "$CWD"

# Ideally I can do configuration here with env variables, not there yet

# Prepare dataset 
python examples/train/rlm/rlm_dataset.py
python examples/train/rlm/multi_paper_dataset.py

