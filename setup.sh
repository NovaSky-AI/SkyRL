# Initialize upstream git repo
git init
git remote add origin https://github.com/alphaXiv/SkyRL.git
git fetch origin daniel/overlong
git checkout -f -B daniel/overlong origin/daniel/overlong

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

# Install AWS
apt update
apt install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
aws --version
# Ideally I can do configuration here with env variables, not there yet

# Prepare dataset 
python examples/train/rlm/rlm_dataset.py

