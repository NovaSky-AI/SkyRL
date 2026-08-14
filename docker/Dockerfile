# NOTE: the tag still says cu128 because Anyscale publishes no cu130 Ray image at
# 2.56.0 (cu130 tags start at Ray 2.57.0, and none of them are `slim`). The base
# tag only decides which CUDA the image ships preinstalled; every CUDA library we
# actually load at runtime comes from the pip stack, which is cu13 throughout, and
# the toolkit installed below repoints /usr/local/cuda at 13.0. Switch this to a
# cu130 tag whenever the Ray pin moves to 2.57.0.
FROM anyscale/ray:2.56.0-slim-py312-cu128

RUN sudo apt-get update -y && sudo apt-get install -y wget kmod libxml2 build-essential libnuma-dev

# the cuda compiler here is needed for deepspeed
RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run \
    && sudo sh cuda_13.0.2_580.95.05_linux.run --silent --toolkit && rm -rf cuda_13.0.2_580.95.05_linux.run

RUN curl -LsSf https://astral.sh/uv/0.9.4/install.sh | sh
RUN echo "export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook" >> /home/ray/.bashrc

RUN sudo apt-get update \
    && sudo apt-get install -y openssh-server iputils-ping net-tools iproute2 traceroute netcat \
    libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev tzdata \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt update && sudo apt install --fix-broken && sudo apt install -y default-jre-headless openjdk-8-jdk \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*
