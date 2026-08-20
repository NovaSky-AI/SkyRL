# NOTE: this is the full (non-slim) cu130 Ray image. At 2.57.0 Anyscale's `slim`
# variants stop at cu129, so a genuinely-CUDA-13 base means giving up `slim`: 8.0 GB
# compressed against 3.2 GB for slim-cu128. What that buys is a base with no CUDA 12
# tree at all, so there is nothing for Transformer Engine's fused attention to trip
# over when it scans for libcudart -- rather than a cu128 base whose CUDA 12 tree is
# merely kept off the loader path by repointing /usr/local/cuda. Every CUDA library we
# actually load at runtime still comes from the pip stack, which is cu13 throughout.
#
# Unlike the slim images, this base's PATH does NOT include ~/.local/bin, which is
# where the uv installer below puts its binary -- hence the explicit ENV PATH.
FROM anyscale/ray:2.57.0-py312-cu130

RUN sudo apt-get update -y && sudo apt-get install -y wget kmod libxml2 build-essential libnuma-dev

# No CUDA toolkit install here: the cu130 base already ships one, including the nvcc
# that deepspeed needs. Verified in the base image:
#   $ docker run --rm anyscale/ray:2.57.0-py312-cu130 nvcc --version
#   Cuda compilation tools, release 13.0, V13.0.88
# The cu128-based predecessor of this file had to install the toolkit by runfile to get
# a CUDA 13 nvcc; doing that now would only layer a runfile install over the base's
# apt-managed toolkit in the same /usr/local/cuda-13.0 prefix, for ~4 GB and no gain.

RUN curl -LsSf https://astral.sh/uv/0.9.4/install.sh | sh
# The uv installer writes to ~/.local/bin, which this base (unlike the slim images)
# leaves off PATH. Without this, `uv` is not found either by the build steps below or
# by the CI entrypoints at runtime.
ENV PATH=/home/ray/.local/bin:${PATH}
RUN echo "export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook" >> /home/ray/.bashrc

RUN sudo apt-get update \
    && sudo apt-get install -y openssh-server iputils-ping net-tools iproute2 traceroute netcat \
    libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev tzdata \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt update && sudo apt install --fix-broken && sudo apt install -y default-jre-headless openjdk-8-jdk \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*
