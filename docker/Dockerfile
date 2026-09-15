FROM anyscale/ray:2.57.0-py312-cu130

RUN sudo apt-get update -y && sudo apt-get install -y wget kmod libxml2 build-essential libnuma-dev

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
