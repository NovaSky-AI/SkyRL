FROM anyscale/ray:2.44.0-slim-py312-cu124

RUN sudo apt-get update -y && sudo apt-get install -y wget kmod libxml2 build-essential libnuma-dev

# the cuda compiler here is needed for deepspeed
RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
RUN sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN echo "export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook" >> /home/ray/.bashrc
RUN sudo apt-get update \
    && sudo apt-get install -y openssh-server iputils-ping net-tools iproute2 traceroute netcat \
    libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev tzdata
RUN sudo apt update && sudo apt install --fix-broken && sudo apt install -y default-jre-headless openjdk-8-jdk
# NOTE: vllm installation in base environment is needed for uv + vLLM to work
RUN pip install vllm==0.8.5 
RUN pip install ray==2.44.0