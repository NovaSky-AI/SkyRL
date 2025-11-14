sudo apt-get update -y && sudo apt-get install -y wget kmod libxml2 build-essential libnuma-dev


echo "export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook" >> /root/.bashrc

sudo apt-get update \
  && sudo apt-get install -y openssh-server iputils-ping net-tools iproute2 traceroute netcat \
  libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev tzdata \
  && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*




sudo apt update && sudo apt install --fix-broken && sudo apt install -y default-jre-headless openjdk-8-jdk \
  && sudo apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*



# ---------- PyTorch + cuDNN + Transformer Engine ----------
# PyTorch + cuDNN + Transformer Engine
uv pip install --no-cache-dir "torch==2.7.1" "nvidia-cudnn-cu12>=9.3" && \
  CUDNN_PATH="$(python -c 'import inspect, nvidia.cudnn as c, os; print(os.path.dirname(inspect.getfile(c)))')" && \
  sudo mkdir -p /opt && sudo ln -sfn "$CUDNN_PATH" /opt/cudnn && \
  echo "/opt/cudnn/lib" | sudo tee /etc/ld.so.conf.d/cudnn.conf >/dev/null && sudo ldconfig




export CUDNN_PATH="/opt/cudnn"
export CPATH="${CUDNN_PATH}/include:${CPATH}"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH}"


uv pip install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]==2.5.0"
# uv pip install --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
# uv pip install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]"
# --------------------


