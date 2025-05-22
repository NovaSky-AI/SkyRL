# SkyRL: Installation

## Pre-requisites

> [!TIP]
> For an easy-to-use Dockerfile, see [Dockerfile.skyrl](./docker/Dockerfile.skyrl)


The main prerequisites are: 
- [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) (versions greater than 12.4 might also work)
- `build-essential`: This is needed for `torch-memory-saver`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation): We use the `uv` + `ray` integration to easily manage dependencies in multi-node training.
- `python` 3.12
- `ray` 2.43.0


Once installed, configure ray to use `uv` with 

```
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```


## Installation dry run

Execute the following command from the root project directory:

```bash
uv run --isolated --frozen python -c 'import ray; ray.init(); print("Success!")'
```

This will trigger a fresh environment build on your system. We use extras to isolate conflicting dependencies between tasks. We recommend performing the same dry run for the task of interest: 

### Installation dry run for Swebench

```bash
uv run --isolated --extra swebench --frozen python -c 'import ray; ray.init(); print("Success!")'
```

### Installation dry run for SQL


```bash
uv run --isolated --extra sql --frozen python -c 'import ray; ray.init(); print("Success!")'
```

## Common installation issues

1. "Failed to build `torch-memory-saver==0.0.5` .....  cannot find -lcuda: No such file or directory" 

With a CPU head node, you might encounter installation issues with `torch-memory-saver`. The main problem is that the CUDA binaries need to be found at `/usr/lib/` for the installation to be successful. To fix this, you need to install CUDA and make sure your CUDA libraries are linked in `/usr/lib`. For example, 

```bash
sudo ln -s /usr/local/cuda-12.4/compat/libcuda.so /usr/lib/libcuda.so
sudo ln -s /usr/local/cuda-12.4/compat/libcuda.so.1 /usr/lib/libcuda.so.1
```