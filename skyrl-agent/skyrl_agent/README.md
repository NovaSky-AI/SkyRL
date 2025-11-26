## SkyRL-Agent (package)

SkyRL-Agent is a modular agent layer for training and evaluating AI agents across diverse tasks and backends. It provides:

- Agent runners with configurable trajectories and dispatchers
- Pluggable inference backends (OpenAI-compatible HTTP, SkyRL-Train, VERL, Tinker)
- Task abstractions (e.g., SWE-bench, web research) with tool integrations (bash, editor, browser, finish, search, etc.)
- YAML-driven configuration for reproducible experiments

This README documents the `skyrl_agent` Python package located here. For the broader monorepo, see the root `README.md`.


### Requirements
- Python 3.12 (enforced by the repo’s `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for dependency management


### Install
Install from the monorepo root to ensure local path dependencies (e.g., OpenHands, VERL) are wired correctly:

```bash
cd /home/ray/default/sa-new
uv venv
uv sync
```

Optional extras (install one or more as needed):

```bash
# VERL integration (+ vLLM pin)
uv pip install -e ".[verl]"

# SkyRL-Train integration
uv pip install -e ".[skyrl-train]"

# Tinker integration
uv pip install -e ".[tinker]"
```


### Repository layout (package)
- `skyrl_agent/agents/`: Agent abstractions and runners (`base.py`), mappings (`mapping.py`), implementations (ReAct, OpenHands CodeAct).
- `skyrl_agent/config/`: Utilities for loading/validating task YAMLs.
- `skyrl_agent/dispatcher/`: Async dispatchers for parallel agent execution.
- `skyrl_agent/functional/`: Chat templates, function-calling, histories, utilities.
- `skyrl_agent/integrations/`: Inference/training backends (`openai`, `skyrl_train`, `verl`, `tinker`).
- `skyrl_agent/tasks/`: Task definitions and verifiers (SWE-bench, web research, math, etc.).
- `skyrl_agent/tools/`: Tool implementations (bash, browser, search, editor, finish, etc.).
- `skyrl_agent/auto.py`: `AutoAgentRunner` convenience entrypoint.


### Quickstart (OpenAI-compatible server)
Run any OpenAI-compatible server (e.g., vLLM OpenAI API). Example:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
```

Then run a minimal agent program using an example task YAML. The OpenAI backend asserts `OPENAI_API_KEY` is present (set a dummy value if using a local server):

```bash
export OPENAI_API_KEY=dummy
python - <<'PY'
import asyncio
from transformers import AutoTokenizer
from skyrl_agent import AutoAgentRunner
from skyrl_agent.integrations.openai import OpenAIBackend

# Model and API server should match your serving setup
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
api_url = "http://127.0.0.1:8000"

tokenizer = AutoTokenizer.from_pretrained(model_name)
infer_engine = OpenAIBackend(
    infer_engine=None,
    cfg={"model_name": model_name, "api_url": api_url}
)

task_yaml = "/home/ray/default/sa-new/examples/run_skyrl/skyrl_oh.yaml"
runner = AutoAgentRunner.from_task(task_yaml, infer_engine, tokenizer)

async def main():
    # Input batch is task-dependent. A single minimal instance example:
    batch = [{"instance": {"prompt": "Hello!"}}]
    result = await runner.run(batch, val_mode=False)
    print(result)

asyncio.run(main())
PY
```


### Using YAML task configs
Tasks and agent behavior are configured via YAML. See:
- `/home/ray/default/sa-new/examples/run_skyrl/skyrl_oh.yaml` (OpenHands CodeAct + SWE-bench task)
- `/home/ray/default/sa-new/examples/run_skyrl/skyrl_web_research_hle.yaml`
- `/home/ray/default/sa-new/examples/run_verl/*.yaml`
- `/home/ray/default/sa-new/examples/run_tinker/*.yaml`

Key knobs inside YAML (illustrative):
- `agent_cls`: Agent class path (e.g., `skyrl_agent.agents.oh_codeact.OHCodeActAgent`)
- `task`: Task class path (e.g., `skyrl_agent.tasks.swebench.utils.SWEBenchTask`)
- `tools.*`: Tool enabling switches
- `generator.*`: Inference backend (`infer_backend`), sampling params, iteration limits, thinking flags, etc.
- `dispatcher.*`: Parallelism controls

The `AutoAgentRunner.from_task(path, infer_engine, tokenizer)` will:
1) Load the YAML
2) Map `agent_cls` to a trajectory runner
3) Use the provided backend and tokenizer to execute trajectories


### Examples
Ready-to-run scripts and configs live under:
- `/home/ray/default/sa-new/examples/run_skyrl/` (SkyRL-Train flow, OpenHands-agent task YAMLs)
- `/home/ray/default/sa-new/examples/run_verl/` (VERL PPO trainer + agent tasks)
- `/home/ray/default/sa-new/examples/run_tinker/` (Tinker integration)
- `/home/ray/default/sa-new/examples/run_openai/` (simple OpenAI-style ReAct)

You can launch the SkyRL-Train workflow with the provided script (edit paths/models to your environment):

```bash
cd /home/ray/default/sa-new
bash ./examples/run_skyrl/skyrl_oh.sh
```

This invokes:

```bash
uv run --directory . --frozen --env-file .env skyrl_agent.integrations.skyrl_train.skyrl_train_main \
  +generator.task="./examples/run_skyrl/skyrl_oh.yaml" \
  # ... many hydra-style overrides for placement, sampling, batch sizes, saving, etc.
```


### Notes and tips
- Python 3.12 is required by the repo’s `pyproject.toml`.
- For OpenAI-compatible HTTP usage with local servers, set `OPENAI_API_KEY` to any non-empty value.
- Tooling like SandboxFusion may be required for certain tasks (e.g., code execution). Follow their upstream setup instructions.
- For VERL training/eval, prefer the `verl` extra and consult `/home/ray/default/sa-new/examples/run_verl/` plus the VERL docs in `/home/ray/default/sa-new/verl/`.
- For Tinker, see `/home/ray/default/sa-new/skyrl_agent/integrations/tinker/README.md` and `/home/ray/default/sa-new/examples/run_tinker/`.


### License
See the repository’s root license file and any submodule licenses (e.g., OpenHands, VERL).



