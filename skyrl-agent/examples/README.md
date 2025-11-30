## Examples

This directory contains runnable examples for multiple tasks. Each section outlines how to set up any required services, prepare datasets, and launch training or inference.

### 1) SWE Training

- Setup remote runtime/server:
  - Refer to the [SkyRL-OpenHands](https://github.com/NovaSky-AI/SkyRL-OpenHands) documentation to set up a remote sandbox server and cache train/eval images.
  - After setup, configure your the remote server URL and API key in the environment file (i.e., `.env`).

- Prepare dataset:
  - Run the dataset preparation script:
    ```bash
    python ./data/swe_data.py --output SWE_DATA_PATH
    ```

- Launch training (modify the corresponding path in the script first):
  - VERL-based:
    ```bash
    bash ./examples/run_verl/verl_oh.sh
    ```
  - SkyRL-Train-based:
    ```bash
    bash ./examples/run_skyrl/skyrl_swe.sh
    ```

- Run inference (demo):
Launch an OpenAI API-compatible serving (e.g., vLLM or similar), then configure the `api_url` in the corresponding YAML (typically under the backend config) to point to your serving endpoint. Then run:
    ```bash
    python ./examples/run_openai/test_vllm_oh_demo.py
    ```

### 2) MemAgent Training

- Prepare dataset:
  ```bash
  python ./data/memagent.py --output-dir MEM_DATA_DIR
  ```

- Configure API:
  - Set the OpenAI API key in the environment file (i.e., `.env`); by default we use GPT-5-nano as the LLM judge for reward calculation.

- Backend note:
  - MemAgent currently supports only the Tinker backend for step-wise training. Set your Tinker API key in the environment file (i.e., `.env`).

- Launch training (modify the corresponding path in the script first):
  ```bash
  bash ./examples/run_tinker/tinker_memagent.sh
  ```

### 3) Deep Research (web_research_hle.sh)

- Quick setup: `uv venv && uv sync`.

- Required `.env`: `WANDB_API_KEY`, `GOOGLE_SEARCH_KEY` (Serper key), `JINA_API_KEYS`, `WEB_SUMMARY_API_BASE`, `WEB_SUMMARY_MODEL` (e.g., `Qwen/Qwen3-32B`), `SKYAGENT_WEB_CACHE_DIR`, `STEM_LLM_JUDGE_URL`; optional blocklists.

- Dataset:
  ```bash
  python ./data/deep_research.py --output-dir DR_DATA_DIR
  ```

- Web summary (required):
  - Point `WEB_SUMMARY_API_BASE` to your remote OpenAI-compatible endpoint (e.g., `http://host:port/v1`).
  - Keep the model name in `WEB_SUMMARY_MODEL`.
- Optional router (for load-balancing/failover):
  ```bash
  SUMMARY_UPSTREAMS=http://host:port/v1 \
  SUMMARY_MODEL=Qwen/Qwen3-32B \
  PORT=8080 \
  bash services/run_router.sh
  ```
  then set `WEB_SUMMARY_API_BASE=http://<router-host>:8080/v1`.

- Optional: `TRAIN_OUTPUT_DIR`, `ROLLOUT_DIR`, `VAL_ROLLOUT_DIR` for storage paths.

- Run:
  ```bash
  bash ./examples/run_verl/web_research_hle.sh
  ```

### 4) OSWorld

Placeholder for now. 

### 5) BrowseComp-Plus (Dense Retrieval)

- Prepare dataset/index. First download the decrypted dataset following [official instruction](https://github.com/texttron/BrowseComp-Plus?tab=readme-ov-file#-downloading-the-dataset). Then run:
  ```bash
  python ./data/browsecomp-plus.py --input DECRYPTED_JSON_PATH --output BC_DATA
  ```
- Download Pre-built Index for `Qwen/Qwen3-Embedding-8B`:
  ```bash
  huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir FAISS_INDEX_PATH
  ```

- Serve embedding model:
  - Start an OpenAI-compatible embedding server using `Qwen/Qwen3-Embedding-8B` as the embedding model. For example:
    ```bash
    vllm serve Qwen/Qwen3-Embedding-8B \
    --port 8000 \
    --task embed \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --dtype float16
    ```
  - Configure your `.env` with:
    - `FAISS_EMBEDDING_API_URL` (embedding server base URL)
    - `FAISS_EMBEDDING_MODEL_NAME` (e.g., `Qwen/Qwen3-Embedding-8B` model name used by your server)
    - `FAISS_INDEX_PATH` (file path to the downloaded pre-built index)

- Launch eval (modify the corresponding path in the script first):
  ```bash
  bash ./examples/run_verl/verl_browsecomp.sh
  ```
