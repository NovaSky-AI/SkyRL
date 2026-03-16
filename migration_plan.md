1. Stage 1: Support for basic full fine-tuning in the new stack (excludes fully async) (In Progress)  
   1. Cover all relevant unit and integration tests for full FT on the new stack. Specifically, need to migrate the following test: `test_inference_engine_client_http_endpoint.py`  
   2. Performance benchmarking for basic full FT for the new stack  
      1. Add basic timing breakdown in the new stack  
         1. `RemoteInferenceClient`: Add a `.get_timing_metrics()` for aggregate time spent in ser/ deser to router, overall time spent in vllm server calls, detokenization, etc.  
      2. Compare performance with the old stack for (1) Single engine, single turn (2) Single engine, multi-turn (3) Multi-engine, single turn (4) Multi-engine, multi-turn  
   3. E2E Tests:  
      1. GSM8K colocated  
      2. GSM8K non-colocated  
2. Stage 2: Support for fully async RL (full fine-tuning)
   1. Migrate to the new vLLM APIs (Done)
   2. Cover existing fully async RL unit tests (`test_pause_and_continue_generation.py`) (Done)
      - `test_continue_generation_generate_vllm_engine_generation` works with `_SKYRL_USE_NEW_INFERENCE=1` (retry-on-abort in `RemoteInferenceClient.generate()`)
      - Tests 1 (HTTP chat completions) and 3 (abort via `client.engines[0]`) have API divergence with new path
      - New path equivalents in `tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_pause_and_continue_generation.py`
   3. Performance benchmarking for fully async RL for the new stack  
      1. Single engine \+ single-turn and Multi-engine \+ multi-turn)  
   4. E2E Tests:  
      1. GSM8K Fully Async RL  
3. Stage 3: LoRA Support in the new stack  
   1. Add support for LoRA in the new stack using vLLM's native dynamic loading of LoRA adapters. Cover existing LoRA unit tests (`test_lora.py`)  
   2. Performance benchmarking for LoRA weight update for the new stack  
      1. Single engine, single turn  
   3. E2E Tests:  
      1. GSM8K LoRA  
4. Stage 4: Misc  
   1. Support for FlashRL  
   2. Better performance breakdown  
      1. Support `/metrics` endpoint in the router to aggregate metrics from vllm servers
