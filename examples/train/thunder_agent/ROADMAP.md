# R2EGym 32B ThunderAgent Recipe - Implementation Roadmap

## 原则
- 每个 Phase 严格独立，Phase N 不依赖 Phase N+1 的代码
- 每个 Phase 结束后 git commit，确保可回滚
- 验收标准不冗余：每个检查项对应一个真实的运行时失败模式
- 所有测试使用目标 venv 做真实 import，不做 AST 静态分析

---

## Phase 1: Fix `ThunderAgentRemoteInferenceClient.pause()` signature

**状态**: ✅ DONE (commit `5043d403`)

**文件**: `examples/train/thunder_agent/skyrl_integration/remote_inference_client.py`

**改动**:
- 添加 `clear_cache: bool = False` 参数
- 默认模式 `ABORT` -> `KEEP`
- `super().pause()` 调用时透传 `clear_cache`

**验收标准**:
1. ✅ `python -c "from examples.train.thunder_agent.skyrl_integration.remote_inference_client import ThunderAgentRemoteInferenceClient; import inspect; sig = inspect.signature(ThunderAgentRemoteInferenceClient.pause); assert 'clear_cache' in sig.parameters"` 返回 0
2. ✅ `python -c "from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient, PauseMode; from examples.train.thunder_agent.skyrl_integration.remote_inference_client import ThunderAgentRemoteInferenceClient; import inspect; a = set(inspect.signature(RemoteInferenceClient.pause).parameters); b = set(inspect.signature(ThunderAgentRemoteInferenceClient.pause).parameters); assert a.issubset(b), f'missing: {a - b}'"` 返回 0
3. ✅ `PauseMode.ABORT` 不出现在 `ThunderAgentRemoteInferenceClient.pause` 的默认参数中

---

## Phase 2: Add `ThunderAgentHarborDataset`

**状态**: ✅ DONE (commit `f9c5658d`)

**文件**: `examples/train/thunder_agent/skyrl_integration/harbor_dataset.py` (新建)

**功能**:
- `max_tasks`: 限制加载任务数
- `MANIFEST.json` curated subset 支持
- sorted directory traversal（稳定顺序）
- path-based canonical UID（`task_path.resolve().as_posix()`）
- duplicate UID filtering

**验收标准**:
1. ✅ `python -c "from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset; ds = ThunderAgentHarborDataset(['/tmp'], max_tasks=5); len(ds) <= 5"` 返回 0
2. ✅ `python -c "... ds = ThunderAgentHarborDataset([...]); uids = [item['uid'] for item in ds]; assert len(uids) == len(set(uids))"` 返回 0
3. ✅ 同一数据集实例化两次，返回的 `uid` 列表完全一致（sorted traversal + stable UID）

---

## Phase 3: Add Harbor-specific config classes

**状态**: ✅ DONE (commit `e4bdd8a4`)

**文件**: `examples/train/thunder_agent/training_config.py` (修改)

**改动**:
- 新增 `ThunderAgentHarborGeneratorConfig(ThunderAgentGeneratorConfig)`，含 `rate_limit: RateLimiterConfig`
- 新增 `ThunderAgentHarborConfig(ThunderAgentConfig)`，含 `harbor_trial_config`、`max_train_tasks`、`max_eval_tasks`
- `generator` 字段类型改为 `ThunderAgentHarborGeneratorConfig`

**验收标准**:
1. ✅ `python -c "from examples.train.thunder_agent.training_config import ThunderAgentHarborConfig; cfg = ThunderAgentHarborConfig.from_cli_overrides([...]); print('OK')"` 返回 0
2. ✅ `cfg.harbor_trial_config['agent']['name'] == 'mini-swe-agent'`
3. ✅ `cfg.generator.rate_limit.enabled == True`
4. ✅ `cfg.max_train_tasks == 256`

---

## Phase 4: Add `ThunderAgentHarborGenerator`

**状态**: ✅ DONE (commit `ef061149`)

**文件**: `examples/train/thunder_agent/skyrl_integration/harbor_generator.py` (新建)

**功能**:
- proxy URL 检测（`inference_engine_client.proxy_url`）
- sampling params 注入（temperature、top_p、max_tokens、top_k、min_p、repetition_penalty）
- per-attempt fresh UUID routing IDs（`session_id` + `program_id`）
- `finally` block 中 best-effort program release
- hard-failure circuit breaker（`RewardFileNotFoundError`、`VerifierTimeoutError`）
- `rollout_logprobs` 收集（`collect_rollout_details` -> `completion_token_ids` + `logprobs`）
- 从 Harbor rollout details 提取 token IDs

**验收标准**:
1. ✅ `python -c "from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator; print('OK')"` 返回 0
2. ✅ `_attach_trial_routing_ids(config, session_id)` 正确注入 routing IDs
3. ✅ `_best_effort_release_program()` 在 `finally` 中被调用（代码审查）
4. ✅ `_apply_sampling_params_to_trial_config()` 正确处理 temperature/top_p/max_tokens/top_k
5. ✅ `generate()` 返回的 `GeneratorOutput` 包含 `rollout_logprobs`

---

## Phase 5: Add `main_harbor_thunder_agent.py` entrypoint

**状态**: ✅ DONE (commit `2bbcbd8b`)

**文件**: `examples/train/thunder_agent/main_harbor_thunder_agent.py` (新建)

**功能**:
- `HarborThunderAgentFullyAsyncExp(FullyAsyncThunderAgentExp)`
- `get_generator()` 返回 `ThunderAgentHarborGenerator`
- `get_train_dataset()` / `get_eval_dataset()` 返回 `ThunderAgentHarborDataset`
- Harbor default config deep merge

**验收标准**:
1. ✅ `python -c "from examples.train.thunder_agent.main_harbor_thunder_agent import HarborThunderAgentFullyAsyncExp; print('OK')"` 返回 0
2. ✅ `issubclass(HarborThunderAgentFullyAsyncExp, FullyAsyncThunderAgentExp)` 为 True
3. ✅ `get_generator()` 方法签名正确（参数: self, cfg, tokenizer, inference_engine_client）
4. ✅ `_deep_merge()` 递归合并工作正常
5. ✅ `HARBOR_DEFAULT_CONFIG.exists()` 为 True

---

## Phase 6: Add default harbor trial config YAML

**状态**: ✅ DONE (commit `8fc9f132`)

**文件**: `examples/train/thunder_agent/harbor_trial_config/default.yaml` (新建)

**内容**:
- agent: mini-swe-agent, hosted_vllm/{model_name}
- environment: docker, 2 CPU, 4GB mem/storage
- verifier: disable=false

**验收标准**:
1. ✅ `python -c "import yaml; data = yaml.safe_load(open('examples/train/thunder_agent/harbor_trial_config/default.yaml')); assert data['agent']['name'] == 'mini-swe-agent'; assert data['environment']['type'] == 'docker'"` 返回 0
2. ✅ YAML 语法合法，无解析错误

---

## Phase 7: Add `run_harbor_thunder_agent_32b.sh`

**状态**: ✅ DONE (commit `8fc9f132`)

**文件**: `examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh` (新建)

**参数对齐** `thunderagent_medium_hard_256_10epoch_no_preflight`:
- TRAIN_DATA=r2egym-train256-medium-hard-v1
- EVAL_DATA=r2egym-eval64-medium-hard-v1
- MAX_TRAIN_TASKS=256, MAX_EVAL_TASKS=64
- FULL_EPOCHS=10, EVAL_INTERVAL_STEPS=4
- USE_KL_LOSS=false, KL_LOSS_COEF=0.0
- ROLLOUT_ENGINES=4, ROLLOUT_TP_SIZE=2
- HARBOR_AGENT_MAX_TURNS=25
- AGENT_TIMEOUT_SEC=9000
- generator.sampling_params.logprobs=1
- generator.rate_limit.enabled=true

**验收标准**:
1. ✅ `bash -n examples/train/thunder_agent/run_harbor_thunder_agent_32b.sh` 返回 0
2. ✅ 脚本中显式设置 `_SKYRL_USE_NEW_INFERENCE=1`
3. ✅ 脚本中显式设置 `generator.inference_engine.external_server_urls`
4. ✅ 脚本中显式设置 `generator.inference_engine.thunder_agent_mode=tr`
5. ✅ `USE_KL_LOSS=false` 和 `KL_LOSS_COEF=0.0` 出现在脚本中

---

## Phase 8: Integration validation

**状态**: ✅ DONE

**验证内容**:
- 端到端 import chain
- Config override 全链路
- 无未定义符号

**验收标准**:
1. ✅ `python -c "from examples.train.thunder_agent.main_harbor_thunder_agent import HarborThunderAgentFullyAsyncExp; print('OK')"` 返回 0
2. ✅ `python -c "from examples.train.thunder_agent.training_config import ThunderAgentHarborConfig; cfg = ThunderAgentHarborConfig.from_cli_overrides([...]); print('OK')"` 返回 0
3. ✅ `python -c "from examples.train.thunder_agent.skyrl_integration.harbor_generator import ThunderAgentHarborGenerator; print('OK')"` 返回 0
4. ✅ `python -c "from examples.train.thunder_agent.skyrl_integration.harbor_dataset import ThunderAgentHarborDataset; print('OK')"` 返回 0
5. ✅ 所有改动文件都在 `examples/train/thunder_agent/` 目录内
6. ✅ 完整 import chain: config -> dataset -> generator -> remote client -> entrypoint 全部通过

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `skyrl_integration/remote_inference_client.py` | 修改 | Phase 1: pause() 签名修复 |
| `skyrl_integration/harbor_dataset.py` | 新建 | Phase 2: Harbor dataset wrapper |
| `training_config.py` | 修改 | Phase 3: Harbor config classes |
| `skyrl_integration/harbor_generator.py` | 新建 | Phase 4: Harbor generator |
| `main_harbor_thunder_agent.py` | 新建 | Phase 5: Entrypoint |
| `harbor_trial_config/default.yaml` | 新建 | Phase 6: Default trial config |
| `run_harbor_thunder_agent_32b.sh` | 新建 | Phase 7: Launch script |
| `ROADMAP.md` | 新建 | 本文件 |
