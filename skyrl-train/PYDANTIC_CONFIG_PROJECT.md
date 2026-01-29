# SkyRL Pydantic Configuration System

**Status:** 🚧 In Progress
**Started:** 2026-01-25
**Goal:** Modernize SkyRL's configuration system with type-safe Pydantic models while maintaining full backward compatibility

---

## 🎯 Project Goals

### Primary Objectives

1. **Type-Safe Configuration**: Replace untyped YAML-only configs with Pydantic models providing:
   - Full type checking at development time
   - IDE autocomplete and IntelliSense support
   - Runtime validation of configuration values
   - Clear documentation through type annotations

2. **Python-First Development**: Enable users to write training scripts in pure Python:
   - Build configs programmatically with full type safety
   - Compose configurations from reusable components
   - Eliminate string-based CLI overrides in favor of Python code
   - **Northstar**: Convert all `.sh` files in `examples/` to `.py` files

3. **Backward Compatibility**: Maintain 100% compatibility with existing workflows:
   - All existing `.sh` scripts continue to work unchanged
   - YAML + Hydra CLI overrides still supported
   - No breaking changes to existing user code
   - Seamless migration path from YAML to Python

4. **Unified API**: Single entry point (`run_training()`) that accepts both:
   - `DictConfig` from YAML/Hydra (existing)
   - `SkyRLConfig` from Pydantic (new)

### Future Goals

- [ ] Add CLI override support for Python-based configs
- [ ] Migrate validation logic from `validate_cfg()` to Pydantic validators
- [ ] Create config builder utilities for common use cases
- [ ] Add config serialization/deserialization helpers
- [ ] Documentation and migration guides

---

## 📊 Current Progress

### ✅ Completed

#### 1. Pydantic Models (`skyrl_train/config/configs.py`)
- [x] Complete 1:1 mapping to `ppo_base_config.yaml` structure
- [x] 30+ Pydantic model classes with full type annotations
- [x] Default values matching YAML defaults
- [x] Nested configuration hierarchy:
  - `SkyRLConfig` (root)
  - `DataConfig`, `TrainerConfig`, `GeneratorConfig`, `EnvironmentConfig`
  - `PolicyConfig`, `RefConfig`, `CriticConfig`
  - `AlgorithmConfig`, `PlacementConfig`, `OptimizerConfig`
  - `LoRAConfig`, `FSDPConfig`, `SamplingParamsConfig`
  - Megatron-specific configs
  - And more...

#### 2. Helper Functions
- [x] `dictconfig_to_pydantic()` - Convert Hydra DictConfig → Pydantic
- [x] `pydantic_to_dictconfig()` - Convert Pydantic → DictConfig
- [x] `create_default_config()` - Generate default config programmatically
- [x] `set_nested_attr()` / `get_nested_attr()` - Nested field access
- [x] `load_config_from_yaml()` - Load YAML without decorators

#### 3. Unified Entry Point (`main_base.py`)
- [x] `run_training(cfg: Union[DictConfig, SkyRLConfig])` - Accepts both types
- [x] Automatic type detection and conversion
- [x] Backward compatible `@hydra.main` decorator preserved
- [x] Removed redundant `main_with_pydantic()` function

#### 4. Python Examples
- [x] `examples/gsm8k/run_gsm8k.py` - Production-ready Python config
- [x] `examples/gsm8k/test_run_gsm8k.py` - Minimal test config
- [x] Compositional config building (RLlib-style)
- [x] Environment variable support for overrides

#### 5. Testing & Validation
- [x] Unit tests for config creation and conversion
- [x] Integration test: Full training run with Pydantic config
- [x] Backward compatibility tests: YAML configs still work
- [x] Roundtrip conversion tests (no data loss)

#### 6. Dependencies
- [x] Added `pydantic>=2.0.0` to `pyproject.toml`

---

## 🏗️ Architecture

### Configuration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Sources                     │
├──────────────────────────┬──────────────────────────────────┤
│   YAML + Hydra CLI       │     Python + Pydantic            │
│   (Existing)             │     (New)                        │
│                          │                                  │
│  ppo_base_config.yaml    │  from skyrl_train.config import  │
│  + CLI overrides         │      SkyRLConfig                 │
│                          │                                  │
│  ↓                       │  cfg = SkyRLConfig(...)          │
│  DictConfig              │  ↓                               │
│                          │  SkyRLConfig                     │
└──────────────────────────┴──────────────────────────────────┘
                          │
                          ↓
              ┌───────────────────────┐
              │   run_training()      │
              │  (Unified Entry)      │
              └───────────────────────┘
                          │
                          ↓
              ┌───────────────────────┐
              │   Normalize to        │
              │   DictConfig          │
              └───────────────────────┘
                          │
                          ↓
              ┌───────────────────────┐
              │   validate_cfg()      │
              └───────────────────────┘
                          │
                          ↓
              ┌───────────────────────┐
              │   Training Pipeline   │
              └───────────────────────┘
```

### File Structure

```
skyrl-train/
├── skyrl_train/
│   ├── config/
│   │   ├── configs.py              # ✨ NEW: Pydantic models
│   │   ├── ppo_base_config.yaml    # Existing YAML config
│   │   └── ...
│   └── entrypoints/
│       └── main_base.py            # ✨ UPDATED: run_training()
└── examples/
    └── gsm8k/
        ├── run_gsm8k.sh            # Existing YAML-based
        ├── run_gsm8k.py            # ✨ NEW: Python-based
        └── test_run_gsm8k.py       # ✨ NEW: Test script
```

---

## 📖 Usage Examples

### Python-Based Configuration (New)

```python
from skyrl_train.config.configs import (
    SkyRLConfig, DataConfig, TrainerConfig, GeneratorConfig,
    PolicyConfig, ModelConfig, OptimizerConfig
)
from skyrl_train.entrypoints.main_base import run_training

# Build config compositionally
cfg = SkyRLConfig(
    data=DataConfig(
        train_data=["~/data/gsm8k/train.parquet"],
        val_data=["~/data/gsm8k/validation.parquet"],
    ),
    trainer=TrainerConfig(
        policy=PolicyConfig(
            model=ModelConfig(path="Qwen/Qwen2.5-1.5B"),
            optimizer_config=OptimizerConfig(lr=1e-6),
        ),
        epochs=20,
        # ... other params with full type safety!
    ),
    # ... generator, environment configs
)

# Run training
run_training(cfg)
```

### YAML-Based Configuration (Existing - Still Works!)

```bash
# All existing scripts work unchanged
bash examples/gsm8k/run_gsm8k.sh

# Or with Hydra overrides
uv run -m skyrl_train.entrypoints.main_base \
    trainer.epochs=30 \
    trainer.policy.model.path="Qwen/Qwen2.5-7B"
```

### Hybrid Approach

```python
from skyrl_train.config.configs import load_config_from_yaml

# Load base config from YAML
cfg = load_config_from_yaml(
    "config/ppo_base_config.yaml",
    overrides=["trainer.epochs=10"]
)

# Modify with type-safe Python
cfg.trainer.policy.model.path = "custom/model"
cfg.trainer.algorithm.advantage_estimator = "gae"

# Run
run_training(cfg)
```

---

## 🚧 Known Limitations & TODOs

### Current Limitations

1. **No CLI overrides for Python configs**: When using pure Python configs, you can't override via CLI like Hydra does
   - **Workaround**: Use environment variables or modify Python code
   - **Planned**: Add argparse-based override system

2. **Validation still in `validate_cfg()`**: Business logic validation hasn't been migrated to Pydantic
   - **Status**: Intentional - keeping existing validation for now
   - **TODO**: Gradually migrate to Pydantic validators

3. **No config inheritance/composition**: Can't easily extend base configs
   - **Planned**: Add config builder utilities and presets

### Upcoming Work

- [ ] **CLI Override Support**: Add ability to override Pydantic configs via CLI
  ```bash
  python examples/gsm8k/run_gsm8k.py --trainer.epochs=30
  ```

- [ ] **Validation Migration**: Move validation from `validate_cfg()` to Pydantic
  - Batch size relationships
  - GPU count validation
  - Path existence checks
  - Cross-field dependencies

- [ ] **Documentation**:
  - Migration guide from YAML to Python
  - Best practices for config composition
  - API reference for all config classes

- [ ] **Config Presets**: Pre-built configs for common scenarios
  ```python
  from skyrl_train.config.presets import gsm8k_1_5b, sql_7b
  cfg = gsm8k_1_5b()  # Returns pre-configured SkyRLConfig
  ```

- [ ] **Convert More Examples**: Update remaining `.sh` files to `.py`

---

## 🧪 Testing

### Test Coverage

✅ **Unit Tests**
- Config creation and modification
- Type validation
- DictConfig ↔ Pydantic conversion
- Nested field access

✅ **Integration Tests**
- Full training run with Pydantic config (GSM8K, 4 GPUs)
- YAML config backward compatibility
- Roundtrip conversions

### Running Tests

```bash
# Quick validation
uv run python -c "
from skyrl_train.config.configs import create_default_config
from examples.gsm8k.run_gsm8k import get_gsm8k_config
cfg = get_gsm8k_config()
print('✓ Config creation works')
"

# Full integration test
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --extra vllm examples/gsm8k/test_run_gsm8k.py
```

---

## 🎓 Design Decisions

### Why Pydantic v2?
- Industry standard for Python configuration
- Excellent performance (Rust core)
- Rich validation capabilities
- Great IDE support

### Why Keep YAML Support?
- Backward compatibility is critical
- Many users prefer YAML for simplicity
- Existing CI/CD pipelines depend on it
- Gradual migration is safer

### Why Normalize to DictConfig Internally?
- Minimal changes to existing codebase
- Validation logic already works with DictConfig
- Reduces risk of bugs during transition
- Can be refactored later to use Pydantic natively

### Why Compositional Config Building?
- Inspired by RLlib's API design
- More readable than mutation-based style
- Type checkers can validate the entire tree
- Easier to test individual components

---

## 📝 Migration Guide

### Converting a `.sh` Script to `.py`

**Before (run_example.sh):**
```bash
uv run -m skyrl_train.entrypoints.main_base \
    data.train_data="['$DATA_DIR/train.parquet']" \
    trainer.policy.model.path="Qwen/Qwen2.5-1.5B" \
    trainer.epochs=20 \
    generator.backend=vllm
```

**After (run_example.py):**
```python
from skyrl_train.config.configs import (
    SkyRLConfig, DataConfig, TrainerConfig, GeneratorConfig,
    # ... import needed configs
)
from skyrl_train.entrypoints.main_base import run_training
import os

data_dir = os.getenv("DATA_DIR", "~/data")

cfg = SkyRLConfig(
    data=DataConfig(
        train_data=[f"{data_dir}/train.parquet"],
    ),
    trainer=TrainerConfig(
        policy=PolicyConfig(
            model=ModelConfig(path="Qwen/Qwen2.5-1.5B"),
        ),
        epochs=20,
    ),
    generator=GeneratorConfig(
        backend="vllm",
    ),
)

run_training(cfg)
```

---

## 🤝 Contributing

When adding new config fields:

1. Add to appropriate Pydantic model in `configs.py`
2. Add to YAML config for backward compatibility
3. Update type hints and defaults
4. Add to documentation
5. Test both YAML and Python paths

---

## 📚 References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Hydra Documentation](https://hydra.cc/)
- [RLlib Config API](https://docs.ray.io/en/latest/rllib/getting-started.html)

---

**Last Updated:** 2026-01-25
**Maintained By:** SkyRL Team
