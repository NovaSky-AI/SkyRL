# Move to pythonic configurations in SkyRL

[Sumanth Hegde](mailto:sumanthrh@anyscale.com)  
Jan 30, 2026

## Summary

This short doc focuses on configuration and instantiation in SkyRL. Currently, we use Hydra for config management and while it has served us well, it is very CLI / YAML centric and lacks structured typing. Further, our current configuration structure and instantiation APIs for different components simply passes the full training configuration, which tightly couples different components (ex: `Worker` needs full training cfg). 

## Proposal

The proposal is to (a) migrate to configuration dataclasses everywhere and (b) refactor current abstractions to be instantiated from these standalone configs.

In summary: 

1. Move towards python dataclasses that capture configs for our internal components.   
2. Configurations should be nested in a way that the corresponding component can be instantiated without depending on information from another component (ex: Requiring the generator configuration to instantiate a PolicyWorker for SFT is not good).  
3.  The CLI experience will allow for simple overrides and move away from the heavyweight hydra-based experience at the moment.

## Design Overview

### Configurations

The configurations will mostly mimic the current hierarchies with a few changes:

1. TrainerConfig  
   1. PlacementConfig  
   2. PolicyConfig  
      1. ModelConfig  
      2. OptimizerConfig  
      3. FSDPConfig  
      4. MegatronConfig  
   3. AlgorithmConfig  
      1. SAPOConfig  
      2. …  
   4. …  
2. GeneratorConfig  
   1. InferenceEngineConfig  
3. EnvironmentConfig

InferenceEngineConfig is the only new configuration hierarchy introduced, and this will contain all engine instantiation related parameters (including number of inference engines, TP/PP, etc)

Dataclasses in detail:

```py
@dataclass
class GeneratorConfig:
    inference_engine: InferenceEngineConfig
    max_turns: int
    chat_template: str
    sampling_params: SamplingParams
    eval_sampling_params: SamplingParams
    use_conversation_multi_turn: bool
    n_samples_per_prompt: int
    eval_n_samples_per_prompt: int

@dataclass
class TrainerConfig:
    strategy: StrategyEnum
    placement: PlacementConfig
    policy: PolicyConfig       # contains ModelConfig, OptimizerConfig, FSDPConfig, MegatronConfig, ulysses_sequence_parallel_size
    critic: CriticConfig       # contains ModelConfig, OptimizerConfig, FSDPConfig, MegatronConfig, ulysses_sequence_parallel_size
    ref: RefConfig             # contains ModelConfig, FSDPConfig, MegatronConfig, ulysses_sequence_parallel_size
    algorithm: AlgorithmConfig # contains policy_mini_batch_size, critic_mini_batch_size, temperature (derived), ...
    use_torch_compile: bool = False
    record_memory: bool = False
```

```py
@dataclass
class ModelConfig:
    model_path: str
    lora_config: Optional[LoraConfig] = None
    model_config_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoraConfig:
    rank: int = 0
    alpha: int = 16
    dropout: float = 0.0
    lora_sync_path: str = "/tmp/skyrl_lora_sync"
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None
    # FSDP: maps to `init_lora_weights` in PEFT
    # Megatron: maps to `lora_A_init_method`; supports "xavier", "normal", "kaiming", "zero"
    init_method: str = "kaiming"

@dataclass
class InferenceEngineConfig:
    model_dtype: str = "bfloat16"
    run_engines_locally: bool = True
    num_inference_engines: int = 1
    backend: str = "vllm"
    weight_sync_backend: str = "nccl"
    weight_transfer_threshold_cuda_ipc_GB: float = 1.0
    inference_engine_tensor_parallel_size: int = 4
    inference_engine_pipeline_parallel_size: int = 1
    inference_engine_expert_parallel_size: int = 1
    inference_engine_data_parallel_size: int = 1
    n_samples_per_prompt: int = 5
    async_engine: bool = True
    batched: bool = False
    max_input_length: int = 0  # defaults to max_prompt_length; resolved during config validation
    vllm_v1_disable_multiproc: bool = True
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    enforce_eager: bool = True
    fully_sharded_loras: bool = False
    gpu_memory_utilization: float = 0.8
    max_num_seqs: int = 1024
    remote_inference_engine_urls: List[str] = field(default_factory=lambda: ["127.0.0.1:8001"])
    enable_http_endpoint: bool = False
    http_endpoint_host: str = "127.0.0.1"
    http_endpoint_port: int = 8000
    engine_init_kwargs: Dict[str, Any] = field(default_factory=dict)

```

### 

### Instantiation APIs

**InferenceEngineClient**

`InferenceEngineClient.__init__(self, engines: List[InferenceEngineInterface], tokenizer: PreTrainedTokenizerBase, full_config: DictConfig) -> InferenceEngineClient` 

`InferenceEngineClient.__init__(self, engines: List[InferenceEngineInterface], tokenizer: PreTrainedTokenizerBase, model_path: str, lora_cfg: LoRAConfig, inference_engine_config: InferenceEngineConfig) -> InferenceEngineClient` 

**SkyRLGymGenerator**

`SkyRLGymGenerator.__init__(self, generator_cfg: DictConfig, skyrl_gym_cfg: DictConfig, inference_engine_client: InferenceEngineClient, tokenizer, model_name: str) -> SkyRLGymGenerator`

`SkyRLGymGenerator.__init__(self, model_path: str, lora_config: LoRAConfig, generator_cfg: GeneratorConfig, skyrl_gym_cfg: SkyRLGymConfig, inference_engine_client: InferenceEngineClient) -> SkyRLGymGenerator`

```py
class RayPPOTrainer:
    def __init__(
        self,
        trainer_cfg: TrainerConfig,
        tracker: Tracking,
        tokenizer: AutoTokenizer,
        train_dataset: Optional[PromptDataset],
        inference_engine_client: InferenceEngineClient,
        generator: GeneratorInterface,
        colocate_pg: Optional[PlacementGroup] = None,
        eval_dataset: Optional[PromptDataset] = None,
    ):
	pass

class Worker(self, trainer_cfg: TrainerConfig, world_size, rank, local_rank, master_addr, master_port, record_memory=False)
```

### New CLI Experience

The new CLI experience from the user’s point of view will remain similar to the current setup: 

```
python main_base.py trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" generator.backend=vllm
```

However, a key difference is that it will not support additional kwargs that hydra supports with `+`  

Internally, we do the following:

- Entrypoint script allows for any and all user arguments   
- User arguments are parsed via OmegaConf  
- Resolved dictionary from OmegaConf is used to override the default `SkyRLConfig` parameters

Reading the config from the CLI would be supported with

```
SkyRLConfig.from_cli_overrides(args: List[str]) -> SkyRLConfig
```

### Extra fields

Users introduce new generators, new environments, new trainer and want to add additional fields to the existing config objects for their new object. With the new design, the canonical way is for the user to define a subclass of the old config and extend it in the way they prefer

```
class MySkyRLConfig(SkyRLConfig):
	newfield1: str = "newvalue1"
```

And then use it in the script as needed

Customization flow:

```
class MySkyRLConfig(SkyRLConfig):
	newfield1: str = "newvalue1"

class MyBasePPOExp(BasePPOExp):
	@property
	def get_config_cls():
		return MySkyRLConfig
```

```
class MyTrainerConfig(TrainerConfig):
	newfield1: str = "newvalue1"

class MySkyRLConfig(SkyRLConfig):
	trainer: MyTrainerConfig
```

This ensures type hints work well with new scripts. 

**Design notes:**

- Policy worker needs weight sync, so it takes `InferenceEngineClient` / `InferenceEngineConfig`. This config carries more parameters than strictly needed for weight sync (num engines, backend, parallelism sizes, etc.), but extracting a separate `WeightSyncConfig` isn't worth the complexity right now.  
- `n_samples_per_prompt`: This is a generator field currently used in the policy worker implementation. In the new design this will be used to update `policy/critic_mini_batch_size` on `TrainerConfig` during config validation, so workers don't need to know about generation details.  
- `temperature`: passed as an explicit arg to the trainer in the algorithm config— while training with SkyRL, we set this automatically based on generator sampling params \- but the worker components can be used independently, in which case this parameter is needed for scaling logits appropriately

## Comments

- `model` and `lora_path` arguments to the generator extracted from TrainerConfig  
  *A*  
  *Reason:* Model information is best left in one place \- either generator or TrainerConfig or placed at a higher level. It is more natural for the policy configuration to have the full details at init time to avoid config magic

- Eval , checkpointing and logging are still flat;  
  *Alternative*: Have separate config objects for each  
  *Reason*: Simplicity \- it is preferred to have as little nesting as possible  
    
- BasePPOExp \+ RayPPOTrainer structure is left as-is  
  *Reason:* Some of the major complaints on customization have been related to the current hydra-based instantiation and should go away once we move to python based configs.

- Configs as simple dataclasses  
  *Alternative:* Use Pydantic  
  *Reason:* I believe simple dataclasses get the job done here. While we do lose runtime validation I think the *library* experience is better with simple dataclasses (can hack with it in a bunch of ways) similar to configs in vLLM and Transformers.

- Config extensions are straightforward but can lead to modifications for all the parent classes (ex: changing AlgorithmConfig requires defining custom SkyRLConfig, TrainerConfig, etc)  
  *Alternative:* Provide some config extension utility (like WorkerExtension in vLLM) to reduce the number of steps  
  *Reason:* This is more idiomatic and is preferred solely for better typing 

## Migration Plan

**Stage 1: Introduce only backwards-compatible configuration dataclasses for the current YAML; Migrate Tests**

- Introduce all the key configuration classes \- PolicyConfig, ModelConfig, AlgorithmConfig, PlacementConfig, etc.  
- Omit refactoring any configuration hierarchy changes that break backwards compatibility (ex: model\_config\_kwargs)  
- Entrypoint scripts continue to use generic DictConfig \- only migrate tests to new dataclasses.   
- Existing user scripts don’t break  
- Make a release announcing planned future changes

**Stage 2: Finish configuration refactoring; Change instantiation signatures** 

- Complete migration to the new configuration structure  
- New instantiation signatures with fewer cross-config dependencies
- Keep backwards compatibility for old configs \- internally translate older configuration objects to the new configuration  
- Migrate all examples/ to the new CLI and config classes  
- Use OmegaConf for CLI parsing

**Out of scope:** Trainer API subclassing improvements (proposal item 2), Generator API utilities (item 3), and profiling tooling (item 4\) are deferred to future work.  
