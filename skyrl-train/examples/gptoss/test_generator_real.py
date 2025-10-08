import pytest
from omegaconf import OmegaConf
from unittest.mock import AsyncMock, MagicMock
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from gpt_oss_generator_step_wise import GPTOSSGenerator
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.generators.base import GeneratorInput, TrajectoryID
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
import os
from transformers import AutoTokenizer

TEST_DATA_PATH = "dummy"


def get_test_generator_input(
    model: str,
    num_prompts: int = 20,
    n_samples_per_prompt: int = 1,
    max_prompt_length: int = 512,
    data_path: str = TEST_DATA_PATH,
    env_class: str = "gsm8k",
):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Ensure pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PromptDataset(
        datasets=[data_path],
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )

    prompts = []
    env_extras = []
    for i in range(min(num_prompts, len(dataset))):
        prompt_data, _, env_extra, _ = dataset[i]  # dataset returns (messages, env_class, extra, uid)
        prompts.extend([prompt_data] * n_samples_per_prompt)
        env_extras.extend([env_extra] * n_samples_per_prompt)

    env_classes = [env_class] * len(prompts)

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_classes": env_classes,
        "env_extras": env_extras,
    }

    return input_batch


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

assistant_message = """<|channel|>analysis<|message|>Compute 15% of 220 = 33. So new price = 220+33=253.<|end|><|start|>assistant<|channel|>final<|message|>The price increased by 15% means it is multiplied by \(1 + 0.15 = 1.15\).

\[
220 \times 1.15 = 220 \times \left(\frac{115}{100}\right)
               = 220 \times 1.15
               = 220 + 33
               = 253
\]

The new selling price is **$253**.

####<|return|>"""
MOCK_LLM_OUTPUT_IDS = tokenizer.encode(assistant_message)


def create_remote_engines(cfg, tokenizer):
    inference_engines = create_remote_inference_engines(
        urls=cfg.generator.remote_inference_engine_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=cfg.generator.backend,
        tokenizer=tokenizer,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
    )
    return inference_engines


@pytest.fixture
def mock_env():
    mock_env_instance = MagicMock()
    mock_env_instance.step.side_effect = lambda x: BaseTextEnvStepOutput(
        observations=[{"role": "user", "content": "next"}], reward=1.0, done=True, metadata={}
    )
    mock_env_instance.close.return_value = None
    return mock_env_instance


@pytest.fixture
def mock_cfg():
    cfg = get_default_config()
    OmegaConf.update(
        cfg,
        "generator",
        {
            "remote_inference_engine_urls": ["127.0.0.1:8001"],
            "run_engines_locally": False,
            "apply_overlong_filtering": False,
            "max_input_length": 32000,
            "batched": False,
            "max_turns": 1,
            "chat_template_kwargs": {},
            "chat_template": {"source": "name", "name_or_path": None},
            "use_conversation_multi_turn": True,
        },
    )
    cfg.generator.sampling_params.max_generate_length = 4096
    cfg.generator.sampling_params.stop = ["</sql>", "</solution>"]
    cfg.trainer.policy.model.path = MODEL
    return cfg


@pytest.fixture
def mock_llm():
    """
    This replaces InferenceEngineClient, where `.generate()` always returns MOCK_LLM_OUTPUT_IDS
    for each prompt, with corresponding string output "mocked output".
    """
    mock = MagicMock()

    # Mock the new generate method
    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        return {
            "responses": ["mocked output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            # say response gets tokenized to 3 tokens
            "response_logprobs": [[0.1] * len(MOCK_LLM_OUTPUT_IDS)] * num_prompts,
            "response_ids": [MOCK_LLM_OUTPUT_IDS.copy()] * num_prompts,
        }

    mock.generate = AsyncMock(side_effect=mock_generate)
    return mock


@pytest.mark.asyncio
async def test_generate_single_turn(mock_cfg, mock_llm):
    env_config = mock_cfg.environment.skyrl_gym
    inference_engines = create_remote_engines(mock_cfg, tokenizer)
    inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, mock_cfg)

    generator = GPTOSSGenerator(
        generator_cfg=mock_cfg.generator,
        skyrl_gym_cfg=env_config,
        inference_engine_client=inference_engine_client,
        tokenizer=tokenizer,
        model_name="test_model",
    )

    output = await generator.agent_loop(
        prompt=[{"role": "user", "content": "What is 2 + 2?"}],
        env_class="gsm8k",
        env_extras={"reward_spec": {"method": "rule", "ground_truth": "253"}},
        max_tokens=8000,
        max_input_length=512,
        sampling_params=get_sampling_params_for_backend(mock_cfg.generator.backend, mock_cfg.generator.sampling_params),
    )
    print(output)


@pytest.mark.asyncio
async def test_generate_multi_turn(mock_cfg):
    mock_cfg.generator.max_turns = 5
    env_config = mock_cfg.environment.skyrl_gym
    env_config.text2sql.db_path = os.path.expanduser("~/data/sql/db_files/data")
    inference_engines = create_remote_engines(mock_cfg, tokenizer)
    inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, mock_cfg)

    generator = GPTOSSGenerator(
        generator_cfg=mock_cfg.generator,
        skyrl_gym_cfg=env_config,
        inference_engine_client=inference_engine_client,
        tokenizer=tokenizer,
        model_name=MODEL,
    )
    n_samples_per_prompt = 1
    num_prompts = 1

    dataset = get_test_generator_input(
        model=MODEL,
        num_prompts=num_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        max_prompt_length=4096,
        data_path=os.path.expanduser("~/data/sql/validation.parquet"),
        env_class="text2sql",
    )
    dataset["sampling_params"] = get_sampling_params_for_backend(
        mock_cfg.generator.backend, mock_cfg.generator.sampling_params
    )
    dataset["trajectory_ids"] = [
        TrajectoryID(instance_id=i // n_samples_per_prompt, repetition_id=i % n_samples_per_prompt)
        for i in range(n_samples_per_prompt * num_prompts)
    ]
    outputs = await generator.generate(dataset)
    # output = await generator.agent_loop(
    #     prompt=[{"role": "user", "content": "What is 2 + 2?"}],
    #     env_class="gsm8k",
    #     env_extras={"reward_spec": {"method": "rule", "ground_truth": "253"}},
    #     max_tokens=8000,
    #     max_input_length=512,
    #     sampling_params=get_sampling_params_for_backend(mock_cfg.generator.backend, mock_cfg.generator.sampling_params),
    # )
    responses = tokenizer.batch_decode(outputs["response_ids"])
    print(outputs)
    print(responses)
