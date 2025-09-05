"""
uv run --extra dev --extra vllm --isolated pytest tests/gpu/test_eval_only.py
"""

import json
import ray

from skyrl_train.entrypoints.eval_only import EvalPPOExp, TRAIN_METRICS_KEY, EVAL_METRICS_KEY
from skyrl_train.utils.utils import initialize_ray
from tests.gpu.utils import get_test_actor_config, get_test_generator_input


def create_dataset(tmp_path, model_name: str):
    input_batch = get_test_generator_input(model=model_name, num_prompts=1, n_samples_per_prompt=1)
    data = {
        "prompt": input_batch["prompts"][0],
        "env_class": input_batch["env_classes"][0],  # defaults to "gsm8k"
        "data_source": "test",
        **(input_batch["env_extras"][0] or {}),
    }
    path = tmp_path / "data.json"
    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")
    return str(path)


def test_eval_only(tmp_path):
    cfg = get_test_actor_config()

    # Minimize resources and side effects
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.export_path = str(tmp_path)
    cfg.trainer.ckpt_path = str(tmp_path)
    cfg.trainer.dump_eval_results = False

    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.sampling_params.max_generate_length = 4
    cfg.generator.eval_sampling_params.max_generate_length = 4
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.generator.async_engine = True

    cfg.environment.skyrl_gym.max_env_workers = 1

    initialize_ray(cfg)
    try:
        data_path = create_dataset(tmp_path, cfg.trainer.policy.model.path)
        cfg.data.train_data = [data_path]
        cfg.data.val_data = [data_path]
        cfg.trainer.train_batch_size = 1
        cfg.trainer.eval_batch_size = 1
        cfg.trainer.eval_interval = 1

        exp = EvalPPOExp(cfg)
        metrics = exp.run()

        assert TRAIN_METRICS_KEY in metrics, f"Train metrics not found in {metrics}"
        assert EVAL_METRICS_KEY in metrics, f"Eval metrics not found in {metrics}"
        assert isinstance(
            metrics[TRAIN_METRICS_KEY], dict
        ), f"Train metrics is not a dict: {metrics[TRAIN_METRICS_KEY]}"
        assert isinstance(metrics[EVAL_METRICS_KEY], dict), f"Eval metrics is not a dict: {metrics[EVAL_METRICS_KEY]}"
    finally:
        ray.shutdown()
