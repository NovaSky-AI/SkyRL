# utility functions used for CPU tests

from omegaconf import OmegaConf


def example_dummy_config():
    return OmegaConf.create(
        {
            "trainer": {
                "project_name": "unit-test",
                "run_name": "test-run",
                "logger": "tensorboard",
                "micro_train_batch_size_per_gpu": 2,
                "train_batch_size": 2,
                "eval_batch_size": 2,
                "update_epochs_per_batch": 1,
                "epochs": 1,
                "max_prompt_length": 20,
                "gamma": 0.99,
                "lambd": 0.95,
                "use_sample_packing": False,
                "seed": 42,
                "algorithm": {
                    "advantage_estimator": "grpo",
                    "use_kl_estimator_k3": False,
                    "use_abs_kl": False,
                    "kl_estimator_type": "k1",
                    "reward_clip_range": 5.0,
                    "use_kl_loss": True,
                    "kl_loss_coef": 0.0,
                    "lambd": 1.0,
                    "gamma": 1.0,
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "clip_ratio_c": 3.0,
                    "value_clip": 0.2,
                    "policy_loss_type": "regular",
                    "loss_reduction": "token_mean",
                    "grpo_norm_by_std": True,
                },
                "resume_mode": "none",
            },
            "generator": {
                "max_generate_length": 20,
                "n_samples_per_prompt": 1,
                "batched": False,
                "env_class": "gsm8k",
                "max_turns": 1,
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 8000,
            },
        }
    )
