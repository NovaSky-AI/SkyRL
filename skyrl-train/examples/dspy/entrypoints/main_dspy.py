"""
Main entrypoint for training on dspy tasks.
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from examples.dspy.dspy_generator import DSPyGenerator
from examples.dspy.dataset import DSPyDataset


class DSPyExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
      
        return DSPyGenerator(
            generator_cfg=cfg.generator,
            dspy_cfg=cfg.dspy,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            TerminalBenchTaskDataset: The training dataset.
        """
        prompts_dataset = DSPyDataset(
            benchmark_name=self.cfg.dspy.benchmark_name,
            max_num_examples=self.cfg.dspy.max_num_examples,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            TerminalBenchTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = DSPyDataset(
                data_file=self.cfg.data.val_data,
                max_num_examples=self.cfg.data.max_num_examples,
            )   
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = DSPyExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()