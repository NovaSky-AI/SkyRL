"""SDPO/SRPO on the original SDPO Section 3 science and tool-use splits.

All recipe hyperparameters are set by run_sdpo.sh / run_srpo.sh. This entrypoint
only registers the original reward functions and filters prompts with the same
(non-thinking) chat template used for generation.
"""

import datasets

from skyrl.train.dataset import PromptDataset

from ..main_sdpo import SDPOExp, main
from .envs import register_sdpo_envs


class TemplatePromptDataset(PromptDataset):
    """Length-filter prompts with the generator's chat_template_kwargs applied."""

    def __init__(self, *args, template_kwargs=None, **kwargs):
        self.template_kwargs = template_kwargs or {}
        super().__init__(*args, **kwargs)

    def _read_files_and_tokenize(self):
        sources = self.datasets if isinstance(self.datasets, list) else [self.datasets]
        self.dataframe = datasets.concatenate_datasets(
            [datasets.load_dataset("json", data_files=source, split="train", keep_in_memory=True) for source in sources]
        )
        self.dataframe = self.dataframe.filter(
            lambda row: len(
                self.tokenizer.apply_chat_template(
                    row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False, **self.template_kwargs
                )
            )
            <= self.max_prompt_length,
            desc="Filtering prompts longer than max_prompt_length",
        )


class ReproductionExp(SDPOExp):
    @classmethod
    def setup(cls, cfg):
        register_sdpo_envs()

    def _dataset(self, paths):
        return TemplatePromptDataset(
            datasets=paths,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=1,
            template_kwargs=self.cfg.generator.chat_template_kwargs,
        )

    def get_train_dataset(self):
        return self._dataset(self.cfg.data.train_data)

    def get_eval_dataset(self):
        return self._dataset(self.cfg.data.val_data) if self.cfg.trainer.eval_interval > 0 else None


if __name__ == "__main__":
    main(ReproductionExp)
