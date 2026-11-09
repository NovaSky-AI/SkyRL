"""Fully-async training on Harbor tasks through skycap.

``main_harbor_skycap`` with ``FullyAsyncRayPPOTrainer``, as ``main_harbor_fully_async`` is for the
sibling. With ``trainer.fully_async.store_groups_in_ray=true`` each group's rows stay in the object
store of the node whose skycap server composed them until they are trained on.
"""

import sys

import ray
import yaml

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.entrypoints.main_harbor import HARBOR_DEFAULT_CONFIG, _deep_merge
from ..record_index import SkycapRecordIndex
from .main_harbor_skycap import HarborSkycapConfig, HarborSkycapExp, _exposure


class HarborSkycapFullyAsyncExp(HarborSkycapExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        trainer = FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )
        if self.records is not None:
            trainer.add_callback(SkycapRecordIndex(self.records, self.cfg.skycap.wandb.phases))
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapFullyAsyncExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    _exposure(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for Harbor training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
