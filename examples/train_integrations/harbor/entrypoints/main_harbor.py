"""
Main entrypoint for training on Harbor tasks.
"""

import sys

import ray
import hydra
from loguru import logger
from omegaconf import DictConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from ..harbor_generator import HarborGenerator
from ..dataset import HarborTaskDataset


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = HarborExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    import signal
    import time

    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ref = skyrl_entrypoint.remote(cfg)

    # Ray's C code sets SIGINT to SIG_IGN during initialization, which
    # prevents Python from ever seeing KeyboardInterrupt.  We MUST
    # reinstall handlers AFTER initialize_ray() to restore them.
    # For SIGTERM we also install a handler since Ray's C++ crash handler
    # intercepts it otherwise.  See harbor#656 / SkyRL#1160.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda s, f: signal.default_int_handler(s, f))

    try:
        # Poll with time.sleep() (interruptible by signals) + non-blocking
        # ray.wait().  We cannot use ray.get() or ray.wait(timeout=N>0)
        # because those block in C code and may re-mask SIGINT.
        while True:
            time.sleep(1)
            # Reinstall handlers in case ray.wait() overrode them
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, lambda s, f: signal.default_int_handler(s, f))
            ready, _ = ray.wait([ref], timeout=0)
            if ready:
                ray.get(ref)  # immediate — task already finished
                break
    except KeyboardInterrupt:
        # Explicitly cancel the remote task so the worker receives
        # KeyboardInterrupt, allowing asyncio.run() to cancel all async
        # tasks (e.g. Harbor Trial.run()) and trigger their cleanup
        # (sandbox teardown).  Then WAIT for the worker to finish cleanup
        # before exiting — otherwise the driver exits immediately and
        # sandboxes are leaked.  See harbor#656 / SkyRL#1160.
        #
        # Temporarily ignore signals: when Ctrl+C sends SIGINT to the
        # process group, `uv` also dies and sends SIGTERM to us.  Without
        # this, the SIGTERM immediately triggers a "second interrupt" before
        # cleanup can start.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        logger.info("KeyboardInterrupt: cancelling worker, waiting for sandbox cleanup...")
        ray.cancel(ref, force=False)
        # Brief pause to let uv's dying SIGTERM be absorbed while ignored
        time.sleep(0.5)
        # Restore handlers so a deliberate second Ctrl+C can force-kill
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, lambda s, f: signal.default_int_handler(s, f))
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    time.sleep(1)
                    signal.signal(signal.SIGINT, signal.default_int_handler)
                    ready, _ = ray.wait([ref], timeout=0)
                    if ready:
                        try:
                            ray.get(ref)
                        except Exception:
                            pass  # Expected: RayTaskError from cancelled task
                        break
                except KeyboardInterrupt:
                    logger.warning("Second interrupt: force-killing worker (sandboxes may leak)...")
                    ray.cancel(ref, force=True)
                    break
        except Exception:
            pass
        logger.info("Cleanup complete.")
        sys.exit(130)  # 128 + SIGINT(2)


if __name__ == "__main__":
    main()
